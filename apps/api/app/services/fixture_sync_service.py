"""Idempotent synchronisation of upcoming fixtures from football-data.org.

Running a sync twice must leave the database in the same state as running it
once. Two natural keys make that possible: `teams.provider_id` and
`fixtures.provider_id`, both backed by unique indexes. Club names that differ
between sources are resolved through `team_aliases` rather than by string
matching at read time.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.clients.football_data_client import Match, parse_matches
from app.clients.football_data_client import Team as ProviderTeam
from app.models.data_import import DataImport
from app.models.fixture import Fixture
from app.models.team import Team
from app.models.team_alias import TeamAlias
from app.repositories.fixture_repository import SQLAlchemyFixtureRepository
from app.services.fixture_import_service import PROVIDER, FixtureImportService

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Outcome of one synchronisation run."""

    matches_read: int = 0
    teams_created: int = 0
    teams_updated: int = 0
    fixtures_created: int = 0
    fixtures_updated: int = 0
    aliases_created: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return self.fixtures_created + self.fixtures_updated

    def as_dict(self) -> dict[str, Any]:
        return {
            "matches_read": self.matches_read,
            "teams_created": self.teams_created,
            "teams_updated": self.teams_updated,
            "fixtures_created": self.fixtures_created,
            "fixtures_updated": self.fixtures_updated,
            "aliases_created": self.aliases_created,
            "rejected": self.rejected,
        }


def payload_checksum(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonicalised payload, so an unchanged source is detectable."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class FixtureSyncService:
    """Writes provider matches into `teams`, `team_aliases`, and `fixtures`."""

    def __init__(self, session: Session, source: str = PROVIDER):
        self.session = session
        self.source = source
        self.fixtures = SQLAlchemyFixtureRepository(session)
        self.converter = FixtureImportService()

    def sync_payload(
        self, payload: dict[str, Any], source_uri: str, season_start_year: int | None = None
    ) -> SyncReport:
        """Synchronise a raw `/matches` response body and record an audit row."""
        matches = parse_matches(payload)
        report = self.sync_matches(matches)
        report.matches_read = len(payload.get("matches", []))

        unparsed = report.matches_read - len(matches)
        if unparsed > 0:
            report.rejected.append({"reason": "unparseable_payload", "count": unparsed})

        self.session.add(
            DataImport(
                source=self.source,
                source_uri=source_uri,
                season_start_year=season_start_year
                or (matches[0].season_start_year if matches else None),
                checksum=payload_checksum(payload),
                rows_read=report.matches_read,
                rows_accepted=report.accepted,
                rows_rejected=report.matches_read - report.accepted,
                report_json=report.as_dict(),
            )
        )
        self.session.commit()
        return report

    def sync_matches(self, matches: list[Match]) -> SyncReport:
        """Upsert every team and fixture in `matches`."""
        report = SyncReport()

        team_ids: dict[int, int] = {}
        for match in matches:
            for provider_team in (match.home_team, match.away_team):
                if provider_team.id not in team_ids:
                    team_ids[provider_team.id] = self._sync_team(provider_team, report)

        for match in matches:
            self._sync_fixture(match, team_ids, report)

        return report

    def _sync_team(self, provider_team: ProviderTeam, report: SyncReport) -> int:
        provider_id = self.converter.build_team_provider_id(provider_team)
        existed = (
            self.session.query(Team).filter(Team.provider_id == provider_id).one_or_none()
            is not None
        )

        team = self.fixtures.upsert_team(self.converter.to_team_dict(provider_team))
        if existed:
            report.teams_updated += 1
        else:
            report.teams_created += 1

        if self._register_alias(team.id, provider_team.name):
            report.aliases_created += 1

        return team.id

    def _register_alias(self, team_id: int, alias: str) -> bool:
        """Record the provider's spelling for a team, returning True if it is new.

        An alias already claimed by a different team is left untouched and
        logged: silently repointing it would corrupt every historical feature
        derived from that club.
        """
        existing = (
            self.session.query(TeamAlias)
            .filter(TeamAlias.source == self.source, TeamAlias.alias == alias)
            .one_or_none()
        )
        if existing is None:
            self.session.add(TeamAlias(team_id=team_id, source=self.source, alias=alias))
            self.session.flush()
            return True

        if existing.team_id != team_id:
            logger.warning(
                "Alias %r from %s already maps to team %d; refusing to repoint it to %d.",
                alias,
                self.source,
                existing.team_id,
                team_id,
            )
        return False

    def _sync_fixture(self, match: Match, team_ids: dict[int, int], report: SyncReport) -> None:
        provider_id = self.converter.build_fixture_provider_id(match)
        home_team_id = team_ids.get(match.home_team.id)
        away_team_id = team_ids.get(match.away_team.id)

        if home_team_id is None or away_team_id is None:
            report.rejected.append({"reason": "unresolved_team", "provider_id": provider_id})
            return

        already_present = self._fixture_exists(provider_id)
        self.fixtures.upsert_fixture(
            self.converter.to_fixture_dict(match, home_team_id, away_team_id)
        )
        if already_present:
            report.fixtures_updated += 1
        else:
            report.fixtures_created += 1

    def _fixture_exists(self, provider_id: str) -> bool:
        return (
            self.session.query(Fixture.id).filter(Fixture.provider_id == provider_id).first()
            is not None
        )
