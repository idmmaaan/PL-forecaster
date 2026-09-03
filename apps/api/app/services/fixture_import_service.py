"""Converts football-data.org payloads into our domain models."""

from datetime import datetime
from typing import Any

from app.clients.football_data_client import Match
from app.clients.football_data_client import Team as FootballTeam
from app.models.fixture import Fixture, FixtureStatus
from app.models.team import Team

PROVIDER = "football-data.org"


class FixtureImportService:
    """Maps provider objects onto `Team` and `Fixture` rows."""

    def __init__(self) -> None:
        self.provider = PROVIDER

    def build_fixture_provider_id(self, match: Match) -> str:
        """Return the stable natural key used to deduplicate fixtures on re-import."""
        return f"pl-{match.season_start_year}-{match.id}"

    def build_team_provider_id(self, football_team: FootballTeam) -> str:
        """Return the stable natural key used to deduplicate teams on re-import."""
        return f"football-data-{football_team.id}"

    def convert_football_data_match_to_fixture(self, match: Match) -> Fixture:
        """Convert a provider match into a `Fixture`.

        Team foreign keys are left unset: they are resolved against the `teams`
        table during synchronisation, once the teams themselves are persisted.
        """
        return Fixture(
            provider=self.provider,
            provider_id=self.build_fixture_provider_id(match),
            competition_code=match.competition_code,
            season_start_year=match.season_start_year,
            matchday=match.matchday,
            kickoff_at=parse_kickoff(match.kickoff_at),
            status=FixtureStatus(match.status.value),
            home_score=match.home_score,
            away_score=match.away_score,
            result=match.result,
        )

    def convert_football_data_match_to_team(self, football_team: FootballTeam) -> Team:
        """Convert a provider team into a `Team`."""
        return Team(
            provider_id=self.build_team_provider_id(football_team),
            canonical_name=football_team.name,
            short_name=football_team.short_name,
            code=None,  # Not provided by football-data.org v4
            crest_url=football_team.crest_url,
        )

    def convert_football_data_matches_to_domain_models(
        self, matches: list[Match]
    ) -> tuple[list[Team], list[Fixture]]:
        """Convert many matches into unique teams plus one fixture per match."""
        team_lookup: dict[int, Team] = {}

        for match in matches:
            for football_team in (match.home_team, match.away_team):
                if football_team.id not in team_lookup:
                    team_lookup[football_team.id] = self.convert_football_data_match_to_team(
                        football_team
                    )

        fixtures = [self.convert_football_data_match_to_fixture(match) for match in matches]
        return list(team_lookup.values()), fixtures

    def to_team_dict(self, football_team: FootballTeam) -> dict[str, Any]:
        """Return upsert-ready column values for a provider team."""
        return {
            "provider_id": self.build_team_provider_id(football_team),
            "canonical_name": football_team.name,
            "short_name": football_team.short_name,
            "crest_url": football_team.crest_url,
        }

    def to_fixture_dict(self, match: Match, home_team_id: int, away_team_id: int) -> dict[str, Any]:
        """Return upsert-ready column values for a provider match."""
        return {
            "provider": self.provider,
            "provider_id": self.build_fixture_provider_id(match),
            "competition_code": match.competition_code,
            "season_start_year": match.season_start_year,
            "matchday": match.matchday,
            "kickoff_at": parse_kickoff(match.kickoff_at),
            "status": FixtureStatus(match.status.value),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "result": match.result,
        }


def parse_kickoff(value: str) -> datetime:
    """Parse the provider's UTC timestamp into a timezone-aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
