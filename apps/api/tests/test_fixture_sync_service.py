"""Fixture synchronisation driven by a stored provider response.

Uses the same JSON the client tests use, so the importer is exercised against a
realistic payload without network access. Runs against real PostgreSQL because
idempotency is a property of the unique indexes, not of the Python code.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.models import DataImport, Fixture, FixtureStatus, Team, TeamAlias
from app.services.fixture_sync_service import FixtureSyncService, payload_checksum

STORED_RESPONSE = Path(__file__).parent / "fixtures" / "football_data_matches.json"
SOURCE_URI = "https://api.football-data.org/v4/competitions/PL/matches?season=2026"


@pytest.fixture
def payload() -> dict[str, Any]:
    return json.loads(STORED_RESPONSE.read_text())


@pytest.fixture
def service(db_session: Session) -> FixtureSyncService:
    return FixtureSyncService(db_session)


def test_sync_imports_teams_and_fixtures(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    report = service.sync_payload(payload, source_uri=SOURCE_URI, season_start_year=2026)

    assert report.matches_read == 3
    assert report.fixtures_created == 3
    assert report.teams_created == 6
    assert db_session.query(Fixture).count() == 3
    assert db_session.query(Team).count() == 6


def test_sync_uses_provider_names_as_canonical_names(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    service.sync_payload(payload, source_uri=SOURCE_URI)

    team = db_session.query(Team).filter(Team.provider_id == "football-data-57").one()

    assert team.canonical_name == "Arsenal FC"
    assert team.short_name == "Arsenal"


def test_sync_registers_a_provider_alias_per_team(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    """The provider's spelling is recorded so later sources can be reconciled."""
    service.sync_payload(payload, source_uri=SOURCE_URI)

    alias = (
        db_session.query(TeamAlias)
        .filter(TeamAlias.source == "football-data.org", TeamAlias.alias == "Arsenal FC")
        .one()
    )

    assert alias.team.provider_id == "football-data-57"
    assert db_session.query(TeamAlias).count() == 6


def test_sync_maps_fixture_details(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    service.sync_payload(payload, source_uri=SOURCE_URI)

    fixture = db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14621").one()

    assert fixture.competition_code == "PL"
    assert fixture.season_start_year == 2026
    assert fixture.matchday == 4
    assert fixture.status is FixtureStatus.SCHEDULED
    assert fixture.home_team.canonical_name == "Arsenal FC"
    assert fixture.away_team.canonical_name == "Chelsea FC"
    assert fixture.kickoff_at.isoformat() == "2026-09-12T14:00:00+00:00"


def test_sync_stores_final_scores_and_results(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    service.sync_payload(payload, source_uri=SOURCE_URI)

    away_win = db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14622").one()
    draw = db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14623").one()

    assert (away_win.home_score, away_win.away_score, away_win.result) == (1, 3, "A")
    assert away_win.status is FixtureStatus.FINISHED
    assert (draw.home_score, draw.away_score, draw.result) == (2, 2, "D")


def test_sync_is_idempotent(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    """Running the same payload twice must not duplicate a single row."""
    service.sync_payload(payload, source_uri=SOURCE_URI)
    second = service.sync_payload(payload, source_uri=SOURCE_URI)

    assert second.fixtures_created == 0
    assert second.fixtures_updated == 3
    assert second.teams_created == 0
    assert second.teams_updated == 6
    assert second.aliases_created == 0
    assert db_session.query(Fixture).count() == 3
    assert db_session.query(Team).count() == 6
    assert db_session.query(TeamAlias).count() == 6


def test_resync_applies_provider_updates(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    """A rescheduled or completed match must be updated in place."""
    service.sync_payload(payload, source_uri=SOURCE_URI)

    payload["matches"][0]["matchday"] = 5
    payload["matches"][0]["status"] = "FINISHED"
    payload["matches"][0]["score"]["fullTime"] = {"home": 2, "away": 0}
    service.sync_payload(payload, source_uri=SOURCE_URI)

    fixture = db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14621").one()

    assert fixture.matchday == 5
    assert fixture.status is FixtureStatus.FINISHED
    assert (fixture.home_score, fixture.away_score, fixture.result) == (2, 0, "H")
    assert db_session.query(Fixture).count() == 3


def test_sync_records_an_audit_row(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    service.sync_payload(payload, source_uri=SOURCE_URI, season_start_year=2026)

    record = db_session.query(DataImport).one()

    assert record.source == "football-data.org"
    assert record.source_uri == SOURCE_URI
    assert record.season_start_year == 2026
    assert record.checksum == payload_checksum(payload)
    assert (record.rows_read, record.rows_accepted, record.rows_rejected) == (3, 3, 0)
    assert record.report_json["fixtures_created"] == 3


def test_sync_reports_unparseable_matches_without_failing(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    """One malformed match must be reported, not silently dropped or fatal."""
    payload["matches"].append({"id": 99, "status": "SCHEDULED"})

    report = service.sync_payload(payload, source_uri=SOURCE_URI)

    assert report.matches_read == 4
    assert report.fixtures_created == 3
    assert {"reason": "unparseable_payload", "count": 1} in report.rejected

    record = db_session.query(DataImport).one()
    assert (record.rows_read, record.rows_accepted, record.rows_rejected) == (4, 3, 1)


def test_sync_of_an_empty_payload_is_a_no_op(
    service: FixtureSyncService, db_session: Session
) -> None:
    report = service.sync_payload({"matches": []}, source_uri=SOURCE_URI)

    assert report.accepted == 0
    assert db_session.query(Fixture).count() == 0
    assert db_session.query(DataImport).count() == 1


def test_checksum_changes_when_the_payload_changes(payload: dict[str, Any]) -> None:
    """The checksum is what lets a later run detect a changed source."""
    before = payload_checksum(payload)
    payload["matches"][0]["matchday"] = 5

    assert payload_checksum(payload) != before


def test_checksum_is_insensitive_to_key_order() -> None:
    assert payload_checksum({"a": 1, "b": 2}) == payload_checksum({"b": 2, "a": 1})


def test_sync_does_not_repoint_an_alias_claimed_by_another_team(
    service: FixtureSyncService, db_session: Session, payload: dict[str, Any]
) -> None:
    """An ambiguous alias must be left alone rather than silently reassigned."""
    squatter = Team(provider_id="football-data-999", canonical_name="Some Other Club")
    db_session.add(squatter)
    db_session.flush()
    db_session.add(TeamAlias(team_id=squatter.id, source="football-data.org", alias="Arsenal FC"))
    db_session.flush()

    service.sync_payload(payload, source_uri=SOURCE_URI)

    alias = (
        db_session.query(TeamAlias)
        .filter(TeamAlias.source == "football-data.org", TeamAlias.alias == "Arsenal FC")
        .one()
    )
    assert alias.team_id == squatter.id
    # The fixture still resolves through the provider id, not the alias.
    fixture = db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14621").one()
    assert fixture.home_team.provider_id == "football-data-57"
