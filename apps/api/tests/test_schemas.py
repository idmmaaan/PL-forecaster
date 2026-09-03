from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.fixture import Fixture, FixtureStatus
from app.models.team import Team
from app.schemas.fixture import FixtureResponse
from app.schemas.team import TeamSummary


def fixture_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 14621,
        "competition_code": "PL",
        "season_start_year": 2026,
        "matchday": 4,
        "kickoff_at": datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC),
        "status": "SCHEDULED",
        "home_team": {"id": 1, "name": "Arsenal", "crest_url": None},
        "away_team": {"id": 2, "name": "Chelsea", "crest_url": None},
    }
    payload.update(overrides)
    return payload


def test_team_summary_schema() -> None:
    team = TeamSummary(id=1, name="Arsenal", crest_url="https://example.com/arsenal.png")

    assert team.id == 1
    assert team.name == "Arsenal"
    assert team.crest_url == "https://example.com/arsenal.png"


def test_team_summary_crest_url_is_optional() -> None:
    assert TeamSummary(id=1, name="Arsenal").crest_url is None


def test_fixture_response_schema() -> None:
    fixture = FixtureResponse(**fixture_payload())

    assert fixture.id == 14621
    assert fixture.competition_code == "PL"
    assert fixture.season_start_year == 2026
    assert fixture.matchday == 4
    assert fixture.status == "SCHEDULED"
    assert fixture.home_team.name == "Arsenal"
    assert fixture.away_team.name == "Chelsea"


def test_fixture_response_schema_with_crest_urls() -> None:
    fixture = FixtureResponse(
        **fixture_payload(
            home_team={"id": 1, "name": "Arsenal", "crest_url": "https://example.com/a.png"},
            away_team={"id": 2, "name": "Chelsea", "crest_url": "https://example.com/c.png"},
        )
    )

    assert fixture.home_team.crest_url == "https://example.com/a.png"
    assert fixture.away_team.crest_url == "https://example.com/c.png"


def test_fixture_response_requires_nested_teams() -> None:
    payload = fixture_payload()
    del payload["home_team"]

    with pytest.raises(ValidationError):
        FixtureResponse(**payload)


def test_fixture_response_reads_from_orm_objects() -> None:
    """`from_attributes` must map the ORM model, including its status enum."""
    home = Team(id=1, provider_id="football-data-57", canonical_name="Arsenal", crest_url=None)
    away = Team(id=2, provider_id="football-data-61", canonical_name="Chelsea", crest_url=None)
    fixture = Fixture(
        id=14621,
        provider="football-data.org",
        provider_id="pl-2026-14621",
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at=datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC),
        status=FixtureStatus.SCHEDULED,
        home_team_id=1,
        away_team_id=2,
    )
    fixture.home_team = home
    fixture.away_team = away

    response = FixtureResponse.model_validate(fixture)

    assert response.status == "SCHEDULED"
    assert response.home_team.name == "Arsenal"
    assert response.model_dump(mode="json")["status"] == "SCHEDULED"
