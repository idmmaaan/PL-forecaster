from datetime import UTC, datetime

import pytest

from app.clients.football_data_client import Match, MatchStatus
from app.clients.football_data_client import Team as FootballTeam
from app.models.fixture import FixtureStatus
from app.services.fixture_import_service import FixtureImportService, parse_kickoff

ARSENAL = FootballTeam(
    id=1, name="Arsenal", short_name="ARS", crest_url="https://example.com/arsenal.png"
)
CHELSEA = FootballTeam(
    id=2, name="Chelsea", short_name="CHE", crest_url="https://example.com/chelsea.png"
)


def make_match(
    match_id: int = 14621,
    matchday: int = 4,
    kickoff_at: str = "2026-09-12T14:00:00Z",
    home_team: FootballTeam = ARSENAL,
    away_team: FootballTeam = CHELSEA,
    status: MatchStatus = MatchStatus.SCHEDULED,
    home_score: int | None = None,
    away_score: int | None = None,
    result: str | None = None,
) -> Match:
    return Match(
        id=match_id,
        competition_code="PL",
        season_start_year=2026,
        matchday=matchday,
        kickoff_at=kickoff_at,
        status=status,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        result=result,
    )


@pytest.fixture
def service() -> FixtureImportService:
    return FixtureImportService()


def test_convert_match_to_fixture(service: FixtureImportService) -> None:
    fixture = service.convert_football_data_match_to_fixture(make_match())

    assert fixture.provider == "football-data.org"
    assert fixture.provider_id == "pl-2026-14621"
    assert fixture.competition_code == "PL"
    assert fixture.season_start_year == 2026
    assert fixture.matchday == 4
    assert fixture.status is FixtureStatus.SCHEDULED
    assert fixture.home_score is None
    assert fixture.away_score is None
    assert fixture.result is None


def test_convert_match_to_fixture_keeps_kickoff_timezone_aware(
    service: FixtureImportService,
) -> None:
    """`kickoff_at` is a timestamptz column, so the tzinfo must survive import."""
    fixture = service.convert_football_data_match_to_fixture(make_match())

    assert fixture.kickoff_at == datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC)
    assert fixture.kickoff_at.tzinfo is not None


def test_convert_match_to_fixture_maps_provider_status(service: FixtureImportService) -> None:
    """TIMED is a provider status that must survive as a fixture status."""
    fixture = service.convert_football_data_match_to_fixture(make_match(status=MatchStatus.TIMED))

    assert fixture.status is FixtureStatus.TIMED


def test_convert_match_to_fixture_carries_final_score(service: FixtureImportService) -> None:
    fixture = service.convert_football_data_match_to_fixture(
        make_match(status=MatchStatus.FINISHED, home_score=2, away_score=1, result="H")
    )

    assert (fixture.home_score, fixture.away_score, fixture.result) == (2, 1, "H")


def test_convert_team(service: FixtureImportService) -> None:
    team = service.convert_football_data_match_to_team(ARSENAL)

    assert team.provider_id == "football-data-1"
    assert team.canonical_name == "Arsenal"
    assert team.short_name == "ARS"
    assert team.code is None  # Not provided by football-data.org v4
    assert team.crest_url == "https://example.com/arsenal.png"


def test_convert_team_with_missing_optional_fields(service: FixtureImportService) -> None:
    team = service.convert_football_data_match_to_team(
        FootballTeam(id=3, name="Manchester City", short_name=None, crest_url=None)
    )

    assert team.provider_id == "football-data-3"
    assert team.canonical_name == "Manchester City"
    assert team.short_name is None
    assert team.crest_url is None


def test_convert_matches_to_domain_models(service: FixtureImportService) -> None:
    matches = [
        make_match(14621),
        make_match(14622, home_team=CHELSEA, away_team=ARSENAL),
    ]

    teams, fixtures = service.convert_football_data_matches_to_domain_models(matches)

    assert len(teams) == 2
    assert {team.canonical_name for team in teams} == {"Arsenal", "Chelsea"}
    assert [fixture.provider_id for fixture in fixtures] == [
        "pl-2026-14621",
        "pl-2026-14622",
    ]


def test_convert_matches_deduplicates_repeated_teams(service: FixtureImportService) -> None:
    matches = [make_match(14621, matchday=4), make_match(14622, matchday=5)]

    teams, fixtures = service.convert_football_data_matches_to_domain_models(matches)

    assert len(teams) == 2
    assert len(fixtures) == 2


def test_convert_matches_handles_an_empty_batch(service: FixtureImportService) -> None:
    assert service.convert_football_data_matches_to_domain_models([]) == ([], [])


def test_to_fixture_dict_resolves_team_ids(service: FixtureImportService) -> None:
    payload = service.to_fixture_dict(make_match(), home_team_id=1, away_team_id=2)

    assert payload["home_team_id"] == 1
    assert payload["away_team_id"] == 2
    assert payload["provider_id"] == "pl-2026-14621"
    assert payload["status"] is FixtureStatus.SCHEDULED


def test_to_team_dict(service: FixtureImportService) -> None:
    assert service.to_team_dict(ARSENAL) == {
        "provider_id": "football-data-1",
        "canonical_name": "Arsenal",
        "short_name": "ARS",
        "crest_url": "https://example.com/arsenal.png",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-12T14:00:00Z", datetime(2026, 9, 12, 14, 0, tzinfo=UTC)),
        ("2026-09-12T14:00:00+00:00", datetime(2026, 9, 12, 14, 0, tzinfo=UTC)),
    ],
)
def test_parse_kickoff(value: str, expected: datetime) -> None:
    assert parse_kickoff(value) == expected
