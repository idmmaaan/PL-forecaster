from datetime import UTC, datetime

from app.models.fixture import Fixture, FixtureStatus
from app.repositories.in_memory_fixture_repository import InMemoryFixtureRepository


def test_get_fixtures() -> None:
    repo = InMemoryFixtureRepository()
    fixtures = repo.get_fixtures()

    assert len(fixtures) == 3
    assert isinstance(fixtures[0], Fixture)
    assert fixtures[0].id == 14621


def test_get_fixture_by_id() -> None:
    repo = InMemoryFixtureRepository()

    fixture = repo.get_fixture_by_id(14621)
    assert fixture is not None
    assert fixture.id == 14621
    assert fixture.home_team_id == 1
    assert fixture.away_team_id == 2

    assert repo.get_fixture_by_id(99999) is None


def test_get_upcoming_fixtures_respects_limit() -> None:
    repo = InMemoryFixtureRepository()

    assert len(repo.get_upcoming_fixtures(limit=5)) == 3
    assert len(repo.get_upcoming_fixtures(limit=2)) == 2


def test_get_upcoming_fixtures_excludes_finished() -> None:
    repo = InMemoryFixtureRepository()
    played = repo.get_fixture_by_id(14621)
    assert played is not None
    played.status = FixtureStatus.FINISHED

    upcoming_ids = [fixture.id for fixture in repo.get_upcoming_fixtures()]

    assert 14621 not in upcoming_ids
    assert len(upcoming_ids) == 2


def test_fixtures_are_sorted_by_kickoff() -> None:
    repo = InMemoryFixtureRepository()
    kickoffs = [fixture.kickoff_at for fixture in repo.get_fixtures()]

    assert kickoffs == sorted(kickoffs)


def test_fixture_attributes() -> None:
    repo = InMemoryFixtureRepository()
    first_fixture = repo.get_fixtures()[0]

    assert first_fixture.provider == "football-data.org"
    assert first_fixture.competition_code == "PL"
    assert first_fixture.season_start_year == 2026
    assert first_fixture.matchday == 4
    assert first_fixture.status.value == "SCHEDULED"

    assert first_fixture.kickoff_at.tzinfo is not None
    assert first_fixture.kickoff_at == datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC)


def test_team_relationships_are_populated() -> None:
    """The API serialises nested teams, so transient fixtures must carry them."""
    repo = InMemoryFixtureRepository()
    fixture = repo.get_fixture_by_id(14621)

    assert fixture is not None
    assert fixture.home_team.canonical_name == "Arsenal"
    assert fixture.away_team.canonical_name == "Chelsea"


def test_upsert_team_is_idempotent() -> None:
    repo = InMemoryFixtureRepository()
    team_data = {
        "provider_id": "football-data-57",
        "canonical_name": "Arsenal FC",
        "short_name": "Arsenal",
        "crest_url": None,
    }

    first = repo.upsert_team(team_data)
    second = repo.upsert_team(team_data)

    assert first is second
    assert first.id == 1
    assert first.canonical_name == "Arsenal FC"


def test_upsert_team_inserts_unknown_provider_id() -> None:
    repo = InMemoryFixtureRepository()

    team = repo.upsert_team({"provider_id": "football-data-999", "canonical_name": "Luton Town"})

    assert team.id == 7
    assert repo.upsert_team({"provider_id": "football-data-999", "canonical_name": "Luton"}) is team


def test_upsert_fixture_is_idempotent() -> None:
    repo = InMemoryFixtureRepository()
    fixture_data = {
        "provider": "football-data.org",
        "provider_id": "pl-2026-14621",
        "competition_code": "PL",
        "season_start_year": 2026,
        "matchday": 5,
        "kickoff_at": datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC),
        "status": FixtureStatus.SCHEDULED,
        "home_team_id": 1,
        "away_team_id": 2,
    }

    repo.upsert_fixture(fixture_data)

    assert len(repo.get_fixtures()) == 3
    updated = repo.get_fixture_by_id(14621)
    assert updated is not None
    assert updated.matchday == 5
