"""Repository tests against a real PostgreSQL database.

These exercise the SQL that the in-memory double cannot: filtering, ordering,
and idempotent upserts. Every test runs inside a transaction that is rolled
back, and the whole module is skipped when PostgreSQL is not reachable.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.db.seed_data import SEED_TEAMS
from app.models.fixture import Fixture, FixtureStatus
from app.models.team import Team
from app.repositories.fixture_repository import SQLAlchemyFixtureRepository

NOW = datetime.now(UTC)


@pytest.fixture
def repo(db_session: Session) -> SQLAlchemyFixtureRepository:
    return SQLAlchemyFixtureRepository(db_session)


@pytest.fixture
def teams(db_session: Session) -> list[Team]:
    """Two persisted teams with database-assigned ids."""
    created = [
        Team(**{key: value for key, value in team.items() if key != "id"})
        for team in SEED_TEAMS[:2]
    ]
    db_session.add_all(created)
    db_session.flush()
    return created


def make_fixture(
    teams: list[Team],
    provider_id: str,
    kickoff_at: datetime,
    status: FixtureStatus = FixtureStatus.SCHEDULED,
    matchday: int = 4,
) -> Fixture:
    return Fixture(
        provider="football-data.org",
        provider_id=provider_id,
        competition_code="PL",
        season_start_year=2026,
        matchday=matchday,
        kickoff_at=kickoff_at,
        status=status,
        home_team_id=teams[0].id,
        away_team_id=teams[1].id,
    )


def test_get_fixtures_returns_rows_ordered_by_kickoff(
    repo: SQLAlchemyFixtureRepository, db_session: Session, teams: list[Team]
) -> None:
    later = make_fixture(teams, "pl-2026-2", NOW + timedelta(days=2))
    earlier = make_fixture(teams, "pl-2026-1", NOW + timedelta(days=1))
    db_session.add_all([later, earlier])
    db_session.flush()

    fixtures = repo.get_fixtures()

    assert [fixture.provider_id for fixture in fixtures] == ["pl-2026-1", "pl-2026-2"]


def test_get_fixture_by_id_loads_teams(
    repo: SQLAlchemyFixtureRepository, db_session: Session, teams: list[Team]
) -> None:
    fixture = make_fixture(teams, "pl-2026-1", NOW + timedelta(days=1))
    db_session.add(fixture)
    db_session.flush()

    loaded = repo.get_fixture_by_id(fixture.id)

    assert loaded is not None
    assert loaded.home_team.canonical_name == "Arsenal"
    assert loaded.away_team.canonical_name == "Chelsea"


def test_get_fixture_by_id_returns_none_for_unknown_id(
    repo: SQLAlchemyFixtureRepository,
) -> None:
    assert repo.get_fixture_by_id(-1) is None


def test_get_upcoming_fixtures_excludes_past_and_finished(
    repo: SQLAlchemyFixtureRepository, db_session: Session, teams: list[Team]
) -> None:
    db_session.add_all(
        [
            make_fixture(teams, "pl-2026-upcoming", NOW + timedelta(days=1)),
            make_fixture(teams, "pl-2026-timed", NOW + timedelta(days=2), FixtureStatus.TIMED),
            make_fixture(teams, "pl-2026-past", NOW - timedelta(days=1)),
            make_fixture(
                teams, "pl-2026-finished", NOW + timedelta(days=3), FixtureStatus.FINISHED
            ),
        ]
    )
    db_session.flush()

    provider_ids = [fixture.provider_id for fixture in repo.get_upcoming_fixtures()]

    assert provider_ids == ["pl-2026-upcoming", "pl-2026-timed"]


def test_get_upcoming_fixtures_respects_limit(
    repo: SQLAlchemyFixtureRepository, db_session: Session, teams: list[Team]
) -> None:
    db_session.add_all(
        [
            make_fixture(teams, f"pl-2026-{index}", NOW + timedelta(days=index))
            for index in range(1, 5)
        ]
    )
    db_session.flush()

    assert len(repo.get_upcoming_fixtures(limit=2)) == 2


def test_upsert_team_inserts_then_updates(
    repo: SQLAlchemyFixtureRepository, db_session: Session
) -> None:
    inserted = repo.upsert_team(
        {"provider_id": "football-data-999", "canonical_name": "Luton Town", "short_name": "Luton"}
    )
    assert inserted.id is not None

    updated = repo.upsert_team(
        {
            "provider_id": "football-data-999",
            "canonical_name": "Luton Town FC",
            "short_name": "Luton",
        }
    )

    assert updated.id == inserted.id
    assert updated.canonical_name == "Luton Town FC"
    assert db_session.query(Team).filter(Team.provider_id == "football-data-999").count() == 1


def test_upsert_fixture_inserts_then_updates(
    repo: SQLAlchemyFixtureRepository, db_session: Session, teams: list[Team]
) -> None:
    kickoff = NOW + timedelta(days=1)
    payload = {
        "provider": "football-data.org",
        "provider_id": "pl-2026-14621",
        "competition_code": "PL",
        "season_start_year": 2026,
        "matchday": 4,
        "kickoff_at": kickoff,
        "status": FixtureStatus.SCHEDULED,
        "home_team_id": teams[0].id,
        "away_team_id": teams[1].id,
    }

    inserted = repo.upsert_fixture(payload)
    updated = repo.upsert_fixture(
        {**payload, "matchday": 5, "status": FixtureStatus.FINISHED, "result": "H"}
    )

    assert updated.id == inserted.id
    assert updated.matchday == 5
    assert updated.result == "H"
    assert db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14621").count() == 1


def test_result_check_constraint_rejects_unknown_label(
    repo: SQLAlchemyFixtureRepository, db_session: Session, teams: list[Team]
) -> None:
    """The database enforces the H/D/A domain, not just application code."""
    from sqlalchemy.exc import IntegrityError

    fixture = make_fixture(teams, "pl-2026-bad", NOW + timedelta(days=1))
    fixture.result = "X"
    db_session.add(fixture)

    with pytest.raises(IntegrityError):
        db_session.flush()
