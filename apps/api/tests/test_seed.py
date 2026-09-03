"""The seed command must be safe to re-run, including after a migration."""

from sqlalchemy.orm import Session

from app.db.seed import (
    HISTORICAL_SOURCE,
    seed_stub_model_version,
    seed_team_aliases,
    seed_teams_and_fixtures,
)
from app.db.seed_data import SEED_FIXTURES, SEED_TEAMS
from app.models import Fixture, ModelStatus, ModelVersion, Team, TeamAlias


def test_seed_inserts_teams_and_fixtures(db_session: Session) -> None:
    seed_teams_and_fixtures(db_session)

    assert db_session.query(Team).count() == len(SEED_TEAMS)
    assert db_session.query(Fixture).count() == len(SEED_FIXTURES)


def test_seed_links_fixtures_to_the_right_teams(db_session: Session) -> None:
    seed_teams_and_fixtures(db_session)

    fixture = db_session.query(Fixture).filter(Fixture.provider_id == "pl-2026-14621").one()

    assert fixture.home_team.canonical_name == "Arsenal"
    assert fixture.away_team.canonical_name == "Chelsea"


def test_seed_is_idempotent(db_session: Session) -> None:
    seed_teams_and_fixtures(db_session)
    seed_teams_and_fixtures(db_session)

    assert db_session.query(Team).count() == len(SEED_TEAMS)
    assert db_session.query(Fixture).count() == len(SEED_FIXTURES)


def test_seed_registers_historical_aliases(db_session: Session) -> None:
    """Historical CSV spellings must resolve to the canonical team rows."""
    seed_teams_and_fixtures(db_session)
    seed_team_aliases(db_session)

    alias = (
        db_session.query(TeamAlias)
        .filter(TeamAlias.source == HISTORICAL_SOURCE, TeamAlias.alias == "Man City")
        .one()
    )

    assert alias.team.canonical_name == "Manchester City"


def test_seed_aliases_is_idempotent(db_session: Session) -> None:
    seed_teams_and_fixtures(db_session)
    seed_team_aliases(db_session)
    before = db_session.query(TeamAlias).count()

    seed_team_aliases(db_session)

    assert db_session.query(TeamAlias).count() == before


def test_seed_activates_the_stub_model(db_session: Session) -> None:
    model_version = seed_stub_model_version(db_session)

    assert model_version.status is ModelStatus.ACTIVE
    assert model_version.full_version == "dummy-epl-0.1.0"
    assert model_version.activated_at is not None


def test_seed_does_not_demote_an_already_active_model(db_session: Session) -> None:
    """Seeding must never silently replace a deliberately promoted model."""
    promoted = ModelVersion(
        model_name="catboost-epl",
        version="1.0.0",
        adapter_type="catboost",
        artifact_path="ml/artifacts/catboost-epl-1.0.0",
        feature_schema_version="1.0.0",
        status=ModelStatus.ACTIVE,
        metrics_json={},
    )
    db_session.add(promoted)
    db_session.commit()

    stub = seed_stub_model_version(db_session)

    assert stub.status is ModelStatus.CANDIDATE
    db_session.refresh(promoted)
    assert promoted.status is ModelStatus.ACTIVE
