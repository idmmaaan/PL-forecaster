"""Prediction repository tests against real PostgreSQL.

Skipped when PostgreSQL is not reachable; every test rolls back.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.models import (
    FeatureSnapshot,
    Fixture,
    FixtureStatus,
    ModelStatus,
    ModelVersion,
    Prediction,
    Team,
)
from app.repositories.prediction_interface import PredictionRecord
from app.repositories.prediction_repository import SQLAlchemyPredictionRepository
from epl_predictor import Outcome

KICKOFF = datetime.now(UTC) + timedelta(days=1)


@pytest.fixture
def repo(db_session: Session) -> SQLAlchemyPredictionRepository:
    return SQLAlchemyPredictionRepository(db_session)


@pytest.fixture
def fixture(db_session: Session) -> Fixture:
    home = Team(provider_id="football-data-57", canonical_name="Arsenal")
    away = Team(provider_id="football-data-61", canonical_name="Chelsea")
    db_session.add_all([home, away])
    db_session.flush()

    fixture = Fixture(
        provider="football-data.org",
        provider_id="pl-2026-14621",
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at=KICKOFF,
        status=FixtureStatus.SCHEDULED,
        home_team_id=home.id,
        away_team_id=away.id,
    )
    db_session.add(fixture)
    db_session.flush()
    return fixture


@pytest.fixture
def active_model(db_session: Session) -> ModelVersion:
    model = ModelVersion(
        model_name="dummy-epl",
        version="0.1.0",
        adapter_type="dummy",
        artifact_path="ml/artifacts/dummy-epl-0.1.0",
        feature_schema_version="1.0.0",
        status=ModelStatus.ACTIVE,
        activated_at=datetime.now(UTC),
        metrics_json={},
    )
    db_session.add(model)
    db_session.flush()
    return model


def make_record(
    fixture: Fixture,
    model: ModelVersion,
    probabilities: dict[str, float] | None = None,
    features: dict[str, object] | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        fixture_id=fixture.id,
        model_version_id=model.id,
        probabilities=probabilities or {"home_win": 0.4, "draw": 0.3, "away_win": 0.3},
        predicted_outcome=Outcome.HOME_WIN,
        features=features or {"matchday": 4},
        feature_schema_version="1.0.0",
        source_cutoff_at=fixture.kickoff_at,
    )


def test_get_active_model_version(
    repo: SQLAlchemyPredictionRepository, active_model: ModelVersion
) -> None:
    assert repo.get_active_model_version() is not None
    assert repo.get_active_model_version().full_version == "dummy-epl-0.1.0"


def test_get_active_model_version_is_none_when_nothing_is_promoted(
    repo: SQLAlchemyPredictionRepository, db_session: Session
) -> None:
    db_session.add(
        ModelVersion(
            model_name="catboost-epl",
            version="0.1.0",
            adapter_type="catboost",
            artifact_path="somewhere",
            feature_schema_version="1.0.0",
            status=ModelStatus.CANDIDATE,
            metrics_json={},
        )
    )
    db_session.flush()

    assert repo.get_active_model_version() is None


def test_record_prediction_stores_snapshot_and_prediction(
    repo: SQLAlchemyPredictionRepository,
    db_session: Session,
    fixture: Fixture,
    active_model: ModelVersion,
) -> None:
    prediction = repo.record_prediction(make_record(fixture, active_model))

    assert prediction.id is not None
    assert prediction.probabilities == {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}
    assert prediction.predicted_outcome is Outcome.HOME_WIN
    assert db_session.query(FeatureSnapshot).count() == 1
    assert db_session.query(Prediction).count() == 1


def test_recording_twice_updates_one_row(
    repo: SQLAlchemyPredictionRepository,
    db_session: Session,
    fixture: Fixture,
    active_model: ModelVersion,
) -> None:
    """The unique constraint means a second predict must update, not insert."""
    first = repo.record_prediction(make_record(fixture, active_model))
    second = repo.record_prediction(
        make_record(
            fixture, active_model, probabilities={"home_win": 0.2, "draw": 0.3, "away_win": 0.5}
        )
    )

    assert first.id == second.id
    assert db_session.query(Prediction).count() == 1
    assert second.probabilities["away_win"] == 0.5


def test_identical_features_reuse_one_snapshot(
    repo: SQLAlchemyPredictionRepository,
    db_session: Session,
    fixture: Fixture,
    active_model: ModelVersion,
) -> None:
    """Re-predicting an unchanged fixture must not orphan duplicate vectors."""
    repo.record_prediction(make_record(fixture, active_model))
    repo.record_prediction(make_record(fixture, active_model))

    assert db_session.query(FeatureSnapshot).count() == 1


def test_different_features_get_their_own_snapshot(
    repo: SQLAlchemyPredictionRepository,
    db_session: Session,
    fixture: Fixture,
    active_model: ModelVersion,
) -> None:
    repo.record_prediction(make_record(fixture, active_model, features={"matchday": 4}))
    prediction = repo.record_prediction(
        make_record(fixture, active_model, features={"matchday": 4, "elo_difference": 100.0})
    )

    assert db_session.query(FeatureSnapshot).count() == 2
    assert prediction.feature_snapshot.features_json["elo_difference"] == 100.0


def test_get_prediction_by_id_loads_the_full_lineage(
    repo: SQLAlchemyPredictionRepository, fixture: Fixture, active_model: ModelVersion
) -> None:
    stored = repo.record_prediction(make_record(fixture, active_model))

    loaded = repo.get_prediction_by_id(stored.id)

    assert loaded is not None
    assert loaded.fixture.home_team.canonical_name == "Arsenal"
    assert loaded.fixture.away_team.canonical_name == "Chelsea"
    assert loaded.model_version.full_version == "dummy-epl-0.1.0"
    assert loaded.feature_snapshot.feature_schema_version == "1.0.0"


def test_get_prediction_by_id_returns_none_for_unknown_id(
    repo: SQLAlchemyPredictionRepository,
) -> None:
    assert repo.get_prediction_by_id(-1) is None
