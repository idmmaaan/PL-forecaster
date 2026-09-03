"""Tests for probability validation and outcome selection.

The README requires the API to reject invalid model output rather than repair
it, so these cases assert that a broken predictor produces an error.
"""

from typing import Any

import pytest

from app.core.exceptions import FixtureNotFoundError, InvalidPredictionError
from app.repositories.in_memory_fixture_repository import InMemoryFixtureRepository
from app.repositories.in_memory_prediction_repository import InMemoryPredictionRepository
from app.schemas.prediction import Outcome
from app.services.feature_service import FIXTURE_METADATA_SCHEMA_VERSION
from app.services.predictor_service import (
    PredictorService,
    argmax_outcome,
    validate_probabilities,
)
from epl_predictor.predictors.base import Predictor


class StubPredictor(Predictor):
    """Returns whatever probabilities a test asks for, valid or not."""

    def __init__(self, probabilities: dict[str, float]):
        self.probabilities = probabilities

    @property
    def model_name(self) -> str:
        return "stub-epl"

    @property
    def model_version(self) -> str:
        return "9.9.9"

    def fit(self, train_data: Any, validation_data: Any) -> None:
        raise NotImplementedError

    def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
        return dict(self.probabilities)

    def save(self, artifact_path: Any) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, artifact_path: Any) -> "StubPredictor":
        raise NotImplementedError


def make_service(probabilities: dict[str, float]) -> PredictorService:
    fixture_repo = InMemoryFixtureRepository()
    prediction_repo = InMemoryPredictionRepository(fixture_repo)
    model_version = prediction_repo.get_active_model_version()
    assert model_version is not None
    return PredictorService(
        fixture_repo, prediction_repo, StubPredictor(probabilities), model_version
    )


def test_predict_reports_the_registered_model_not_the_adapter() -> None:
    """Response metadata comes from the registry row, which is the audit record."""
    service = make_service({"home_win": 0.2, "draw": 0.3, "away_win": 0.5})

    prediction = service.predict(14621, {})

    assert prediction.model_name == "dummy-epl"
    assert prediction.model_version == "0.1.0"
    assert prediction.predicted_outcome is Outcome.AWAY_WIN


def test_predict_persists_the_prediction_and_returns_its_id() -> None:
    service = make_service({"home_win": 0.4, "draw": 0.3, "away_win": 0.3})

    prediction = service.predict(14621, {})
    stored = service.prediction_repo.get_prediction_by_id(prediction.prediction_id)

    assert stored is not None
    assert stored.fixture_id == 14621
    assert stored.probabilities == {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}


def test_predict_stores_a_feature_snapshot_with_a_source_cutoff() -> None:
    """The stored cutoff is what makes a past prediction auditable for leakage."""
    service = make_service({"home_win": 0.4, "draw": 0.3, "away_win": 0.3})

    prediction = service.predict(14621, {})
    stored = service.prediction_repo.get_prediction_by_id(prediction.prediction_id)

    assert stored is not None
    snapshot = stored.feature_snapshot
    assert snapshot.feature_schema_version == FIXTURE_METADATA_SCHEMA_VERSION
    assert snapshot.features_json["matchday"] == 4
    assert snapshot.source_cutoff_at == service.fixture_repo.get_fixture_by_id(14621).kickoff_at


def test_repeated_predictions_do_not_accumulate_rows() -> None:
    """One canonical prediction per fixture and model version."""
    service = make_service({"home_win": 0.4, "draw": 0.3, "away_win": 0.3})

    first = service.predict(14621, {})
    second = service.predict(14621, {})

    assert first.prediction_id == second.prediction_id


def test_predict_never_reads_the_match_result() -> None:
    """A finished fixture must yield the same features it would have pre-kickoff."""
    service = make_service({"home_win": 0.4, "draw": 0.3, "away_win": 0.3})
    fixture = service.fixture_repo.get_fixture_by_id(14621)
    assert fixture is not None
    fixture.home_score, fixture.away_score, fixture.result = 3, 0, "H"

    prediction = service.predict(14621, {})
    stored = service.prediction_repo.get_prediction_by_id(prediction.prediction_id)

    assert stored is not None
    features = stored.feature_snapshot.features_json
    for leaked in ("home_score", "away_score", "result"):
        assert leaked not in features


def test_predict_raises_for_unknown_fixture() -> None:
    service = make_service({"home_win": 0.4, "draw": 0.3, "away_win": 0.3})

    with pytest.raises(FixtureNotFoundError):
        service.predict(99999, {})


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"home_win": 0.5, "draw": 0.3, "away_win": 0.2}, Outcome.HOME_WIN),
        ({"home_win": 0.2, "draw": 0.5, "away_win": 0.3}, Outcome.DRAW),
        ({"home_win": 0.2, "draw": 0.3, "away_win": 0.5}, Outcome.AWAY_WIN),
    ],
)
def test_argmax_outcome_picks_the_largest_probability(
    probabilities: dict[str, float], expected: Outcome
) -> None:
    assert argmax_outcome(probabilities) is expected


def test_argmax_outcome_breaks_ties_in_canonical_class_order() -> None:
    """A three-way tie must resolve deterministically, not by dict ordering."""
    tie = {"home_win": 1 / 3, "draw": 1 / 3, "away_win": 1 / 3}

    assert argmax_outcome(tie) is Outcome.HOME_WIN


def test_validate_probabilities_returns_valid_input_unchanged() -> None:
    probabilities = {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}

    assert validate_probabilities(probabilities) == probabilities


@pytest.mark.parametrize(
    "probabilities",
    [
        pytest.param({"home_win": 0.4, "draw": 0.3}, id="missing-key"),
        pytest.param({"home_win": 0.4, "draw": 0.3, "away_win": 0.3, "extra": 0.0}, id="extra-key"),
        pytest.param({"home_win": 1.2, "draw": -0.1, "away_win": -0.1}, id="out-of-range"),
        pytest.param({"home_win": 0.4, "draw": 0.4, "away_win": 0.4}, id="sum-above-one"),
        pytest.param({"home_win": 0.1, "draw": 0.1, "away_win": 0.1}, id="sum-below-one"),
        pytest.param({"home_win": "0.4", "draw": 0.3, "away_win": 0.3}, id="non-numeric"),
    ],
)
def test_validate_probabilities_rejects_contract_violations(
    probabilities: dict[str, Any],
) -> None:
    with pytest.raises(InvalidPredictionError):
        validate_probabilities(probabilities)


def test_predict_rejects_a_model_that_breaks_the_contract() -> None:
    """Invalid probabilities must surface as an error, never as a prediction."""
    service = make_service({"home_win": 0.9, "draw": 0.9, "away_win": 0.9})

    with pytest.raises(InvalidPredictionError):
        service.predict(14621, {})
