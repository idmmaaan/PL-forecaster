import pytest
from pydantic import ValidationError

from app.schemas.prediction import Outcome, PredictionResponse, Probabilities


def prediction_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prediction_id": 1042,
        "fixture_id": 14621,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "predicted_outcome": "HOME_WIN",
        "probabilities": {"home_win": 0.51, "draw": 0.27, "away_win": 0.22},
        "model_name": "tabicl-epl",
        "model_version": "0.1.0",
        "feature_schema_version": "1.0.0",
        "created_at": "2026-08-31T15:00:00Z",
    }
    payload.update(overrides)
    return payload


def test_probabilities_accepts_valid_values() -> None:
    probabilities = Probabilities(home_win=0.5, draw=0.3, away_win=0.2)

    assert probabilities.home_win == 0.5
    assert probabilities.draw == 0.3
    assert probabilities.away_win == 0.2


def test_probabilities_accepts_boundary_values() -> None:
    probabilities = Probabilities(home_win=0.0, draw=1.0, away_win=0.0)

    assert probabilities.home_win == 0.0
    assert probabilities.draw == 1.0


@pytest.mark.parametrize(
    "values",
    [
        {"home_win": -0.1, "draw": 0.5, "away_win": 0.6},
        {"home_win": 0.5, "draw": 1.5, "away_win": 0.0},
        {"home_win": 0.5, "draw": 0.3, "away_win": -0.001},
    ],
)
def test_probabilities_rejects_values_outside_unit_interval(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Probabilities(**values)


def test_prediction_response_schema() -> None:
    prediction = PredictionResponse(**prediction_payload())

    assert prediction.prediction_id == 1042
    assert prediction.fixture_id == 14621
    assert prediction.home_team == "Arsenal"
    assert prediction.away_team == "Chelsea"
    assert prediction.predicted_outcome is Outcome.HOME_WIN
    assert prediction.predicted_outcome.value == "HOME_WIN"
    assert prediction.probabilities.home_win == 0.51
    assert prediction.model_name == "tabicl-epl"
    assert prediction.model_version == "0.1.0"
    assert prediction.feature_schema_version == "1.0.0"
    assert prediction.created_at == "2026-08-31T15:00:00Z"


def test_prediction_response_serialises_outcome_as_its_label() -> None:
    """The wire format must be HOME_WIN, not an enum repr."""
    prediction = PredictionResponse(**prediction_payload(predicted_outcome="DRAW"))

    assert prediction.model_dump(mode="json")["predicted_outcome"] == "DRAW"


def test_prediction_response_rejects_unknown_outcome() -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(**prediction_payload(predicted_outcome="home_win"))


def test_prediction_response_probabilities_sum_to_one() -> None:
    prediction = PredictionResponse(**prediction_payload())
    probabilities = prediction.probabilities

    total = probabilities.home_win + probabilities.draw + probabilities.away_win
    assert abs(total - 1.0) < 1e-6
