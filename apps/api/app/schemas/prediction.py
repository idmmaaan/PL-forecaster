from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

from epl_predictor import Outcome

if TYPE_CHECKING:
    from app.models.prediction import Prediction


class Probabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float

    @field_validator("home_win", "draw", "away_win")
    @classmethod
    def probability_values_must_be_between_0_and_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probabilities must be between 0.0 and 1.0")
        return v


class PredictionResponse(BaseModel):
    # `model_name` / `model_version` are part of the published prediction
    # contract, so pydantic's reserved `model_` namespace is disabled here.
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    prediction_id: int
    fixture_id: int
    home_team: str
    away_team: str
    predicted_outcome: Outcome
    probabilities: Probabilities
    model_name: str
    model_version: str
    feature_schema_version: str
    created_at: str  # ISO 8601 timestamp

    @classmethod
    def from_prediction(cls, prediction: "Prediction") -> "PredictionResponse":
        """Build the API response from a stored prediction and its lineage."""
        return cls(
            prediction_id=prediction.id,
            fixture_id=prediction.fixture_id,
            home_team=prediction.fixture.home_team.canonical_name,
            away_team=prediction.fixture.away_team.canonical_name,
            predicted_outcome=prediction.predicted_outcome,
            probabilities=Probabilities(**prediction.probabilities),
            model_name=prediction.model_version.model_name,
            model_version=prediction.model_version.version,
            feature_schema_version=prediction.feature_snapshot.feature_schema_version,
            created_at=prediction.created_at.isoformat(),
        )


__all__ = ["Outcome", "PredictionResponse", "Probabilities"]
