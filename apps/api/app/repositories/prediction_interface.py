from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.models.model_version import ModelVersion
from app.models.prediction import Prediction
from epl_predictor import Outcome


@dataclass(frozen=True)
class PredictionRecord:
    """Everything needed to persist one prediction together with its lineage."""

    fixture_id: int
    model_version_id: int
    probabilities: dict[str, float]
    predicted_outcome: Outcome
    features: dict[str, Any]
    feature_schema_version: str
    #: No feature input may postdate this timestamp; stored for leakage audits.
    source_cutoff_at: datetime


class PredictionRepository(ABC):
    """Persistence for predictions, their feature snapshots, and the model registry."""

    @abstractmethod
    def get_active_model_version(self) -> ModelVersion | None:
        """Return the model currently promoted to ACTIVE, or None if there is none."""

    @abstractmethod
    def record_prediction(self, record: PredictionRecord) -> Prediction:
        """Store a feature snapshot and its prediction, returning the stored row.

        Idempotent per fixture and model version: predicting the same fixture
        twice with the same model updates the existing row instead of inserting
        a duplicate.
        """

    @abstractmethod
    def get_prediction_by_id(self, prediction_id: int) -> Prediction | None:
        """Return one stored prediction, or None when the id is unknown."""
