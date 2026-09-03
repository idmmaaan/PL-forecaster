"""The common predictor interface every candidate model must implement.

Keeping every model behind this interface is what allows the application to load
a logistic-regression artifact today and a fine-tuned tabular foundation model
tomorrow without a single change to the API layer.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from epl_predictor import PROBABILITY_KEYS


class Predictor(ABC):
    """Abstract base class for all predictors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model family name, e.g. `catboost-epl`."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the semantic version of this trained model."""

    @abstractmethod
    def fit(self, train_data: Any, validation_data: Any) -> None:
        """Train or adapt a candidate model.

        Args:
            train_data: Rows from the chronological training period only.
            validation_data: Rows from the validation period, used for early
                stopping and calibration. The final test period must never be
                passed to this method.
        """

    @abstractmethod
    def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
        """Return `home_win`, `draw`, and `away_win` probabilities."""

    @abstractmethod
    def save(self, artifact_path: Path) -> None:
        """Save a complete, reloadable artifact."""

    @classmethod
    @abstractmethod
    def load(cls, artifact_path: Path) -> "Predictor":
        """Load a previously saved artifact."""


def normalise_probabilities(raw: dict[str, float]) -> dict[str, float]:
    """Return probabilities in canonical key order, renormalised to sum to 1.

    Intended for adapters translating a library's raw output into the contract.
    It corrects floating-point drift only: a caller whose values do not already
    form a distribution (negative, or summing to zero) gets an error rather than
    silently rescaled numbers.
    """
    missing = set(PROBABILITY_KEYS) - raw.keys()
    if missing:
        raise ValueError(f"Missing probability keys: {sorted(missing)}")

    values = {key: float(raw[key]) for key in PROBABILITY_KEYS}
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"Negative probability in {values}")

    total = sum(values.values())
    if total <= 0.0:
        raise ValueError(f"Probabilities sum to {total}, cannot normalise")

    return {key: value / total for key, value in values.items()}
