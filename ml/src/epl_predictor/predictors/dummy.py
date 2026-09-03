"""Deterministic stub predictor.

Used to verify the frontend, API, database, and response contract end to end
before any trained model is connected, and as the mandatory baseline that every
real candidate must beat.
"""

import json
from pathlib import Path
from typing import Any

from epl_predictor.predictors.base import Predictor

ARTIFACT_FILENAME = "predictor.json"

#: Roughly the long-run Premier League outcome distribution, rounded so the
#: values stay exact in the API response and obviously sum to 1.0.
FIXED_PROBABILITIES = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}


class DummyPredictor(Predictor):
    """Returns the same probabilities for every fixture, ignoring features."""

    def __init__(self, name: str = "dummy-epl", version: str = "0.1.0"):
        self._model_name = name
        self._model_version = version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def fit(self, train_data: Any = None, validation_data: Any = None) -> None:
        """No-op: the fixed probabilities are not learned from data."""

    def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
        """Return the fixed probabilities, independent of `features`."""
        return dict(FIXED_PROBABILITIES)

    def save(self, artifact_path: Path) -> None:
        """Write the artifact directory, including name and version metadata."""
        artifact_path = Path(artifact_path)
        artifact_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "adapter_type": "dummy",
            "model_name": self._model_name,
            "model_version": self._model_version,
            "probabilities": FIXED_PROBABILITIES,
        }
        (artifact_path / ARTIFACT_FILENAME).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, artifact_path: Path) -> "DummyPredictor":
        """Reload a saved artifact, restoring its name and version."""
        metadata_file = Path(artifact_path) / ARTIFACT_FILENAME
        if not metadata_file.exists():
            return cls()

        payload = json.loads(metadata_file.read_text())
        return cls(
            name=payload.get("model_name", "dummy-epl"),
            version=payload.get("model_version", "0.1.0"),
        )
