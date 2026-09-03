"""Class-frequency baseline: predict the training-set outcome distribution.

This is the number every other model must beat to have earned anything at
all. It uses no features, only the observed rate of home wins, draws, and away
wins in the training period, so its log loss is the entropy of the outcome
distribution. A model that cannot improve on it has learned nothing about
football.

Unlike `DummyPredictor`, whose probabilities are hard-coded, this one is
fitted, which makes it the honest "no information" reference for whichever
seasons a fold happens to train on.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import TARGET_COLUMN
from epl_predictor.predictors.tabular import FloatMatrix, TabularPredictor


class ClassFrequencyPredictor(TabularPredictor):
    """Returns the training-set class distribution for every fixture."""

    adapter_type = "class-frequency"

    def __init__(
        self,
        name: str = "class-frequency-epl",
        version: str = "0.1.0",
        calibrator: Any = None,
    ):
        super().__init__(name=name, version=version, calibrator=calibrator)
        self.frequencies: FloatMatrix = np.full(len(OUTCOME_CLASSES), 1 / len(OUTCOME_CLASSES))

    def _fit_frame(self, train: pd.DataFrame, validation: pd.DataFrame | None) -> None:
        """Count outcomes in the training period.

        Validation rows are deliberately ignored: including them would make
        this baseline depend on data the models it anchors are tuned against.
        """
        counts = train[TARGET_COLUMN].value_counts()
        totals = np.array(
            [float(counts.get(outcome, 0.0)) for outcome in OUTCOME_CLASSES], dtype=np.float64
        )
        if totals.sum() <= 0:
            raise ValueError("Training rows contain no recognised outcomes")
        self.frequencies = totals / totals.sum()

    def _predict_matrix(self, features: pd.DataFrame) -> FloatMatrix:
        return np.tile(self.frequencies, (len(features), 1))

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            "frequencies": dict(
                zip(OUTCOME_CLASSES, (float(value) for value in self.frequencies), strict=True)
            )
        }

    def _save_model(self, artifact_path: Path) -> None:
        (artifact_path / "frequencies.json").write_text(
            json.dumps(self.frequencies.tolist()) + "\n"
        )

    def _load_model(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        frequencies_file = artifact_path / "frequencies.json"
        if frequencies_file.exists():
            self.frequencies = np.asarray(
                json.loads(frequencies_file.read_text()), dtype=np.float64
            )
            return
        stored = metadata.get("frequencies", {})
        self.frequencies = np.asarray(
            [float(stored.get(outcome, 1 / 3)) for outcome in OUTCOME_CLASSES],
            dtype=np.float64,
        )
