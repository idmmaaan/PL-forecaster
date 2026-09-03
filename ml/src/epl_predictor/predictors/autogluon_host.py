"""Shared adapter for the two candidates that AutoGluon hosts.

Mitra and TabPFNMix are both reached through `autogluon.tabular.TabularPredictor`
rather than through their own APIs, so they differ only in a hyperparameter key
and a fine-tuning budget. Everything else — building the labelled frame
AutoGluon wants, pinning it to the single requested model so it cannot quietly
ensemble a decision tree into the result, and reading probability columns back
in canonical order — is identical and lives here.

Pinning matters more than it looks. `TabularPredictor.fit` with default
hyperparameters trains a whole zoo and returns a weighted ensemble; benchmarking
that against CatBoost would answer a different question than the README asks.
Passing an explicit single-model hyperparameter dict keeps the comparison
honest.

The candidates arrive already encoded by `FoundationPredictor`, so every model
in the benchmark sees the same preprocessed matrix.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.predictors.foundation import (
    FoundationPredictor,
    LabelVector,
    require_library,
)
from epl_predictor.predictors.tabular import FloatMatrix

#: Subdirectory of the artifact holding AutoGluon's own predictor directory.
MODEL_DIRNAME = "autogluon"

LABEL_COLUMN = "outcome"


class AutoGluonPredictor(FoundationPredictor):
    """Base for candidates trained through AutoGluon's tabular predictor."""

    #: AutoGluon's hyperparameter key for this candidate, e.g. `"MITRA"`.
    autogluon_key: str

    def __init__(
        self,
        name: str | None = None,
        version: str = "0.1.0",
        calibrator: Any = None,
        max_context_rows: int = 1520,
        device: str | None = None,
        random_state: int = 20260903,
        fine_tune: bool = True,
        fine_tune_steps: int = 50,
        time_limit: int | None = 1800,
    ):
        super().__init__(
            name=name,
            version=version,
            calibrator=calibrator,
            max_context_rows=max_context_rows,
            device=device,
            random_state=random_state,
        )
        self.fine_tune = fine_tune
        self.fine_tune_steps = fine_tune_steps
        self.time_limit = time_limit
        self.predictor: Any = None
        self.feature_names: list[str] = []

    def _hyperparameters(self) -> dict[str, list[dict[str, Any]]]:
        """The single-model hyperparameter dict to fit.

        Subclasses override to add candidate-specific options.
        """
        return {
            self.autogluon_key: [
                {
                    "fine_tune": self.fine_tune,
                    "fine_tune_steps": self.fine_tune_steps,
                }
            ]
        }

    def _labelled_frame(self, features: FloatMatrix, labels: LabelVector | None) -> pd.DataFrame:
        """Wrap an encoded matrix as the labelled frame AutoGluon expects."""
        frame = pd.DataFrame(features, columns=self.feature_names)
        if labels is not None:
            frame[LABEL_COLUMN] = [OUTCOME_CLASSES[int(index)] for index in labels]
        return frame

    def _fit_encoded(self, features: FloatMatrix, labels: LabelVector) -> None:
        tabular = require_library("autogluon.tabular", self.spec)
        self.feature_names = [f"f{index}" for index in range(features.shape[1])]

        self.predictor = tabular.TabularPredictor(
            label=LABEL_COLUMN,
            problem_type="multiclass",
            # Log loss is the project's primary metric, so AutoGluon must
            # select on it too rather than on its accuracy default.
            eval_metric="log_loss",
            verbosity=0,
        )
        self.predictor.fit(
            self._labelled_frame(features, labels),
            hyperparameters=self._hyperparameters(),
            time_limit=self.time_limit,
        )

    def _predict_encoded(self, features: FloatMatrix) -> FloatMatrix:
        if self.predictor is None:
            raise RuntimeError(f"{type(self).__name__} has no predictor; fit or load it first.")
        probabilities = self.predictor.predict_proba(self._labelled_frame(features, None))
        return np.asarray(probabilities.to_numpy(), dtype=np.float64)

    def _class_order(self) -> list[str]:
        """AutoGluon orders probability columns by its own class list."""
        if self.predictor is None:
            return list(OUTCOME_CLASSES)
        return [str(label) for label in self.predictor.class_labels]

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            **super()._extra_metadata(),
            "autogluon_key": self.autogluon_key,
            "fine_tune": self.fine_tune,
            "fine_tune_steps": self.fine_tune_steps,
            "feature_names": self.feature_names,
        }

    def _save_estimator(self, artifact_path: Path) -> None:
        """Clone AutoGluon's predictor directory into the artifact.

        AutoGluon owns its own on-disk layout, so the artifact contains that
        directory verbatim rather than a pickle of the predictor object.
        """
        if self.predictor is None:
            raise RuntimeError("Nothing to save: the predictor was never fitted.")

        target = artifact_path / MODEL_DIRNAME
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.predictor.path, target)

    def _load_estimator(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        tabular = require_library("autogluon.tabular", self.spec)
        self.feature_names = metadata.get("feature_names", [])
        self.fine_tune = metadata.get("fine_tune", self.fine_tune)
        self.fine_tune_steps = metadata.get("fine_tune_steps", self.fine_tune_steps)
        self.predictor = tabular.TabularPredictor.load(str(artifact_path / MODEL_DIRNAME))
