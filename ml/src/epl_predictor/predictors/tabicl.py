"""TabICL adapter: the README's primary foundation-model candidate.

TabICL is an in-context tabular classifier with a scikit-learn-style API, so
the adapter is thin. Two things still need care.

Its `fit` does not train: it stores the encoded training rows, which the model
attends over at prediction time. That makes the artifact carry its context,
and makes prediction cost grow with context size rather than fit cost.

Its checkpoint is downloaded from Hugging Face on first use. The version is
recorded in the artifact metadata, because "TabICL" alone does not identify
the weights that produced a set of probabilities.

Licence: BSD-3-Clause, permissive for the intended use.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from epl_predictor.predictors.foundation import (
    CANDIDATES,
    FoundationPredictor,
    LabelVector,
    require_library,
)
from epl_predictor.predictors.tabular import FloatMatrix

MODEL_FILENAME = "tabicl.joblib"

#: Ensembles over feature and class permutations. TabICL's default is 8; 4 is
#: enough to stabilise probabilities on a table this small and halves the
#: inference cost, which the selection gate measures.
DEFAULT_N_ESTIMATORS = 4


class TabICLPredictor(FoundationPredictor):
    """TabICL over the v1 feature schema."""

    adapter_type = "tabicl"
    spec = CANDIDATES["tabicl"]

    def __init__(
        self,
        name: str | None = None,
        version: str = "0.1.0",
        calibrator: Any = None,
        max_context_rows: int = 1520,
        device: str | None = None,
        random_state: int = 20260903,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
    ):
        super().__init__(
            name=name,
            version=version,
            calibrator=calibrator,
            max_context_rows=max_context_rows,
            device=device,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.classifier: Any = None
        self.checkpoint_version: str | None = None

    def _build_classifier(self) -> Any:
        tabicl = require_library("tabicl", self.spec)
        classifier = tabicl.TabICLClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            device=self.device,
        )
        # Recorded rather than assumed: the default checkpoint changes between
        # releases, and probabilities are not comparable across checkpoints.
        self.checkpoint_version = getattr(classifier, "checkpoint_version", None)
        return classifier

    def _fit_encoded(self, features: FloatMatrix, labels: LabelVector) -> None:
        """Store the context TabICL will attend over at prediction time."""
        self.classifier = self._build_classifier()
        self.classifier.fit(features, labels)

    def _predict_encoded(self, features: FloatMatrix) -> FloatMatrix:
        if self.classifier is None:
            raise RuntimeError("TabICLPredictor has no classifier; call fit or load first.")
        return np.asarray(self.classifier.predict_proba(features), dtype=np.float64)

    def _class_order(self) -> list[str]:
        """TabICL reports columns in `classes_` order, which is the integer
        label order this adapter encoded, so canonical order is preserved."""
        from epl_predictor import OUTCOME_CLASSES

        if self.classifier is None or not hasattr(self.classifier, "classes_"):
            return list(OUTCOME_CLASSES)
        return [OUTCOME_CLASSES[int(index)] for index in self.classifier.classes_]

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            **super()._extra_metadata(),
            "n_estimators": self.n_estimators,
            "checkpoint_version": self.checkpoint_version,
        }

    def _save_estimator(self, artifact_path: Path) -> None:
        # Pickling carries the stored context with the model, which is what
        # makes the artifact self-contained: reloading it does not need the
        # training table to be available or unchanged.
        joblib.dump(self.classifier, artifact_path / MODEL_FILENAME)

    def _load_estimator(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        self.n_estimators = metadata.get("n_estimators", self.n_estimators)
        self.checkpoint_version = metadata.get("checkpoint_version")
        self.classifier = joblib.load(artifact_path / MODEL_FILENAME)
