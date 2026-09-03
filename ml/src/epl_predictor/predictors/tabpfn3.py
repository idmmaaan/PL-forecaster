"""TabPFN-3 adapter: research benchmark only, never promoted.

Two constraints, both from the README, shape this adapter.

**The licence forbids production use.** `CandidateSpec.production_use_allowed`
is False for this candidate, and `assert_promotable` refuses it outright. The
number it produces is a useful research reference point — it says how much
headroom a very strong tabular model finds in these features — but it may not
be served.

**Fine-tuning is out of reach here.** The published fine-tuning example
requires an 80 GB CUDA GPU and rejects non-CUDA execution. This adapter
therefore runs TabPFN in in-context inference mode only, which is what the
README asks for, and does not pretend to offer a fine-tuning path.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.predictors.foundation import (
    CANDIDATES,
    FoundationPredictor,
    LabelVector,
    require_library,
)
from epl_predictor.predictors.tabular import FloatMatrix

MODEL_FILENAME = "tabpfn3.joblib"


class TabPFN3Predictor(FoundationPredictor):
    """TabPFN-3 in in-context inference mode, as a research benchmark."""

    adapter_type = "tabpfn3"
    spec = CANDIDATES["tabpfn3"]

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.classifier: Any = None

    def _fit_encoded(self, features: FloatMatrix, labels: LabelVector) -> None:
        """Store the in-context training rows.

        No gradient step is taken: `fit` for a prior-fitted network means
        handing it the context it will attend over.
        """
        tabpfn = require_library("tabpfn", self.spec)
        self.classifier = tabpfn.TabPFNClassifier(
            device=self.device,
            random_state=self.random_state,
        )
        self.classifier.fit(features, labels)

    def _predict_encoded(self, features: FloatMatrix) -> FloatMatrix:
        if self.classifier is None:
            raise RuntimeError("TabPFN3Predictor has no classifier; fit or load it first.")
        return np.asarray(self.classifier.predict_proba(features), dtype=np.float64)

    def _class_order(self) -> list[str]:
        if self.classifier is None or not hasattr(self.classifier, "classes_"):
            return list(OUTCOME_CLASSES)
        return [OUTCOME_CLASSES[int(index)] for index in self.classifier.classes_]

    def _save_estimator(self, artifact_path: Path) -> None:
        joblib.dump(self.classifier, artifact_path / MODEL_FILENAME)

    def _load_estimator(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        self.classifier = joblib.load(artifact_path / MODEL_FILENAME)
