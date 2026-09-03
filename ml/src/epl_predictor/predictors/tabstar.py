"""TabSTAR adapter: the README's semantic-feature challenger.

TabSTAR's distinguishing idea is that it reads column *names* and categorical
*values* as text, so it can transfer knowledge about what a column means. That
is a real advantage on tables full of natural-language columns.

The README is candid that the advantage is limited here: 26 of the v1 schema's
28 features are numeric, and the two categorical ones are club names. This
adapter therefore preserves the schema's real column names when handing data
over, which is the only way TabSTAR's semantic path can contribute anything at
all — passing `f0, f1, ...` would disable the feature being tested.

Licence needs care: the Hugging Face model card lists CC-BY-4.0 while the
source repository describes its code as MIT. Both are recorded, and the README
requires verifying them separately before any deployment.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from epl_predictor.predictors.foundation import (
    CANDIDATES,
    FoundationPredictor,
    LabelVector,
    require_library,
)
from epl_predictor.predictors.tabular import FloatMatrix

MODEL_FILENAME = "tabstar.joblib"

#: Columns in the order `FoundationPredictor` encodes them, so the encoded
#: matrix can be given back its real names for TabSTAR's semantic path.
ENCODED_COLUMNS = list(CATEGORICAL_FEATURES + NUMERIC_FEATURES)


class TabSTARPredictor(FoundationPredictor):
    """TabSTAR over the v1 feature schema."""

    adapter_type = "tabstar"
    spec = CANDIDATES["tabstar"]

    def __init__(
        self,
        *args: Any,
        verbose: bool = False,
        time_limit: int | None = 1800,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.time_limit = time_limit
        self.classifier: Any = None

    def _named_frame(self, features: FloatMatrix) -> pd.DataFrame:
        """Restore the schema's column names onto an encoded matrix."""
        return pd.DataFrame(features, columns=ENCODED_COLUMNS)

    def _fit_encoded(self, features: FloatMatrix, labels: LabelVector) -> None:
        # The estimator lives in `tabstar.tabstar_model`; the top-level package
        # exports nothing, so importing `tabstar` alone would not find it.
        model_module = require_library("tabstar.tabstar_model", self.spec)
        self.classifier = model_module.TabSTARClassifier(
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
            time_limit=self.time_limit,
        )
        self.classifier.fit(self._named_frame(features), labels)

    def _predict_encoded(self, features: FloatMatrix) -> FloatMatrix:
        if self.classifier is None:
            raise RuntimeError("TabSTARPredictor has no classifier; fit or load it first.")
        return np.asarray(
            self.classifier.predict_proba(self._named_frame(features)), dtype=np.float64
        )

    def _class_order(self) -> list[str]:
        if self.classifier is None or not hasattr(self.classifier, "classes_"):
            return list(OUTCOME_CLASSES)
        return [OUTCOME_CLASSES[int(index)] for index in self.classifier.classes_]

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            **super()._extra_metadata(),
            "encoded_columns": ENCODED_COLUMNS,
            "time_limit": self.time_limit,
        }

    def _save_estimator(self, artifact_path: Path) -> None:
        joblib.dump(self.classifier, artifact_path / MODEL_FILENAME)

    def _load_estimator(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        self.classifier = joblib.load(artifact_path / MODEL_FILENAME)
