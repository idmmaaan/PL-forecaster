"""Multinomial logistic regression baseline.

A linear model on the v1 features. It is the first baseline that actually uses
the feature set, so the gap between it and the class-frequency baseline
measures how much signal the features carry, and the gap between it and
CatBoost measures how much of that signal is non-linear.

Preprocessing lives inside the fitted pipeline rather than beside it, so the
imputation medians, scaling parameters, and club encoding are all saved with
the artifact. Recomputing any of them at inference time would make the model's
output depend on whatever data happened to be loaded.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from epl_predictor.predictors.tabular import FloatMatrix, TabularPredictor

MODEL_FILENAME = "model.joblib"
RANDOM_STATE = 20260903


def build_pipeline(regularisation: float = 1.0, max_iter: int = 2000) -> Pipeline:
    """A preprocessing and estimator pipeline for the v1 feature schema.

    Nulls are imputed with the training median here rather than in the feature
    builder, so the "no history yet" nulls stay honest in the dataset and each
    model decides for itself how to handle them.
    """
    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        (
                            "clubs",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            list(CATEGORICAL_FEATURES),
                        ),
                        (
                            "numbers",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            list(NUMERIC_FEATURES),
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "estimate",
                LogisticRegression(
                    C=regularisation,
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


class LogisticRegressionPredictor(TabularPredictor):
    """Multinomial logistic regression over the v1 feature set."""

    adapter_type = "logistic-regression"

    def __init__(
        self,
        name: str = "logistic-regression-epl",
        version: str = "0.1.0",
        calibrator: Any = None,
        regularisation: float = 1.0,
    ):
        super().__init__(name=name, version=version, calibrator=calibrator)
        self.regularisation = regularisation
        self.pipeline: Pipeline | None = None

    def _fit_frame(self, train: pd.DataFrame, validation: pd.DataFrame | None) -> None:
        """Fit on the training period only.

        `validation` is unused: it is reserved for the calibrator, which the
        base class fits after this returns.
        """
        self.pipeline = build_pipeline(regularisation=self.regularisation)
        self.pipeline.fit(
            train[list(CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES)],
            train[TARGET_COLUMN],
        )

    def _predict_matrix(self, features: pd.DataFrame) -> FloatMatrix:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted")
        raw = self.pipeline.predict_proba(features)
        return _reorder_to_canonical(raw, list(self.pipeline.classes_))

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            "regularisation": self.regularisation,
            "random_state": RANDOM_STATE,
            "imputation": "median, fitted on the training period",
        }

    def _save_model(self, artifact_path: Path) -> None:
        joblib.dump(self.pipeline, artifact_path / MODEL_FILENAME)

    def _load_model(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        self.pipeline = joblib.load(artifact_path / MODEL_FILENAME)
        self.regularisation = float(metadata.get("regularisation", 1.0))


def _reorder_to_canonical(matrix: Any, classes: list[Any]) -> FloatMatrix:
    """Map an estimator's own class order onto the canonical order.

    scikit-learn sorts classes alphabetically, which for these labels gives
    AWAY_WIN, DRAW, HOME_WIN. Handing that back unchanged would swap home and
    away on every prediction with no error anywhere.
    """
    names = [str(value) for value in classes]
    columns = np.asarray(matrix, dtype=np.float64)
    ordered = np.zeros((len(columns), len(OUTCOME_CLASSES)), dtype=np.float64)

    for target, outcome in enumerate(OUTCOME_CLASSES):
        if outcome in names:
            ordered[:, target] = columns[:, names.index(outcome)]

    return ordered
