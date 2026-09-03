"""CatBoost baseline: the bar every foundation-model candidate must clear.

Gradient-boosted trees are the strongest thing to reach for on small
heterogeneous tabular data, and CatBoost in particular handles the two
awkward parts of this feature set natively: club names as categories, and the
"no history yet" nulls, which it routes down their own branch instead of
needing an imputed value.

The README's selection gate is deliberately anchored here. A tabular
foundation model is promoted only if it beats this, so it is a real
possibility that CatBoost simply wins and stays active.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from epl_predictor.predictors.tabular import FloatMatrix, TabularPredictor

MODEL_FILENAME = "model.cbm"
RANDOM_SEED = 20260903

# Early stopping uses the validation period, so the iteration count is an
# upper bound rather than a tuned value.
DEFAULT_ITERATIONS = 1000
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_DEPTH = 6
EARLY_STOPPING_ROUNDS = 100


class CatBoostUnavailableError(RuntimeError):
    """CatBoost is not installed."""


def _import_catboost() -> Any:
    try:
        import catboost
    except ImportError as exc:  # pragma: no cover - exercised by env without extra
        raise CatBoostUnavailableError(
            "CatBoost is not installed. Install it with `uv sync --extra catboost`."
        ) from exc
    return catboost


class CatBoostPredictor(TabularPredictor):
    """Multiclass CatBoost over the v1 feature set."""

    adapter_type = "catboost"

    def __init__(
        self,
        name: str = "catboost-epl",
        version: str = "0.1.0",
        calibrator: Any = None,
        iterations: int = DEFAULT_ITERATIONS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        depth: int = DEFAULT_DEPTH,
    ):
        super().__init__(name=name, version=version, calibrator=calibrator)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.model: Any = None
        self.best_iteration: int | None = None

    def _pool(self, frame: pd.DataFrame, labels: pd.Series | None = None) -> Any:
        """Wrap a feature frame as a CatBoost Pool with declared categories."""
        catboost = _import_catboost()
        features = frame[list(FEATURE_COLUMNS)].copy()
        for column in CATEGORICAL_FEATURES:
            # CatBoost requires categorical values as strings with no nulls.
            features[column] = features[column].astype("string").fillna("unknown")
        for column in features.columns:
            if column not in CATEGORICAL_FEATURES:
                # Nullable extension dtypes are converted to float so pandas.NA
                # becomes the NaN that CatBoost treats as a missing value.
                features[column] = pd.to_numeric(features[column], errors="coerce").astype(
                    "float64"
                )

        return catboost.Pool(
            data=features,
            label=None if labels is None else list(labels),
            cat_features=list(CATEGORICAL_FEATURES),
        )

    def _fit_frame(self, train: pd.DataFrame, validation: pd.DataFrame | None) -> None:
        """Fit with early stopping against the validation period when available."""
        catboost = _import_catboost()

        self.model = catboost.CatBoostClassifier(
            loss_function="MultiClass",
            # Declaring the names rather than a count keeps CatBoost's output
            # columns in the canonical order instead of its own sorted one.
            class_names=list(OUTCOME_CLASSES),
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=RANDOM_SEED,
            allow_writing_files=False,
            verbose=False,
        )

        train_pool = self._pool(train, train[TARGET_COLUMN])
        evaluation_pool = (
            self._pool(validation, validation[TARGET_COLUMN]) if validation is not None else None
        )

        self.model.fit(
            train_pool,
            eval_set=evaluation_pool,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if evaluation_pool else None,
            verbose=False,
        )
        self.best_iteration = (
            int(self.model.get_best_iteration()) if evaluation_pool else self.iterations
        )

    def _predict_matrix(self, features: pd.DataFrame) -> FloatMatrix:
        if self.model is None:
            raise RuntimeError("CatBoost model is not fitted")
        raw = self.model.predict_proba(self._pool(features))
        return _reorder_to_canonical(raw, [str(value) for value in self.model.classes_])

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": RANDOM_SEED,
            "best_iteration": self.best_iteration,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "null_handling": "native; missing values take their own split branch",
        }

    def _save_model(self, artifact_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted CatBoost model")
        self.model.save_model(str(artifact_path / MODEL_FILENAME))

    def _load_model(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        catboost = _import_catboost()
        self.model = catboost.CatBoostClassifier()
        self.model.load_model(str(artifact_path / MODEL_FILENAME))
        self.iterations = int(metadata.get("iterations", DEFAULT_ITERATIONS))
        self.learning_rate = float(metadata.get("learning_rate", DEFAULT_LEARNING_RATE))
        self.depth = int(metadata.get("depth", DEFAULT_DEPTH))
        self.best_iteration = metadata.get("best_iteration")


def _reorder_to_canonical(matrix: Any, classes: list[str]) -> FloatMatrix:
    """Map CatBoost's class order onto the canonical order."""
    columns = np.asarray(matrix, dtype=np.float64)
    ordered = np.zeros((len(columns), len(OUTCOME_CLASSES)), dtype=np.float64)

    for target, outcome in enumerate(OUTCOME_CLASSES):
        if outcome in classes:
            ordered[:, target] = columns[:, classes.index(outcome)]

    return ordered
