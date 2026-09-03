"""Shared base for predictors trained on the v1 feature table.

The `Predictor` contract is row-at-a-time (`predict_proba` takes one feature
dict) because that is what the API needs. Training and evaluation, however,
work on whole frames. This class supplies the bridge once, so every adapter
implements only `_fit_frame` and `_predict_matrix` and inherits identical
handling of column order, class order, and probability normalisation.

Getting that shared handling right in one place matters: a permuted column or
class order would not raise an error anywhere, it would just quietly degrade
every metric.
"""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from epl_predictor import OUTCOME_CLASSES, PROBABILITY_KEYS
from epl_predictor.features.builder import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from epl_predictor.predictors.base import Predictor
from epl_predictor.predictors.calibration import Calibrator, IdentityCalibrator, load_calibrator

METADATA_FILENAME = "predictor.json"
CALIBRATOR_FILENAME = "calibrator.joblib"

FloatMatrix = npt.NDArray[np.float64]


class NotFittedError(RuntimeError):
    """A predictor was asked for probabilities before being fitted."""


class TabularPredictor(Predictor):
    """A predictor over the v1 feature schema.

    Subclasses implement `_fit_frame` and `_predict_matrix`; everything about
    the schema, the class order, calibration, and the artifact layout is
    handled here.
    """

    adapter_type: str = "tabular"

    def __init__(self, name: str, version: str = "0.1.0", calibrator: Calibrator | None = None):
        self._model_name = name
        self._model_version = version
        self.calibrator: Calibrator = calibrator or IdentityCalibrator()
        self.fitted = False
        self.feature_schema_version = FEATURE_SCHEMA_VERSION

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    # --- Subclass responsibilities ------------------------------------------

    @abstractmethod
    def _fit_frame(self, train: pd.DataFrame, validation: pd.DataFrame | None) -> None:
        """Fit on the training frame, optionally using validation for tuning."""

    @abstractmethod
    def _predict_matrix(self, features: pd.DataFrame) -> FloatMatrix:
        """Return uncalibrated (n, 3) probabilities in canonical class order."""

    def _extra_metadata(self) -> dict[str, Any]:
        """Adapter-specific values to record in the artifact."""
        return {}

    # --- Shared behaviour ---------------------------------------------------

    def fit(self, train_data: Any, validation_data: Any = None) -> None:
        """Fit the model, then fit any calibrator on the validation period only.

        Calibration must never see the training rows it is correcting, nor the
        test period. Fitting it here, on `validation_data`, is what keeps that
        true for every adapter.
        """
        train = _require_frame(train_data, "train_data")
        validation = (
            _require_frame(validation_data, "validation_data")
            if validation_data is not None and len(validation_data)
            else None
        )

        self._fit_frame(train, validation)
        self.fitted = True

        if validation is not None:
            self.calibrator.fit(
                self._predict_matrix(features_frame(validation)),
                validation[TARGET_COLUMN],
            )

    def predict_matrix(self, features: pd.DataFrame) -> FloatMatrix:
        """Calibrated (n, 3) probabilities for a frame of fixtures."""
        if not self.fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before predicting.")
        raw = self._predict_matrix(features_frame(features))
        return normalise_matrix(self.calibrator.transform(raw))

    def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
        """Probabilities for one fixture, keyed as the API contract requires."""
        row = pd.DataFrame([{column: features.get(column) for column in FEATURE_COLUMNS}])
        matrix = self.predict_matrix(row)
        return dict(zip(PROBABILITY_KEYS, (float(value) for value in matrix[0]), strict=True))

    # --- Artifacts ----------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Everything needed to interpret this artifact's predictions."""
        return {
            "adapter_type": self.adapter_type,
            "model_name": self._model_name,
            "model_version": self._model_version,
            "feature_schema_version": self.feature_schema_version,
            "class_order": list(OUTCOME_CLASSES),
            "probability_keys": list(PROBABILITY_KEYS),
            "feature_columns": list(FEATURE_COLUMNS),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "numeric_features": list(NUMERIC_FEATURES),
            "calibration": self.calibrator.as_dict(),
            **self._extra_metadata(),
        }

    def save(self, artifact_path: Path) -> None:
        """Write metadata, the fitted calibrator, and the model to a directory."""
        artifact_path = Path(artifact_path)
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / METADATA_FILENAME).write_text(
            json.dumps(self.metadata(), indent=2, default=str) + "\n"
        )
        # The calibrator is pickled rather than described in metadata: an
        # isotonic mapping cannot be rebuilt from its method name, and an
        # artifact that reloaded as uncalibrated would silently change every
        # probability it returned.
        joblib.dump(self.calibrator, artifact_path / CALIBRATOR_FILENAME)
        self._save_model(artifact_path)

    @abstractmethod
    def _save_model(self, artifact_path: Path) -> None:
        """Persist the fitted estimator itself."""

    @classmethod
    def load(cls, artifact_path: Path) -> "TabularPredictor":
        """Reload an artifact written by `save`.

        Raises:
            FileNotFoundError: The artifact directory has no metadata.
        """
        artifact_path = Path(artifact_path)
        metadata_file = artifact_path / METADATA_FILENAME
        if not metadata_file.exists():
            raise FileNotFoundError(f"No {METADATA_FILENAME} in {artifact_path}")

        metadata = json.loads(metadata_file.read_text())
        calibrator_file = artifact_path / CALIBRATOR_FILENAME
        calibrator = (
            joblib.load(calibrator_file)
            if calibrator_file.exists()
            else load_calibrator(metadata.get("calibration"))
        )
        predictor = cls(
            name=metadata["model_name"],
            version=metadata["model_version"],
            calibrator=calibrator,
        )
        predictor.feature_schema_version = metadata.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION
        )
        predictor._load_model(artifact_path, metadata)
        predictor.fitted = True
        return predictor

    @abstractmethod
    def _load_model(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        """Restore the fitted estimator from an artifact directory."""


def _require_frame(data: Any, argument: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{argument} must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError(f"{argument} is empty")
    return data


def features_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Select the v1 feature columns in schema order.

    Raises:
        ValueError: A schema column is absent.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Feature frame is missing column(s): {', '.join(missing)}")
    return data[list(FEATURE_COLUMNS)]


def normalise_matrix(matrix: FloatMatrix) -> FloatMatrix:
    """Clip to non-negative and rescale each row to sum to one.

    Guards the API's contract at the last possible moment: a row that does not
    sum to one would be rejected downstream, and floating-point drift in an
    adapter should not be allowed to cause that.
    """
    clipped = np.clip(np.asarray(matrix, dtype=np.float64), 0.0, None)
    totals = clipped.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("A predicted probability row sums to zero and cannot be normalised")
    return clipped / totals
