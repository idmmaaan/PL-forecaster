"""Probability calibration, fitted on the chronological validation period only.

A model can rank fixtures well and still state its confidence badly: "70 %
home win" is only useful if the home side actually wins about 70 % of such
matches. Log loss punishes miscalibration directly, so correcting it is often
a larger win than changing the model.

Three methods are offered for comparison, as the README asks:

- `IdentityCalibrator` leaves the output alone, the control condition;
- `TemperatureScaler` sharpens or softens every probability with one shared
  parameter, which cannot reorder outcomes and cannot overfit;
- `IsotonicCalibrator` and `SigmoidCalibrator` fit a per-class mapping, which
  is more flexible and correspondingly easier to overfit on 380 matches.

Each calibrator is fitted on validation predictions and serialises into the
model artifact, because a probability is not reproducible without the
transformation that produced it.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.evaluation.metrics import as_label_indices

FloatMatrix = npt.NDArray[np.float64]

# Probabilities are floored before any logarithm so a zero cannot produce an
# infinite objective during fitting.
EPSILON = 1e-12

# Temperature below 1 sharpens, above 1 softens. The bounds are wide enough to
# cover any realistic miscalibration without letting the optimiser wander.
TEMPERATURE_BOUNDS = (0.2, 5.0)


@runtime_checkable
class Calibrator(Protocol):
    """Transforms raw probabilities into calibrated ones."""

    method: str

    def fit(self, probabilities: Any, labels: Any) -> None:
        """Fit on validation predictions and their observed outcomes."""

    def transform(self, probabilities: Any) -> FloatMatrix:
        """Return calibrated probabilities in canonical class order."""

    def as_dict(self) -> dict[str, Any]:
        """Serialise into the model artifact."""


def _as_matrix(probabilities: Any) -> FloatMatrix:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != len(OUTCOME_CLASSES):
        raise ValueError(
            f"Expected {len(OUTCOME_CLASSES)} probability columns, got {matrix.shape[1]}"
        )
    return matrix


def _renormalise(matrix: FloatMatrix) -> FloatMatrix:
    clipped = np.clip(matrix, EPSILON, None)
    return clipped / clipped.sum(axis=1, keepdims=True)


class IdentityCalibrator:
    """Leaves probabilities untouched: the uncalibrated control condition."""

    method = "identity"

    def fit(self, probabilities: Any, labels: Any) -> None:
        """Nothing to fit."""

    def transform(self, probabilities: Any) -> FloatMatrix:
        return _renormalise(_as_matrix(probabilities))

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method}


class TemperatureScaler:
    """Single-parameter scaling of the log probabilities.

    Raising every probability to the power 1/T and renormalising sharpens the
    distribution when T < 1 and flattens it when T > 1. Because one parameter
    is shared across all classes and all rows, the argmax never changes: this
    fixes confidence without touching ranking, and cannot overfit a validation
    season of 380 matches.
    """

    method = "temperature"

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, probabilities: Any, labels: Any) -> None:
        """Choose the temperature that minimises validation log loss."""
        matrix = _renormalise(_as_matrix(probabilities))
        indices = as_label_indices(labels)
        log_probabilities = np.log(matrix)
        rows = np.arange(len(indices))

        def negative_log_likelihood(temperature: float) -> float:
            scaled = log_probabilities / temperature
            # Subtract the row max before exponentiating, for numerical safety.
            scaled -= scaled.max(axis=1, keepdims=True)
            normalised = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return float(-normalised[rows, indices].mean())

        result = minimize_scalar(
            negative_log_likelihood, bounds=TEMPERATURE_BOUNDS, method="bounded"
        )
        self.temperature = float(result.x)

    def transform(self, probabilities: Any) -> FloatMatrix:
        matrix = _renormalise(_as_matrix(probabilities))
        return _renormalise(matrix ** (1.0 / self.temperature))

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature}


class _PerClassCalibrator:
    """Shared machinery for one-vs-rest calibration followed by renormalising.

    Fitting each class independently means the three calibrated values do not
    naturally sum to one, so they are rescaled afterwards. That rescaling is
    the reason these methods can, unlike temperature scaling, change which
    outcome is most likely.
    """

    method = "per-class"

    def __init__(self) -> None:
        self.models: list[Any] = []

    def _new_model(self) -> Any:
        raise NotImplementedError

    def _fit_model(self, model: Any, predicted: FloatMatrix, observed: FloatMatrix) -> None:
        raise NotImplementedError

    def _apply_model(self, model: Any, predicted: FloatMatrix) -> FloatMatrix:
        raise NotImplementedError

    def fit(self, probabilities: Any, labels: Any) -> None:
        matrix = _renormalise(_as_matrix(probabilities))
        indices = as_label_indices(labels)

        self.models = []
        for column in range(len(OUTCOME_CLASSES)):
            observed = (indices == column).astype(np.float64)
            model = self._new_model()
            if len(np.unique(observed)) < 2:
                # This outcome never occurred in validation, so there is
                # nothing to learn; fall back to the raw probability.
                self.models.append(None)
                continue
            self._fit_model(model, matrix[:, column], observed)
            self.models.append(model)

    def transform(self, probabilities: Any) -> FloatMatrix:
        matrix = _renormalise(_as_matrix(probabilities))
        if not self.models:
            return matrix

        calibrated = np.empty_like(matrix)
        for column, model in enumerate(self.models):
            if model is None:
                calibrated[:, column] = matrix[:, column]
            else:
                calibrated[:, column] = self._apply_model(model, matrix[:, column])
        return _renormalise(calibrated)


class IsotonicCalibrator(_PerClassCalibrator):
    """Per-class isotonic regression: flexible, monotone, and easy to overfit."""

    method = "isotonic"

    def _new_model(self) -> IsotonicRegression:
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def _fit_model(
        self, model: IsotonicRegression, predicted: FloatMatrix, observed: FloatMatrix
    ) -> None:
        model.fit(predicted, observed)

    def _apply_model(self, model: IsotonicRegression, predicted: FloatMatrix) -> FloatMatrix:
        return np.asarray(model.predict(predicted), dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "fitted_classes": len(self.models)}


class SigmoidCalibrator(_PerClassCalibrator):
    """Per-class Platt scaling: a two-parameter logistic fit per outcome."""

    method = "sigmoid"

    def _new_model(self) -> LogisticRegression:
        return LogisticRegression(solver="lbfgs")

    def _fit_model(
        self, model: LogisticRegression, predicted: FloatMatrix, observed: FloatMatrix
    ) -> None:
        model.fit(_logit(predicted).reshape(-1, 1), observed)

    def _apply_model(self, model: LogisticRegression, predicted: FloatMatrix) -> FloatMatrix:
        probabilities = model.predict_proba(_logit(predicted).reshape(-1, 1))
        return np.asarray(probabilities[:, 1], dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "fitted_classes": len(self.models)}


def _logit(probabilities: FloatMatrix) -> FloatMatrix:
    """Log odds, clipped so 0 and 1 do not become infinite."""
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


CALIBRATORS: dict[str, type[Any]] = {
    IdentityCalibrator.method: IdentityCalibrator,
    TemperatureScaler.method: TemperatureScaler,
    IsotonicCalibrator.method: IsotonicCalibrator,
    SigmoidCalibrator.method: SigmoidCalibrator,
}


def make_calibrator(method: str) -> Calibrator:
    """Build a calibrator by name.

    Raises:
        ValueError: The method is unknown.
    """
    if method not in CALIBRATORS:
        raise ValueError(
            f"Unknown calibration method {method!r}; expected one of {sorted(CALIBRATORS)}"
        )
    calibrator: Calibrator = CALIBRATORS[method]()
    return calibrator


def load_calibrator(payload: dict[str, Any] | None) -> Calibrator:
    """Rebuild a calibrator from its serialised form in an artifact.

    Per-class calibrators cannot be reconstructed from metadata alone: their
    fitted mappings live in the joblib model file, so an artifact that stored
    only the method name is restored as uncalibrated rather than silently
    pretending to be calibrated.
    """
    if not payload:
        return IdentityCalibrator()

    method = payload.get("method", IdentityCalibrator.method)
    if method == TemperatureScaler.method:
        return TemperatureScaler(temperature=float(payload.get("temperature", 1.0)))
    if method in {IsotonicCalibrator.method, SigmoidCalibrator.method}:
        return IdentityCalibrator()
    return IdentityCalibrator()
