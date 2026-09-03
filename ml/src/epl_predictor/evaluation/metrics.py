"""Metrics for three-class match-outcome probabilities.

Multiclass log loss is the primary selection metric because it scores all
three probabilities and punishes confident mistakes far harder than uncertain
ones. Accuracy alone is misleading here: a model that never predicts a draw
can still look respectable on accuracy while being useless for the outcome
that decides roughly a quarter of Premier League matches. Draw precision and
recall are therefore reported as first-class numbers, not buried in a
per-class table.

Every function takes probabilities in the canonical class order from
`epl_predictor.OUTCOME_CLASSES`, so a column can never silently mean a
different outcome than a caller intends.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from epl_predictor import OUTCOME_CLASSES, Outcome

# Probabilities are clipped before taking a logarithm: a model that assigns
# exactly zero to the outcome that happened would otherwise score infinite
# loss and make every comparison meaningless.
PROBABILITY_FLOOR = 1e-15

DRAW_INDEX = OUTCOME_CLASSES.index(Outcome.DRAW.value)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class ShapeMismatchError(ValueError):
    """Probabilities and labels do not describe the same set of matches."""


def as_probability_matrix(probabilities: Any) -> FloatArray:
    """Coerce input into an (n, 3) float matrix in canonical class order.

    Raises:
        ShapeMismatchError: The input is not shaped like three-class output.
    """
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(OUTCOME_CLASSES):
        raise ShapeMismatchError(
            f"Expected probabilities with {len(OUTCOME_CLASSES)} columns in the order "
            f"{list(OUTCOME_CLASSES)}, got shape {matrix.shape}."
        )
    return matrix


def as_label_indices(labels: Any) -> IntArray:
    """Convert outcome labels into canonical class indices.

    Raises:
        ValueError: A label is not one of the canonical outcome classes.
    """
    lookup = {name: index for index, name in enumerate(OUTCOME_CLASSES)}
    indices: list[int] = []

    for label in np.asarray(labels).ravel():
        if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
            if not 0 <= int(label) < len(OUTCOME_CLASSES):
                raise ValueError(f"Class index {label} is outside the three outcomes")
            indices.append(int(label))
            continue
        name = str(label)
        if name not in lookup:
            raise ValueError(
                f"Unknown outcome label {name!r}; expected one of {list(OUTCOME_CLASSES)}"
            )
        indices.append(lookup[name])

    return np.asarray(indices, dtype=np.int64)


def _aligned(probabilities: Any, labels: Any) -> tuple[FloatArray, IntArray]:
    matrix = as_probability_matrix(probabilities)
    indices = as_label_indices(labels)
    if len(matrix) != len(indices):
        raise ShapeMismatchError(f"Got {len(matrix)} probability rows for {len(indices)} labels.")
    if not len(matrix):
        raise ShapeMismatchError("Cannot evaluate an empty set of predictions")
    return matrix, indices


def log_loss(probabilities: Any, labels: Any) -> float:
    """Mean negative log probability of the observed outcomes. Lower is better."""
    matrix, indices = _aligned(probabilities, labels)
    observed = matrix[np.arange(len(indices)), indices]
    return float(-np.mean(np.log(np.clip(observed, PROBABILITY_FLOOR, 1.0))))


def brier_score(probabilities: Any, labels: Any) -> float:
    """Multiclass Brier score: mean squared error against the one-hot truth.

    Ranges from 0 (perfect) to 2 (confidently wrong on every match).
    """
    matrix, indices = _aligned(probabilities, labels)
    truth = np.zeros_like(matrix)
    truth[np.arange(len(indices)), indices] = 1.0
    return float(np.mean(np.sum((matrix - truth) ** 2, axis=1)))


def accuracy(probabilities: Any, labels: Any) -> float:
    """Share of matches whose most likely outcome was the one that happened."""
    matrix, indices = _aligned(probabilities, labels)
    return float(np.mean(np.argmax(matrix, axis=1) == indices))


def confusion_matrix(probabilities: Any, labels: Any) -> IntArray:
    """Counts of actual (rows) against predicted (columns), in class order."""
    matrix, indices = _aligned(probabilities, labels)
    predicted = np.argmax(matrix, axis=1)
    counts = np.zeros((len(OUTCOME_CLASSES), len(OUTCOME_CLASSES)), dtype=np.int64)
    for actual, guess in zip(indices, predicted, strict=True):
        counts[actual, guess] += 1
    return counts


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1, and support for one outcome class."""

    outcome: str
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Zero rather than NaN when a class was never predicted or never occurred."""
    return float(numerator / denominator) if denominator else 0.0


def per_class_metrics(probabilities: Any, labels: Any) -> dict[str, ClassMetrics]:
    """Precision, recall, and F1 for each of the three outcomes."""
    counts = confusion_matrix(probabilities, labels)
    results: dict[str, ClassMetrics] = {}

    for index, outcome in enumerate(OUTCOME_CLASSES):
        true_positive = int(counts[index, index])
        predicted = int(counts[:, index].sum())
        support = int(counts[index, :].sum())

        precision = _safe_ratio(true_positive, predicted)
        recall = _safe_ratio(true_positive, support)
        results[outcome] = ClassMetrics(
            outcome=outcome,
            precision=precision,
            recall=recall,
            f1=_safe_ratio(2 * precision * recall, precision + recall),
            support=support,
            predicted=predicted,
        )

    return results


def expected_calibration_error(probabilities: Any, labels: Any, bins: int = 10) -> float:
    """Expected calibration error over the predicted-class confidence.

    Confidences are grouped into equal-width bins; the error is the
    support-weighted mean gap between a bin's mean confidence and its observed
    accuracy. Zero means stated confidence matches reality.
    """
    matrix, indices = _aligned(probabilities, labels)
    confidence = np.max(matrix, axis=1)
    correct = (np.argmax(matrix, axis=1) == indices).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        # Upper-inclusive on the final bin so a confidence of exactly 1 counts.
        in_bin = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            in_bin |= confidence == 0.0
        if not in_bin.any():
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        total += gap * in_bin.sum() / len(confidence)

    return float(total)


def calibration_curve(
    probabilities: Any, labels: Any, outcome: Outcome | str = Outcome.DRAW, bins: int = 10
) -> list[dict[str, float]]:
    """Reliability curve for one outcome, for the calibration plot in a report."""
    matrix, indices = _aligned(probabilities, labels)
    name = outcome.value if isinstance(outcome, Outcome) else str(outcome)
    column = OUTCOME_CLASSES.index(name)

    predicted = matrix[:, column]
    observed = (indices == column).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[dict[str, float]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (predicted > lower) & (predicted <= upper)
        if lower == 0.0:
            in_bin |= predicted == 0.0
        if not in_bin.any():
            continue
        curve.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "mean_predicted": float(predicted[in_bin].mean()),
                "observed_rate": float(observed[in_bin].mean()),
                "count": int(in_bin.sum()),
            }
        )
    return curve


@dataclass
class ResourceMetrics:
    """Cost of using a model, which the README weighs alongside accuracy."""

    inference_latency_ms: float | None = None
    peak_memory_mb: float | None = None
    artifact_size_bytes: int | None = None
    training_duration_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Every metric the README requires for one model on one period."""

    model_name: str
    period: str
    rows: int
    log_loss: float
    brier_score: float
    accuracy: float
    expected_calibration_error: float
    draw_precision: float
    draw_recall: float
    draw_f1: float
    draw_predicted: int
    draw_support: int
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "period": self.period,
            "rows": self.rows,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "draw_precision": self.draw_precision,
            "draw_recall": self.draw_recall,
            "draw_f1": self.draw_f1,
            "draw_predicted": self.draw_predicted,
            "draw_support": self.draw_support,
            "per_class": {name: metrics.as_dict() for name, metrics in self.per_class.items()},
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_classes": list(OUTCOME_CLASSES),
            "resources": self.resources.as_dict(),
        }


def evaluate(
    probabilities: Any,
    labels: Any,
    model_name: str = "unnamed",
    period: str = "unspecified",
    resources: ResourceMetrics | None = None,
    bins: int = 10,
) -> EvaluationResult:
    """Score one model's probabilities over one evaluation period."""
    matrix, indices = _aligned(probabilities, labels)
    per_class = per_class_metrics(matrix, indices)
    draw = per_class[Outcome.DRAW.value]

    return EvaluationResult(
        model_name=model_name,
        period=period,
        rows=len(matrix),
        log_loss=log_loss(matrix, indices),
        brier_score=brier_score(matrix, indices),
        accuracy=accuracy(matrix, indices),
        expected_calibration_error=expected_calibration_error(matrix, indices, bins=bins),
        draw_precision=draw.precision,
        draw_recall=draw.recall,
        draw_f1=draw.f1,
        draw_predicted=draw.predicted,
        draw_support=draw.support,
        per_class=per_class,
        confusion_matrix=confusion_matrix(matrix, indices).tolist(),
        resources=resources or ResourceMetrics(),
    )
