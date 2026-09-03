"""Metrics, walk-forward backtesting, and comparison reports."""

from epl_predictor.evaluation.backtest import (
    BacktestResult,
    Fold,
    InsufficientHistoryError,
    walk_forward_backtest,
    walk_forward_folds,
)
from epl_predictor.evaluation.metrics import (
    ClassMetrics,
    EvaluationResult,
    ResourceMetrics,
    ShapeMismatchError,
    accuracy,
    brier_score,
    calibration_curve,
    confusion_matrix,
    evaluate,
    expected_calibration_error,
    log_loss,
    per_class_metrics,
)
from epl_predictor.evaluation.report import (
    ComparisonReport,
    beats_baseline,
    write_report,
)

__all__ = [
    "BacktestResult",
    "ClassMetrics",
    "ComparisonReport",
    "EvaluationResult",
    "Fold",
    "InsufficientHistoryError",
    "ResourceMetrics",
    "ShapeMismatchError",
    "accuracy",
    "beats_baseline",
    "brier_score",
    "calibration_curve",
    "confusion_matrix",
    "evaluate",
    "expected_calibration_error",
    "log_loss",
    "per_class_metrics",
    "walk_forward_backtest",
    "walk_forward_folds",
    "write_report",
]
