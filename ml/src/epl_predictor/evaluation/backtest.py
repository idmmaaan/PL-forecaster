"""Walk-forward backtesting over whole seasons.

A random train/test split would let a model learn from matches that had not
happened yet, so the README forbids it as the main evaluation method. This
module instead rolls the origin forward: for each evaluation season, the model
is fitted on everything that preceded it and scored on that season alone.

The result is a set of out-of-sample scores that each respect chronology, plus
an aggregate over all of them. Because the model is refitted per fold, the
scores also reveal whether performance is stable across seasons or depends on
one lucky year.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from epl_predictor.evaluation.metrics import (
    EvaluationResult,
    ResourceMetrics,
    evaluate,
)
from epl_predictor.features.builder import TARGET_COLUMN

# Fitting a fold on a handful of matches produces noise, not a baseline.
MINIMUM_TRAINING_ROWS = 200


class InsufficientHistoryError(ValueError):
    """A fold has too little preceding data to fit on."""


@dataclass
class Fold:
    """One walk-forward step: fit on the past, score one season."""

    season: int
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_seasons(self) -> list[int]:
        return sorted(int(year) for year in self.train["season_start_year"].unique())

    def as_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "train_rows": len(self.train),
            "test_rows": len(self.test),
            "train_seasons": self.train_seasons,
        }


def walk_forward_folds(
    table: pd.DataFrame,
    evaluation_seasons: Sequence[int],
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> list[Fold]:
    """Build one fold per evaluation season, training only on earlier seasons.

    Raises:
        InsufficientHistoryError: A requested season has too little history.
    """
    folds: list[Fold] = []

    for season in sorted(evaluation_seasons):
        train = table[table["season_start_year"] < season]
        test = table[table["season_start_year"] == season]

        if test.empty:
            continue
        if len(train) < minimum_training_rows:
            raise InsufficientHistoryError(
                f"Season {season} has only {len(train)} preceding rows, fewer than the "
                f"{minimum_training_rows} required to fit a fold."
            )

        folds.append(
            Fold(
                season=season,
                train=train.reset_index(drop=True),
                test=test.reset_index(drop=True),
            )
        )

    return folds


@dataclass
class BacktestResult:
    """Per-season results plus their aggregate."""

    model_name: str
    folds: list[EvaluationResult] = field(default_factory=list)
    fold_definitions: list[dict[str, Any]] = field(default_factory=list)
    aggregate: EvaluationResult | None = None

    @property
    def seasons(self) -> list[int]:
        return [int(result.period) for result in self.folds]

    def metric_by_season(self, metric: str = "log_loss") -> dict[int, float]:
        return {int(result.period): float(getattr(result, metric)) for result in self.folds}

    def mean(self, metric: str = "log_loss") -> float:
        """Unweighted mean across folds, so no season dominates by size."""
        values = [float(getattr(result, metric)) for result in self.folds]
        return float(np.mean(values)) if values else float("nan")

    def standard_deviation(self, metric: str = "log_loss") -> float:
        """Spread across folds: a large value means the model is season-sensitive."""
        values = [float(getattr(result, metric)) for result in self.folds]
        return float(np.std(values)) if len(values) > 1 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "seasons": self.seasons,
            "folds": [result.as_dict() for result in self.folds],
            "fold_definitions": self.fold_definitions,
            "aggregate": self.aggregate.as_dict() if self.aggregate else None,
            "mean_log_loss": self.mean("log_loss"),
            "log_loss_standard_deviation": self.standard_deviation("log_loss"),
            "mean_draw_recall": self.mean("draw_recall"),
        }


# A fold trainer receives the fold's training rows and returns a callable that
# turns test rows into an (n, 3) probability matrix in canonical class order.
FoldTrainer = Callable[[pd.DataFrame], Callable[[pd.DataFrame], Any]]


def walk_forward_backtest(
    table: pd.DataFrame,
    train_fold: FoldTrainer,
    evaluation_seasons: Sequence[int],
    model_name: str = "unnamed",
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS,
) -> BacktestResult:
    """Refit and score a model once per evaluation season.

    `train_fold` is called with each fold's training rows only. It never sees
    the test rows, which is what keeps every fold's score out-of-sample.
    """
    folds = walk_forward_folds(table, evaluation_seasons, minimum_training_rows)
    result = BacktestResult(model_name=model_name)

    pooled_probabilities: list[Any] = []
    pooled_labels: list[Any] = []

    for fold in folds:
        started = time.perf_counter()
        predict = train_fold(fold.train)
        training_seconds = time.perf_counter() - started

        started = time.perf_counter()
        probabilities = predict(fold.test)
        latency_ms = (time.perf_counter() - started) * 1000 / max(len(fold.test), 1)

        labels = fold.test[TARGET_COLUMN]
        result.folds.append(
            evaluate(
                probabilities,
                labels,
                model_name=model_name,
                period=str(fold.season),
                resources=ResourceMetrics(
                    inference_latency_ms=latency_ms,
                    training_duration_seconds=training_seconds,
                ),
            )
        )
        result.fold_definitions.append(fold.as_dict())

        pooled_probabilities.append(np.asarray(probabilities, dtype=np.float64))
        pooled_labels.append(np.asarray(labels))

    if pooled_probabilities:
        result.aggregate = evaluate(
            np.vstack(pooled_probabilities),
            np.concatenate(pooled_labels),
            model_name=model_name,
            period=f"{folds[0].season}-{folds[-1].season}",
        )

    return result
