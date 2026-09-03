"""Train and compare the mandatory baselines, including calibration methods.

Run with `make baselines`. Writes a comparison report to `ml/reports/`.

## Why calibration is compared on a third period

Fitting a calibrator on the validation season and then scoring it on that same
season would flatter it: isotonic regression in particular can fit 380 matches
closely enough to look excellent and generalise poorly. The final test season
cannot be used either, because the README reserves it and it must stay
untouched during selection.

So the training period is split once more. Models are fitted on the earlier
training seasons, calibrators on the last training season, and every number
reported here comes from the validation season, which none of them has seen:

    fit model       2010/11 - 2022/23
    fit calibrator  2023/24
    report on       2024/25       (validation)
    never touched   2025/26       (final test)
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epl_predictor.data.ingest import DEFAULT_PROCESSED_DIR, DEFAULT_REPORTS_DIR
from epl_predictor.evaluation.backtest import walk_forward_backtest
from epl_predictor.evaluation.metrics import EvaluationResult, ResourceMetrics, evaluate
from epl_predictor.evaluation.report import ComparisonReport, write_report
from epl_predictor.features.builder import (
    DEFAULT_SPLIT,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    load_features,
)
from epl_predictor.predictors.calibration import make_calibrator
from epl_predictor.predictors.catboost import CatBoostPredictor
from epl_predictor.predictors.class_frequency import ClassFrequencyPredictor
from epl_predictor.predictors.dummy import FIXED_PROBABILITIES
from epl_predictor.predictors.logistic_regression import LogisticRegressionPredictor
from epl_predictor.predictors.tabular import TabularPredictor

logger = logging.getLogger(__name__)

CALIBRATION_METHODS = ("identity", "temperature", "isotonic", "sigmoid")

# The last training season is reserved for fitting calibrators.
CALIBRATION_SEASONS = (2023,)

PredictorFactory = Callable[[], TabularPredictor]

BASELINES: dict[str, PredictorFactory] = {
    "class-frequency": ClassFrequencyPredictor,
    "logistic-regression": LogisticRegressionPredictor,
    "catboost": CatBoostPredictor,
}


@dataclass
class Periods:
    """The four chronological periods this experiment uses."""

    model_fit: pd.DataFrame
    calibration_fit: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_fit_rows": len(self.model_fit),
            "model_fit_seasons": _seasons(self.model_fit),
            "calibration_fit_rows": len(self.calibration_fit),
            "calibration_fit_seasons": _seasons(self.calibration_fit),
            "validation_rows": len(self.validation),
            "validation_seasons": _seasons(self.validation),
            "test_rows": len(self.test),
            "test_seasons": _seasons(self.test),
        }


def _seasons(frame: pd.DataFrame) -> list[int]:
    return sorted(int(year) for year in frame["season_start_year"].unique())


def build_periods(
    table: pd.DataFrame, calibration_seasons: tuple[int, ...] = CALIBRATION_SEASONS
) -> Periods:
    """Split the feature table into model-fit, calibration, validation, and test."""
    seasons = table["season_start_year"]
    train_seasons = [season for season in DEFAULT_SPLIT.train if season not in calibration_seasons]

    return Periods(
        model_fit=table[seasons.isin(train_seasons)].reset_index(drop=True),
        calibration_fit=table[seasons.isin(calibration_seasons)].reset_index(drop=True),
        validation=table[seasons.isin(DEFAULT_SPLIT.validation)].reset_index(drop=True),
        test=table[seasons.isin(DEFAULT_SPLIT.test)].reset_index(drop=True),
    )


def evaluate_dummy(validation: pd.DataFrame) -> EvaluationResult:
    """Score the hard-coded stub, the floor every other model must clear."""
    probabilities = np.tile(list(FIXED_PROBABILITIES.values()), (len(validation), 1))
    return evaluate(
        probabilities,
        validation[TARGET_COLUMN],
        model_name="dummy",
        period="validation",
    )


def train_and_score(
    name: str, factory: PredictorFactory, periods: Periods, calibration: str
) -> tuple[EvaluationResult, TabularPredictor]:
    """Fit one baseline with one calibration method and score it on validation."""
    import time

    predictor = factory()
    predictor.calibrator = make_calibrator(calibration)

    started = time.perf_counter()
    predictor.fit(periods.model_fit, periods.calibration_fit)
    training_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = predictor.predict_matrix(periods.validation)
    latency_ms = (time.perf_counter() - started) * 1000 / max(len(periods.validation), 1)

    result = evaluate(
        probabilities,
        periods.validation[TARGET_COLUMN],
        model_name=f"{name} ({calibration})",
        period="validation",
        resources=ResourceMetrics(
            inference_latency_ms=latency_ms, training_duration_seconds=training_seconds
        ),
    )
    return result, predictor


def run(
    table: pd.DataFrame, backtest: bool = True
) -> tuple[ComparisonReport, dict[str, TabularPredictor]]:
    """Train every baseline under every calibration method and compare them."""
    periods = build_periods(table)
    report = ComparisonReport(
        title="Baseline and calibration comparison",
        period="validation season 2024/25",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    report.notes.append(
        "Models fitted on "
        f"{periods.as_dict()['model_fit_seasons'][0]}-"
        f"{periods.as_dict()['model_fit_seasons'][-1]}, calibrators fitted on "
        f"{periods.as_dict()['calibration_fit_seasons']}, all numbers from the "
        "validation season. The final test season was not used."
    )

    report.add(evaluate_dummy(periods.validation))
    best_per_baseline: dict[str, TabularPredictor] = {}

    for name, factory in BASELINES.items():
        scored: list[tuple[EvaluationResult, TabularPredictor]] = []
        for calibration in CALIBRATION_METHODS:
            result, predictor = train_and_score(name, factory, periods, calibration)
            report.add(result)
            scored.append((result, predictor))
            logger.info(
                "%-20s %-12s log_loss=%.4f  draw_recall=%.3f  ece=%.4f",
                name,
                calibration,
                result.log_loss,
                result.draw_recall,
                result.expected_calibration_error,
            )
        best_result, best_predictor = min(scored, key=lambda pair: pair[0].log_loss)
        best_per_baseline[name] = best_predictor
        report.notes.append(f"Best calibration for {name}: {best_result.model_name}.")

    if backtest:
        _add_backtest_note(report, table)

    return report, best_per_baseline


def _add_backtest_note(report: ComparisonReport, table: pd.DataFrame) -> None:
    """Walk-forward backtest the strongest baseline across several seasons.

    A single validation season can be a lucky year. Refitting per season shows
    whether the ranking holds.
    """

    def train_fold(train: pd.DataFrame) -> Callable[[pd.DataFrame], Any]:
        predictor = CatBoostPredictor()
        # The final fifth of each fold's history stands in for a calibration
        # period, keeping the fold's own test season untouched.
        cut = int(len(train) * 0.8)
        predictor.calibrator = make_calibrator("temperature")
        predictor.fit(train.iloc[:cut], train.iloc[cut:])
        return predictor.predict_matrix

    seasons = [2021, 2022, 2023, 2024]
    result = walk_forward_backtest(table, train_fold, seasons, model_name="catboost")

    report.notes.append(
        "Walk-forward backtest of catboost (temperature): mean log loss "
        f"{result.mean('log_loss'):.4f} "
        f"(sd {result.standard_deviation('log_loss'):.4f}) over seasons "
        f"{result.seasons}; per season "
        + ", ".join(f"{season}={value:.4f}" for season, value in result.metric_by_season().items())
        + "."
    )
    report.notes.append(f"Walk-forward mean draw recall: {result.mean('draw_recall'):.3f}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and compare the baselines.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the walk-forward backtest, which refits CatBoost per season.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        table = load_features(args.processed_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    report, _ = run(table, backtest=not args.skip_backtest)
    written = write_report(report, args.reports_dir / "baselines", name="baselines")

    best = report.best()
    logger.info("")
    logger.info("Best on log loss: %s", best.model_name if best else "none")
    logger.info("Wrote %s", written["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
