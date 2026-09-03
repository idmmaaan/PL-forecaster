"""Train a candidate model and write a complete, reloadable artifact.

Run with `make train` (see `--help` for the options). Training is CLI-only by
design: the API loads artifacts but never creates them.

An artifact is only useful if a prediction made from it can be reproduced
later, so it contains everything needed to do that: the fitted model, the
calibrator, the league state the features come from, the feature schema
version, the class order, the split ranges, the metrics, the library versions,
the random seeds, and the commit hash.
"""

import argparse
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from epl_predictor import OUTCOME_CLASSES, __version__
from epl_predictor.data.ingest import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REPORTS_DIR,
    load_canonical,
)
from epl_predictor.evaluation.metrics import EvaluationResult, ResourceMetrics, evaluate
from epl_predictor.features.builder import (
    DEFAULT_SPLIT,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    SeasonSplit,
    build_training_table,
)
from epl_predictor.features.context import FeatureContext
from epl_predictor.predictors.calibration import make_calibrator
from epl_predictor.predictors.catboost import CatBoostPredictor
from epl_predictor.predictors.class_frequency import ClassFrequencyPredictor
from epl_predictor.predictors.logistic_regression import LogisticRegressionPredictor
from epl_predictor.predictors.tabular import TabularPredictor

logger = logging.getLogger(__name__)

ML_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = ML_ROOT / "artifacts"
MODEL_CARD_FILENAME = "model_card.md"
TRAINING_METADATA_FILENAME = "training.json"


class AdapterFactory(Protocol):
    """Constructs an adapter with its own default model name.

    Typed as a factory rather than as `type[TabularPredictor]` because the
    training script only ever chooses the version; each adapter names itself.
    """

    def __call__(self, *, version: str) -> TabularPredictor: ...


ADAPTERS: dict[str, AdapterFactory] = {
    "catboost": CatBoostPredictor,
    "logistic-regression": LogisticRegressionPredictor,
    "class-frequency": ClassFrequencyPredictor,
}

# The last training season doubles as the calibration period, so a calibrator
# is never fitted on rows the model was fitted on.
DEFAULT_CALIBRATION_SEASONS = (2023,)


def git_commit() -> str | None:
    """Current commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ML_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def library_versions() -> dict[str, str]:
    """Versions of every library that can change a prediction."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "epl_predictor": __version__,
    }
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "catboost", "joblib"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # noqa: BLE001 - a missing optional library is fine
            continue
    return versions


@dataclass
class TrainingOutcome:
    """What a training run produced."""

    predictor: TabularPredictor
    artifact_path: Path
    validation: EvaluationResult
    metadata: dict[str, Any]


def train_candidate(
    adapter: str,
    matches: pd.DataFrame,
    version: str,
    calibration: str = "identity",
    split: SeasonSplit = DEFAULT_SPLIT,
    calibration_seasons: tuple[int, ...] = DEFAULT_CALIBRATION_SEASONS,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    evaluate_on_test: bool = False,
) -> TrainingOutcome:
    """Build features, fit one candidate, and write its artifact.

    The test period is evaluated only when `evaluate_on_test` is set, which
    the README reserves for a single final measurement after selection.

    Raises:
        ValueError: The adapter is unknown or a period is empty.
    """
    if adapter not in ADAPTERS:
        raise ValueError(f"Unknown adapter {adapter!r}; expected one of {sorted(ADAPTERS)}")

    table, state = build_training_table(matches)
    context = FeatureContext.from_matches(state, matches)

    seasons = table["season_start_year"]
    model_seasons = [s for s in split.train if s not in calibration_seasons]
    model_fit = table[seasons.isin(model_seasons)].reset_index(drop=True)
    calibration_fit = table[seasons.isin(calibration_seasons)].reset_index(drop=True)
    validation = table[seasons.isin(split.validation)].reset_index(drop=True)

    if model_fit.empty:
        raise ValueError("No training rows: check the split against the ingested seasons")
    if validation.empty:
        raise ValueError("No validation rows: check the split against the ingested seasons")

    predictor = ADAPTERS[adapter](version=version)
    predictor.calibrator = make_calibrator(calibration)

    started = time.perf_counter()
    predictor.fit(model_fit, calibration_fit if not calibration_fit.empty else None)
    training_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = predictor.predict_matrix(validation)
    latency_ms = (time.perf_counter() - started) * 1000 / max(len(validation), 1)

    artifact_path = artifact_root / f"{predictor.model_name}-{version}"
    predictor.save(artifact_path)
    context.save(artifact_path)

    validation_result = evaluate(
        probabilities,
        validation[TARGET_COLUMN],
        model_name=predictor.model_name,
        period="validation",
        resources=ResourceMetrics(
            inference_latency_ms=latency_ms,
            training_duration_seconds=training_seconds,
            artifact_size_bytes=_directory_size(artifact_path),
        ),
    )

    metadata = _training_metadata(
        predictor=predictor,
        adapter=adapter,
        calibration=calibration,
        split=split,
        calibration_seasons=calibration_seasons,
        model_fit=model_fit,
        validation=validation,
        context=context,
        validation_result=validation_result,
    )

    if evaluate_on_test:
        test = table[seasons.isin(split.test)].reset_index(drop=True)
        if not test.empty:
            test_result = evaluate(
                predictor.predict_matrix(test),
                test[TARGET_COLUMN],
                model_name=predictor.model_name,
                period="test",
            )
            metadata["metrics"]["test"] = test_result.as_dict()

    (artifact_path / TRAINING_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    (artifact_path / MODEL_CARD_FILENAME).write_text(write_model_card(metadata))

    return TrainingOutcome(
        predictor=predictor,
        artifact_path=artifact_path,
        validation=validation_result,
        metadata=metadata,
    )


def _training_metadata(
    predictor: TabularPredictor,
    adapter: str,
    calibration: str,
    split: SeasonSplit,
    calibration_seasons: tuple[int, ...],
    model_fit: pd.DataFrame,
    validation: pd.DataFrame,
    context: FeatureContext,
    validation_result: EvaluationResult,
) -> dict[str, Any]:
    return {
        "model_name": predictor.model_name,
        "model_version": predictor.model_version,
        "adapter_type": adapter,
        "calibration": calibration,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "class_order": list(OUTCOME_CLASSES),
        "trained_at": datetime.now(UTC).isoformat(),
        "splits": {
            **split.as_dict(),
            "calibration": list(calibration_seasons),
            "model_fit_rows": len(model_fit),
            "validation_rows": len(validation),
        },
        "date_ranges": {
            "trained_from": str(model_fit["kickoff_date"].min().date()),
            "trained_until": str(model_fit["kickoff_date"].max().date()),
            "validated_from": str(validation["kickoff_date"].min().date()),
            "validated_until": str(validation["kickoff_date"].max().date()),
        },
        "feature_context": context.metadata(),
        "metrics": {"validation": validation_result.as_dict()},
        "library_versions": library_versions(),
        "git_commit": git_commit(),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_model_card(metadata: dict[str, Any]) -> str:
    """Render the model card the README requires with every artifact."""
    validation = metadata["metrics"]["validation"]
    ranges = metadata["date_ranges"]

    lines = [
        f"# {metadata['model_name']} {metadata['model_version']}",
        "",
        "## What this model does",
        "",
        "Predicts probabilities of a home win, draw, or away win for a Premier",
        "League fixture, from information available before kickoff only.",
        "",
        "## How it was built",
        "",
        f"- Adapter: `{metadata['adapter_type']}`",
        f"- Calibration: `{metadata['calibration']}`",
        f"- Feature schema: `{metadata['feature_schema_version']}`",
        f"- Class order: {', '.join(metadata['class_order'])}",
        f"- Trained on {ranges['trained_from']} to {ranges['trained_until']} "
        f"({metadata['splits']['model_fit_rows']} matches)",
        f"- Calibrated on season(s) {metadata['splits']['calibration']}",
        f"- Validated on {ranges['validated_from']} to {ranges['validated_until']} "
        f"({metadata['splits']['validation_rows']} matches)",
        f"- Commit: `{metadata['git_commit'] or 'unknown'}`",
        "",
        "## Validation performance",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Log loss (primary) | {validation['log_loss']:.4f} |",
        f"| Brier score | {validation['brier_score']:.4f} |",
        f"| Accuracy | {validation['accuracy']:.3f} |",
        f"| Draw recall | {validation['draw_recall']:.3f} |",
        f"| Draw precision | {validation['draw_precision']:.3f} |",
        f"| Calibration error | {validation['expected_calibration_error']:.4f} |",
        "",
        "## Intended use and limits",
        "",
        "- For match-outcome probabilities on Premier League fixtures only.",
        "- Not built or validated for betting.",
        "- Features come from the league state at the end of training",
        f"  ({metadata['feature_context']['last_match_date']}). Predictions for",
        "  fixtures long after that date use stale form and Elo, so the model",
        "  should be retrained as results arrive.",
        "- A club absent from training is handled but predicted with no history.",
        "",
        "## Known weaknesses",
        "",
        f"- Draw recall is {validation['draw_recall']:.3f}. Draws are the minority",
        "  outcome and are rarely the most likely single result, so this model is",
        "  much better at ranking home and away wins than at identifying draws.",
        "- Bookmaker odds are deliberately excluded, so the model does not track",
        "  market information such as injuries or lineup news.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a candidate model.")
    parser.add_argument("--adapter", default="catboost", choices=sorted(ADAPTERS))
    parser.add_argument("--version", default="1.0.0", help="Semantic version to assign.")
    parser.add_argument(
        "--calibration",
        default="identity",
        choices=["identity", "temperature", "isotonic", "sigmoid"],
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--evaluate-on-test",
        action="store_true",
        help="Also score the reserved test season. Use once, after selection.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        matches = load_canonical(args.processed_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    outcome = train_candidate(
        adapter=args.adapter,
        matches=matches,
        version=args.version,
        calibration=args.calibration,
        artifact_root=args.artifact_root,
        evaluate_on_test=args.evaluate_on_test,
    )

    logger.info("Wrote artifact to %s", outcome.artifact_path)
    logger.info(
        "Validation: log_loss=%.4f  brier=%.4f  accuracy=%.3f  draw_recall=%.3f",
        outcome.validation.log_loss,
        outcome.validation.brier_score,
        outcome.validation.accuracy,
        outcome.validation.draw_recall,
    )
    logger.info("Register it with `make promote version=%s`.", args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
