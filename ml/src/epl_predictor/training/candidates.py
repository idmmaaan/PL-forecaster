"""Benchmark the Hugging Face candidates against CatBoost on one harness.

Run with `make candidates`. Writes a comparison report and a gate decision per
candidate to `ml/reports/candidates/`.

The point of this script is that every model is measured the same way. Each
candidate is fitted on the same seasons, calibrated on the same season, scored
on the same season, timed the same way, and saved and reloaded to check the
artifact reproduces its own probabilities. CatBoost runs through this script
too, rather than having its number copied from the baseline report, so no
difference in harness can be mistaken for a difference in model.

Candidates whose optional dependency group is not installed are reported as
skipped, with the install command. That is deliberately not silent: a report
that simply omitted TabSTAR would read as though TabSTAR had been beaten.
"""

import argparse
import json
import logging
import resource
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epl_predictor.data.ingest import DEFAULT_PROCESSED_DIR, DEFAULT_REPORTS_DIR
from epl_predictor.evaluation.gate import GateDecision, evaluate_gate
from epl_predictor.evaluation.metrics import EvaluationResult, ResourceMetrics, evaluate
from epl_predictor.evaluation.report import ComparisonReport, write_report
from epl_predictor.features.builder import (
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    load_features,
)
from epl_predictor.predictors.calibration import make_calibrator
from epl_predictor.predictors.catboost import CatBoostPredictor
from epl_predictor.predictors.foundation import CANDIDATES, MissingDependencyError
from epl_predictor.predictors.tabular import TabularPredictor
from epl_predictor.training.baselines import Periods, build_periods

logger = logging.getLogger(__name__)

#: The baseline every candidate must beat, per the README's selection gate.
BASELINE_KEY = "catboost"

#: Calibration method applied to every model here. Temperature scaling is used
#: uniformly rather than tuned per model, because a candidate that only wins
#: after a bespoke calibration search has not been compared fairly.
CALIBRATION_METHOD = "temperature"


def build_predictor(key: str, version: str = "0.1.0") -> TabularPredictor:
    """Construct one model by key, importing its library only when asked.

    Raises:
        MissingDependencyError: The candidate's optional group is not installed.
        ValueError: The key is not a known model.
    """
    if key == BASELINE_KEY:
        return CatBoostPredictor(version=version)

    if key not in CANDIDATES:
        raise ValueError(f"Unknown model {key!r}; expected one of {sorted(all_keys())}")

    # Imported lazily and individually: importing the package's candidate
    # modules eagerly would drag torch into every process that touches the
    # predictors package, including the API.
    if key == "tabicl":
        from epl_predictor.predictors.tabicl import TabICLPredictor

        return TabICLPredictor(version=version)
    if key == "mitra":
        from epl_predictor.predictors.mitra import MitraPredictor

        return MitraPredictor(version=version)
    if key == "tabstar":
        from epl_predictor.predictors.tabstar import TabSTARPredictor

        return TabSTARPredictor(version=version)
    if key == "tabpfn-mix":
        from epl_predictor.predictors.tabpfn_mix import TabPFNMixPredictor

        return TabPFNMixPredictor(version=version)

    from epl_predictor.predictors.tabpfn3 import TabPFN3Predictor

    return TabPFN3Predictor(version=version)


def all_keys() -> list[str]:
    """Every model this script can benchmark, baseline first."""
    return [BASELINE_KEY, *CANDIDATES]


@dataclass
class CandidateRun:
    """One model's measured outcome, or the reason it did not run."""

    key: str
    result: EvaluationResult | None = None
    probabilities: Any | None = None
    round_trip_matches: bool | None = None
    schema_matches: bool | None = None
    skipped_reason: str | None = None
    decision: GateDecision | None = None

    @property
    def ran(self) -> bool:
        return self.result is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "ran": self.ran,
            "skipped_reason": self.skipped_reason,
            "result": self.result.as_dict() if self.result else None,
            "round_trip_matches": self.round_trip_matches,
            "schema_matches": self.schema_matches,
            "decision": self.decision.as_dict() if self.decision else None,
        }


@dataclass
class BenchmarkOutcome:
    """Every model's run, plus the gate decisions between them."""

    runs: list[CandidateRun] = field(default_factory=list)
    report: ComparisonReport | None = None

    def by_key(self, key: str) -> CandidateRun | None:
        return next((run for run in self.runs if run.key == key), None)

    def promotable(self) -> list[CandidateRun]:
        """Candidates that passed every gate criterion."""
        return [run for run in self.runs if run.decision is not None and run.decision.accepted]


def is_license_block(error: BaseException) -> bool:
    """Whether a candidate failed because nobody has accepted its licence.

    TabPFN-3 refuses to download weights until a human accepts its terms
    interactively. That is a legal step, not a defect, and the report should
    say so instead of filing it beside genuine runtime failures.
    """
    name = type(error).__name__.lower()
    text = str(error).lower()
    return "license" in name or "licence" in name or "license acceptance" in text


def peak_memory_mb() -> float:
    """Peak resident set size of this process, in megabytes.

    Measured for the whole process rather than per model, so it is only
    meaningful as a high-water mark after a model has been fitted. Reported
    because the README's gate includes a memory requirement; a per-model
    figure would need a subprocess per candidate.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return usage / divisor


def directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def single_fixture_latency_ms(
    predictor: TabularPredictor, validation: pd.DataFrame, repeats: int = 10
) -> float:
    """Median time to predict one fixture on its own.

    The API answers one fixture per request, so this is the latency a user
    experiences. Amortising a 380-row batch instead would flatter an
    in-context model badly: those models attend over their stored training
    rows on every call, so a single row costs almost as much as a batch.

    The median of several repeats is used rather than the mean, so one
    unlucky scheduling hiccup does not decide a gate criterion.
    """
    row = validation.head(1)
    timings = []
    for _ in range(max(repeats, 1)):
        started = time.perf_counter()
        predictor.predict_matrix(row)
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings))


def run_one(key: str, periods: Periods, artifact_root: Path) -> CandidateRun:
    """Fit, score, and round-trip one model through the shared harness."""
    run = CandidateRun(key=key)

    try:
        predictor = build_predictor(key)
    except MissingDependencyError as exc:
        run.skipped_reason = str(exc)
        logger.warning("Skipping %s: %s", key, exc)
        return run

    predictor.calibrator = make_calibrator(CALIBRATION_METHOD)

    try:
        started = time.perf_counter()
        predictor.fit(periods.model_fit, periods.calibration_fit)
        training_seconds = time.perf_counter() - started
        probabilities = predictor.predict_matrix(periods.validation)
        latency_ms = single_fixture_latency_ms(predictor, periods.validation)
    except MissingDependencyError as exc:
        # Several candidates only touch their library while fitting, so a
        # missing group surfaces here rather than at construction.
        run.skipped_reason = str(exc)
        logger.warning("Skipping %s: %s", key, exc)
        return run
    except Exception as exc:  # noqa: BLE001 - a candidate failing is a result
        # A candidate that cannot run here is reported as such. Inventing a
        # metric for it would be worse than an empty row.
        if is_license_block(exc):
            run.skipped_reason = (
                f"licence not accepted: {exc}. This is a consent step rather than a "
                "defect: the weights may not be downloaded until a human accepts "
                "the licence."
            )
            logger.warning("Skipping %s: its licence has not been accepted.", key)
        else:
            run.skipped_reason = f"failed to run: {type(exc).__name__}: {exc}"
            logger.warning("Candidate %s failed: %s", key, exc)
        return run

    artifact_path = artifact_root / f"{predictor.model_name}-{predictor.model_version}"
    predictor.save(artifact_path)
    run.round_trip_matches = _round_trips(predictor, artifact_path, periods.validation)
    run.schema_matches = predictor.feature_schema_version == FEATURE_SCHEMA_VERSION

    run.probabilities = probabilities
    run.result = evaluate(
        probabilities,
        periods.validation[TARGET_COLUMN],
        model_name=predictor.model_name,
        period="validation",
        resources=ResourceMetrics(
            inference_latency_ms=latency_ms,
            training_duration_seconds=training_seconds,
            peak_memory_mb=peak_memory_mb(),
            artifact_size_bytes=directory_size_bytes(artifact_path),
        ),
    )

    logger.info(
        "%-12s log_loss=%.4f  draw_recall=%.3f  ece=%.4f  %.1f ms/fixture  %.1f MB artifact",
        key,
        run.result.log_loss,
        run.result.draw_recall,
        run.result.expected_calibration_error,
        latency_ms,
        (run.result.resources.artifact_size_bytes or 0) / (1024 * 1024),
    )
    return run


def _round_trips(
    predictor: TabularPredictor, artifact_path: Path, validation: pd.DataFrame
) -> bool:
    """Whether a reloaded artifact reproduces the same probabilities.

    Part of the gate: an artifact that loads but predicts differently is worse
    than one that fails to load, because nothing would report the difference.
    """
    sample = validation.head(32)
    try:
        reloaded = type(predictor).load(artifact_path)
        return bool(
            np.allclose(
                predictor.predict_matrix(sample), reloaded.predict_matrix(sample), atol=1e-8
            )
        )
    except Exception as exc:  # noqa: BLE001 - a failed round trip is a result
        logger.warning("Round trip failed for %s: %s", predictor.model_name, exc)
        return False


def run(
    table: pd.DataFrame, keys: list[str] | None = None, artifact_root: Path | None = None
) -> BenchmarkOutcome:
    """Benchmark the baseline and the requested candidates on one harness."""
    periods = build_periods(table)
    keys = keys or all_keys()
    if BASELINE_KEY not in keys:
        keys = [BASELINE_KEY, *keys]

    report = ComparisonReport(
        title="Hugging Face candidates against the CatBoost baseline",
        period="validation season 2024/25",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    report.notes.append(
        f"All models fitted on {periods.as_dict()['model_fit_seasons'][0]}-"
        f"{periods.as_dict()['model_fit_seasons'][-1]}, calibrated with "
        f"{CALIBRATION_METHOD} scaling on "
        f"{periods.as_dict()['calibration_fit_seasons']}, scored on the "
        "validation season. The final test season was not used."
    )

    outcome = BenchmarkOutcome(report=report)

    with tempfile.TemporaryDirectory(prefix="epl-candidates-") as scratch:
        root = artifact_root or Path(scratch)
        for key in keys:
            candidate_run = run_one(key, periods, root)
            outcome.runs.append(candidate_run)
            if candidate_run.result is not None:
                report.add(candidate_run.result)

    baseline_run = outcome.by_key(BASELINE_KEY)
    if baseline_run is None or baseline_run.result is None:
        report.notes.append(
            "The CatBoost baseline did not run, so no candidate could be gated against it."
        )
        return outcome

    for candidate_run in outcome.runs:
        if candidate_run.key == BASELINE_KEY:
            continue
        if candidate_run.result is None:
            report.notes.append(f"{candidate_run.key}: {candidate_run.skipped_reason}")
            continue

        spec = CANDIDATES[candidate_run.key]
        candidate_run.decision = evaluate_gate(
            candidate=candidate_run.result,
            baseline=baseline_run.result,
            probabilities=candidate_run.probabilities,
            license_summary=spec.license_summary,
            production_use_allowed=spec.production_use_allowed,
            schema_matches=candidate_run.schema_matches,
            round_trip_matches=candidate_run.round_trip_matches,
        )
        report.notes.append(candidate_run.decision.summary())

    promotable = outcome.promotable()
    report.notes.append(
        "Selection: "
        + (
            ", ".join(run.key for run in promotable) + " may be promoted."
            if promotable
            else f"no candidate clears the gate, so {BASELINE_KEY} stays active."
        )
    )
    return outcome


def write_decisions(outcome: BenchmarkOutcome, reports_dir: Path) -> Path:
    """Write the machine-readable gate decisions beside the comparison report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / "gate-decisions.json"
    target.write_text(
        json.dumps(
            {
                "baseline": BASELINE_KEY,
                "calibration": CALIBRATION_METHOD,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "candidates": [run.as_dict() for run in outcome.runs],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=all_keys(),
        help="Benchmark only these models; CatBoost is always included.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Keep the benchmark artifacts here instead of a temporary directory.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        table = load_features(args.processed_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    outcome = run(table, keys=args.only, artifact_root=args.artifact_root)
    if outcome.report is None:
        return 1

    reports_dir = args.reports_dir / "candidates"
    written = write_report(outcome.report, reports_dir, name="candidates")
    decisions = write_decisions(outcome, reports_dir)

    logger.info("")
    for note in outcome.report.notes:
        logger.info("%s", note)
    logger.info("")
    logger.info("Wrote %s", written["markdown"])
    logger.info("Wrote %s", decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
