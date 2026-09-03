"""The README's selection gate, as executable criteria.

The README lists seven conditions a candidate must meet before it may replace
the active model. Written out as prose they are easy to apply selectively — it
is tempting to promote a model that wins on log loss and quietly skip the draw
check, or to benchmark a non-commercial model and forget its licence by the
time a decision is made. Encoding all seven here means a promotion decision
comes with the full list of reasons, and a rejection records which criterion
failed.

Two of the criteria deserve comment.

**Draw performance is protected explicitly.** Draws are roughly a quarter of
Premier League results and the hardest class to predict, so a model can lower
log loss while giving up on draws entirely. That trade is refused by default.

**The licence is a hard gate, not a note.** TabPFN-3 may be benchmarked and
may not be served, and no metric can override that.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from epl_predictor.evaluation.metrics import EvaluationResult
from epl_predictor.evaluation.report import MINIMUM_LOG_LOSS_IMPROVEMENT

#: Draw recall may not fall by more than this against the active model. Zero
#: would reject a candidate for statistical noise on a 380-match season; a
#: little slack keeps the gate meaningful rather than merely strict.
MAX_DRAW_RECALL_LOSS = 0.02

#: Probabilities beyond this distance from summing to one indicate a broken
#: adapter rather than floating-point drift.
PROBABILITY_SUM_TOLERANCE = 1e-6

#: Expected calibration error above this counts as "not reasonably calibrated".
#: A candidate this far off should be recalibrated before being reconsidered.
MAX_EXPECTED_CALIBRATION_ERROR = 0.10

#: Per-fixture inference budget. The API predicts one fixture per request, so
#: this is a user-visible latency, not a batch throughput figure.
MAX_INFERENCE_LATENCY_MS = 250.0

#: Resident memory ceiling for serving a model on a development machine.
MAX_PEAK_MEMORY_MB = 4096.0


@dataclass(frozen=True)
class Criterion:
    """One gate condition and how the candidate fared against it."""

    name: str
    passed: bool
    detail: str
    #: False when the condition could not be evaluated, e.g. no memory
    #: measurement was taken. Unmeasured criteria never pass silently.
    measured: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "measured": self.measured,
            "detail": self.detail,
        }


@dataclass
class GateDecision:
    """The verdict on one candidate, with every criterion's reasoning."""

    candidate: str
    baseline: str
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(criterion.passed for criterion in self.criteria)

    @property
    def failures(self) -> list[Criterion]:
        return [criterion for criterion in self.criteria if not criterion.passed]

    def summary(self) -> str:
        if self.accepted:
            return f"{self.candidate} passes the selection gate against {self.baseline}."
        reasons = "; ".join(criterion.detail for criterion in self.failures)
        return f"{self.candidate} is rejected against {self.baseline}: {reasons}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "accepted": self.accepted,
            "summary": self.summary(),
            "criteria": [criterion.as_dict() for criterion in self.criteria],
        }


def evaluate_gate(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
    probabilities: Any | None = None,
    license_summary: str = "",
    production_use_allowed: bool = True,
    schema_matches: bool | None = None,
    round_trip_matches: bool | None = None,
    minimum_improvement: float = MINIMUM_LOG_LOSS_IMPROVEMENT,
    max_draw_recall_loss: float = MAX_DRAW_RECALL_LOSS,
    max_calibration_error: float = MAX_EXPECTED_CALIBRATION_ERROR,
    max_latency_ms: float = MAX_INFERENCE_LATENCY_MS,
    max_memory_mb: float = MAX_PEAK_MEMORY_MB,
) -> GateDecision:
    """Apply all seven criteria to one candidate against the active model.

    Args:
        candidate: The candidate's metrics on the comparison period.
        baseline: The active model's metrics on the same period.
        probabilities: The candidate's raw output, checked for validity. When
            omitted, the validity criterion is reported as unmeasured, which
            counts as a failure rather than a pass.
        license_summary: Recorded with the decision and with the registry row.
        production_use_allowed: False for a licence that forbids serving.
        schema_matches: Whether the candidate consumes the active feature
            schema, normally the result of a compatibility test.
        round_trip_matches: Whether saving and reloading the artifact
            reproduced identical probabilities.
    """
    criteria = [
        _log_loss_criterion(candidate, baseline, minimum_improvement),
        _draw_criterion(candidate, baseline, max_draw_recall_loss),
        _probability_criterion(candidate, probabilities, max_calibration_error),
        _boolean_criterion(
            "feature schema compatibility",
            schema_matches,
            "consumes the active feature schema",
            "does not consume the active feature schema",
        ),
        _boolean_criterion(
            "artifact save and load",
            round_trip_matches,
            "reloads to identical probabilities",
            "does not reload to identical probabilities",
        ),
        _resource_criterion(candidate, max_latency_ms, max_memory_mb),
        _license_criterion(license_summary, production_use_allowed),
    ]
    return GateDecision(
        candidate=candidate.model_name, baseline=baseline.model_name, criteria=criteria
    )


def _log_loss_criterion(
    candidate: EvaluationResult, baseline: EvaluationResult, minimum_improvement: float
) -> Criterion:
    improvement = baseline.log_loss - candidate.log_loss
    passed = improvement >= minimum_improvement
    return Criterion(
        name="primary metric",
        passed=passed,
        detail=(
            f"log loss {candidate.log_loss:.4f} vs {baseline.log_loss:.4f} "
            f"({improvement:+.4f}"
            + ("" if passed else f", needs at least {minimum_improvement:+.4f}")
            + ")"
        ),
    )


def _draw_criterion(
    candidate: EvaluationResult, baseline: EvaluationResult, max_loss: float
) -> Criterion:
    change = candidate.draw_recall - baseline.draw_recall
    passed = change >= -max_loss
    return Criterion(
        name="draw performance",
        passed=passed,
        detail=(
            f"draw recall {candidate.draw_recall:.3f} vs {baseline.draw_recall:.3f} "
            f"({change:+.3f}"
            + ("" if passed else f", may not fall by more than {max_loss:.3f}")
            + ")"
        ),
    )


def _probability_criterion(
    candidate: EvaluationResult, probabilities: Any | None, max_calibration_error: float
) -> Criterion:
    if probabilities is None:
        return Criterion(
            name="valid and calibrated probabilities",
            passed=False,
            measured=False,
            detail="no probability matrix was supplied, so validity was not checked",
        )

    matrix = np.asarray(probabilities, dtype=np.float64)
    problems = []
    if not np.all(np.isfinite(matrix)):
        problems.append("contains non-finite values")
    if matrix.size and (matrix.min() < 0.0 or matrix.max() > 1.0):
        problems.append(f"values outside [0, 1] (min {matrix.min():.4f}, max {matrix.max():.4f})")
    totals = matrix.sum(axis=1) if matrix.ndim == 2 else np.array([matrix.sum()])
    worst = float(np.max(np.abs(totals - 1.0))) if totals.size else 0.0
    if worst > PROBABILITY_SUM_TOLERANCE:
        problems.append(f"rows deviate from summing to one by up to {worst:.2e}")
    if candidate.expected_calibration_error > max_calibration_error:
        problems.append(
            f"calibration error {candidate.expected_calibration_error:.4f} "
            f"exceeds {max_calibration_error:.2f}"
        )

    return Criterion(
        name="valid and calibrated probabilities",
        passed=not problems,
        detail=(
            "; ".join(problems)
            if problems
            else (
                f"valid distributions, calibration error {candidate.expected_calibration_error:.4f}"
            )
        ),
    )


def _boolean_criterion(
    name: str, value: bool | None, pass_detail: str, fail_detail: str
) -> Criterion:
    if value is None:
        return Criterion(name=name, passed=False, measured=False, detail=f"{name} was not checked")
    return Criterion(name=name, passed=value, detail=pass_detail if value else fail_detail)


def _resource_criterion(
    candidate: EvaluationResult, max_latency_ms: float, max_memory_mb: float
) -> Criterion:
    latency = candidate.resources.inference_latency_ms
    memory = candidate.resources.peak_memory_mb

    if latency is None:
        return Criterion(
            name="local resources",
            passed=False,
            measured=False,
            detail="inference latency was not measured",
        )

    problems = []
    if latency > max_latency_ms:
        problems.append(f"latency {latency:.1f} ms/fixture exceeds {max_latency_ms:.0f} ms")
    if memory is not None and memory > max_memory_mb:
        problems.append(f"peak memory {memory:.0f} MB exceeds {max_memory_mb:.0f} MB")

    measured_memory = f", peak memory {memory:.0f} MB" if memory is not None else ""
    return Criterion(
        name="local resources",
        passed=not problems,
        detail=(
            "; ".join(problems)
            if problems
            else f"latency {latency:.1f} ms/fixture{measured_memory}"
        ),
    )


def _license_criterion(license_summary: str, production_use_allowed: bool) -> Criterion:
    if not license_summary:
        return Criterion(
            name="licence",
            passed=False,
            measured=False,
            detail="no licence summary was recorded",
        )
    if not production_use_allowed:
        return Criterion(
            name="licence",
            passed=False,
            detail=f"{license_summary} does not permit production use",
        )
    return Criterion(
        name="licence", passed=True, detail=f"{license_summary} permits the intended use"
    )
