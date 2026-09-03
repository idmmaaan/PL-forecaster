"""Comparison reports for model selection.

Reports are written as both JSON (for the model registry and for diffing
between runs) and Markdown (for a human deciding whether to promote). The
Markdown always leads with log loss and always shows draw recall beside it,
so a model that wins on accuracy while ignoring draws cannot look good by
omission.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.evaluation.metrics import EvaluationResult

# Baseline that a candidate must beat on log loss to be worth promoting. The
# README's gate is explicit that novelty is not an acceptance criterion.
MINIMUM_LOG_LOSS_IMPROVEMENT = 0.001


@dataclass
class ComparisonReport:
    """Several models scored over the same period."""

    title: str
    period: str
    feature_schema_version: str
    results: list[EvaluationResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add(self, result: EvaluationResult) -> None:
        self.results.append(result)

    def ranked(self, metric: str = "log_loss") -> list[EvaluationResult]:
        """Results best-first. Log loss, Brier, and calibration error sort ascending."""
        lower_is_better = metric in {
            "log_loss",
            "brier_score",
            "expected_calibration_error",
        }
        return sorted(
            self.results,
            key=lambda result: float(getattr(result, metric)),
            reverse=not lower_is_better,
        )

    def best(self, metric: str = "log_loss") -> EvaluationResult | None:
        ranked = self.ranked(metric)
        return ranked[0] if ranked else None

    def as_dict(self) -> dict[str, Any]:
        best = self.best()
        return {
            "title": self.title,
            "period": self.period,
            "feature_schema_version": self.feature_schema_version,
            "generated_at": self.generated_at,
            "primary_metric": "log_loss",
            "best_model": best.model_name if best else None,
            "results": [result.as_dict() for result in self.results],
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        """Render the report for a human reviewer."""
        lines = [
            f"# {self.title}",
            "",
            f"- Period: **{self.period}**",
            f"- Feature schema: **{self.feature_schema_version}**",
            f"- Generated: {self.generated_at}",
            "- Primary metric: **multiclass log loss** (lower is better)",
            "",
        ]

        if not self.results:
            lines += ["_No results._", ""]
            return "\n".join(lines)

        lines += [
            "## Ranking",
            "",
            "| Model | Log loss | Brier | Accuracy | Draw recall | Draw precision | ECE |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for result in self.ranked():
            lines.append(
                f"| {result.model_name} "
                f"| {result.log_loss:.4f} "
                f"| {result.brier_score:.4f} "
                f"| {result.accuracy:.3f} "
                f"| {result.draw_recall:.3f} "
                f"| {result.draw_precision:.3f} "
                f"| {result.expected_calibration_error:.4f} |"
            )

        lines += ["", "## Per-class detail", ""]
        for result in self.ranked():
            lines += [
                f"### {result.model_name}",
                "",
                f"{result.rows} matches.",
                "",
                "| Outcome | Precision | Recall | F1 | Support | Predicted |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for outcome in OUTCOME_CLASSES:
                metrics = result.per_class[outcome]
                lines.append(
                    f"| {outcome} "
                    f"| {metrics.precision:.3f} "
                    f"| {metrics.recall:.3f} "
                    f"| {metrics.f1:.3f} "
                    f"| {metrics.support} "
                    f"| {metrics.predicted} |"
                )
            lines += ["", _confusion_block(result), ""]

        if self.notes:
            lines += ["## Notes", ""] + [f"- {note}" for note in self.notes] + [""]

        return "\n".join(lines)


def _confusion_block(result: EvaluationResult) -> str:
    """Confusion matrix as a Markdown table, actual down and predicted across."""
    header = "| Actual \\ Predicted | " + " | ".join(OUTCOME_CLASSES) + " |"
    divider = "| --- " * (len(OUTCOME_CLASSES) + 1) + "|"
    rows = [
        f"| **{outcome}** | " + " | ".join(str(count) for count in row) + " |"
        for outcome, row in zip(OUTCOME_CLASSES, result.confusion_matrix, strict=True)
    ]
    return "\n".join([header, divider, *rows])


def beats_baseline(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
    minimum_improvement: float = MINIMUM_LOG_LOSS_IMPROVEMENT,
    allow_draw_recall_loss: float = 0.0,
) -> tuple[bool, list[str]]:
    """Apply the README's selection gate to one candidate.

    A candidate is accepted only if it measurably improves log loss *and* does
    not damage draw performance. Returns the verdict and the reasons behind
    it, so a rejection can be recorded rather than silently discarded.
    """
    reasons: list[str] = []

    improvement = baseline.log_loss - candidate.log_loss
    if improvement < minimum_improvement:
        reasons.append(
            f"log loss {candidate.log_loss:.4f} does not improve on "
            f"{baseline.model_name}'s {baseline.log_loss:.4f} by at least "
            f"{minimum_improvement}"
        )
    else:
        reasons.append(f"log loss improves by {improvement:.4f}")

    draw_change = candidate.draw_recall - baseline.draw_recall
    if draw_change < -allow_draw_recall_loss:
        reasons.append(
            f"draw recall falls from {baseline.draw_recall:.3f} to {candidate.draw_recall:.3f}"
        )
    else:
        reasons.append(f"draw recall changes by {draw_change:+.3f}")

    accepted = improvement >= minimum_improvement and draw_change >= -allow_draw_recall_loss
    return accepted, reasons


def write_report(
    report: ComparisonReport, reports_dir: Path, name: str = "comparison"
) -> dict[str, Path]:
    """Write the report as JSON and Markdown, plus `latest` copies."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.replace(":", "").replace("-", "").split(".")[0]

    payload = json.dumps(report.as_dict(), indent=2, default=str) + "\n"
    markdown = report.to_markdown()

    written: dict[str, Path] = {}
    for label, suffix, content in (
        ("json", "json", payload),
        ("markdown", "md", markdown),
    ):
        target = reports_dir / f"{name}-{stamp}.{suffix}"
        target.write_text(content)
        (reports_dir / f"{name}-latest.{suffix}").write_text(content)
        written[label] = target

    return written
