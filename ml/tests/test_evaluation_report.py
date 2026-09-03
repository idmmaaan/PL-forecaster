"""Comparison reports and the README's promotion gate.

The gate is the part that has teeth: a candidate is promoted only if it
measurably improves log loss without damaging draw performance. These tests
pin both halves, including the case where a candidate wins on log loss but
loses draws and must still be rejected.
"""

import json
from pathlib import Path

import pytest

from epl_predictor.evaluation.metrics import EvaluationResult, evaluate
from epl_predictor.evaluation.report import (
    ComparisonReport,
    beats_baseline,
    write_report,
)

HOME, DRAW, AWAY = "HOME_WIN", "DRAW", "AWAY_WIN"


def scored(
    model_name: str,
    log_loss: float,
    draw_recall: float = 0.2,
    accuracy: float = 0.5,
    brier: float = 0.6,
) -> EvaluationResult:
    """An EvaluationResult with chosen headline numbers, for gate tests."""
    result = evaluate([[0.5, 0.3, 0.2]], [HOME], model_name=model_name, period="2024")
    result.log_loss = log_loss
    result.draw_recall = draw_recall
    result.accuracy = accuracy
    result.brier_score = brier
    return result


@pytest.fixture
def report() -> ComparisonReport:
    report = ComparisonReport(
        title="Baseline comparison", period="2024", feature_schema_version="1.0.0"
    )
    report.add(scored("catboost", log_loss=0.98, draw_recall=0.18, accuracy=0.53))
    report.add(scored("logistic-regression", log_loss=1.01, draw_recall=0.10))
    report.add(scored("class-frequency", log_loss=1.06, draw_recall=0.0))
    return report


# --- Ranking ----------------------------------------------------------------


def test_the_report_ranks_by_log_loss_ascending(report: ComparisonReport) -> None:
    assert [result.model_name for result in report.ranked()] == [
        "catboost",
        "logistic-regression",
        "class-frequency",
    ]


def test_the_best_model_is_the_lowest_log_loss(report: ComparisonReport) -> None:
    best = report.best()

    assert best is not None
    assert best.model_name == "catboost"


def test_accuracy_ranks_descending(report: ComparisonReport) -> None:
    """Higher is better for accuracy, unlike the loss metrics."""
    assert report.ranked("accuracy")[0].model_name == "catboost"


def test_calibration_error_ranks_ascending(report: ComparisonReport) -> None:
    report.results[1].expected_calibration_error = 0.01
    report.results[0].expected_calibration_error = 0.20

    assert report.ranked("expected_calibration_error")[0].model_name == ("logistic-regression")


def test_an_empty_report_has_no_best_model() -> None:
    empty = ComparisonReport(title="none", period="2024", feature_schema_version="1.0.0")

    assert empty.best() is None
    assert "_No results._" in empty.to_markdown()


# --- Serialisation ----------------------------------------------------------


def test_the_report_records_the_schema_it_was_built_on(report: ComparisonReport) -> None:
    """A metric is meaningless without knowing which features produced it."""
    payload = report.as_dict()

    assert payload["feature_schema_version"] == "1.0.0"
    assert payload["primary_metric"] == "log_loss"
    assert payload["best_model"] == "catboost"
    assert payload["period"] == "2024"
    assert len(payload["results"]) == 3


def test_the_markdown_leads_with_log_loss_and_shows_draw_recall(
    report: ComparisonReport,
) -> None:
    """Draw recall must be visible next to the headline, not buried."""
    markdown = report.to_markdown()

    assert "Primary metric: **multiclass log loss**" in markdown
    assert "| Model | Log loss | Brier | Accuracy | Draw recall |" in markdown
    assert "catboost" in markdown
    assert markdown.index("catboost") < markdown.index("class-frequency")


def test_the_markdown_includes_per_class_and_confusion_detail(
    report: ComparisonReport,
) -> None:
    markdown = report.to_markdown()

    assert "## Per-class detail" in markdown
    assert "| Actual \\ Predicted | HOME_WIN | DRAW | AWAY_WIN |" in markdown


def test_notes_are_carried_into_the_report(report: ComparisonReport) -> None:
    report.notes.append("TabPFN-3 skipped: requires an 80 GB CUDA GPU.")

    assert "TabPFN-3 skipped" in report.to_markdown()
    assert report.as_dict()["notes"] == ["TabPFN-3 skipped: requires an 80 GB CUDA GPU."]


def test_writing_a_report_produces_json_markdown_and_latest_copies(
    report: ComparisonReport, tmp_path: Path
) -> None:
    written = write_report(report, tmp_path, name="baselines")

    assert written["json"].exists()
    assert written["markdown"].exists()
    assert (tmp_path / "baselines-latest.json").exists()
    assert (tmp_path / "baselines-latest.md").exists()

    payload = json.loads((tmp_path / "baselines-latest.json").read_text())
    assert payload["best_model"] == "catboost"
    assert (tmp_path / "baselines-latest.md").read_text() == report.to_markdown()


# --- The promotion gate -----------------------------------------------------


def test_a_clear_improvement_is_accepted() -> None:
    accepted, reasons = beats_baseline(
        scored("candidate", log_loss=0.95, draw_recall=0.25),
        scored("catboost", log_loss=0.98, draw_recall=0.20),
    )

    assert accepted
    assert any("log loss improves" in reason for reason in reasons)


def test_a_worse_log_loss_is_rejected() -> None:
    accepted, reasons = beats_baseline(
        scored("candidate", log_loss=1.02),
        scored("catboost", log_loss=0.98),
    )

    assert not accepted
    assert any("does not improve" in reason for reason in reasons)


def test_an_immaterial_improvement_is_rejected() -> None:
    """Novelty is not an acceptance criterion, so noise-level gains do not count."""
    accepted, _ = beats_baseline(
        scored("candidate", log_loss=0.979_9),
        scored("catboost", log_loss=0.980_0),
    )

    assert not accepted


def test_a_candidate_that_wins_on_log_loss_but_loses_draws_is_rejected() -> None:
    """The gate's real purpose: protect draw performance during promotion."""
    accepted, reasons = beats_baseline(
        scored("candidate", log_loss=0.90, draw_recall=0.05),
        scored("catboost", log_loss=0.98, draw_recall=0.20),
    )

    assert not accepted
    assert any("draw recall falls" in reason for reason in reasons)


def test_a_tolerance_for_draw_recall_can_be_granted_explicitly() -> None:
    accepted, _ = beats_baseline(
        scored("candidate", log_loss=0.90, draw_recall=0.18),
        scored("catboost", log_loss=0.98, draw_recall=0.20),
        allow_draw_recall_loss=0.03,
    )

    assert accepted


def test_equal_draw_recall_does_not_block_promotion() -> None:
    accepted, _ = beats_baseline(
        scored("candidate", log_loss=0.90, draw_recall=0.20),
        scored("catboost", log_loss=0.98, draw_recall=0.20),
    )

    assert accepted


def test_the_gate_always_explains_itself() -> None:
    """A rejection has to be recordable, so reasons are returned either way."""
    for candidate_loss in (0.90, 1.10):
        _, reasons = beats_baseline(
            scored("candidate", log_loss=candidate_loss),
            scored("catboost", log_loss=0.98),
        )
        assert len(reasons) == 2
