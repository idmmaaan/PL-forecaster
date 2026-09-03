"""Tests for the README's selection gate.

The gate exists to stop a candidate being promoted on one favourable number,
so most of these tests are about refusal: a model that wins on log loss but
abandons draws, or has an incompatible licence, or was never measured, must
not pass. The unmeasured cases matter as much as the failing ones, because a
criterion that silently passes when nobody checked it is worse than no gate.
"""

from typing import Any

import numpy as np
import pytest

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.evaluation.gate import (
    MAX_DRAW_RECALL_LOSS,
    MAX_EXPECTED_CALIBRATION_ERROR,
    MAX_INFERENCE_LATENCY_MS,
    evaluate_gate,
)
from epl_predictor.evaluation.metrics import ResourceMetrics, evaluate

RNG = np.random.default_rng(20260903)


def scored(
    log_loss_target: float = 1.0,
    draw_recall_target: float = 0.2,
    name: str = "model",
    latency_ms: float | None = 10.0,
    memory_mb: float | None = 512.0,
    calibration_error: float | None = None,
):
    """An EvaluationResult with the fields the gate reads set explicitly.

    The gate only ever reads a handful of numbers, so they are set directly
    rather than reverse-engineered from probabilities that happen to produce
    them; that keeps each test's intent legible.
    """
    labels = [OUTCOME_CLASSES[index % 3] for index in range(30)]
    probabilities = np.full((30, 3), 1 / 3)
    result = evaluate(
        probabilities,
        labels,
        model_name=name,
        period="validation",
        resources=ResourceMetrics(inference_latency_ms=latency_ms, peak_memory_mb=memory_mb),
    )
    result.log_loss = log_loss_target
    result.draw_recall = draw_recall_target
    if calibration_error is not None:
        result.expected_calibration_error = calibration_error
    else:
        result.expected_calibration_error = 0.01
    return result


def valid_probabilities(rows: int = 30):
    matrix = RNG.random((rows, len(OUTCOME_CLASSES)))
    return matrix / matrix.sum(axis=1, keepdims=True)


def passing_kwargs(**overrides):
    """Arguments under which every criterion passes, for one-at-a-time breaking."""
    defaults: dict[str, Any] = {
        "candidate": scored(log_loss_target=0.90, draw_recall_target=0.25, name="candidate"),
        "baseline": scored(log_loss_target=1.00, draw_recall_target=0.20, name="baseline"),
        "probabilities": valid_probabilities(),
        "license_summary": "Apache-2.0",
        "production_use_allowed": True,
        "schema_matches": True,
        "round_trip_matches": True,
    }
    defaults.update(overrides)
    return defaults


class TestAcceptance:
    def test_a_candidate_meeting_every_criterion_is_accepted(self) -> None:
        decision = evaluate_gate(**passing_kwargs())

        assert decision.accepted
        assert decision.failures == []
        assert "passes the selection gate" in decision.summary()

    def test_all_seven_readme_criteria_are_evaluated(self) -> None:
        """The README lists seven conditions; none may be quietly dropped."""
        decision = evaluate_gate(**passing_kwargs())

        assert [criterion.name for criterion in decision.criteria] == [
            "primary metric",
            "draw performance",
            "valid and calibrated probabilities",
            "feature schema compatibility",
            "artifact save and load",
            "local resources",
            "licence",
        ]


class TestPrimaryMetric:
    def test_a_candidate_that_does_not_improve_log_loss_is_rejected(self) -> None:
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(log_loss_target=1.00, draw_recall_target=0.25, name="c")
            )
        )

        assert not decision.accepted
        assert [c.name for c in decision.failures] == ["primary metric"]

    def test_a_marginally_worse_candidate_is_rejected(self) -> None:
        """Novelty is not an acceptance criterion, so ties do not pass."""
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(log_loss_target=1.0000001, draw_recall_target=0.25, name="c")
            )
        )

        assert not decision.accepted

    def test_the_reason_quotes_both_log_losses(self) -> None:
        decision = evaluate_gate(**passing_kwargs())

        detail = decision.criteria[0].detail
        assert "0.9000" in detail
        assert "1.0000" in detail


class TestDrawPerformance:
    def test_a_candidate_that_abandons_draws_is_rejected(self) -> None:
        """The failure mode the README calls out: winning on log loss by
        giving up on the hardest class."""
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(log_loss_target=0.80, draw_recall_target=0.0, name="c"),
                baseline=scored(log_loss_target=1.00, draw_recall_target=0.30, name="b"),
            )
        )

        assert not decision.accepted
        assert [c.name for c in decision.failures] == ["draw performance"]

    def test_a_small_draw_recall_dip_is_tolerated(self) -> None:
        """Zero tolerance would reject candidates for noise on 380 matches."""
        baseline_recall = 0.30
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=baseline_recall - MAX_DRAW_RECALL_LOSS / 2,
                    name="c",
                ),
                baseline=scored(log_loss_target=1.00, draw_recall_target=baseline_recall, name="b"),
            )
        )

        assert decision.accepted

    def test_a_dip_beyond_the_allowance_is_rejected(self) -> None:
        baseline_recall = 0.30
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=baseline_recall - MAX_DRAW_RECALL_LOSS * 2,
                    name="c",
                ),
                baseline=scored(log_loss_target=1.00, draw_recall_target=baseline_recall, name="b"),
            )
        )

        assert not decision.accepted


class TestProbabilityValidity:
    def test_rows_that_do_not_sum_to_one_are_rejected(self) -> None:
        broken = valid_probabilities()
        broken[3] = [0.5, 0.5, 0.5]

        decision = evaluate_gate(**passing_kwargs(probabilities=broken))

        assert not decision.accepted
        assert "summing to one" in decision.criteria[2].detail

    def test_negative_probabilities_are_rejected(self) -> None:
        broken = valid_probabilities()
        broken[0] = [-0.1, 0.6, 0.5]

        decision = evaluate_gate(**passing_kwargs(probabilities=broken))

        assert not decision.accepted
        assert "outside [0, 1]" in decision.criteria[2].detail

    def test_non_finite_probabilities_are_rejected(self) -> None:
        broken = valid_probabilities()
        broken[1] = [np.nan, 0.5, 0.5]

        decision = evaluate_gate(**passing_kwargs(probabilities=broken))

        assert not decision.accepted
        assert "non-finite" in decision.criteria[2].detail

    def test_a_badly_calibrated_candidate_is_rejected(self) -> None:
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=0.25,
                    name="c",
                    calibration_error=MAX_EXPECTED_CALIBRATION_ERROR * 2,
                )
            )
        )

        assert not decision.accepted
        assert "calibration error" in decision.criteria[2].detail

    def test_unsupplied_probabilities_count_as_a_failure_not_a_pass(self) -> None:
        """An unchecked criterion must never pass silently."""
        decision = evaluate_gate(**passing_kwargs(probabilities=None))

        assert not decision.accepted
        assert decision.criteria[2].measured is False


class TestSchemaAndArtifactCriteria:
    @pytest.mark.parametrize("field", ["schema_matches", "round_trip_matches"])
    def test_a_failing_check_rejects_the_candidate(self, field: str) -> None:
        decision = evaluate_gate(**passing_kwargs(**{field: False}))

        assert not decision.accepted

    @pytest.mark.parametrize("field", ["schema_matches", "round_trip_matches"])
    def test_an_unchecked_criterion_rejects_the_candidate(self, field: str) -> None:
        decision = evaluate_gate(**passing_kwargs(**{field: None}))

        assert not decision.accepted
        assert any(c.measured is False for c in decision.failures)


class TestResourceCriteria:
    def test_a_slow_candidate_is_rejected(self) -> None:
        """The API predicts one fixture per request, so latency is user-visible."""
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=0.25,
                    name="c",
                    latency_ms=MAX_INFERENCE_LATENCY_MS * 4,
                )
            )
        )

        assert not decision.accepted
        assert "latency" in decision.criteria[5].detail

    def test_a_memory_hungry_candidate_is_rejected(self) -> None:
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=0.25,
                    name="c",
                    memory_mb=1_000_000.0,
                )
            )
        )

        assert not decision.accepted
        assert "peak memory" in decision.criteria[5].detail

    def test_unmeasured_latency_rejects_the_candidate(self) -> None:
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=0.25,
                    name="c",
                    latency_ms=None,
                )
            )
        )

        assert not decision.accepted
        assert decision.criteria[5].measured is False

    def test_missing_memory_alone_does_not_reject(self) -> None:
        """Latency is required; a memory reading is reported when available."""
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(
                    log_loss_target=0.90,
                    draw_recall_target=0.25,
                    name="c",
                    memory_mb=None,
                )
            )
        )

        assert decision.accepted


class TestLicenceCriterion:
    def test_a_non_commercial_licence_is_rejected_however_good_the_metrics(self) -> None:
        """TabPFN-3's situation: benchmarkable, never promotable."""
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(log_loss_target=0.10, draw_recall_target=0.90, name="c"),
                license_summary="Prior Labs non-commercial model licence",
                production_use_allowed=False,
            )
        )

        assert not decision.accepted
        assert [c.name for c in decision.failures] == ["licence"]

    def test_an_unrecorded_licence_rejects_the_candidate(self) -> None:
        decision = evaluate_gate(**passing_kwargs(license_summary=""))

        assert not decision.accepted
        assert decision.criteria[6].measured is False


class TestDecisionReporting:
    def test_a_rejection_lists_every_failed_criterion(self) -> None:
        """A rejection must be recordable, not just a boolean."""
        decision = evaluate_gate(
            **passing_kwargs(
                candidate=scored(log_loss_target=1.5, draw_recall_target=0.0, name="c"),
                license_summary="",
            )
        )

        names = {criterion.name for criterion in decision.failures}
        assert names == {"primary metric", "draw performance", "licence"}
        assert decision.candidate == "c"
        assert decision.baseline == "baseline"

    def test_the_decision_serialises_for_the_registry(self) -> None:
        payload = evaluate_gate(**passing_kwargs()).as_dict()

        assert payload["accepted"] is True
        assert len(payload["criteria"]) == 7
        assert all(
            {"name", "passed", "measured", "detail"} <= c.keys() for c in payload["criteria"]
        )
