"""Outcome metrics, checked against hand-computed values where possible.

The draw metrics get their own attention because the README makes draw
performance part of the promotion rules: a model that never predicts a draw
must be visibly bad here, not merely unremarkable.
"""

import math
from typing import Any

import numpy as np
import pytest

from epl_predictor import OUTCOME_CLASSES, Outcome
from epl_predictor.evaluation.metrics import (
    PROBABILITY_FLOOR,
    ClassMetrics,
    ResourceMetrics,
    ShapeMismatchError,
    accuracy,
    as_label_indices,
    as_probability_matrix,
    brier_score,
    calibration_curve,
    confusion_matrix,
    evaluate,
    expected_calibration_error,
    log_loss,
    per_class_metrics,
)

HOME, DRAW, AWAY = OUTCOME_CLASSES
UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def test_canonical_class_order_is_home_draw_away() -> None:
    """Every probability matrix in the project is interpreted in this order."""
    assert OUTCOME_CLASSES == ("HOME_WIN", "DRAW", "AWAY_WIN")


# --- Input handling ---------------------------------------------------------


def test_a_single_prediction_is_accepted_as_one_row() -> None:
    assert as_probability_matrix(UNIFORM).shape == (1, 3)


@pytest.mark.parametrize("bad", [[[0.5, 0.5]], [[0.25, 0.25, 0.25, 0.25]], [[[0.3]]]])
def test_wrongly_shaped_probabilities_are_refused(bad: Any) -> None:
    with pytest.raises(ShapeMismatchError):
        as_probability_matrix(bad)


def test_labels_may_be_names_or_class_indices() -> None:
    assert as_label_indices([HOME, DRAW, AWAY]).tolist() == [0, 1, 2]
    assert as_label_indices([0, 1, 2]).tolist() == [0, 1, 2]


def test_an_unknown_label_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown outcome label"):
        as_label_indices(["SCORE_DRAW"])


def test_an_out_of_range_class_index_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the three outcomes"):
        as_label_indices([3])


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ShapeMismatchError, match="probability rows"):
        log_loss([UNIFORM, UNIFORM], [HOME])


def test_an_empty_evaluation_is_refused() -> None:
    """Silently returning 0.0 would look like a perfect score."""
    with pytest.raises(ShapeMismatchError, match="empty"):
        log_loss(np.empty((0, 3)), [])


# --- Log loss ---------------------------------------------------------------


def test_log_loss_of_a_uniform_prediction_is_log_three() -> None:
    assert log_loss([UNIFORM], [HOME]) == pytest.approx(math.log(3))


def test_log_loss_of_a_perfect_prediction_is_zero() -> None:
    assert log_loss([[1.0, 0.0, 0.0]], [HOME]) == pytest.approx(0.0)


def test_log_loss_uses_only_the_probability_of_what_happened() -> None:
    assert log_loss([[0.6, 0.3, 0.1]], [HOME]) == pytest.approx(-math.log(0.6))
    assert log_loss([[0.6, 0.3, 0.1]], [DRAW]) == pytest.approx(-math.log(0.3))


def test_log_loss_punishes_confident_mistakes_harder_than_hedged_ones() -> None:
    """This is precisely why log loss is the primary selection metric."""
    confident_and_wrong = log_loss([[0.98, 0.01, 0.01]], [AWAY])
    hedged_and_wrong = log_loss([[0.4, 0.3, 0.3]], [AWAY])

    assert confident_and_wrong > hedged_and_wrong


def test_a_zero_probability_on_the_true_outcome_stays_finite() -> None:
    """An infinite score would make every model comparison meaningless."""
    loss = log_loss([[1.0, 0.0, 0.0]], [DRAW])

    assert math.isfinite(loss)
    assert loss == pytest.approx(-math.log(PROBABILITY_FLOOR))


def test_log_loss_averages_over_matches() -> None:
    both = log_loss([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]], [HOME, DRAW])

    assert both == pytest.approx(-math.log(0.6))


# --- Brier ------------------------------------------------------------------


def test_brier_of_a_perfect_prediction_is_zero() -> None:
    assert brier_score([[1.0, 0.0, 0.0]], [HOME]) == pytest.approx(0.0)


def test_brier_of_a_confidently_wrong_prediction_is_two() -> None:
    assert brier_score([[1.0, 0.0, 0.0]], [AWAY]) == pytest.approx(2.0)


def test_brier_of_a_uniform_prediction_is_two_thirds() -> None:
    assert brier_score([UNIFORM], [HOME]) == pytest.approx(2 / 3)


# --- Accuracy and confusion -------------------------------------------------


def test_accuracy_counts_the_most_likely_outcome() -> None:
    probabilities = [[0.6, 0.3, 0.1], [0.1, 0.3, 0.6], [0.3, 0.4, 0.3]]

    assert accuracy(probabilities, [HOME, AWAY, DRAW]) == pytest.approx(1.0)
    assert accuracy(probabilities, [AWAY, AWAY, DRAW]) == pytest.approx(2 / 3)


def test_the_confusion_matrix_places_actuals_on_rows() -> None:
    counts = confusion_matrix([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]], [HOME, AWAY])

    assert counts[0][0] == 1, "a home win predicted as a home win"
    assert counts[2][0] == 1, "an away win predicted as a home win"
    assert counts.sum() == 2


# --- Per-class and draw metrics ---------------------------------------------


def test_per_class_metrics_cover_all_three_outcomes() -> None:
    metrics = per_class_metrics([UNIFORM], [HOME])

    assert set(metrics) == set(OUTCOME_CLASSES)
    assert all(isinstance(value, ClassMetrics) for value in metrics.values())


def test_a_model_that_never_predicts_a_draw_scores_zero_on_draws() -> None:
    """The failure mode the README singles out: good-looking, draw-blind."""
    probabilities = [[0.5, 0.2, 0.3]] * 4
    labels = [HOME, HOME, DRAW, DRAW]

    metrics = per_class_metrics(probabilities, labels)
    draw = metrics[Outcome.DRAW.value]

    assert accuracy(probabilities, labels) == pytest.approx(0.5)
    assert draw.recall == 0.0
    assert draw.precision == 0.0
    assert draw.f1 == 0.0
    assert draw.predicted == 0
    assert draw.support == 2


def test_precision_and_recall_are_computed_per_class() -> None:
    #                       predicted:  HOME    DRAW    HOME    DRAW
    probabilities = [[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.6, 0.3, 0.1], [0.3, 0.6, 0.1]]
    labels = [HOME, DRAW, DRAW, DRAW]

    metrics = per_class_metrics(probabilities, labels)

    home = metrics[HOME]
    assert home.precision == pytest.approx(0.5), "1 of 2 home predictions was right"
    assert home.recall == pytest.approx(1.0), "the only home win was found"

    draw = metrics[DRAW]
    assert draw.precision == pytest.approx(1.0), "both draw predictions were right"
    assert draw.recall == pytest.approx(2 / 3), "2 of 3 draws were found"
    assert draw.f1 == pytest.approx(0.8)


def test_a_class_that_never_occurs_scores_zero_rather_than_nan() -> None:
    metrics = per_class_metrics([[0.6, 0.3, 0.1]], [HOME])

    assert metrics[AWAY].recall == 0.0
    assert metrics[AWAY].precision == 0.0
    assert metrics[AWAY].support == 0


# --- Calibration ------------------------------------------------------------


def test_a_perfectly_calibrated_model_has_no_calibration_error() -> None:
    """Sixty per cent confidence that is right sixty per cent of the time."""
    probabilities = [[0.6, 0.2, 0.2]] * 10
    labels = [HOME] * 6 + [AWAY] * 4

    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0, abs=1e-9)


def test_an_overconfident_model_has_calibration_error() -> None:
    probabilities = [[0.95, 0.03, 0.02]] * 10
    labels = [HOME] * 5 + [AWAY] * 5

    error = expected_calibration_error(probabilities, labels)

    assert error == pytest.approx(0.45, abs=0.01)


def test_calibration_error_is_bounded_by_one() -> None:
    always_wrong = expected_calibration_error([[1.0, 0.0, 0.0]] * 5, [AWAY] * 5)

    assert 0.0 <= always_wrong <= 1.0


def test_the_calibration_curve_reports_predicted_against_observed() -> None:
    probabilities = [[0.2, 0.6, 0.2]] * 10
    labels = [DRAW] * 6 + [HOME] * 4

    curve = calibration_curve(probabilities, labels, outcome=Outcome.DRAW)

    assert len(curve) == 1
    assert curve[0]["mean_predicted"] == pytest.approx(0.6)
    assert curve[0]["observed_rate"] == pytest.approx(0.6)
    assert curve[0]["count"] == 10


def test_the_calibration_curve_defaults_to_the_draw_class() -> None:
    curve = calibration_curve([[0.2, 0.6, 0.2]] * 4, [DRAW] * 4)

    assert curve[0]["mean_predicted"] == pytest.approx(0.6)


# --- Aggregate result -------------------------------------------------------


def test_evaluate_collects_every_required_metric() -> None:
    probabilities = [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]
    labels = [HOME, DRAW, AWAY]

    result = evaluate(probabilities, labels, model_name="stub", period="2024")

    assert result.model_name == "stub"
    assert result.period == "2024"
    assert result.rows == 3
    assert result.accuracy == pytest.approx(1.0)
    assert result.log_loss == pytest.approx(-math.log(0.5))
    assert result.draw_recall == pytest.approx(1.0)
    assert set(result.per_class) == set(OUTCOME_CLASSES)
    assert len(result.confusion_matrix) == 3


def test_the_result_serialises_with_its_class_order() -> None:
    """A confusion matrix without its class order cannot be read later."""
    result = evaluate([UNIFORM], [HOME], model_name="stub")
    payload = result.as_dict()

    assert payload["confusion_matrix_classes"] == list(OUTCOME_CLASSES)
    assert payload["per_class"][DRAW]["outcome"] == DRAW
    assert payload["log_loss"] == pytest.approx(math.log(3))


def test_resource_metrics_travel_with_the_result() -> None:
    result = evaluate(
        [UNIFORM],
        [HOME],
        resources=ResourceMetrics(
            inference_latency_ms=1.5, artifact_size_bytes=2048, peak_memory_mb=64.0
        ),
    )

    assert result.resources.inference_latency_ms == 1.5
    assert result.as_dict()["resources"]["artifact_size_bytes"] == 2048


def test_resource_metrics_default_to_unmeasured() -> None:
    """None means "not measured", which is different from zero cost."""
    result = evaluate([UNIFORM], [HOME])

    assert result.resources.inference_latency_ms is None
    assert result.resources.artifact_size_bytes is None
