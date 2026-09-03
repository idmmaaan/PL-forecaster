"""Probability calibration.

The properties that matter operationally: temperature scaling cannot reorder
outcomes (so it can never turn a home win into an away win), every calibrator
returns a valid distribution, and a calibrator fitted on a well-calibrated
model leaves it roughly alone.
"""

from typing import Any

import numpy as np
import pytest

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.evaluation.metrics import expected_calibration_error, log_loss
from epl_predictor.predictors.calibration import (
    CALIBRATORS,
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
    TemperatureScaler,
    load_calibrator,
    make_calibrator,
)

HOME, DRAW, AWAY = OUTCOME_CLASSES


def overconfident(rows: int = 200, seed: int = 3) -> tuple[np.ndarray, list[str]]:
    """Predictions that are far more certain than they deserve to be.

    Every row says 90 % home win, but the home side wins only half the time,
    so a calibrator has something real to correct.
    """
    rng = np.random.default_rng(seed)
    probabilities = np.tile([0.90, 0.05, 0.05], (rows, 1))
    labels = [HOME if value < 0.5 else AWAY for value in rng.random(rows)]
    return probabilities, labels


def well_calibrated(rows: int = 300, seed: int = 5) -> tuple[np.ndarray, list[str]]:
    """Predictions whose stated confidence matches reality."""
    rng = np.random.default_rng(seed)
    probabilities = np.tile([0.5, 0.25, 0.25], (rows, 1))
    labels = [[HOME, DRAW, AWAY][index] for index in rng.choice(3, size=rows, p=[0.5, 0.25, 0.25])]
    return probabilities, labels


ALL_METHODS = ["identity", "temperature", "isotonic", "sigmoid"]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_calibrator_returns_a_valid_distribution(method: str) -> None:
    probabilities, labels = overconfident()
    calibrator = make_calibrator(method)
    calibrator.fit(probabilities, labels)

    calibrated = calibrator.transform(probabilities)

    assert calibrated.shape == probabilities.shape
    assert np.all(calibrated >= 0.0)
    assert np.allclose(calibrated.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_calibrator_satisfies_the_protocol(method: str) -> None:
    assert isinstance(make_calibrator(method), Calibrator)


def test_an_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        make_calibrator("magic")


def test_the_registry_lists_every_readme_method() -> None:
    assert set(CALIBRATORS) == {"identity", "temperature", "isotonic", "sigmoid"}


# --- Identity ---------------------------------------------------------------


def test_the_identity_calibrator_is_the_uncalibrated_control() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1]])
    calibrator = IdentityCalibrator()
    calibrator.fit(probabilities, [HOME])

    assert np.allclose(calibrator.transform(probabilities), probabilities)
    assert calibrator.as_dict() == {"method": "identity"}


# --- Temperature ------------------------------------------------------------


def test_temperature_scaling_improves_an_overconfident_model() -> None:
    probabilities, labels = overconfident()
    calibrator = TemperatureScaler()
    calibrator.fit(probabilities, labels)

    before = expected_calibration_error(probabilities, labels)
    after = expected_calibration_error(calibrator.transform(probabilities), labels)

    assert after < before
    assert calibrator.temperature > 1.0, "an overconfident model needs softening"


def test_temperature_scaling_reduces_log_loss_it_was_fitted_on() -> None:
    probabilities, labels = overconfident()
    calibrator = TemperatureScaler()
    calibrator.fit(probabilities, labels)

    assert log_loss(calibrator.transform(probabilities), labels) < log_loss(probabilities, labels)


def test_temperature_scaling_never_reorders_the_outcomes() -> None:
    """One shared parameter cannot change which outcome is most likely.

    This is the reason to prefer it: it fixes confidence without touching the
    model's ranking, so accuracy is guaranteed unchanged.
    """
    probabilities = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])

    for temperature in (0.3, 0.9, 1.0, 2.5, 4.0):
        calibrated = TemperatureScaler(temperature).transform(probabilities)
        assert np.array_equal(np.argmax(calibrated, axis=1), np.argmax(probabilities, axis=1))


def test_a_temperature_of_one_leaves_probabilities_alone() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1]])

    assert np.allclose(TemperatureScaler(1.0).transform(probabilities), probabilities)


def test_a_low_temperature_sharpens_and_a_high_one_softens() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1]])

    sharpened = TemperatureScaler(0.5).transform(probabilities)
    softened = TemperatureScaler(3.0).transform(probabilities)

    assert sharpened[0, 0] > 0.6
    assert softened[0, 0] < 0.6


def test_temperature_stays_within_its_bounds() -> None:
    """Bounded fitting keeps the optimiser from chasing a degenerate solution."""
    probabilities, labels = overconfident()
    calibrator = TemperatureScaler()
    calibrator.fit(probabilities, labels)

    assert 0.2 <= calibrator.temperature <= 5.0


def test_the_fitted_temperature_is_serialised() -> None:
    """A probability is not reproducible without the transform that made it."""
    calibrator = TemperatureScaler(2.5)
    payload = calibrator.as_dict()

    assert payload == {"method": "temperature", "temperature": 2.5}
    restored = load_calibrator(payload)
    assert isinstance(restored, TemperatureScaler)
    assert restored.temperature == 2.5


def test_a_well_calibrated_model_is_left_near_alone() -> None:
    probabilities, labels = well_calibrated()
    calibrator = TemperatureScaler()
    calibrator.fit(probabilities, labels)

    assert calibrator.temperature == pytest.approx(1.0, abs=0.25)


# --- Per-class methods ------------------------------------------------------


@pytest.mark.parametrize("factory", [IsotonicCalibrator, SigmoidCalibrator])
def test_per_class_calibrators_fit_one_model_per_outcome(factory: Any) -> None:
    probabilities, labels = overconfident()
    calibrator = factory()
    calibrator.fit(probabilities, labels)

    assert len(calibrator.models) == len(OUTCOME_CLASSES)
    assert calibrator.as_dict()["fitted_classes"] == 3


@pytest.mark.parametrize("factory", [IsotonicCalibrator, SigmoidCalibrator])
def test_a_class_absent_from_validation_is_left_uncalibrated(factory: Any) -> None:
    """Nothing can be learned about an outcome that never occurred."""
    probabilities = np.tile([0.5, 0.25, 0.25], (50, 1))
    labels = [HOME] * 25 + [AWAY] * 25

    calibrator = factory()
    calibrator.fit(probabilities, labels)

    draw_index = OUTCOME_CLASSES.index(DRAW)
    assert calibrator.models[draw_index] is None
    assert np.allclose(calibrator.transform(probabilities).sum(axis=1), 1.0)


@pytest.mark.parametrize("factory", [IsotonicCalibrator, SigmoidCalibrator])
def test_an_unfitted_per_class_calibrator_passes_probabilities_through(
    factory: Any,
) -> None:
    probabilities = np.array([[0.6, 0.3, 0.1]])

    assert np.allclose(factory().transform(probabilities), probabilities)


def test_isotonic_calibration_can_reorder_outcomes() -> None:
    """The trade-off against temperature scaling, stated explicitly.

    Per-class fitting followed by renormalising is more flexible but may change
    which outcome is most likely, so accuracy is not preserved.
    """
    rng = np.random.default_rng(11)
    probabilities = rng.dirichlet([2.0, 2.0, 2.0], size=400)
    labels = [[HOME, DRAW, AWAY][index] for index in rng.choice(3, size=400, p=[0.2, 0.3, 0.5])]

    calibrator = IsotonicCalibrator()
    calibrator.fit(probabilities, labels)
    calibrated = calibrator.transform(probabilities)

    assert not np.array_equal(np.argmax(calibrated, axis=1), np.argmax(probabilities, axis=1))


# --- Input handling ---------------------------------------------------------


def test_a_single_row_is_accepted() -> None:
    assert TemperatureScaler(1.5).transform([0.6, 0.3, 0.1]).shape == (1, 3)


def test_a_wrong_column_count_is_refused() -> None:
    with pytest.raises(ValueError, match="probability columns"):
        IdentityCalibrator().transform([[0.5, 0.5]])


def test_zero_probabilities_survive_calibration() -> None:
    """A zero must not become a NaN or an infinite log odds."""
    calibrated = TemperatureScaler(2.0).transform([[1.0, 0.0, 0.0]])

    assert np.all(np.isfinite(calibrated))
    assert calibrated.sum() == pytest.approx(1.0)


def test_loading_no_calibration_gives_the_identity() -> None:
    assert isinstance(load_calibrator(None), IdentityCalibrator)
    assert isinstance(load_calibrator({}), IdentityCalibrator)


def test_loading_a_per_class_method_from_metadata_alone_does_not_pretend() -> None:
    """The fitted mapping lives in the artifact, not in the metadata.

    Returning a fresh isotonic calibrator here would silently apply an
    unfitted transform, so the identity is returned instead and the real one
    is restored from the pickled artifact.
    """
    restored = load_calibrator({"method": "isotonic", "fitted_classes": 3})

    assert isinstance(restored, IdentityCalibrator)
