from pathlib import Path
from typing import Any

import pytest

from epl_predictor import PROBABILITY_KEYS
from epl_predictor.predictors.base import Predictor
from epl_predictor.predictors.dummy import DummyPredictor

EXPECTED = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}


def test_dummy_predictor_implements_the_interface() -> None:
    assert isinstance(DummyPredictor(), Predictor)


def test_dummy_predictor_initialization() -> None:
    predictor = DummyPredictor()
    assert predictor.model_name == "dummy-epl"
    assert predictor.model_version == "0.1.0"

    custom = DummyPredictor("custom-dummy", "2.0.0")
    assert custom.model_name == "custom-dummy"
    assert custom.model_version == "2.0.0"


def test_dummy_predictor_deterministic_output() -> None:
    """Identical calls and differing features must give identical probabilities."""
    predictor = DummyPredictor()

    results = [
        predictor.predict_proba({"some_feature": 1.0}),
        predictor.predict_proba({"another_feature": 2.0}),
        predictor.predict_proba({"home_team": "Arsenal", "away_team": "Chelsea"}),
    ]

    assert all(result == EXPECTED for result in results)


def test_dummy_predictor_probability_sum() -> None:
    probabilities = DummyPredictor().predict_proba({"test_feature": 1.0})

    assert abs(sum(probabilities.values()) - 1.0) < 1e-9


def test_dummy_predictor_returns_all_classes_in_canonical_order() -> None:
    probabilities = DummyPredictor().predict_proba({"test": 1.0})

    assert tuple(probabilities) == PROBABILITY_KEYS
    assert all(isinstance(value, float) for value in probabilities.values())


def test_dummy_predictor_probability_ranges() -> None:
    probabilities = DummyPredictor().predict_proba({"test": 1.0})

    for key, value in probabilities.items():
        assert 0.0 <= value <= 1.0, f"Probability {key} ({value}) must be between 0 and 1"


def test_dummy_predictor_output_is_not_shared_between_calls() -> None:
    """A caller mutating the result must not corrupt later predictions."""
    predictor = DummyPredictor()

    predictor.predict_proba({})["home_win"] = 99.0

    assert predictor.predict_proba({}) == EXPECTED


@pytest.mark.parametrize(
    "features",
    [
        pytest.param({}, id="empty"),
        pytest.param({"simple": 1}, id="scalar"),
        pytest.param({"home_team": "Arsenal", "away_team": "Chelsea"}, id="strings"),
        pytest.param({"numeric": [1, 2, 3]}, id="list-value"),
        pytest.param({"nested": {"elo": 1500}}, id="nested-dict"),
    ],
)
def test_dummy_predictor_with_various_features(features: dict[str, Any]) -> None:
    assert DummyPredictor().predict_proba(features) == EXPECTED


def test_dummy_predictor_fit_is_a_noop() -> None:
    predictor = DummyPredictor()

    predictor.fit(train_data=None, validation_data=None)

    assert predictor.predict_proba({}) == EXPECTED


def test_dummy_predictor_save_load_round_trip(tmp_path: Path) -> None:
    artifact_path = tmp_path / "dummy-epl-0.1.0"
    DummyPredictor().save(artifact_path)

    assert (artifact_path / "predictor.json").exists()

    loaded = DummyPredictor.load(artifact_path)
    assert isinstance(loaded, DummyPredictor)
    assert loaded.model_name == "dummy-epl"
    assert loaded.model_version == "0.1.0"
    assert loaded.predict_proba({}) == EXPECTED


def test_dummy_predictor_save_preserves_custom_name_and_version(tmp_path: Path) -> None:
    """Reloading an artifact must not silently reset its identity to defaults."""
    DummyPredictor("custom-dummy", "2.0.0").save(tmp_path / "artifact")

    loaded = DummyPredictor.load(tmp_path / "artifact")

    assert loaded.model_name == "custom-dummy"
    assert loaded.model_version == "2.0.0"


def test_dummy_predictor_save_creates_missing_directories(tmp_path: Path) -> None:
    artifact_path = tmp_path / "nested" / "deeper" / "artifact"

    DummyPredictor().save(artifact_path)

    assert (artifact_path / "predictor.json").exists()
