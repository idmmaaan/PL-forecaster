from pathlib import Path
from typing import Any

import pytest

from epl_predictor.predictors.base import Predictor, normalise_probabilities


class MockPredictor(Predictor):
    """Concrete implementation used to exercise the base class contract."""

    def __init__(self, name: str = "test-model", version: str = "0.1.0"):
        self._model_name = name
        self._model_version = version
        self.fitted_with: tuple[Any, Any] | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def fit(self, train_data: Any, validation_data: Any) -> None:
        self.fitted_with = (train_data, validation_data)

    def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
        return {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}

    def save(self, artifact_path: Path) -> None:
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / "model_saved").touch()

    @classmethod
    def load(cls, artifact_path: Path) -> "MockPredictor":
        return cls()


def test_abstract_base_class_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Predictor()  # type: ignore[abstract]


def test_interface_requires_every_documented_method() -> None:
    """The four methods in AGENTS.md must all be abstract on the base class."""
    assert Predictor.__abstractmethods__ >= {
        "fit",
        "predict_proba",
        "save",
        "load",
        "model_name",
        "model_version",
    }


def test_incomplete_implementation_is_rejected() -> None:
    """Omitting fit() must fail at construction, not at training time."""

    class MissingFit(Predictor):
        @property
        def model_name(self) -> str:
            return "missing-fit"

        @property
        def model_version(self) -> str:
            return "0.1.0"

        def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
            return {"home_win": 1.0, "draw": 0.0, "away_win": 0.0}

        def save(self, artifact_path: Path) -> None:
            return None

        @classmethod
        def load(cls, artifact_path: Path) -> "Predictor":
            return cls()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        MissingFit()  # type: ignore[abstract]


def test_predictor_properties() -> None:
    predictor = MockPredictor("sample-model", "2.5.1")

    assert predictor.model_name == "sample-model"
    assert predictor.model_version == "2.5.1"


def test_predict_proba_returns_the_three_outcome_classes() -> None:
    predictor = MockPredictor()

    result = predictor.predict_proba(
        {"home_team": "Arsenal", "away_team": "Chelsea", "elo_difference": 150.0}
    )

    assert set(result) == {"home_win", "draw", "away_win"}
    assert all(isinstance(probability, float) for probability in result.values())


def test_fit_receives_train_and_validation_data_separately() -> None:
    """The interface must keep the validation period distinct from training."""
    predictor = MockPredictor()

    predictor.fit("train-rows", "validation-rows")

    assert predictor.fitted_with == ("train-rows", "validation-rows")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    predictor = MockPredictor("saved-model", "1.0.0")
    artifact_path = tmp_path / "artifact"

    predictor.save(artifact_path)

    assert (artifact_path / "model_saved").exists()
    assert isinstance(MockPredictor.load(artifact_path), MockPredictor)


def test_normalise_probabilities_corrects_floating_point_drift() -> None:
    normalised = normalise_probabilities({"home_win": 0.4, "draw": 0.3, "away_win": 0.3000001})

    assert abs(sum(normalised.values()) - 1.0) < 1e-12
    assert tuple(normalised) == ("home_win", "draw", "away_win")


def test_normalise_probabilities_reorders_into_canonical_key_order() -> None:
    normalised = normalise_probabilities({"away_win": 0.3, "home_win": 0.4, "draw": 0.3})

    assert tuple(normalised) == ("home_win", "draw", "away_win")


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"home_win": 0.5, "draw": 0.5}, id="missing-key"),
        pytest.param({"home_win": -0.5, "draw": 1.0, "away_win": 0.5}, id="negative"),
        pytest.param({"home_win": 0.0, "draw": 0.0, "away_win": 0.0}, id="all-zero"),
    ],
)
def test_normalise_probabilities_rejects_non_distributions(raw: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        normalise_probabilities(raw)
