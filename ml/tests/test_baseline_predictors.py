"""The trainable baselines, behind the shared `Predictor` interface.

Each adapter is checked for the same things: it honours the interface, it
returns probabilities in canonical class order, and it survives a save/load
round trip unchanged. Class order gets particular attention because scikit-
learn and CatBoost both sort their classes alphabetically, which would swap
home and away without raising anything.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from epl_predictor import OUTCOME_CLASSES, PROBABILITY_KEYS
from epl_predictor.features.builder import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
    build_training_table,
)
from epl_predictor.predictors.base import Predictor
from epl_predictor.predictors.calibration import TemperatureScaler
from epl_predictor.predictors.catboost import CatBoostPredictor
from epl_predictor.predictors.class_frequency import ClassFrequencyPredictor
from epl_predictor.predictors.logistic_regression import LogisticRegressionPredictor
from epl_predictor.predictors.tabular import NotFittedError, TabularPredictor

CLUBS = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Everton", "Fulham"]

ADAPTERS: list[type[TabularPredictor]] = [
    ClassFrequencyPredictor,
    LogisticRegressionPredictor,
    CatBoostPredictor,
]


def canonical_matches(seasons: int = 3, seed: int = 4) -> pd.DataFrame:
    """A deterministic multi-season fixture list with a home-win skew."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for offset in range(seasons):
        season = 2020 + offset
        kickoff = date(season, 8, 8)
        for home in CLUBS:
            for away in CLUBS:
                if home == away:
                    continue
                home_goals = int(rng.integers(0, 4))
                away_goals = int(rng.integers(0, 3))
                rows.append(
                    {
                        "match_id": f"{season}:{home}:{away}",
                        "season_start_year": season,
                        "kickoff_date": pd.Timestamp(kickoff),
                        "home_team": home,
                        "away_team": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "result": (
                            "H"
                            if home_goals > away_goals
                            else "A"
                            if home_goals < away_goals
                            else "D"
                        ),
                        "home_shots_on_target": home_goals + 3,
                        "away_shots_on_target": away_goals + 2,
                    }
                )
                kickoff += timedelta(days=4)

    return pd.DataFrame(rows).sort_values("kickoff_date").reset_index(drop=True)


@pytest.fixture(scope="module")
def periods() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = build_training_table(canonical_matches())[0]
    train = table[table["season_start_year"] < 2022].reset_index(drop=True)
    validation = table[table["season_start_year"] == 2022].reset_index(drop=True)
    return train, validation


@pytest.fixture(params=ADAPTERS, ids=[adapter.adapter_type for adapter in ADAPTERS])
def fitted(
    request: pytest.FixtureRequest, periods: tuple[pd.DataFrame, pd.DataFrame]
) -> TabularPredictor:
    train, validation = periods
    predictor: TabularPredictor = request.param()
    predictor.fit(train, validation)
    return predictor


# --- Interface conformance --------------------------------------------------


def test_every_adapter_implements_the_predictor_interface(
    fitted: TabularPredictor,
) -> None:
    assert isinstance(fitted, Predictor)
    assert fitted.model_name
    assert fitted.model_version


def test_predicting_before_fitting_is_refused() -> None:
    """Better a clear error than probabilities from an untrained model."""
    with pytest.raises(NotFittedError):
        ClassFrequencyPredictor().predict_matrix(pd.DataFrame())


def test_fitting_on_a_non_frame_is_refused() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        ClassFrequencyPredictor().fit([{"home_team": "Arsenal"}])


def test_fitting_on_an_empty_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        ClassFrequencyPredictor().fit(pd.DataFrame({TARGET_COLUMN: []}))


# --- Output contract --------------------------------------------------------


def test_a_frame_of_fixtures_yields_a_valid_distribution_per_row(
    fitted: TabularPredictor, periods: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    matrix = fitted.predict_matrix(periods[1])

    assert matrix.shape == (len(periods[1]), 3)
    assert np.all(matrix >= 0.0)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_a_single_fixture_yields_the_contract_keys(
    fitted: TabularPredictor, periods: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    features = periods[1].iloc[0][list(FEATURE_COLUMNS)].to_dict()

    probabilities = fitted.predict_proba(features)

    assert list(probabilities) == list(PROBABILITY_KEYS)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())


def test_single_row_and_batch_prediction_agree(
    fitted: TabularPredictor, periods: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """The API path and the evaluation path must not diverge."""
    validation = periods[1]
    batch = fitted.predict_matrix(validation)
    single = fitted.predict_proba(validation.iloc[3][list(FEATURE_COLUMNS)].to_dict())

    assert list(single.values()) == pytest.approx(list(batch[3]), abs=1e-9)


def test_prediction_is_deterministic(
    fitted: TabularPredictor, periods: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    first = fitted.predict_matrix(periods[1])
    second = fitted.predict_matrix(periods[1])

    assert np.array_equal(first, second)


def test_a_missing_feature_column_is_refused(fitted: TabularPredictor) -> None:
    with pytest.raises(ValueError, match="missing column"):
        fitted.predict_matrix(pd.DataFrame({"home_team": ["Arsenal"]}))


def lopsided_matches(dominant: str, seed: int = 9) -> pd.DataFrame:
    """A fixture list where `dominant` ("H", "D", or "A") is 90 % of results.

    Deliberately unambiguous: any adapter that permuted its class columns
    would put its largest probability on the wrong outcome, which a
    realistically balanced dataset is too noisy to reveal.
    """
    scores = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}
    others = [key for key in scores if key != dominant]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for offset in range(3):
        season = 2020 + offset
        kickoff = date(season, 8, 8)
        for home in CLUBS:
            for away in CLUBS:
                if home == away:
                    continue
                result = dominant if rng.random() < 0.9 else others[int(rng.integers(0, 2))]
                home_goals, away_goals = scores[result]
                rows.append(
                    {
                        "match_id": f"{season}:{home}:{away}",
                        "season_start_year": season,
                        "kickoff_date": pd.Timestamp(kickoff),
                        "home_team": home,
                        "away_team": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "result": result,
                        "home_shots_on_target": home_goals + 3,
                        "away_shots_on_target": away_goals + 2,
                    }
                )
                kickoff += timedelta(days=4)

    return pd.DataFrame(rows).sort_values("kickoff_date").reset_index(drop=True)


@pytest.mark.parametrize(
    ("dominant", "expected_outcome"),
    [("H", "HOME_WIN"), ("D", "DRAW"), ("A", "AWAY_WIN")],
)
@pytest.mark.parametrize("factory", ADAPTERS, ids=[a.adapter_type for a in ADAPTERS])
def test_probabilities_are_reported_in_canonical_class_order(
    factory: type[TabularPredictor], dominant: str, expected_outcome: str
) -> None:
    """The bug this guards: sklearn and CatBoost both sort their own classes.

    Both order labels alphabetically, giving AWAY_WIN, DRAW, HOME_WIN. Handing
    that back unchanged would swap home and away on every prediction without
    raising anything anywhere.
    """
    table = build_training_table(lopsided_matches(dominant))[0]
    train = table[table["season_start_year"] < 2022].reset_index(drop=True)
    validation = table[table["season_start_year"] == 2022].reset_index(drop=True)

    predictor = factory()
    predictor.fit(train, validation)
    mean = predictor.predict_matrix(validation).mean(axis=0)

    assert OUTCOME_CLASSES[int(np.argmax(mean))] == expected_outcome, (
        f"{predictor.model_name} put its largest probability on "
        f"{OUTCOME_CLASSES[int(np.argmax(mean))]} when {expected_outcome} "
        f"was 90 % of results"
    )


@pytest.mark.parametrize("factory", ADAPTERS, ids=[a.adapter_type for a in ADAPTERS])
def test_the_estimators_own_class_order_is_recorded(
    factory: type[TabularPredictor], periods: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Whatever order the library chose, the adapter must know what it was."""
    train, validation = periods
    predictor = factory()
    predictor.fit(train, validation)

    assert predictor.metadata()["class_order"] == list(OUTCOME_CLASSES)


# --- Artifacts --------------------------------------------------------------


def test_an_artifact_round_trips_unchanged(
    fitted: TabularPredictor, periods: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    before = fitted.predict_matrix(periods[1])

    fitted.save(tmp_path / "artifact")
    reloaded = type(fitted).load(tmp_path / "artifact")

    assert reloaded.model_name == fitted.model_name
    assert reloaded.model_version == fitted.model_version
    assert np.allclose(reloaded.predict_matrix(periods[1]), before)


def test_the_artifact_records_the_schema_and_class_order(
    fitted: TabularPredictor, tmp_path: Path
) -> None:
    """An artifact whose class order is unknown cannot be interpreted later."""
    import json

    fitted.save(tmp_path / "artifact")
    metadata = json.loads((tmp_path / "artifact" / "predictor.json").read_text())

    assert metadata["adapter_type"] == fitted.adapter_type
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["class_order"] == list(OUTCOME_CLASSES)
    assert metadata["probability_keys"] == list(PROBABILITY_KEYS)
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)


def test_loading_from_an_empty_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="predictor.json"):
        ClassFrequencyPredictor.load(tmp_path)


def test_a_fitted_calibrator_survives_the_round_trip(
    periods: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    """Reloading as uncalibrated would silently change every probability."""
    train, validation = periods
    predictor = LogisticRegressionPredictor(calibrator=TemperatureScaler())
    predictor.fit(train, validation)
    before = predictor.predict_matrix(validation)

    predictor.save(tmp_path / "artifact")
    reloaded = LogisticRegressionPredictor.load(tmp_path / "artifact")

    assert isinstance(reloaded.calibrator, TemperatureScaler)
    assert reloaded.calibrator.temperature == pytest.approx(predictor.calibrator.temperature)
    assert np.allclose(reloaded.predict_matrix(validation), before)


# --- Adapter-specific behaviour ---------------------------------------------


def test_the_class_frequency_baseline_learns_the_training_distribution(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, validation = periods
    predictor = ClassFrequencyPredictor()
    predictor.fit(train, validation)

    observed = train[TARGET_COLUMN].value_counts(normalize=True)
    for index, outcome in enumerate(OUTCOME_CLASSES):
        assert predictor.frequencies[index] == pytest.approx(observed.get(outcome, 0.0))


def test_the_class_frequency_baseline_ignores_its_features(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """It is the no-information reference, so every fixture must score alike."""
    train, validation = periods
    predictor = ClassFrequencyPredictor()
    predictor.fit(train, validation)

    matrix = predictor.predict_matrix(validation)

    assert np.allclose(matrix, matrix[0])


def test_the_class_frequency_baseline_does_not_learn_from_validation(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, validation = periods

    with_validation = ClassFrequencyPredictor()
    with_validation.fit(train, validation)
    without = ClassFrequencyPredictor()
    without.fit(train)

    assert np.allclose(with_validation.frequencies, without.frequencies)


def test_logistic_regression_uses_its_features(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A model that ignored the features would be the class-frequency baseline."""
    train, validation = periods
    predictor = LogisticRegressionPredictor()
    predictor.fit(train, validation)

    matrix = predictor.predict_matrix(validation)

    assert matrix[:, 0].std() > 0.01


def test_logistic_regression_imputes_nulls_rather_than_failing(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Opening-weekend fixtures legitimately have null form and position."""
    train, validation = periods
    predictor = LogisticRegressionPredictor()
    predictor.fit(train, validation)

    debut = validation.iloc[0][list(FEATURE_COLUMNS)].to_dict()
    debut["home_points_last_5"] = None
    debut["home_league_position_before_match"] = None

    probabilities = predictor.predict_proba(debut)

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_catboost_stops_early_against_the_validation_period(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, validation = periods
    predictor = CatBoostPredictor(iterations=500)
    predictor.fit(train, validation)

    assert predictor.best_iteration is not None
    assert predictor.best_iteration <= 500


def test_catboost_handles_nulls_natively(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """No imputation step, so a null must reach the model and be routed."""
    train, validation = periods
    predictor = CatBoostPredictor(iterations=50)
    predictor.fit(train, validation)

    features = validation.iloc[0][list(FEATURE_COLUMNS)].to_dict()
    features["home_elo"] = None
    features["home_points_last_5"] = None

    probabilities = predictor.predict_proba(features)

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_catboost_handles_an_unseen_club(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A promoted club appears in fixtures before it appears in training."""
    train, validation = periods
    predictor = CatBoostPredictor(iterations=50)
    predictor.fit(train, validation)

    features = validation.iloc[0][list(FEATURE_COLUMNS)].to_dict()
    features["home_team"] = "Newly Promoted FC"

    probabilities = predictor.predict_proba(features)

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_logistic_regression_handles_an_unseen_club(
    periods: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, validation = periods
    predictor = LogisticRegressionPredictor()
    predictor.fit(train, validation)

    features = validation.iloc[0][list(FEATURE_COLUMNS)].to_dict()
    features["away_team"] = "Newly Promoted FC"

    probabilities = predictor.predict_proba(features)

    assert sum(probabilities.values()) == pytest.approx(1.0)
