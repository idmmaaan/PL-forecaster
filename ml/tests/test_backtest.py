"""Walk-forward backtesting.

The property that matters is that a fold's training rows all precede its test
season. A random split would break it, which is why the README forbids one as
the main evaluation method.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.evaluation.backtest import (
    InsufficientHistoryError,
    walk_forward_backtest,
    walk_forward_folds,
)
from epl_predictor.features.builder import TARGET_COLUMN

HOME, DRAW, AWAY = OUTCOME_CLASSES


def table(seasons: dict[int, int]) -> pd.DataFrame:
    """A feature table with a given number of rows per season."""
    rows: list[dict[str, Any]] = []
    for season, count in sorted(seasons.items()):
        for index in range(count):
            rows.append(
                {
                    "match_id": f"{season}:{index}",
                    "season_start_year": season,
                    "home_elo": 1500.0 + index,
                    TARGET_COLUMN: [HOME, DRAW, AWAY][index % 3],
                }
            )
    return pd.DataFrame(rows)


def uniform_trainer(train: pd.DataFrame) -> Callable[[pd.DataFrame], Any]:
    """A trainer that ignores its input and always predicts one third each."""
    assert not train.empty
    return lambda test: np.full((len(test), 3), 1 / 3)


@pytest.fixture
def four_seasons() -> pd.DataFrame:
    return table({2020: 300, 2021: 300, 2022: 300, 2023: 300})


# --- Fold construction ------------------------------------------------------


def test_a_fold_trains_only_on_earlier_seasons(four_seasons: pd.DataFrame) -> None:
    """The defining property of walk-forward evaluation."""
    folds = walk_forward_folds(four_seasons, [2022, 2023])

    assert [fold.season for fold in folds] == [2022, 2023]
    assert folds[0].train_seasons == [2020, 2021]
    assert folds[1].train_seasons == [2020, 2021, 2022]


def test_a_folds_test_rows_are_exactly_its_season(four_seasons: pd.DataFrame) -> None:
    fold = walk_forward_folds(four_seasons, [2022])[0]

    assert fold.test["season_start_year"].unique().tolist() == [2022]
    assert len(fold.test) == 300


def test_no_fold_trains_on_its_own_test_rows(four_seasons: pd.DataFrame) -> None:
    for fold in walk_forward_folds(four_seasons, [2022, 2023]):
        assert set(fold.train["match_id"]) & set(fold.test["match_id"]) == set()
        assert fold.season not in fold.train_seasons


def test_training_data_grows_as_the_origin_rolls_forward(
    four_seasons: pd.DataFrame,
) -> None:
    folds = walk_forward_folds(four_seasons, [2022, 2023])

    assert len(folds[1].train) > len(folds[0].train)


def test_folds_are_ordered_even_when_seasons_are_requested_out_of_order(
    four_seasons: pd.DataFrame,
) -> None:
    folds = walk_forward_folds(four_seasons, [2023, 2022])

    assert [fold.season for fold in folds] == [2022, 2023]


def test_a_season_with_no_rows_is_skipped(four_seasons: pd.DataFrame) -> None:
    folds = walk_forward_folds(four_seasons, [2022, 2030])

    assert [fold.season for fold in folds] == [2022]


def test_a_season_without_enough_history_is_refused(four_seasons: pd.DataFrame) -> None:
    """Fitting on a handful of matches produces noise, not a baseline."""
    with pytest.raises(InsufficientHistoryError, match="2020"):
        walk_forward_folds(four_seasons, [2020])


def test_the_minimum_history_requirement_is_configurable(
    four_seasons: pd.DataFrame,
) -> None:
    folds = walk_forward_folds(four_seasons, [2021], minimum_training_rows=100)

    assert folds[0].train_seasons == [2020]


def test_a_fold_describes_itself_for_the_report(four_seasons: pd.DataFrame) -> None:
    payload = walk_forward_folds(four_seasons, [2022])[0].as_dict()

    assert payload == {
        "season": 2022,
        "train_rows": 600,
        "test_rows": 300,
        "train_seasons": [2020, 2021],
    }


# --- Running a backtest -----------------------------------------------------


def test_a_backtest_scores_every_evaluation_season(four_seasons: pd.DataFrame) -> None:
    result = walk_forward_backtest(
        four_seasons, uniform_trainer, [2022, 2023], model_name="uniform"
    )

    assert result.seasons == [2022, 2023]
    assert all(fold.model_name == "uniform" for fold in result.folds)
    assert all(fold.rows == 300 for fold in result.folds)


def test_the_trainer_never_receives_the_test_rows(four_seasons: pd.DataFrame) -> None:
    """Recorded explicitly, because this is the leak a backtest must not have."""
    seen: list[set[str]] = []

    def recording_trainer(train: pd.DataFrame) -> Callable[[pd.DataFrame], Any]:
        seen.append(set(train["match_id"]))
        return lambda test: np.full((len(test), 3), 1 / 3)

    walk_forward_backtest(four_seasons, recording_trainer, [2022, 2023])

    for index, season in enumerate([2022, 2023]):
        test_ids = set(four_seasons[four_seasons["season_start_year"] == season]["match_id"])
        assert seen[index] & test_ids == set()


def test_the_aggregate_pools_every_fold(four_seasons: pd.DataFrame) -> None:
    result = walk_forward_backtest(four_seasons, uniform_trainer, [2022, 2023])

    assert result.aggregate is not None
    assert result.aggregate.rows == 600
    assert result.aggregate.period == "2022-2023"


def test_a_uniform_model_scores_log_three_every_season(
    four_seasons: pd.DataFrame,
) -> None:
    import math

    result = walk_forward_backtest(four_seasons, uniform_trainer, [2022, 2023])

    for value in result.metric_by_season("log_loss").values():
        assert value == pytest.approx(math.log(3))
    assert result.mean("log_loss") == pytest.approx(math.log(3))
    assert result.standard_deviation("log_loss") == pytest.approx(0.0)


def test_the_spread_across_folds_is_reported() -> None:
    """A large spread means performance depends on which season it was tested on."""
    rows = table({2020: 300, 2021: 300, 2022: 30, 2023: 30})

    def season_sensitive(train: pd.DataFrame) -> Callable[[pd.DataFrame], Any]:
        confident = len(train) > 620
        return lambda test: np.tile(
            [0.9, 0.05, 0.05] if confident else [1 / 3, 1 / 3, 1 / 3], (len(test), 1)
        )

    result = walk_forward_backtest(rows, season_sensitive, [2022, 2023])

    assert result.standard_deviation("log_loss") > 0.1


def test_resource_timings_are_captured_per_fold(four_seasons: pd.DataFrame) -> None:
    result = walk_forward_backtest(four_seasons, uniform_trainer, [2022])
    resources = result.folds[0].resources

    assert resources.training_duration_seconds is not None
    assert resources.training_duration_seconds >= 0.0
    assert resources.inference_latency_ms is not None


def test_the_backtest_serialises_for_a_report(four_seasons: pd.DataFrame) -> None:
    payload = walk_forward_backtest(
        four_seasons, uniform_trainer, [2022, 2023], model_name="uniform"
    ).as_dict()

    assert payload["model_name"] == "uniform"
    assert payload["seasons"] == [2022, 2023]
    assert len(payload["folds"]) == 2
    assert payload["fold_definitions"][0]["train_seasons"] == [2020, 2021]
    assert payload["aggregate"]["rows"] == 600
    assert "mean_log_loss" in payload
    assert "mean_draw_recall" in payload


def test_a_backtest_over_no_seasons_produces_no_aggregate(
    four_seasons: pd.DataFrame,
) -> None:
    result = walk_forward_backtest(four_seasons, uniform_trainer, [])

    assert result.folds == []
    assert result.aggregate is None
    assert np.isnan(result.mean("log_loss"))
