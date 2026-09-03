"""Feature table assembly, schema, dtypes, and chronological splits.

Leakage itself is covered by `test_leakage.py`. This module pins the published
v1 contract: which columns exist, in what order, with what dtypes, and how the
seasons are partitioned.
"""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import (
    CATEGORICAL_FEATURES,
    DEFAULT_SPLIT,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    NULLABLE_INTEGER_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TRAINING_COLUMNS,
    SeasonSplit,
    build_fixture_features,
    build_report,
    build_training_table,
    derive_matchday,
    features_of,
    split_table,
    targets_of,
)
from epl_predictor.features.state import LeagueState
from epl_predictor.features.validation import assert_schema

CANONICAL_COLUMNS_USED = (
    "match_id",
    "season_start_year",
    "kickoff_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_shots_on_target",
    "away_shots_on_target",
)


def match(
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    day: int,
    season: int = 2024,
) -> dict[str, Any]:
    return {
        "match_id": f"{season}:{home}:{away}",
        "season_start_year": season,
        "kickoff_date": pd.Timestamp(date(season, 8, 17) + timedelta(days=day)),
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": ("H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"),
        "home_shots_on_target": 6,
        "away_shots_on_target": 3,
    }


def canonical(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(CANONICAL_COLUMNS_USED))


@pytest.fixture
def small_table() -> pd.DataFrame:
    return build_training_table(
        canonical(
            match("Arsenal", "Chelsea", 2, 0, day=0),
            match("Liverpool", "Everton", 1, 1, day=1),
            match("Chelsea", "Liverpool", 0, 3, day=8),
        )
    )[0]


# --- Published schema -------------------------------------------------------


def test_the_schema_version_is_published() -> None:
    assert FEATURE_SCHEMA_VERSION == "1.0.0"


def test_the_feature_set_matches_the_readme_v1_list() -> None:
    """Pinned so that adding or renaming a feature is a deliberate schema change."""
    assert FEATURE_COLUMNS == (
        "home_team",
        "away_team",
        "season_start_year",
        "matchday",
        "home_points_last_5",
        "away_points_last_5",
        "home_goals_for_last_5",
        "away_goals_for_last_5",
        "home_goals_against_last_5",
        "away_goals_against_last_5",
        "home_goal_difference_last_5",
        "away_goal_difference_last_5",
        "home_home_points_last_5",
        "away_away_points_last_5",
        "home_shots_on_target_last_5",
        "away_shots_on_target_last_5",
        "home_clean_sheet_rate_last_5",
        "away_clean_sheet_rate_last_5",
        "home_elo",
        "away_elo",
        "elo_difference",
        "elo_expected_home_score",
        "home_rest_days",
        "away_rest_days",
        "rest_days_difference",
        "home_league_position_before_match",
        "away_league_position_before_match",
        "league_position_difference",
    )


def test_categorical_and_numeric_features_partition_the_schema() -> None:
    assert set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES) == set(FEATURE_COLUMNS)
    assert set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES) == set()


def test_the_training_table_carries_identifiers_features_and_target(
    small_table: pd.DataFrame,
) -> None:
    assert list(small_table.columns) == list(TRAINING_COLUMNS)
    assert "match_id" in small_table.columns
    assert TARGET_COLUMN in small_table.columns


def test_the_feature_frame_passes_its_own_schema_check(small_table: pd.DataFrame) -> None:
    assert_schema(features_of(small_table))


def test_the_schema_check_rejects_a_reordered_frame(small_table: pd.DataFrame) -> None:
    reordered = features_of(small_table)[list(reversed(FEATURE_COLUMNS))]

    with pytest.raises(ValueError, match="schema order"):
        assert_schema(reordered)


def test_the_schema_check_rejects_a_missing_column(small_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing home_elo"):
        assert_schema(features_of(small_table).drop(columns=["home_elo"]))


# --- Values -----------------------------------------------------------------


def test_the_target_uses_the_canonical_outcome_classes(small_table: pd.DataFrame) -> None:
    assert small_table[TARGET_COLUMN].tolist() == ["HOME_WIN", "DRAW", "AWAY_WIN"]
    assert set(small_table[TARGET_COLUMN]) <= set(OUTCOME_CLASSES)


def test_targets_are_categorical_in_canonical_class_order(
    small_table: pd.DataFrame,
) -> None:
    """Class order must be fixed, or probability columns would not line up."""
    targets = targets_of(small_table)

    assert list(targets.cat.categories) == list(OUTCOME_CLASSES)


def test_elo_difference_and_expectation_agree_with_the_ratings(
    small_table: pd.DataFrame,
) -> None:
    row = small_table.iloc[2]

    assert row["elo_difference"] == pytest.approx(row["home_elo"] - row["away_elo"])
    assert 0.0 < row["elo_expected_home_score"] < 1.0


def test_a_positive_position_difference_means_the_home_side_is_higher() -> None:
    """Lower position numbers are better, so the difference is inverted."""
    table = build_training_table(
        canonical(
            match("Leader", "Bottom", 5, 0, day=0),
            match("Leader", "Bottom", 1, 0, day=7),
        )
    )[0]
    row = table.iloc[1]

    assert row["home_league_position_before_match"] == 1
    assert row["away_league_position_before_match"] == 2
    assert row["league_position_difference"] == 1


def test_rest_days_difference_is_home_minus_away() -> None:
    table = build_training_table(
        canonical(
            match("Arsenal", "Chelsea", 1, 0, day=0),
            match("Liverpool", "Everton", 1, 0, day=4),
            match("Arsenal", "Liverpool", 1, 0, day=10),
        )
    )[0]
    row = table.iloc[2]

    assert row["home_rest_days"] == 10
    assert row["away_rest_days"] == 6
    assert row["rest_days_difference"] == 4


def test_a_difference_is_null_when_either_side_is_unknown(
    small_table: pd.DataFrame,
) -> None:
    first = small_table.iloc[0]

    assert pd.isna(first["home_rest_days"])
    assert pd.isna(first["rest_days_difference"])
    assert pd.isna(first["league_position_difference"])


def test_matchday_advances_with_the_further_along_club() -> None:
    state = LeagueState()
    home = state.snapshot("Arsenal", date(2024, 8, 17), 2024, is_home=True)

    assert derive_matchday(home, home) == 1

    for day in range(3):
        state.record_match(
            kickoff_date=date(2024, 8, 17) + timedelta(days=day * 7),
            season_start_year=2024,
            home_team="Arsenal",
            away_team=f"Rival {day}",
            home_goals=1,
            away_goals=0,
        )

    played = state.snapshot("Arsenal", date(2024, 9, 20), 2024, is_home=True)
    fresh = state.snapshot("Newcastle United", date(2024, 9, 20), 2024, is_home=False)

    assert derive_matchday(played, fresh) == 4


# --- Dtypes -----------------------------------------------------------------


def test_rolling_features_use_nullable_integers(small_table: pd.DataFrame) -> None:
    """Float dtype would let "no history" decay into NaN and then into 0.0."""
    for column in NULLABLE_INTEGER_FEATURES:
        assert small_table[column].dtype == "Int16", column


def test_elo_columns_are_floats(small_table: pd.DataFrame) -> None:
    for column in ("home_elo", "away_elo", "elo_difference", "elo_expected_home_score"):
        assert small_table[column].dtype == "float64"


def test_clean_sheet_rates_are_nullable_floats(small_table: pd.DataFrame) -> None:
    assert small_table["home_clean_sheet_rate_last_5"].dtype == "Float64"


def test_team_columns_are_strings(small_table: pd.DataFrame) -> None:
    assert small_table["home_team"].dtype == "string"
    assert small_table["away_team"].dtype == "string"


# --- Reports ----------------------------------------------------------------


def test_the_build_report_records_the_schema_version_and_nulls(
    small_table: pd.DataFrame,
) -> None:
    report = build_report(small_table)

    assert report.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert report.rows_built == 3
    assert report.seasons == [2024]
    assert report.null_counts["home_points_last_5"] == 3
    assert report.as_dict()["feature_schema_version"] == FEATURE_SCHEMA_VERSION


def test_the_report_omits_columns_without_nulls(small_table: pd.DataFrame) -> None:
    report = build_report(small_table)

    assert "home_elo" not in report.null_counts
    assert "home_team" not in report.null_counts


# --- Inference path ---------------------------------------------------------


def test_an_upcoming_fixture_is_described_from_the_same_state() -> None:
    """Inference must reuse the training code path, or the two would drift."""
    table, state = build_training_table(
        canonical(
            match("Arsenal", "Chelsea", 2, 0, day=0),
            match("Chelsea", "Arsenal", 0, 1, day=7),
        )
    )

    features = build_fixture_features(
        state, "Arsenal", "Chelsea", date(2024, 9, 1), season_start_year=2024
    )

    assert set(features) == set(FEATURE_COLUMNS)
    assert features["home_team"] == "Arsenal"
    assert features["home_elo"] == pytest.approx(state.elo.rating("Arsenal"))
    assert features["matchday"] == 3
    assert len(table) == 2


def test_describing_a_fixture_does_not_mutate_the_state() -> None:
    """Predicting the same fixture twice must give the same answer."""
    _, state = build_training_table(canonical(match("Arsenal", "Chelsea", 2, 0, day=0)))

    first = build_fixture_features(state, "Arsenal", "Chelsea", date(2024, 9, 1), 2024)
    second = build_fixture_features(state, "Arsenal", "Chelsea", date(2024, 9, 1), 2024)

    assert first == second


# --- Chronological splits ---------------------------------------------------


def test_the_default_split_matches_the_readme() -> None:
    assert DEFAULT_SPLIT.train == tuple(range(2010, 2024))
    assert DEFAULT_SPLIT.validation == (2024,)
    assert DEFAULT_SPLIT.test == (2025,)
    assert DEFAULT_SPLIT.as_dict()["validation"] == [2024]


def test_the_split_leaves_the_live_shadow_season_out() -> None:
    """2026/27 belongs to neither training nor evaluation."""
    covered = set(DEFAULT_SPLIT.train) | set(DEFAULT_SPLIT.validation) | set(DEFAULT_SPLIT.test)

    assert 2026 not in covered


def test_overlapping_splits_are_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SeasonSplit(train=(2010, 2011), validation=(2011,), test=(2012,))


def test_a_validation_season_before_training_is_refused() -> None:
    """A later season must never be used to fit an earlier one."""
    with pytest.raises(ValueError, match="after every training season"):
        SeasonSplit(train=(2012,), validation=(2011,), test=(2013,))


def test_a_test_season_before_validation_is_refused() -> None:
    with pytest.raises(ValueError, match="after every validation season"):
        SeasonSplit(train=(2010,), validation=(2012,), test=(2011,))


def test_splitting_partitions_the_table_by_season() -> None:
    table = build_training_table(
        canonical(
            match("Arsenal", "Chelsea", 1, 0, day=0, season=2010),
            match("Arsenal", "Chelsea", 1, 0, day=0, season=2024),
            match("Arsenal", "Chelsea", 1, 0, day=0, season=2025),
        )
    )[0]

    parts = split_table(table)

    assert parts["train"]["season_start_year"].tolist() == [2010]
    assert parts["validation"]["season_start_year"].tolist() == [2024]
    assert parts["test"]["season_start_year"].tolist() == [2025]


def test_splitting_never_shares_a_match_between_parts() -> None:
    table = build_training_table(
        canonical(
            match("Arsenal", "Chelsea", 1, 0, day=0, season=2010),
            match("Arsenal", "Chelsea", 1, 0, day=0, season=2024),
            match("Arsenal", "Chelsea", 1, 0, day=0, season=2025),
        )
    )[0]

    parts = split_table(table)
    ids = [set(part["match_id"]) for part in parts.values()]

    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()


def test_building_an_empty_dataset_yields_an_empty_typed_table() -> None:
    table, state = build_training_table(canonical())

    assert table.empty
    assert list(table.columns) == list(TRAINING_COLUMNS)
    assert state.elo.as_dict() == {}
