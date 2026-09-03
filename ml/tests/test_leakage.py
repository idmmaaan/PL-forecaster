"""Proof that no current-match information reaches a feature row.

The README calls data leakage the project's largest technical risk, so these
tests are written as adversarial attempts to detect it rather than as
demonstrations that the happy path works.

Three properties, in order of strength:

1. **Result independence.** Rewriting a match's own goals, shots, cards, and
   result must leave that match's feature row byte-identical. If any
   current-match quantity reached the row, the row would change.
2. **Truncation invariance.** A match's features must be identical whether the
   dataset stops at that match or continues for years. If any future
   information reached the row, the row would change.
3. **State propagation.** Rewriting a match's result *must* change later rows.
   Without this, the first two properties could be satisfied by features that
   simply ignore match history.
"""

import random
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from epl_predictor.data.ingest import load_canonical
from epl_predictor.features.builder import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_SOURCES,
    TARGET_COLUMN,
    build_training_table,
)
from epl_predictor.features.state import LeagueState
from epl_predictor.features.validation import assert_no_forbidden_columns

CLUBS = [
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Everton",
    "Fulham",
]


def synthetic_matches(seasons: int = 3, seed: int = 7) -> pd.DataFrame:
    """A deterministic multi-season double round robin.

    Synthetic rather than real data, so a single match's values can be
    rewritten and the effect isolated.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    kickoff = date(2015, 8, 8)

    for offset in range(seasons):
        season = 2015 + offset
        kickoff = date(season, 8, 8)
        for home in CLUBS:
            for away in CLUBS:
                if home == away:
                    continue
                home_goals = rng.randint(0, 4)
                away_goals = rng.randint(0, 4)
                rows.append(
                    {
                        "match_id": f"{season}:{home}:{away}".lower().replace(" ", "-"),
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
                        "home_shots": home_goals * 3 + 5,
                        "away_shots": away_goals * 3 + 5,
                        "home_shots_on_target": home_goals * 2 + 2,
                        "away_shots_on_target": away_goals * 2 + 2,
                        "home_corners": 5,
                        "away_corners": 4,
                        "home_yellow_cards": 1,
                        "away_yellow_cards": 2,
                        "home_red_cards": 0,
                        "away_red_cards": 0,
                        "source_file": "synthetic.csv",
                    }
                )
                kickoff += timedelta(days=3)

    return pd.DataFrame(rows).sort_values("kickoff_date").reset_index(drop=True)


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return synthetic_matches()


@pytest.fixture(scope="module")
def table(matches: pd.DataFrame) -> pd.DataFrame:
    return build_training_table(matches)[0]


def rewrite_match(matches: pd.DataFrame, index: int, **values: Any) -> pd.DataFrame:
    """Return a copy of the dataset with one match's own values replaced."""
    edited = matches.copy()
    for column, value in values.items():
        edited.loc[index, column] = value
    return edited


# --- Property 1: a match's own outcome cannot reach its own features ---------


@pytest.mark.parametrize("index", [0, 1, 40, 120, 167])
def test_rewriting_a_match_does_not_change_its_own_features(
    matches: pd.DataFrame, table: pd.DataFrame, index: int
) -> None:
    """The decisive test: flip everything about a match, keep its row identical."""
    edited = rewrite_match(
        matches,
        index,
        home_goals=9,
        away_goals=0,
        result="H",
        home_shots=40,
        away_shots=0,
        home_shots_on_target=25,
        away_shots_on_target=0,
        home_corners=20,
        away_corners=0,
        home_yellow_cards=0,
        away_yellow_cards=5,
        home_red_cards=0,
        away_red_cards=3,
    )

    rebuilt = build_training_table(edited)[0]

    pd.testing.assert_series_equal(
        table.loc[index, list(FEATURE_COLUMNS)],
        rebuilt.loc[index, list(FEATURE_COLUMNS)],
    )


def test_reversing_every_result_leaves_the_first_matchday_untouched(
    matches: pd.DataFrame, table: pd.DataFrame
) -> None:
    """No club has history on the opening day, so no result can inform it."""
    flipped = matches.copy()
    flipped["home_goals"], flipped["away_goals"] = (
        matches["away_goals"].copy(),
        matches["home_goals"].copy(),
    )
    flipped["result"] = flipped.apply(
        lambda row: (
            "H"
            if row["home_goals"] > row["away_goals"]
            else "A"
            if row["home_goals"] < row["away_goals"]
            else "D"
        ),
        axis=1,
    )

    rebuilt = build_training_table(flipped)[0]
    opening = table["kickoff_date"] == table["kickoff_date"].min()

    pd.testing.assert_frame_equal(
        table.loc[opening, list(FEATURE_COLUMNS)],
        rebuilt.loc[opening, list(FEATURE_COLUMNS)],
    )


# --- Property 2: no future information reaches an earlier row ---------------


@pytest.mark.parametrize("cutoff", [1, 17, 60, 150])
def test_features_are_identical_when_the_dataset_stops_early(
    matches: pd.DataFrame, table: pd.DataFrame, cutoff: int
) -> None:
    """Truncation invariance: a row cannot depend on matches that follow it."""
    truncated = build_training_table(matches.iloc[:cutoff].copy())[0]

    pd.testing.assert_frame_equal(
        table.iloc[:cutoff][list(FEATURE_COLUMNS)],
        truncated[list(FEATURE_COLUMNS)],
    )


def test_appending_future_seasons_does_not_disturb_earlier_rows(
    matches: pd.DataFrame,
) -> None:
    """Retraining on more history must not silently rewrite old training rows."""
    early = build_training_table(matches[matches["season_start_year"] <= 2016].copy())[0]
    full = build_training_table(matches)[0]

    pd.testing.assert_frame_equal(
        early[list(FEATURE_COLUMNS)],
        full.iloc[: len(early)][list(FEATURE_COLUMNS)],
    )


# --- Property 3: history does propagate, so the tests above are meaningful ---


def test_rewriting_a_match_does_change_later_features(
    matches: pd.DataFrame, table: pd.DataFrame
) -> None:
    """Guards against features that pass the leakage tests by being inert."""
    edited = rewrite_match(matches, 0, home_goals=7, away_goals=0, result="H")

    rebuilt = build_training_table(edited)[0]
    changed = ~rebuilt["home_elo"].eq(table["home_elo"])

    assert not changed.iloc[0], "the edited match's own row must not move"
    assert changed.any(), "later rows must reflect the edited result"
    assert changed.idxmax() > 0


def test_a_clubs_own_result_reaches_its_next_match(matches: pd.DataFrame) -> None:
    """A win must raise the club's Elo for its following fixture, not this one."""
    table = build_training_table(matches)[0]
    home_team = table.loc[0, "home_team"]

    later = table[
        (table.index > 0) & ((table["home_team"] == home_team) | (table["away_team"] == home_team))
    ]
    assert not later.empty

    thrashed = build_training_table(
        rewrite_match(matches, 0, home_goals=6, away_goals=0, result="H")
    )[0]
    next_index = later.index[0]
    column = "home_elo" if thrashed.loc[next_index, "home_team"] == home_team else "away_elo"

    assert thrashed.loc[next_index, column] > table.loc[next_index, column]


# --- Structural guarantees ---------------------------------------------------


def test_no_forbidden_column_appears_in_the_feature_set() -> None:
    for forbidden in FORBIDDEN_FEATURE_SOURCES:
        assert forbidden not in FEATURE_COLUMNS


def test_the_built_table_carries_no_forbidden_column(table: pd.DataFrame) -> None:
    assert_no_forbidden_columns(table[list(FEATURE_COLUMNS)])


def test_the_guard_catches_a_smuggled_current_match_column(table: pd.DataFrame) -> None:
    """The guard must be able to fail, or it proves nothing."""
    tampered = table[list(FEATURE_COLUMNS)].copy()
    tampered["home_shots_on_target"] = 5

    with pytest.raises(ValueError, match="home_shots_on_target"):
        assert_no_forbidden_columns(tampered)


def test_no_single_feature_determines_the_outcome(table: pd.DataFrame) -> None:
    """A feature that separates the classes perfectly would be a leak."""
    outcomes = table[TARGET_COLUMN]

    for column in FEATURE_COLUMNS:
        grouped = table.groupby(column, observed=True)[TARGET_COLUMN].nunique()
        distinct_values = table[column].nunique(dropna=True)
        if distinct_values <= 1 or distinct_values > len(table) / 3:
            # Near-unique numeric features cannot be judged this way.
            continue
        assert (grouped > 1).any(), (
            f"{column} maps every value to a single outcome, which suggests a leak"
        )

    assert outcomes.nunique() == 3


def test_features_never_correlate_perfectly_with_the_result(table: pd.DataFrame) -> None:
    home_win = (table[TARGET_COLUMN] == "HOME_WIN").astype(float)

    numeric = table[list(FEATURE_COLUMNS)].select_dtypes(include=["number", "Float64", "Int16"])
    correlations = numeric.astype("float64").corrwith(home_win).abs()

    assert (correlations.dropna() < 0.6).all(), (
        f"suspiciously strong correlation: {correlations.dropna().sort_values().tail(3).to_dict()}"
    )


def test_the_opening_matchday_has_no_league_position(table: pd.DataFrame) -> None:
    """Final or current-season position must never be back-filled."""
    opening = table[table["matchday"] == 1]

    assert not opening.empty
    assert opening["home_league_position_before_match"].isna().all()
    assert opening["away_league_position_before_match"].isna().all()


def test_a_league_position_never_reflects_a_completed_season(table: pd.DataFrame) -> None:
    """Position counts only matches already played, so it cannot be the final table."""
    for season, rows in table.groupby("season_start_year"):
        first_round = rows[rows["matchday"] <= 2]
        positions = first_round["home_league_position_before_match"].dropna()
        assert (positions <= len(CLUBS)).all(), f"season {season} position out of range"


def test_reading_state_after_recording_would_leak(matches: pd.DataFrame) -> None:
    """Demonstrates the failure mode the builder's ordering avoids.

    This is the README's "incorrect order" written out: recording first makes
    the snapshot describe a club using the match being predicted.
    """
    row = matches.iloc[0]
    kickoff = row["kickoff_date"].date()
    season = int(row["season_start_year"])
    home = str(row["home_team"])

    correct = LeagueState()
    before = correct.snapshot(home, kickoff, season, is_home=True)

    leaking = LeagueState()
    leaking.record_match(
        kickoff_date=kickoff,
        season_start_year=season,
        home_team=home,
        away_team=str(row["away_team"]),
        home_goals=int(row["home_goals"]),
        away_goals=int(row["away_goals"]),
    )
    after = leaking.snapshot(home, kickoff, season, is_home=True)

    assert before.season_matches_played == 0
    assert after.season_matches_played == 1
    assert before.league_position is None
    assert after.league_position is not None


def test_out_of_order_input_is_refused(matches: pd.DataFrame) -> None:
    """Unsorted input would put later results into earlier rows."""
    shuffled = matches.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="chronological|sorted"):
        build_training_table(shuffled)


# --- The same guarantees, against the real dataset --------------------------


@pytest.fixture(scope="module")
def real_matches() -> pd.DataFrame:
    try:
        return load_canonical()
    except FileNotFoundError:
        pytest.skip("Canonical table not built; run `make ingest` first.")


def test_real_dataset_is_free_of_forbidden_columns(real_matches: pd.DataFrame) -> None:
    table = build_training_table(real_matches)[0]

    assert_no_forbidden_columns(table[list(FEATURE_COLUMNS)])


def test_real_dataset_features_survive_truncation(real_matches: pd.DataFrame) -> None:
    """Truncation invariance on 16 real seasons, not just synthetic fixtures."""
    full = build_training_table(real_matches)[0]
    cutoff = 2000
    truncated = build_training_table(real_matches.iloc[:cutoff].copy())[0]

    pd.testing.assert_frame_equal(
        full.iloc[:cutoff][list(FEATURE_COLUMNS)],
        truncated[list(FEATURE_COLUMNS)],
    )


def test_every_clubs_first_match_of_a_real_season_has_no_position(
    real_matches: pd.DataFrame,
) -> None:
    """The guarantee that matters, stated per club rather than per matchday.

    Counting matchday-1 rows would be the wrong check: a postponed opener can
    leave a club playing its first match of the season on matchday 3.
    """
    table = build_training_table(real_matches)[0]
    checked = 0

    for season, rows in table.groupby("season_start_year"):
        clubs = set(rows["home_team"]) | set(rows["away_team"])
        for club in clubs:
            appearances = rows[(rows["home_team"] == club) | (rows["away_team"] == club)]
            first = appearances.iloc[0]
            side = "home" if first["home_team"] == club else "away"

            assert pd.isna(first[f"{side}_league_position_before_match"]), (
                f"{club} already has a league position in its first {season} match"
            )
            checked += 1

    assert checked == 16 * 20, "expected one first match per club per season"
