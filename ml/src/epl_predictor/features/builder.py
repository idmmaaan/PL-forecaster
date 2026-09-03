"""Build the version 1.0.0 feature set from the canonical match table.

The ordering rule from the README is the whole point of this module:

    for match in matches_sorted_by_kickoff:
        features = build_features_from_prior_state(match)
        save_training_row(features, target=match.result)
        update_team_state_with_completed_match(match)

Reversing the last two lines would describe each match using its own result.
`build_training_table` implements exactly this order, and the leakage tests
assert that no current-match quantity — goals, shots, corners, cards, or final
position — can reach a feature row.

## Null policy

A feature is null when the value is genuinely unknown, never zero-filled:

- rolling windows are null until a club has five matches of history;
- `venue_points_last_5` is null until it has five matches at that venue;
- `shots_on_target_last_5` is null when any match in the window lacks the
  statistic, which is common before the mid-2000s;
- `league_position_before_match` is null on a club's first match of a season;
- `rest_days` is null for a club's first match in the dataset.

Imputation belongs to each predictor's preprocessing, not here, so that every
model sees the same honest input.

## Schema version

`FEATURE_SCHEMA_VERSION` covers the column list, every window and cap, the
null policy above, and the Elo constants in `elo.py`. Changing any of them
requires a new version.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from epl_predictor import OUTCOME_CLASSES, RESULT_TO_OUTCOME
from epl_predictor.data.ingest import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_REPORTS_DIR,
    load_canonical,
)
from epl_predictor.features.elo import EloRatings, expected_home_score
from epl_predictor.features.state import LeagueState, TeamSnapshot

FEATURE_SCHEMA_VERSION = "1.0.0"

IDENTIFIER_COLUMNS: tuple[str, ...] = ("match_id", "kickoff_date")
TARGET_COLUMN = "outcome"
RESULT_COLUMN = "result"

CATEGORICAL_FEATURES: tuple[str, ...] = ("home_team", "away_team")

NUMERIC_FEATURES: tuple[str, ...] = (
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

FEATURE_COLUMNS: tuple[str, ...] = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)

TRAINING_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *FEATURE_COLUMNS,
    RESULT_COLUMN,
    TARGET_COLUMN,
)

# Quantities that describe the match being predicted. None may ever appear in
# a feature row; `test_leakage.py` enforces this against the real dataset.
FORBIDDEN_FEATURE_SOURCES: tuple[str, ...] = (
    "home_goals",
    "away_goals",
    "result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
)


def _difference(home: float | None, away: float | None) -> float | None:
    """Home minus away, or None when either side is unknown."""
    if home is None or away is None:
        return None
    return home - away


def build_feature_dict(
    home: TeamSnapshot,
    away: TeamSnapshot,
    season_start_year: int,
    matchday: int,
) -> dict[str, Any]:
    """Assemble one feature row from two pre-kickoff snapshots.

    Both snapshots must have been taken before the match was recorded. This
    function has no access to the match's outcome, which is what makes that
    guarantee checkable.
    """
    return {
        "home_team": home.team,
        "away_team": away.team,
        "season_start_year": season_start_year,
        "matchday": matchday,
        "home_points_last_5": home.points_last_5,
        "away_points_last_5": away.points_last_5,
        "home_goals_for_last_5": home.goals_for_last_5,
        "away_goals_for_last_5": away.goals_for_last_5,
        "home_goals_against_last_5": home.goals_against_last_5,
        "away_goals_against_last_5": away.goals_against_last_5,
        "home_goal_difference_last_5": home.goal_difference_last_5,
        "away_goal_difference_last_5": away.goal_difference_last_5,
        "home_home_points_last_5": home.venue_points_last_5,
        "away_away_points_last_5": away.venue_points_last_5,
        "home_shots_on_target_last_5": home.shots_on_target_last_5,
        "away_shots_on_target_last_5": away.shots_on_target_last_5,
        "home_clean_sheet_rate_last_5": home.clean_sheet_rate_last_5,
        "away_clean_sheet_rate_last_5": away.clean_sheet_rate_last_5,
        "home_elo": home.elo,
        "away_elo": away.elo,
        "elo_difference": home.elo - away.elo,
        "elo_expected_home_score": expected_home_score(home.elo, away.elo),
        "home_rest_days": home.rest_days,
        "away_rest_days": away.rest_days,
        "rest_days_difference": _difference(home.rest_days, away.rest_days),
        "home_league_position_before_match": home.league_position,
        "away_league_position_before_match": away.league_position,
        # Position is better when lower, so away minus home keeps a positive
        # value meaning "the home side is placed higher".
        "league_position_difference": _difference(away.league_position, home.league_position),
    }


def derive_matchday(home: TeamSnapshot, away: TeamSnapshot) -> int:
    """Round number for a fixture, derived from matches already played.

    The canonical table has no matchday column, so it is inferred from how far
    into the season the two clubs are. Using the further-advanced club keeps
    the value stable when one side has a game in hand.
    """
    return max(home.season_matches_played, away.season_matches_played) + 1


@dataclass
class FeatureBuildReport:
    """What a feature build produced."""

    feature_schema_version: str
    rows_built: int
    seasons: list[int]
    null_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "rows_built": self.rows_built,
            "seasons": self.seasons,
            "null_counts": self.null_counts,
        }


def build_training_table(
    matches: pd.DataFrame, elo: EloRatings | None = None
) -> tuple[pd.DataFrame, LeagueState]:
    """Turn canonical matches into a leakage-free training table.

    Returns the table and the `LeagueState` left standing after the final
    match, which is exactly the state needed to describe the next fixture.

    Raises:
        ValueError: `matches` is not sorted by kickoff date.
    """
    _require_chronological(matches)

    state = LeagueState(elo=elo)
    rows: list[dict[str, Any]] = []

    for match in matches.itertuples(index=False):
        season_start_year = int(match.season_start_year)
        kickoff_date = pd.Timestamp(match.kickoff_date).date()
        home_team = str(match.home_team)
        away_team = str(match.away_team)

        state.begin_season(season_start_year)

        # 1. Describe the fixture from state that predates it.
        home = state.snapshot(home_team, kickoff_date, season_start_year, is_home=True)
        away = state.snapshot(away_team, kickoff_date, season_start_year, is_home=False)

        # 2. Save the row, pairing those features with the observed result.
        row = build_feature_dict(home, away, season_start_year, derive_matchday(home, away))
        row["match_id"] = str(match.match_id)
        row["kickoff_date"] = kickoff_date
        row[RESULT_COLUMN] = str(match.result)
        row[TARGET_COLUMN] = RESULT_TO_OUTCOME[str(match.result)].value
        rows.append(row)

        # 3. Only now does the completed match become part of the state.
        state.record_match(
            kickoff_date=kickoff_date,
            season_start_year=season_start_year,
            home_team=home_team,
            away_team=away_team,
            home_goals=int(match.home_goals),
            away_goals=int(match.away_goals),
            home_shots_on_target=_optional_int(match.home_shots_on_target),
            away_shots_on_target=_optional_int(match.away_shots_on_target),
        )

    return _typed_table(rows), state


def _optional_int(value: Any) -> int | None:
    """Read a nullable integer from the canonical table."""
    if value is None or pd.isna(value):
        return None
    return int(value)


def _require_chronological(matches: pd.DataFrame) -> None:
    if "kickoff_date" not in matches.columns:
        raise ValueError("Canonical matches must include a kickoff_date column")
    dates = pd.to_datetime(matches["kickoff_date"])
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Matches must be sorted by kickoff_date before building features; "
            "out-of-order input would leak later results into earlier rows."
        )


NULLABLE_INTEGER_FEATURES: tuple[str, ...] = (
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
    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",
    "home_league_position_before_match",
    "away_league_position_before_match",
    "league_position_difference",
)


def _typed_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble feature rows with stable column order and honest dtypes."""
    table = pd.DataFrame(rows, columns=list(TRAINING_COLUMNS))

    table["kickoff_date"] = pd.to_datetime(table["kickoff_date"])
    table["season_start_year"] = table["season_start_year"].astype("int16")
    table["matchday"] = table["matchday"].astype("int16")
    for column in NULLABLE_INTEGER_FEATURES:
        # Nullable integers keep "no history yet" distinct from a real zero.
        table[column] = table[column].astype("Int16")
    for column in ("home_elo", "away_elo", "elo_difference", "elo_expected_home_score"):
        table[column] = table[column].astype("float64")
    for column in ("home_clean_sheet_rate_last_5", "away_clean_sheet_rate_last_5"):
        table[column] = table[column].astype("Float64")
    for column in ("match_id", "home_team", "away_team", RESULT_COLUMN, TARGET_COLUMN):
        table[column] = table[column].astype("string")

    return table


def build_report(table: pd.DataFrame) -> FeatureBuildReport:
    """Summarise a built table, including where nulls remain."""
    null_counts = {
        column: int(table[column].isna().sum())
        for column in FEATURE_COLUMNS
        if table[column].isna().any()
    }
    return FeatureBuildReport(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        rows_built=len(table),
        seasons=sorted(int(year) for year in table["season_start_year"].unique()),
        null_counts=null_counts,
    )


def build_fixture_features(
    state: LeagueState,
    home_team: str,
    away_team: str,
    kickoff_date: Any,
    season_start_year: int,
) -> dict[str, Any]:
    """Describe a single upcoming fixture from existing state.

    This is the inference-time counterpart of `build_training_table`: it reads
    the same state through the same code path but records nothing, because the
    match has not been played.
    """
    kickoff = pd.Timestamp(kickoff_date).date()
    home = state.snapshot(home_team, kickoff, season_start_year, is_home=True)
    away = state.snapshot(away_team, kickoff, season_start_year, is_home=False)
    return build_feature_dict(home, away, season_start_year, derive_matchday(home, away))


@dataclass(frozen=True)
class SeasonSplit:
    """Chronological train / validation / test periods, by season start year.

    Seasons are whole holdout periods rather than random rows, per the
    README's split rules. The test seasons must stay untouched during feature
    selection and tuning.
    """

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        overlap = (
            (set(self.train) & set(self.validation))
            | (set(self.train) & set(self.test))
            | (set(self.validation) & set(self.test))
        )
        if overlap:
            raise ValueError(f"Splits overlap on season(s): {sorted(overlap)}")
        if self.train and self.validation and max(self.train) >= min(self.validation):
            raise ValueError("Validation seasons must come after every training season")
        if self.validation and self.test and max(self.validation) >= min(self.test):
            raise ValueError("Test seasons must come after every validation season")

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


# The README's initial split: train 2010/11-2023/24, validate 2024/25,
# test 2025/26. 2026/27 is left out entirely as the live shadow season.
DEFAULT_SPLIT = SeasonSplit(
    train=tuple(range(2010, 2024)),
    validation=(2024,),
    test=(2025,),
)


def split_table(table: pd.DataFrame, split: SeasonSplit = DEFAULT_SPLIT) -> dict[str, pd.DataFrame]:
    """Partition a feature table into its chronological periods."""
    seasons = table["season_start_year"]
    return {
        "train": table[seasons.isin(split.train)].reset_index(drop=True),
        "validation": table[seasons.isin(split.validation)].reset_index(drop=True),
        "test": table[seasons.isin(split.test)].reset_index(drop=True),
    }


def features_of(table: pd.DataFrame) -> pd.DataFrame:
    """The feature columns only, in schema order."""
    return table[list(FEATURE_COLUMNS)]


FEATURES_FILENAME = "features.parquet"


def write_features(table: pd.DataFrame, processed_dir: Path | None = None) -> Path:
    """Write the feature table as Parquet beside the canonical matches."""
    target_dir = processed_dir or DEFAULT_PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / FEATURES_FILENAME
    table.to_parquet(target, index=False)
    return target


def load_features(processed_dir: Path | None = None) -> pd.DataFrame:
    """Read the feature table written by a previous build.

    Raises:
        FileNotFoundError: The table has not been built yet.
    """
    target = (processed_dir or DEFAULT_PROCESSED_DIR) / FEATURES_FILENAME
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Run `python -m epl_predictor.features.builder` first."
        )
    return pd.read_parquet(target)


def main(argv: list[str] | None = None) -> int:
    """Build the feature table from the canonical matches and report on it."""
    import argparse
    import json
    import logging

    # Imported here because the guard module reads this module's schema.
    from epl_predictor.features.validation import assert_no_forbidden_columns

    parser = argparse.ArgumentParser(description="Build the v1 feature table.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    try:
        matches = load_canonical(args.processed_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    table, state = build_training_table(matches)
    assert_no_forbidden_columns(features_of(table))

    target = write_features(table, args.processed_dir)
    report = build_report(table)

    report_dir = args.reports_dir / "features"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["splits"] = DEFAULT_SPLIT.as_dict()
    payload["split_sizes"] = {
        name: len(part) for name, part in split_table(table, DEFAULT_SPLIT).items()
    }
    payload["elo_top_10"] = dict(
        sorted(state.elo.as_dict().items(), key=lambda item: -item[1])[:10]
    )
    (report_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    logger.info(
        "Built %d rows of feature schema %s for seasons %d-%d -> %s",
        report.rows_built,
        report.feature_schema_version,
        report.seasons[0] if report.seasons else 0,
        report.seasons[-1] if report.seasons else 0,
        target,
    )
    for name, size in payload["split_sizes"].items():
        logger.info("  %-10s %d rows", name, size)
    return 0 if report.rows_built else 1


if __name__ == "__main__":
    raise SystemExit(main())


def targets_of(table: pd.DataFrame) -> pd.Series:
    """The outcome column, as canonical class labels."""
    return table[TARGET_COLUMN].astype(pd.CategoricalDtype(categories=list(OUTCOME_CLASSES)))
