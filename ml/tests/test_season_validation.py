"""Per-season validation of Football-Data.co.uk CSVs.

The governing rule from the README is "avoid silently filling critical
fields": a row whose score, result, date, or club cannot be trusted is dropped
and reported, while an absent optional statistic stays null.
"""

from typing import Any

import pandas as pd
import pytest

from epl_predictor.data.validation import (
    CANONICAL_COLUMNS,
    STATISTIC_COLUMNS,
    IssueCode,
    Severity,
    build_match_id,
    derive_result,
    parse_source_date,
    season_label,
    validate_season_frame,
)


def raw_row(**overrides: Any) -> dict[str, Any]:
    """One well-formed source row, in the source's own column names."""
    row: dict[str, Any] = {
        "Div": "E0",
        "Date": "14/08/2010",
        "HomeTeam": "Man United",
        "AwayTeam": "Newcastle",
        "FTHG": "3",
        "FTAG": "0",
        "FTR": "H",
        "HS": "18",
        "AS": "9",
        "HST": "11",
        "AST": "4",
        "HC": "7",
        "AC": "5",
        "HY": "1",
        "AY": "2",
        "HR": "0",
        "AR": "0",
    }
    row.update(overrides)
    return row


def raw_frame(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [raw_row()], dtype=object)


def codes(report: Any) -> set[IssueCode]:
    return {issue.code for issue in report.issues}


def test_a_clean_season_is_accepted_in_full() -> None:
    frame, report = validate_season_frame(raw_frame(), 2010, source_file="E0_1011.csv")

    assert (report.rows_read, report.rows_accepted, report.rows_rejected) == (1, 1, 0)
    assert report.is_usable
    assert list(frame.columns) == list(CANONICAL_COLUMNS)


def test_accepted_rows_carry_the_canonical_schema() -> None:
    frame, _ = validate_season_frame(raw_frame(), 2010, source_file="E0_1011.csv")
    row = frame.iloc[0]

    assert row["match_id"] == "2010:manchester-united:newcastle-united"
    assert row["season_start_year"] == 2010
    assert row["kickoff_date"] == pd.Timestamp("2010-08-14")
    assert row["home_team"] == "Manchester United"
    assert row["away_team"] == "Newcastle United"
    assert (row["home_goals"], row["away_goals"], row["result"]) == (3, 0, "H")
    assert (row["home_shots"], row["away_shots"]) == (18, 9)
    assert (row["home_shots_on_target"], row["away_shots_on_target"]) == (11, 4)
    assert row["source_file"] == "E0_1011.csv"


def test_a_missing_required_column_makes_the_whole_file_unusable() -> None:
    """One absent required column means no row in the file can be trusted."""
    frame = raw_frame().drop(columns=["FTR"])

    result, report = validate_season_frame(frame, 2010)

    assert not report.is_usable
    assert result.empty
    assert IssueCode.MISSING_REQUIRED_COLUMN in codes(report)
    assert list(result.columns) == list(CANONICAL_COLUMNS)


def test_missing_statistic_columns_are_a_warning_not_a_failure() -> None:
    """Older seasons lack shot data; their goals and results are still usable."""
    frame = raw_frame().drop(columns=["HST", "AST"])

    result, report = validate_season_frame(frame, 2003)

    assert report.is_usable
    assert report.rows_accepted == 1
    assert report.missing_statistic_columns == ["HST", "AST"]
    assert result.iloc[0]["home_shots_on_target"] is pd.NA
    assert IssueCode.MISSING_STATISTIC_COLUMN in codes(report)


@pytest.mark.parametrize("missing_field", ["FTHG", "FTAG", "FTR"])
def test_a_row_without_a_usable_target_is_rejected(missing_field: str) -> None:
    """An unplayed match has no target to learn from, so it cannot be filled in."""
    frame = raw_frame(raw_row(**{missing_field: None}))

    result, report = validate_season_frame(frame, 2025)

    assert (report.rows_read, report.rows_accepted) == (1, 0)
    assert result.empty
    assert IssueCode.MISSING_CRITICAL_VALUE in codes(report)


def test_rows_for_an_in_progress_season_are_rejected_not_zero_filled() -> None:
    played = raw_row()
    upcoming = raw_row(HomeTeam="Arsenal", AwayTeam="Chelsea", FTHG=None, FTAG=None, FTR=None)

    frame, report = validate_season_frame(raw_frame(played, upcoming), 2025)

    assert (report.rows_read, report.rows_accepted, report.rows_rejected) == (2, 1, 1)
    assert frame["home_goals"].tolist() == [3]


def test_trailing_blank_rows_are_not_counted_as_rows_read() -> None:
    """In-progress season files end with empty lines; they are not real rows."""
    blank = dict.fromkeys(raw_row())

    _, report = validate_season_frame(raw_frame(raw_row(), blank, blank), 2025)

    assert (report.rows_read, report.rows_accepted, report.rows_rejected) == (1, 1, 0)


def test_an_unparseable_date_is_rejected() -> None:
    frame = raw_frame(raw_row(Date="not-a-date"))

    _, report = validate_season_frame(frame, 2010)

    assert report.rows_accepted == 0
    assert IssueCode.UNPARSEABLE_DATE in codes(report)


def test_an_unmapped_club_is_rejected_and_named_in_the_report() -> None:
    """The operator needs to know which spelling to add to the alias table."""
    frame = raw_frame(raw_row(HomeTeam="Manchester Utd"))

    _, report = validate_season_frame(frame, 2010)

    assert report.rows_accepted == 0
    unknown = next(i for i in report.issues if i.code is IssueCode.UNKNOWN_TEAM)
    assert "Manchester Utd" in unknown.message
    assert unknown.severity is Severity.ERROR


def test_a_club_listed_against_itself_is_rejected() -> None:
    frame = raw_frame(raw_row(AwayTeam="Man United"))

    _, report = validate_season_frame(frame, 2010)

    assert report.rows_accepted == 0
    assert IssueCode.TEAM_PLAYS_ITSELF in codes(report)


def test_a_result_outside_h_d_a_is_rejected() -> None:
    frame = raw_frame(raw_row(FTR="X"))

    _, report = validate_season_frame(frame, 2010)

    assert report.rows_accepted == 0
    assert IssueCode.INVALID_RESULT in codes(report)


def test_a_result_contradicting_its_score_is_rejected() -> None:
    """The source disagreeing with itself would inject a wrong training target."""
    frame = raw_frame(raw_row(FTHG="0", FTAG="3", FTR="H"))

    _, report = validate_season_frame(frame, 2010)

    assert report.rows_accepted == 0
    assert IssueCode.RESULT_SCORE_MISMATCH in codes(report)


def test_a_repeated_match_is_rejected_keeping_the_first() -> None:
    first = raw_row(FTHG="3", FTAG="0", FTR="H")
    repeat = raw_row(FTHG="1", FTAG="1", FTR="D")

    frame, report = validate_season_frame(raw_frame(first, repeat), 2010)

    assert (report.rows_read, report.rows_accepted) == (2, 1)
    assert frame.iloc[0]["home_goals"] == 3
    assert IssueCode.DUPLICATE_MATCH in codes(report)


def test_the_reverse_fixture_is_not_a_duplicate() -> None:
    """Home and away meetings are different matches in the same season."""
    home = raw_row(HomeTeam="Man United", AwayTeam="Newcastle")
    away = raw_row(HomeTeam="Newcastle", AwayTeam="Man United", FTR="D", FTHG="1", FTAG="1")

    frame, report = validate_season_frame(raw_frame(home, away), 2010)

    assert report.rows_accepted == 2
    assert frame["match_id"].nunique() == 2


def test_a_null_statistic_in_a_present_column_is_kept_and_warned_about() -> None:
    """A missing shot count must stay null rather than becoming a real zero."""
    frame, report = validate_season_frame(raw_frame(raw_row(HST=None)), 2010)

    assert report.rows_accepted == 1
    assert frame.iloc[0]["home_shots_on_target"] is pd.NA
    null_issue = next(i for i in report.issues if i.code is IssueCode.NULL_STATISTIC)
    assert null_issue.severity is Severity.WARNING
    assert "home_shots_on_target" in null_issue.message


def test_statistics_stay_nullable_integers() -> None:
    """Float dtype would let a null silently become NaN and then 0.0 downstream."""
    frame, _ = validate_season_frame(raw_frame(raw_row(HST=None)), 2010)

    for column in STATISTIC_COLUMNS.values():
        assert frame[column].dtype == "Int16"


def test_scores_expressed_as_floats_are_read_as_integers() -> None:
    """pandas writes an integer column containing blanks as `3.0`."""
    frame, report = validate_season_frame(raw_frame(raw_row(FTHG="3.0", FTAG="0.0")), 2010)

    assert report.rows_accepted == 1
    assert (frame.iloc[0]["home_goals"], frame.iloc[0]["away_goals"]) == (3, 0)


def test_the_report_serialises_for_the_import_log() -> None:
    _, report = validate_season_frame(raw_frame(raw_row(FTR="X")), 2010)
    payload = report.as_dict()

    assert payload["season_start_year"] == 2010
    assert payload["rows_rejected"] == 1
    assert payload["is_usable"] is True
    assert payload["issues"][0]["code"] == IssueCode.INVALID_RESULT.value
    assert payload["issues"][0]["severity"] == "ERROR"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("14/08/2010", "2010-08-14"),
        ("14/08/10", "2010-08-14"),
        ("01/01/21", "2021-01-01"),
        ("2010-08-14", "2010-08-14"),
    ],
)
def test_both_source_date_formats_parse_day_first(raw: str, expected: str) -> None:
    """`14/08/10` is 14 August 2010, not 10 August 2014."""
    assert parse_source_date(raw) == pd.Timestamp(expected)


@pytest.mark.parametrize("raw", [None, "", "   ", "nan", "rubbish"])
def test_unusable_dates_parse_to_none(raw: Any) -> None:
    assert parse_source_date(raw) is None


@pytest.mark.parametrize(
    ("home", "away", "expected"), [(3, 0, "H"), (0, 3, "A"), (1, 1, "D"), (0, 0, "D")]
)
def test_result_is_derived_from_the_score(home: int, away: int, expected: str) -> None:
    assert derive_result(home, away) == expected


@pytest.mark.parametrize(
    ("start_year", "expected"), [(2010, "2010/11"), (2024, "2024/25"), (1999, "1999/00")]
)
def test_season_label_matches_football_convention(start_year: int, expected: str) -> None:
    assert season_label(start_year) == expected


def test_match_id_is_stable_and_direction_sensitive() -> None:
    home = build_match_id(2010, "Manchester United", "Newcastle United")

    assert home == build_match_id(2010, "Manchester United", "Newcastle United")
    assert home != build_match_id(2010, "Newcastle United", "Manchester United")
    assert home != build_match_id(2011, "Manchester United", "Newcastle United")


def test_match_id_survives_the_ampersand_in_brighton() -> None:
    assert build_match_id(2024, "Brighton & Hove Albion", "Arsenal") == (
        "2024:brighton-and-hove-albion:arsenal"
    )


def test_an_empty_season_file_produces_an_empty_canonical_frame() -> None:
    frame, report = validate_season_frame(pd.DataFrame(columns=list(raw_row())), 2026)

    assert report.is_usable
    assert (report.rows_read, report.rows_accepted) == (0, 0)
    assert list(frame.columns) == list(CANONICAL_COLUMNS)
