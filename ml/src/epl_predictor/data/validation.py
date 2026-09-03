"""Per-season validation of Football-Data.co.uk Premier League CSVs.

The source is a set of hand-maintained files whose schema drifts between
seasons: date formats alternate between two-digit and four-digit years, match
statistics columns appear and disappear, and files for an in-progress season
carry trailing blank rows. Validation converts one season's raw frame into the
canonical schema, rejecting rows it cannot trust instead of filling them in.

The README's rule is explicit: never silently fill a critical field. So a row
missing its score or result is dropped and reported, while a row missing an
optional statistic is kept with a null and reported as a warning.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from epl_predictor.data.teams import try_normalise_team_name

FEATURE_SOURCE_SCHEMA_VERSION = "1.0.0"

# Without these a match cannot be used for training at all.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
)

# Present from the mid-2000s onwards; mapped to canonical names when available.
STATISTIC_COLUMNS: dict[str, str] = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
}

CANONICAL_COLUMNS: tuple[str, ...] = (
    "match_id",
    "season_start_year",
    "kickoff_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    *STATISTIC_COLUMNS.values(),
    "source_file",
)

VALID_RESULTS: frozenset[str] = frozenset({"H", "D", "A"})


class Severity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


class IssueCode(StrEnum):
    MISSING_REQUIRED_COLUMN = "missing_required_column"
    MISSING_STATISTIC_COLUMN = "missing_statistic_column"
    EMPTY_ROW = "empty_row"
    MISSING_CRITICAL_VALUE = "missing_critical_value"
    UNPARSEABLE_DATE = "unparseable_date"
    UNKNOWN_TEAM = "unknown_team"
    TEAM_PLAYS_ITSELF = "team_plays_itself"
    INVALID_RESULT = "invalid_result"
    RESULT_SCORE_MISMATCH = "result_score_mismatch"
    DUPLICATE_MATCH = "duplicate_match"
    NULL_STATISTIC = "null_statistic"


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a season file."""

    code: IssueCode
    severity: Severity
    message: str
    count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "count": self.count,
        }


@dataclass
class SeasonValidationReport:
    """What validation made of one season file."""

    season_start_year: int
    source_file: str
    rows_read: int = 0
    rows_accepted: int = 0
    missing_statistic_columns: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def rows_rejected(self) -> int:
        return self.rows_read - self.rows_accepted

    @property
    def is_usable(self) -> bool:
        """False when a required column was absent, so no row could be trusted."""
        return not any(issue.code is IssueCode.MISSING_REQUIRED_COLUMN for issue in self.issues)

    def add(self, code: IssueCode, severity: Severity, message: str, count: int = 1) -> None:
        self.issues.append(ValidationIssue(code, severity, message, count))

    def as_dict(self) -> dict[str, Any]:
        return {
            "season_start_year": self.season_start_year,
            "source_file": self.source_file,
            "rows_read": self.rows_read,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "is_usable": self.is_usable,
            "missing_statistic_columns": self.missing_statistic_columns,
            "issues": [issue.as_dict() for issue in self.issues],
        }


def season_label(start_year: int) -> str:
    """Render a season as it is written in football, e.g. 2010 -> "2010/11"."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def parse_source_date(value: Any) -> pd.Timestamp | None:
    """Parse a Football-Data.co.uk date, which may use a 2- or 4-digit year.

    Both `14/08/10` and `14/08/2010` occur, always day-first. `dayfirst` alone
    is not enough: pandas would read `14/08/10` as year 2014 in some versions,
    so the two widths are tried explicitly before any fallback.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat"}:
        return None

    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed

    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def build_match_id(season_start_year: int, home_team: str, away_team: str) -> str:
    """Deterministic natural key for a league match.

    A league season contains exactly one meeting per ordered pair of clubs, so
    this triple identifies a match without depending on the row's position or
    on a date that the source sometimes revises.
    """
    home = home_team.lower().replace(" ", "-").replace("&", "and")
    away = away_team.lower().replace(" ", "-").replace("&", "and")
    return f"{season_start_year}:{home}:{away}"


def _coerce_int(value: Any) -> int | None:
    """Read an integer that the CSV may express as `2`, `2.0`, `"2"`, or blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", ""}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def derive_result(home_goals: int, away_goals: int) -> str:
    """Map a full-time score to its H/D/A label."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def validate_season_frame(
    frame: pd.DataFrame, season_start_year: int, source_file: str = ""
) -> tuple[pd.DataFrame, SeasonValidationReport]:
    """Convert one season's raw frame into the canonical schema.

    Returns the accepted rows and a report describing everything rejected. The
    frame is always returned with `CANONICAL_COLUMNS`, empty if unusable, so
    callers can concatenate seasons without branching.
    """
    report = SeasonValidationReport(season_start_year=season_start_year, source_file=source_file)

    missing_required = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_required:
        report.add(
            IssueCode.MISSING_REQUIRED_COLUMN,
            Severity.ERROR,
            f"Season {season_label(season_start_year)} is missing required "
            f"column(s): {', '.join(missing_required)}. The whole file is unusable.",
            count=len(missing_required),
        )
        return empty_canonical_frame(), report

    report.missing_statistic_columns = [
        column for column in STATISTIC_COLUMNS if column not in frame.columns
    ]
    if report.missing_statistic_columns:
        report.add(
            IssueCode.MISSING_STATISTIC_COLUMN,
            Severity.WARNING,
            f"Season {season_label(season_start_year)} has no "
            f"{', '.join(report.missing_statistic_columns)}; those features will be null.",
            count=len(report.missing_statistic_columns),
        )

    rows: list[dict[str, Any]] = []
    seen_match_ids: set[str] = set()
    counts: dict[IssueCode, int] = {}
    unknown_names: set[str] = set()
    null_statistics: dict[str, int] = {}

    def reject(code: IssueCode) -> None:
        counts[code] = counts.get(code, 0) + 1

    for _, raw in frame.iterrows():
        if raw.isna().all():
            # Trailing blank lines are normal in an in-progress season file and
            # are not counted as rows read.
            continue

        report.rows_read += 1

        home_source = raw.get("HomeTeam")
        away_source = raw.get("AwayTeam")
        home_goals = _coerce_int(raw.get("FTHG"))
        away_goals = _coerce_int(raw.get("FTAG"))
        result = raw.get("FTR")
        result = str(result).strip().upper() if pd.notna(result) else None

        if pd.isna(home_source) or pd.isna(away_source):
            reject(IssueCode.MISSING_CRITICAL_VALUE)
            continue
        if home_goals is None or away_goals is None or not result:
            # An unplayed or void match: no target to learn from.
            reject(IssueCode.MISSING_CRITICAL_VALUE)
            continue

        kickoff = parse_source_date(raw.get("Date"))
        if kickoff is None:
            reject(IssueCode.UNPARSEABLE_DATE)
            continue

        home_team = try_normalise_team_name(str(home_source))
        away_team = try_normalise_team_name(str(away_source))
        if home_team is None or away_team is None:
            if home_team is None:
                unknown_names.add(str(home_source).strip())
            if away_team is None:
                unknown_names.add(str(away_source).strip())
            reject(IssueCode.UNKNOWN_TEAM)
            continue

        if home_team == away_team:
            reject(IssueCode.TEAM_PLAYS_ITSELF)
            continue
        if result not in VALID_RESULTS:
            reject(IssueCode.INVALID_RESULT)
            continue
        if result != derive_result(home_goals, away_goals):
            # Source disagrees with itself; trusting either value would put a
            # wrong training target into the dataset.
            reject(IssueCode.RESULT_SCORE_MISMATCH)
            continue

        match_id = build_match_id(season_start_year, home_team, away_team)
        if match_id in seen_match_ids:
            reject(IssueCode.DUPLICATE_MATCH)
            continue
        seen_match_ids.add(match_id)

        row: dict[str, Any] = {
            "match_id": match_id,
            "season_start_year": season_start_year,
            "kickoff_date": kickoff.date(),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "source_file": source_file,
        }
        for source_column, canonical in STATISTIC_COLUMNS.items():
            value = _coerce_int(raw.get(source_column)) if source_column in frame else None
            if value is None and source_column in frame.columns:
                null_statistics[canonical] = null_statistics.get(canonical, 0) + 1
            row[canonical] = value

        rows.append(row)

    report.rows_accepted = len(rows)
    _record_rejections(report, counts, season_start_year, unknown_names)
    _record_null_statistics(report, null_statistics, season_start_year)

    return build_canonical_frame(rows), report


_REJECTION_MESSAGES: dict[IssueCode, str] = {
    IssueCode.MISSING_CRITICAL_VALUE: "missing a team, score, or result",
    IssueCode.UNPARSEABLE_DATE: "an unparseable date",
    IssueCode.UNKNOWN_TEAM: "a club spelling absent from the alias table",
    IssueCode.TEAM_PLAYS_ITSELF: "the same club listed home and away",
    IssueCode.INVALID_RESULT: "a result outside H/D/A",
    IssueCode.RESULT_SCORE_MISMATCH: "a result that contradicts its score",
    IssueCode.DUPLICATE_MATCH: "a duplicate of an earlier match",
}


def _record_rejections(
    report: SeasonValidationReport,
    counts: dict[IssueCode, int],
    season_start_year: int,
    unknown_names: set[str],
) -> None:
    for code, description in _REJECTION_MESSAGES.items():
        count = counts.get(code, 0)
        if not count:
            continue
        message = (
            f"Season {season_label(season_start_year)}: rejected {count} row(s) with {description}."
        )
        if code is IssueCode.UNKNOWN_TEAM and unknown_names:
            message += f" Unmapped: {', '.join(sorted(unknown_names))}."
        report.add(code, Severity.ERROR, message, count=count)


def _record_null_statistics(
    report: SeasonValidationReport, null_statistics: dict[str, int], season_start_year: int
) -> None:
    for column, count in sorted(null_statistics.items()):
        report.add(
            IssueCode.NULL_STATISTIC,
            Severity.WARNING,
            f"Season {season_label(season_start_year)}: {column} is null in "
            f"{count} accepted row(s).",
            count=count,
        )


def empty_canonical_frame() -> pd.DataFrame:
    """An empty frame carrying the canonical schema and its dtypes."""
    return build_canonical_frame([])


def build_canonical_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble canonical rows into a typed frame with stable column order."""
    frame = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))

    frame["kickoff_date"] = pd.to_datetime(frame["kickoff_date"])
    frame["season_start_year"] = frame["season_start_year"].astype("int16")
    for column in ("home_goals", "away_goals"):
        frame[column] = frame[column].astype("int8")
    for column in STATISTIC_COLUMNS.values():
        # Nullable integers: an absent statistic must stay absent, not become 0.
        frame[column] = frame[column].astype("Int16")
    for column in ("match_id", "home_team", "away_team", "result", "source_file"):
        frame[column] = frame[column].astype("string")

    return frame
