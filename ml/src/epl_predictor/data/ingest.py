"""Download and canonicalise Football-Data.co.uk Premier League seasons.

Raw CSVs are written once and never modified, each beside a sidecar recording
its source URL, SHA-256 checksum, and download time. Re-running the ingest
reuses a cached file whose checksum still matches, so the pipeline is
reproducible offline and a silently changed upstream file is detectable.

Validated seasons are concatenated into one canonical Parquet table sorted by
kickoff date, which is the only input the feature builder reads.

Usage:
    python -m epl_predictor.data.ingest --seasons 2010:2025
    python -m epl_predictor.data.ingest --seasons 2024,2025 --refresh
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from epl_predictor.data.validation import (
    CANONICAL_COLUMNS,
    SeasonValidationReport,
    Severity,
    empty_canonical_frame,
    season_label,
    validate_season_frame,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "football-data.co.uk"
SOURCE_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

ML_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ML_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ML_ROOT / "data" / "processed"
DEFAULT_REPORTS_DIR = ML_ROOT / "reports"
CANONICAL_MATCHES_FILENAME = "matches.parquet"

# Football-Data.co.uk publishes statistics columns from 2000/01 onwards.
EARLIEST_SUPPORTED_SEASON = 2000


def season_code(start_year: int) -> str:
    """Render a season as the source's four-digit URL segment, e.g. 2010 -> "1011"."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_url(start_year: int) -> str:
    return SOURCE_URL_TEMPLATE.format(code=season_code(start_year))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RawSeasonFile:
    """A downloaded season CSV and its provenance."""

    season_start_year: int
    path: Path
    source_url: str
    checksum: str
    size_bytes: int
    downloaded_at: str
    from_cache: bool = False

    def metadata_path(self) -> Path:
        return self.path.with_suffix(".meta.json")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload

    def write_metadata(self) -> None:
        payload = self.as_dict()
        payload.pop("from_cache")
        payload["source"] = SOURCE_NAME
        self.metadata_path().write_text(json.dumps(payload, indent=2) + "\n")


@dataclass
class IngestReport:
    """Outcome of one ingest run, written to `ml/reports/` as JSON."""

    started_at: str
    seasons_requested: list[int] = field(default_factory=list)
    seasons_ingested: list[int] = field(default_factory=list)
    seasons_failed: dict[str, str] = field(default_factory=dict)
    raw_files: list[dict[str, Any]] = field(default_factory=list)
    season_reports: list[dict[str, Any]] = field(default_factory=list)
    rows_read: int = 0
    rows_accepted: int = 0
    cross_season_duplicates: int = 0
    output_path: str | None = None
    finished_at: str | None = None

    @property
    def rows_rejected(self) -> int:
        return self.rows_read - self.rows_accepted

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows_rejected"] = self.rows_rejected
        return payload


class SeasonDownloadError(RuntimeError):
    """A season CSV could not be downloaded."""


def download_season(
    start_year: int, raw_dir: Path = DEFAULT_RAW_DIR, refresh: bool = False
) -> RawSeasonFile:
    """Fetch one season CSV, reusing the cached copy unless `refresh` is set.

    Raises:
        SeasonDownloadError: The request failed or returned a non-200 status.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"E0_{season_code(start_year)}.csv"
    url = season_url(start_year)

    if target.exists() and not refresh:
        cached = _load_cached(start_year, target, url)
        if cached is not None:
            logger.info("Season %s: using cached %s", season_label(start_year), target.name)
            return cached

    import requests  # imported lazily so offline callers can use cached files

    logger.info("Season %s: downloading %s", season_label(start_year), url)
    try:
        response = requests.get(url, timeout=60, headers={"User-Agent": "EPL-AI-Predictor/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SeasonDownloadError(f"Could not download {url}: {exc}") from exc

    if not response.content.strip():
        raise SeasonDownloadError(f"{url} returned an empty body")

    target.write_bytes(response.content)
    raw_file = RawSeasonFile(
        season_start_year=start_year,
        path=target,
        source_url=url,
        checksum=sha256_of(target),
        size_bytes=target.stat().st_size,
        downloaded_at=datetime.now(UTC).isoformat(),
    )
    raw_file.write_metadata()
    return raw_file


def _load_cached(start_year: int, target: Path, url: str) -> RawSeasonFile | None:
    """Describe an existing raw file, re-deriving its checksum from disk.

    A sidecar whose checksum no longer matches the bytes means the raw file was
    edited in place, which breaks the reproducibility guarantee, so the file is
    treated as absent and re-downloaded.
    """
    checksum = sha256_of(target)
    metadata_path = target.with_suffix(".meta.json")

    downloaded_at = datetime.now(UTC).isoformat()
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        recorded = metadata.get("checksum")
        if recorded and recorded != checksum:
            logger.warning(
                "Season %s: %s no longer matches its recorded checksum; re-downloading.",
                season_label(start_year),
                target.name,
            )
            return None
        downloaded_at = metadata.get("downloaded_at", downloaded_at)

    raw_file = RawSeasonFile(
        season_start_year=start_year,
        path=target,
        source_url=url,
        checksum=checksum,
        size_bytes=target.stat().st_size,
        downloaded_at=downloaded_at,
        from_cache=True,
    )
    if not metadata_path.exists():
        raw_file.write_metadata()
    return raw_file


def read_season_csv(path: Path) -> pd.DataFrame:
    """Read a raw season CSV without letting pandas guess away the blank rows.

    Everything is read as text so validation controls all coercion, and both
    encodings the source has used over the years are tried.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=True, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8 or Latin-1")


def ingest_seasons(
    seasons: list[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    refresh: bool = False,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, IngestReport]:
    """Download, validate, and canonicalise the requested seasons.

    Returns the canonical match table and the run's report. A season that fails
    to download or validate is recorded and skipped, so one bad season does not
    discard the rest.
    """
    report = IngestReport(
        started_at=datetime.now(UTC).isoformat(), seasons_requested=sorted(seasons)
    )
    frames: list[pd.DataFrame] = []

    for start_year in sorted(seasons):
        try:
            raw_file = download_season(start_year, raw_dir=raw_dir, refresh=refresh)
        except SeasonDownloadError as exc:
            logger.error("%s", exc)
            report.seasons_failed[str(start_year)] = str(exc)
            continue

        report.raw_files.append(raw_file.as_dict())

        try:
            raw_frame = read_season_csv(raw_file.path)
        except (ValueError, pd.errors.ParserError) as exc:
            logger.error("Season %s: unreadable CSV: %s", season_label(start_year), exc)
            report.seasons_failed[str(start_year)] = f"unreadable CSV: {exc}"
            continue

        frame, season_report = validate_season_frame(
            raw_frame, start_year, source_file=raw_file.path.name
        )
        report.season_reports.append(season_report.as_dict())
        report.rows_read += season_report.rows_read
        report.rows_accepted += season_report.rows_accepted
        _log_season(season_report)

        if not season_report.is_usable:
            report.seasons_failed[str(start_year)] = "missing required columns"
            continue

        report.seasons_ingested.append(start_year)
        frames.append(frame)

    matches = _combine(frames, report)

    if write_outputs:
        report.output_path = str(write_canonical(matches, processed_dir))
        write_report(report, reports_dir)

    report.finished_at = datetime.now(UTC).isoformat()
    return matches, report


def _combine(frames: list[pd.DataFrame], report: IngestReport) -> pd.DataFrame:
    """Concatenate seasons, drop cross-season duplicates, and sort chronologically.

    Chronological order is not cosmetic: the feature builder depends on it to
    guarantee that a match's features are derived only from earlier matches.
    """
    if not frames:
        return empty_canonical_frame()

    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["match_id"], keep="first")
    report.cross_season_duplicates = before - len(combined)
    if report.cross_season_duplicates:
        logger.warning(
            "Dropped %d duplicate match(es) spanning season files.",
            report.cross_season_duplicates,
        )

    combined = combined.sort_values(
        ["kickoff_date", "home_team", "away_team"], kind="mergesort"
    ).reset_index(drop=True)
    return combined[list(CANONICAL_COLUMNS)]


def write_canonical(matches: pd.DataFrame, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """Write the canonical match table as Parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    target = processed_dir / CANONICAL_MATCHES_FILENAME
    matches.to_parquet(target, index=False)
    logger.info("Wrote %d matches to %s", len(matches), target)
    return target


def load_canonical(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Read the canonical match table written by a previous ingest run.

    Raises:
        FileNotFoundError: The table has not been built yet.
    """
    target = processed_dir / CANONICAL_MATCHES_FILENAME
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Run `python -m epl_predictor.data.ingest` first."
        )
    return pd.read_parquet(target)


def write_report(report: IngestReport, reports_dir: Path = DEFAULT_REPORTS_DIR) -> Path:
    """Write the ingest report, keeping a timestamped copy alongside the latest."""
    target_dir = reports_dir / "ingest"
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = report.started_at.replace(":", "").replace("-", "").split(".")[0]
    payload = json.dumps(report.as_dict(), indent=2, default=str) + "\n"
    timestamped = target_dir / f"ingest-{stamp}.json"
    timestamped.write_text(payload)
    (target_dir / "latest.json").write_text(payload)
    logger.info("Wrote ingest report to %s", timestamped)
    return timestamped


def _log_season(season_report: SeasonValidationReport) -> None:
    logger.info(
        "Season %s: %d/%d rows accepted.",
        season_label(season_report.season_start_year),
        season_report.rows_accepted,
        season_report.rows_read,
    )
    for issue in season_report.issues:
        log = logger.error if issue.severity is Severity.ERROR else logger.warning
        log("  %s", issue.message)


def parse_seasons(spec: str) -> list[int]:
    """Parse a season selection like `2010:2025` or `2010,2011,2024`.

    Raises:
        ValueError: The specification is malformed or out of range.
    """
    seasons: set[int] = set()

    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start_text, _, end_text = chunk.partition(":")
            start, end = _season_year(start_text), _season_year(end_text)
            if start > end:
                raise ValueError(f"Season range {chunk!r} runs backwards")
            seasons.update(range(start, end + 1))
        else:
            seasons.add(_season_year(chunk))

    if not seasons:
        raise ValueError(f"No seasons found in {spec!r}")
    return sorted(seasons)


def _season_year(text: str) -> int:
    try:
        year = int(text.strip())
    except ValueError:
        raise ValueError(f"{text!r} is not a season start year") from None
    if year < EARLIEST_SUPPORTED_SEASON or year > 2100:
        raise ValueError(
            f"Season {year} is outside the supported range ({EARLIEST_SUPPORTED_SEASON} onwards)"
        )
    return year


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        default="2010:2025",
        help="Season start years: a range like 2010:2025, a list like 2024,2025, or both.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download season files even when a valid cached copy exists.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as exc:
        parser.error(str(exc))

    matches, report = ingest_seasons(
        seasons,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        refresh=args.refresh,
    )

    logger.info(
        "Ingested %d season(s): %d/%d rows accepted, %d rejected.",
        len(report.seasons_ingested),
        report.rows_accepted,
        report.rows_read,
        report.rows_rejected,
    )
    if report.seasons_failed:
        for season, reason in sorted(report.seasons_failed.items()):
            logger.error("Season %s failed: %s", season, reason)
        return 1
    return 0 if len(matches) else 1


if __name__ == "__main__":
    raise SystemExit(main())
