"""Ingest orchestration: caching, checksums, canonical output, and reports.

No test here touches the network. Downloads are stubbed so the caching and
provenance rules can be checked deterministically, which is also what makes
the pipeline reproducible offline.
"""

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from epl_predictor.data import ingest
from epl_predictor.data.ingest import (
    CANONICAL_MATCHES_FILENAME,
    SeasonDownloadError,
    download_season,
    ingest_seasons,
    load_canonical,
    parse_seasons,
    read_season_csv,
    season_code,
    season_url,
    sha256_of,
)
from epl_predictor.data.validation import CANONICAL_COLUMNS, validate_season_frame

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR"


def csv_for(*rows: str) -> bytes:
    return ("\n".join([HEADER, *rows]) + "\n").encode()


def raw_frame_for(date: str) -> pd.DataFrame:
    """A one-row source frame, for tests that work on frames rather than files."""
    row = f"E0,{date},Man United,Newcastle,3,0,H,18,9,11,4,7,5,1,2,0,0"
    return pd.read_csv(io.StringIO("\n".join([HEADER, row])), dtype=str)


SEASON_2010 = csv_for(
    "E0,14/08/2010,Man United,Newcastle,3,0,H,18,9,11,4,7,5,1,2,0,0",
    "E0,15/08/2010,Arsenal,Liverpool,1,1,D,15,10,6,3,8,4,2,1,0,1",
)
SEASON_2011 = csv_for(
    "E0,13/08/2011,Chelsea,Man City,1,1,D,12,14,5,7,6,6,1,1,0,0",
)


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def bodies() -> dict[int, bytes]:
    return {2010: SEASON_2010, 2011: SEASON_2011}


@pytest.fixture
def stub_download(
    monkeypatch: pytest.MonkeyPatch, bodies: dict[int, bytes]
) -> dict[str, list[str]]:
    """Serve season CSVs from memory and record every URL requested."""
    calls: dict[str, list[str]] = {"urls": []}

    def fake_get(url: str, **_: Any) -> FakeResponse:
        calls["urls"].append(url)
        for start_year, body in bodies.items():
            if season_code(start_year) in url:
                return FakeResponse(body)
        return FakeResponse(b"", status_code=404)

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "reports_dir": tmp_path / "reports",
    }


def run_ingest(
    seasons: list[int], dirs: dict[str, Path], **kwargs: Any
) -> tuple[pd.DataFrame, ingest.IngestReport]:
    return ingest_seasons(seasons, **dirs, **kwargs)


@pytest.mark.parametrize(
    ("start_year", "code"), [(2010, "1011"), (1999, "9900"), (2024, "2425"), (2000, "0001")]
)
def test_season_code_matches_the_source_url_scheme(start_year: int, code: str) -> None:
    assert season_code(start_year) == code
    assert season_url(start_year) == f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def test_download_writes_the_raw_file_and_its_provenance(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    raw = download_season(2010, raw_dir=dirs["raw_dir"])

    assert raw.path.read_bytes() == SEASON_2010
    assert raw.checksum == sha256_of(raw.path)
    assert raw.size_bytes == len(SEASON_2010)
    assert not raw.from_cache

    metadata = json.loads(raw.metadata_path().read_text())
    assert metadata["source"] == "football-data.co.uk"
    assert metadata["source_url"] == season_url(2010)
    assert metadata["checksum"] == raw.checksum
    assert metadata["downloaded_at"]


def test_a_second_download_reuses_the_cached_file(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    """Re-running the pipeline must not re-fetch an unchanged season."""
    first = download_season(2010, raw_dir=dirs["raw_dir"])
    second = download_season(2010, raw_dir=dirs["raw_dir"])

    assert len(stub_download["urls"]) == 1
    assert second.from_cache
    assert second.checksum == first.checksum
    assert second.downloaded_at == first.downloaded_at


def test_refresh_forces_a_new_download(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    download_season(2010, raw_dir=dirs["raw_dir"])
    refreshed = download_season(2010, raw_dir=dirs["raw_dir"], refresh=True)

    assert len(stub_download["urls"]) == 2
    assert not refreshed.from_cache


def test_a_raw_file_edited_in_place_is_re_downloaded(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    """Raw files are immutable evidence. A checksum mismatch means tampering."""
    raw = download_season(2010, raw_dir=dirs["raw_dir"])
    raw.path.write_bytes(csv_for("E0,14/08/2010,Man United,Newcastle,9,0,H,,,,,,,,,,"))

    recovered = download_season(2010, raw_dir=dirs["raw_dir"])

    assert len(stub_download["urls"]) == 2
    assert recovered.path.read_bytes() == SEASON_2010
    assert recovered.checksum == raw.checksum


def test_a_cached_file_without_a_sidecar_gets_one(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    raw = download_season(2010, raw_dir=dirs["raw_dir"])
    raw.metadata_path().unlink()

    recovered = download_season(2010, raw_dir=dirs["raw_dir"])

    assert recovered.from_cache
    assert json.loads(recovered.metadata_path().read_text())["checksum"] == raw.checksum


def test_a_failed_download_raises(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    with pytest.raises(SeasonDownloadError):
        download_season(2019, raw_dir=dirs["raw_dir"])


def test_ingest_builds_a_canonical_table_across_seasons(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    matches, report = run_ingest([2010, 2011], dirs)

    assert report.seasons_ingested == [2010, 2011]
    assert (report.rows_read, report.rows_accepted, report.rows_rejected) == (3, 3, 0)
    assert list(matches.columns) == list(CANONICAL_COLUMNS)
    assert matches["season_start_year"].tolist() == [2010, 2010, 2011]


def test_the_canonical_table_is_sorted_chronologically(
    stub_download: dict[str, list[str]], dirs: dict[str, Path], bodies: dict[int, bytes]
) -> None:
    """The feature builder relies on this order to avoid leaking the future."""
    bodies[2010] = csv_for(
        "E0,15/08/2010,Arsenal,Liverpool,1,1,D,15,10,6,3,8,4,2,1,0,1",
        "E0,14/08/2010,Man United,Newcastle,3,0,H,18,9,11,4,7,5,1,2,0,0",
    )

    matches, _ = run_ingest([2010], dirs)

    assert matches["kickoff_date"].is_monotonic_increasing
    assert matches["home_team"].tolist() == ["Manchester United", "Arsenal"]


def test_ingest_writes_the_parquet_table(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    matches, report = run_ingest([2010], dirs)
    target = dirs["processed_dir"] / CANONICAL_MATCHES_FILENAME

    assert report.output_path == str(target)
    assert target.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(target), matches)


def test_canonical_dtypes_survive_the_parquet_round_trip(
    stub_download: dict[str, list[str]], dirs: dict[str, Path], bodies: dict[int, bytes]
) -> None:
    """A null shot count must still be null after reloading, never 0."""
    bodies[2010] = csv_for("E0,14/08/2010,Man United,Newcastle,3,0,H,,,,,,,,,,")

    run_ingest([2010], dirs)
    reloaded = load_canonical(dirs["processed_dir"])

    assert reloaded.iloc[0]["home_shots"] is pd.NA
    assert reloaded["home_shots"].dtype == "Int16"
    assert reloaded["home_goals"].dtype == "int8"


def test_load_canonical_explains_itself_when_nothing_is_built(
    dirs: dict[str, Path],
) -> None:
    with pytest.raises(FileNotFoundError, match="epl_predictor.data.ingest"):
        load_canonical(dirs["processed_dir"])


def test_ingest_writes_a_timestamped_report_and_a_latest_pointer(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    run_ingest([2010], dirs)
    report_dir = dirs["reports_dir"] / "ingest"

    latest = json.loads((report_dir / "latest.json").read_text())
    timestamped = sorted(report_dir.glob("ingest-*.json"))

    assert len(timestamped) == 1
    assert json.loads(timestamped[0].read_text()) == latest
    assert latest["seasons_ingested"] == [2010]
    assert latest["rows_accepted"] == 2
    assert latest["raw_files"][0]["checksum"]
    assert latest["season_reports"][0]["season_start_year"] == 2010


def test_a_season_that_fails_to_download_does_not_discard_the_others(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    matches, report = run_ingest([2010, 2019], dirs)

    assert report.seasons_ingested == [2010]
    assert "2019" in report.seasons_failed
    assert len(matches) == 2


def test_an_unusable_season_is_reported_and_skipped(
    stub_download: dict[str, list[str]], dirs: dict[str, Path], bodies: dict[int, bytes]
) -> None:
    bodies[2011] = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,13/08/2011,Chelsea,Man City,1,1\n"

    matches, report = run_ingest([2010, 2011], dirs)

    assert report.seasons_ingested == [2010]
    assert report.seasons_failed["2011"] == "missing required columns"
    assert matches["season_start_year"].unique().tolist() == [2010]


def test_a_fixture_listed_in_the_wrong_season_file_keeps_that_files_season() -> None:
    """A season file defines the season, so an overlapping row is a distinct match.

    Football-Data.co.uk occasionally repeats a fixture in an adjacent file. The
    `match_id` is keyed on the file's season, so the two rows do not collide and
    the canonical table stays unique.
    """
    early = validate_season_frame(raw_frame_for("14/08/2010"), 2010)[0]
    late = validate_season_frame(raw_frame_for("14/08/2010"), 2011)[0]

    combined = ingest._combine([early, late], ingest.IngestReport(started_at=""))

    assert combined["match_id"].is_unique
    assert combined["season_start_year"].tolist() == [2010, 2011]


def test_the_same_season_ingested_twice_is_deduplicated() -> None:
    """The last defence against a season file being counted twice."""
    frame = validate_season_frame(raw_frame_for("14/08/2010"), 2010)[0]
    report = ingest.IngestReport(started_at="")

    combined = ingest._combine([frame, frame], report)

    assert report.cross_season_duplicates == 1
    assert len(combined) == 1


def test_ingest_of_no_usable_season_yields_an_empty_canonical_frame(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    matches, report = run_ingest([2019], dirs)

    assert matches.empty
    assert list(matches.columns) == list(CANONICAL_COLUMNS)
    assert report.seasons_ingested == []


def test_write_outputs_can_be_disabled_for_dry_runs(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    _, report = run_ingest([2010], dirs, write_outputs=False)

    assert report.output_path is None
    assert not (dirs["processed_dir"] / CANONICAL_MATCHES_FILENAME).exists()
    assert not dirs["reports_dir"].exists()


def test_a_latin_1_encoded_file_is_readable(tmp_path: Path) -> None:
    """Older season files are not UTF-8."""
    target = tmp_path / "E0_0304.csv"
    target.write_bytes(
        csv_for("E0,16/08/2003,Man United,Newcastle,3,0,H,,,,,,,,,,").replace(
            b"Newcastle", "Newcastlé".encode("latin-1")
        )
    )

    frame = read_season_csv(target)

    assert frame.iloc[0]["AwayTeam"] == "Newcastlé"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2010", [2010]),
        ("2010:2012", [2010, 2011, 2012]),
        ("2024,2025", [2024, 2025]),
        ("2010:2011,2024", [2010, 2011, 2024]),
        ("2010, 2010 ,2011", [2010, 2011]),
    ],
)
def test_season_specifications_parse(spec: str, expected: list[int]) -> None:
    assert parse_seasons(spec) == expected


@pytest.mark.parametrize("spec", ["", "abc", "2012:2010", "1970", "2010:abc", ","])
def test_malformed_season_specifications_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_seasons(spec)


def test_the_cli_reports_success(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    exit_code = ingest.main(
        [
            "--seasons",
            "2010:2011",
            "--raw-dir",
            str(dirs["raw_dir"]),
            "--processed-dir",
            str(dirs["processed_dir"]),
            "--reports-dir",
            str(dirs["reports_dir"]),
        ]
    )

    assert exit_code == 0
    assert len(load_canonical(dirs["processed_dir"])) == 3


def test_the_cli_fails_when_a_season_cannot_be_ingested(
    stub_download: dict[str, list[str]], dirs: dict[str, Path]
) -> None:
    exit_code = ingest.main(
        [
            "--seasons",
            "2019",
            "--raw-dir",
            str(dirs["raw_dir"]),
            "--processed-dir",
            str(dirs["processed_dir"]),
            "--reports-dir",
            str(dirs["reports_dir"]),
        ]
    )

    assert exit_code == 1
