"""Fetch upcoming fixtures from football-data.org into PostgreSQL.

Run with `make sync-fixtures` (or `python -m app.db.sync_fixtures`). Requires
FOOTBALL_DATA_API_TOKEN. Safe to re-run: fixtures and teams are keyed on their
provider ids, so a second run updates rather than duplicates.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import aiohttp

from app.clients.football_data_client import FootballDataClient
from app.core.database import get_session_factory
from app.services.fixture_sync_service import FixtureSyncService

logger = logging.getLogger(__name__)


async def fetch_payload(season: str) -> tuple[dict[str, Any], str]:
    """Fetch the raw `/matches` body for one season, plus the URL it came from."""
    client = FootballDataClient()
    payload = await client.fetch_premier_league_matches_payload(season)
    return payload, client.matches_url(season)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        default="2026",
        help="Season start year, e.g. 2026 for the 2026/27 season.",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        help="Read a stored /matches response instead of calling the API.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.from_file:
        payload = json.loads(args.from_file.read_text())
        source_uri = str(args.from_file)
    else:
        try:
            payload, source_uri = asyncio.run(fetch_payload(args.season))
        except ValueError as exc:
            logger.error("%s", exc)
            logger.error("Set FOOTBALL_DATA_API_TOKEN in .env, or pass --from-file.")
            return 2
        except aiohttp.ClientError as exc:
            logger.error("Could not reach football-data.org: %s", exc)
            return 1

    with get_session_factory()() as session:
        report = FixtureSyncService(session).sync_payload(
            payload, source_uri=source_uri, season_start_year=int(args.season)
        )

    logger.info(
        "Read %d matches: %d fixtures created, %d updated, %d teams created, "
        "%d aliases created, %d rejected.",
        report.matches_read,
        report.fixtures_created,
        report.fixtures_updated,
        report.teams_created,
        report.aliases_created,
        report.matches_read - report.accepted,
    )
    for rejection in report.rejected:
        logger.warning("Rejected: %s", rejection)

    return 0


if __name__ == "__main__":
    sys.exit(main())
