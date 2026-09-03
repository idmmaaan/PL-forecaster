"""Deterministic seed fixtures for local development and tests.

Shared by the in-memory repository and the `seed` CLI command so the API behaves
identically whether it is backed by PostgreSQL or by the in-memory double.
Provider ids match football-data.org team ids, which keeps a later real import
idempotent against these rows.
"""

from datetime import UTC, datetime
from typing import Any

PROVIDER = "football-data.org"
COMPETITION_CODE = "PL"
SEASON_START_YEAR = 2026

SEED_TEAMS: list[dict[str, Any]] = [
    {
        "id": 1,
        "provider_id": "football-data-57",
        "canonical_name": "Arsenal",
        "short_name": "Arsenal",
        "code": "ARS",
        "crest_url": "https://crests.football-data.org/57.png",
    },
    {
        "id": 2,
        "provider_id": "football-data-61",
        "canonical_name": "Chelsea",
        "short_name": "Chelsea",
        "code": "CHE",
        "crest_url": "https://crests.football-data.org/61.png",
    },
    {
        "id": 3,
        "provider_id": "football-data-64",
        "canonical_name": "Liverpool",
        "short_name": "Liverpool",
        "code": "LIV",
        "crest_url": "https://crests.football-data.org/64.png",
    },
    {
        "id": 4,
        "provider_id": "football-data-65",
        "canonical_name": "Manchester City",
        "short_name": "Man City",
        "code": "MCI",
        "crest_url": "https://crests.football-data.org/65.png",
    },
    {
        "id": 5,
        "provider_id": "football-data-73",
        "canonical_name": "Tottenham Hotspur",
        "short_name": "Tottenham",
        "code": "TOT",
        "crest_url": "https://crests.football-data.org/73.png",
    },
    {
        "id": 6,
        "provider_id": "football-data-67",
        "canonical_name": "Newcastle United",
        "short_name": "Newcastle",
        "code": "NEW",
        "crest_url": "https://crests.football-data.org/67.png",
    },
]

SEED_FIXTURES: list[dict[str, Any]] = [
    {
        "id": 14621,
        "provider": PROVIDER,
        "provider_id": "pl-2026-14621",
        "competition_code": COMPETITION_CODE,
        "season_start_year": SEASON_START_YEAR,
        "matchday": 4,
        "kickoff_at": datetime(2026, 9, 12, 14, 0, 0, tzinfo=UTC),
        "home_team_id": 1,
        "away_team_id": 2,
    },
    {
        "id": 14622,
        "provider": PROVIDER,
        "provider_id": "pl-2026-14622",
        "competition_code": COMPETITION_CODE,
        "season_start_year": SEASON_START_YEAR,
        "matchday": 4,
        "kickoff_at": datetime(2026, 9, 12, 16, 30, 0, tzinfo=UTC),
        "home_team_id": 3,
        "away_team_id": 4,
    },
    {
        "id": 14623,
        "provider": PROVIDER,
        "provider_id": "pl-2026-14623",
        "competition_code": COMPETITION_CODE,
        "season_start_year": SEASON_START_YEAR,
        "matchday": 4,
        "kickoff_at": datetime(2026, 9, 13, 13, 0, 0, tzinfo=UTC),
        "home_team_id": 5,
        "away_team_id": 6,
    },
]
