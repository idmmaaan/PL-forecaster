"""Adapter for the football-data.org v4 API.

The provider is wrapped behind this client so schema or plan changes stay
contained here, and so stored JSON responses can drive the importer tests.
"""

import asyncio
import logging
from enum import StrEnum
from typing import Any

import aiohttp
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

PREMIER_LEAGUE_CODE = "PL"


class MatchStatus(StrEnum):
    """Match statuses reported by football-data.org."""

    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"
    CANCELLED = "CANCELLED"


class Team(BaseModel):
    """A team as described by the provider."""

    id: int
    name: str
    short_name: str | None = None
    crest_url: str | None = None

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> "Team":
        return cls(
            id=payload["id"],
            name=payload["name"],
            short_name=payload.get("shortName"),
            crest_url=payload.get("crest"),
        )


class Match(BaseModel):
    """A match as described by the provider, flattened into snake_case fields."""

    id: int
    competition_code: str
    season_start_year: int
    matchday: int
    kickoff_at: str  # ISO 8601, as returned by the provider
    status: MatchStatus
    home_team: Team
    away_team: Team
    home_score: int | None = None
    away_score: int | None = None
    result: str | None = None

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> "Match":
        """Build a `Match` from the provider's nested JSON representation."""
        full_time = payload.get("score", {}).get("fullTime", {})
        home_score = full_time.get("home")
        away_score = full_time.get("away")

        return cls(
            id=payload["id"],
            competition_code=payload.get("competition", {}).get("code", PREMIER_LEAGUE_CODE),
            season_start_year=_season_start_year(payload.get("season", {})),
            matchday=payload["matchday"],
            kickoff_at=payload["utcDate"],
            status=MatchStatus(payload["status"]),
            home_team=Team.from_api_payload(payload["homeTeam"]),
            away_team=Team.from_api_payload(payload["awayTeam"]),
            home_score=home_score,
            away_score=away_score,
            result=derive_result(home_score, away_score),
        )


def _season_start_year(season: dict[str, Any]) -> int:
    """Extract the season's starting year from the provider's season object."""
    if "startDate" in season:
        return int(str(season["startDate"])[:4])
    raise ValueError("Season payload is missing startDate")


def derive_result(home_score: int | None, away_score: int | None) -> str | None:
    """Map a full-time score to the historical H/D/A label, or None if unplayed."""
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


class FootballDataClient:
    """Client for the football-data.org API."""

    def __init__(self, api_token: str | None = None, base_url: str | None = None):
        self.base_url = base_url or settings.football_data_base_url
        self.api_token = api_token or settings.football_data_api_token
        if not self.api_token:
            raise ValueError("FOOTBALL_DATA_API_TOKEN must be set in environment variables")

        self.timeout = aiohttp.ClientTimeout(total=30)

    def _headers(self) -> dict[str, str]:
        return {
            "X-Auth-Token": self.api_token or "",
            "User-Agent": "EPL-AI-Predictor/1.0",
        }

    def matches_url(self, season: str) -> str:
        """URL of the `/matches` collection for one season."""
        return f"{self.base_url}/competitions/{PREMIER_LEAGUE_CODE}/matches?season={season}"

    async def fetch_premier_league_matches_payload(self, season: str = "2026") -> dict[str, Any]:
        """Fetch the raw `/matches` response body for one season.

        The importer stores this body's checksum, so it is returned unparsed.

        Raises:
            ValueError: The API token is not configured, or the response could
                not be processed.
            aiohttp.ClientError: The request failed or returned a non-200 status.
        """
        if not self.api_token:
            raise ValueError("API token not configured")

        url = self.matches_url(season)

        try:
            async with (
                aiohttp.ClientSession(timeout=self.timeout, headers=self._headers()) as session,
                session.get(url) as response,
            ):
                if response.status != 200:
                    logger.error("Failed to fetch matches: HTTP %s", response.status)
                    raise aiohttp.ClientError(f"HTTP {response.status}: Failed to fetch matches")
                payload: dict[str, Any] = await response.json()
                return payload
        except TimeoutError as exc:
            logger.error("Request to football-data.org timed out")
            raise aiohttp.ServerTimeoutError("Request timed out") from exc
        except aiohttp.ClientError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Unexpected error fetching matches: %s", exc)
            raise ValueError(f"Failed to fetch Premier League matches: {exc}") from exc

    async def fetch_premier_league_matches(self, season: str = "2026") -> list[Match]:
        """Fetch Premier League matches for one season.

        Args:
            season: Season start year, e.g. "2026" for the 2026/27 season.

        Returns:
            Successfully parsed matches. Individual unparseable matches are
            logged and skipped so one bad row does not fail the whole import.
        """
        return parse_matches(await self.fetch_premier_league_matches_payload(season))


def parse_matches(data: dict[str, Any]) -> list[Match]:
    """Parse a `/matches` response body into `Match` objects, skipping bad rows."""
    matches: list[Match] = []
    for match_data in data.get("matches", []):
        try:
            matches.append(Match.from_api_payload(match_data))
        except Exception as exc:
            logger.warning("Skipping unparseable match payload: %s", exc)
    return matches
