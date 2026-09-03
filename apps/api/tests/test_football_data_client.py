import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from app.clients.football_data_client import (
    FootballDataClient,
    Match,
    MatchStatus,
    derive_result,
    parse_matches,
)

STORED_RESPONSE = Path(__file__).parent / "fixtures" / "football_data_matches.json"


@pytest.fixture
def stored_payload() -> dict[str, Any]:
    return json.loads(STORED_RESPONSE.read_text())


@pytest.fixture
def client() -> FootballDataClient:
    return FootballDataClient(api_token="test-token-12345")


class FakeResponse:
    """Minimal async context manager standing in for an aiohttp response."""

    def __init__(self, status: int, payload: dict[str, Any] | None = None):
        self.status = status
        self._payload = payload or {}

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class FakeSession:
    """Records the request it was given and returns a canned response."""

    def __init__(self, response: FakeResponse):
        self._response = response
        self.requested_url: str | None = None
        self.headers: dict[str, str] = {}

    def get(self, url: str) -> FakeResponse:
        self.requested_url = url
        return self._response

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def install_fake_session(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> FakeSession:
    session = FakeSession(response)

    def factory(*_: object, **kwargs: object) -> FakeSession:
        session.headers = dict(kwargs.get("headers") or {})
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", factory)
    return session


def test_client_initialization(client: FootballDataClient) -> None:
    assert client.base_url == "https://api.football-data.org/v4"
    assert client.api_token == "test-token-12345"


def test_client_requires_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials fail immediately rather than at request time."""
    monkeypatch.setattr("app.clients.football_data_client.settings.football_data_api_token", None)

    with pytest.raises(ValueError, match="FOOTBALL_DATA_API_TOKEN must be set"):
        FootballDataClient()


async def test_fetch_rejects_a_cleared_token(client: FootballDataClient) -> None:
    """The request path re-checks the token instead of sending an unauthenticated call."""
    client.api_token = None

    with pytest.raises(ValueError, match="API token not configured"):
        await client.fetch_premier_league_matches()


def test_auth_token_is_sent_in_the_documented_header(client: FootballDataClient) -> None:
    headers = client._headers()

    assert headers["X-Auth-Token"] == "test-token-12345"
    assert "X-Response-Time" not in headers


async def test_fetch_premier_league_matches_parses_stored_response(
    client: FootballDataClient, monkeypatch: pytest.MonkeyPatch, stored_payload: dict[str, Any]
) -> None:
    session = install_fake_session(monkeypatch, FakeResponse(200, stored_payload))

    matches = await client.fetch_premier_league_matches("2026")

    assert session.requested_url is not None
    assert "/competitions/PL/matches?season=2026" in session.requested_url
    assert session.headers["X-Auth-Token"] == "test-token-12345"
    assert [match.id for match in matches] == [14621, 14622, 14623]

    scheduled = matches[0]
    assert scheduled.competition_code == "PL"
    assert scheduled.season_start_year == 2026
    assert scheduled.matchday == 4
    assert scheduled.kickoff_at == "2026-09-12T14:00:00Z"
    assert scheduled.status is MatchStatus.SCHEDULED
    assert scheduled.home_team.name == "Arsenal FC"
    assert scheduled.home_team.short_name == "Arsenal"
    assert scheduled.home_team.crest_url == "https://crests.football-data.org/57.png"
    assert scheduled.home_score is None
    assert scheduled.result is None


async def test_fetch_premier_league_matches_maps_final_scores(
    client: FootballDataClient, monkeypatch: pytest.MonkeyPatch, stored_payload: dict[str, Any]
) -> None:
    install_fake_session(monkeypatch, FakeResponse(200, stored_payload))

    matches = await client.fetch_premier_league_matches("2026")
    away_win, draw = matches[1], matches[2]

    assert (away_win.home_score, away_win.away_score, away_win.result) == (1, 3, "A")
    assert (draw.home_score, draw.away_score, draw.result) == (2, 2, "D")


async def test_fetch_premier_league_matches_http_error(
    client: FootballDataClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_session(monkeypatch, FakeResponse(404, {"error": "Not Found"}))

    with pytest.raises(aiohttp.ClientError, match="HTTP 404"):
        await client.fetch_premier_league_matches("2026")


def test_parse_matches_skips_unparseable_rows_and_keeps_the_rest(
    stored_payload: dict[str, Any],
) -> None:
    """One malformed row must not discard an entire import batch."""
    payload = {"matches": [{"id": 1, "status": "SCHEDULED"}, *stored_payload["matches"]]}

    matches = parse_matches(payload)

    assert [match.id for match in matches] == [14621, 14622, 14623]


def test_parse_matches_handles_an_empty_response() -> None:
    assert parse_matches({}) == []


@pytest.mark.parametrize(
    ("home", "away", "expected"),
    [(2, 1, "H"), (1, 2, "A"), (1, 1, "D"), (None, None, None), (1, None, None)],
)
def test_derive_result(home: int | None, away: int | None, expected: str | None) -> None:
    assert derive_result(home, away) == expected


def test_match_from_api_payload_requires_a_season_start_date(
    stored_payload: dict[str, Any],
) -> None:
    payload = dict(stored_payload["matches"][0])
    payload["season"] = {}

    with pytest.raises(ValueError, match="startDate"):
        Match.from_api_payload(payload)
