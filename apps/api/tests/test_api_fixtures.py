from httpx import AsyncClient


async def test_get_fixtures(client: AsyncClient) -> None:
    """GET /api/v1/fixtures returns the fixture list with nested teams."""
    response = await client.get("/api/v1/fixtures")

    assert response.status_code == 200
    fixtures = response.json()
    assert len(fixtures) == 3

    first_fixture = fixtures[0]
    for field in (
        "id",
        "competition_code",
        "season_start_year",
        "matchday",
        "kickoff_at",
        "status",
        "home_team",
        "away_team",
    ):
        assert field in first_fixture

    for team in (first_fixture["home_team"], first_fixture["away_team"]):
        assert "id" in team
        assert "name" in team
        assert "crest_url" in team


async def test_get_fixtures_content(client: AsyncClient) -> None:
    """The seeded fixture data is exposed unchanged."""
    response = await client.get("/api/v1/fixtures")
    fixtures = response.json()

    assert fixtures[0]["id"] == 14621
    assert fixtures[0]["competition_code"] == "PL"
    assert fixtures[0]["season_start_year"] == 2026
    assert fixtures[0]["matchday"] == 4
    assert fixtures[0]["status"] == "SCHEDULED"
    assert fixtures[0]["home_team"]["name"] == "Arsenal"
    assert fixtures[0]["away_team"]["name"] == "Chelsea"


async def test_get_fixtures_ordered_by_kickoff(client: AsyncClient) -> None:
    response = await client.get("/api/v1/fixtures")
    kickoffs = [fixture["kickoff_at"] for fixture in response.json()]

    assert kickoffs == sorted(kickoffs)


async def test_get_fixture_by_id(client: AsyncClient) -> None:
    response = await client.get("/api/v1/fixtures/14622")

    assert response.status_code == 200
    fixture = response.json()
    assert fixture["id"] == 14622
    assert fixture["home_team"]["name"] == "Liverpool"


async def test_get_unknown_fixture_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/v1/fixtures/99999")

    assert response.status_code == 404
    assert "detail" in response.json()
