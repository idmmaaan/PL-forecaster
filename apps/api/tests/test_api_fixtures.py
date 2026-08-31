import pytest
from httpx import AsyncClient
from app.main import app
from app.repositories.fixture_repository import TemporaryFixtureRepository

@pytest.mark.asyncio
async def test_get_fixtures():
    """Test GET /api/v1/fixtures endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/fixtures")
        
        # Test HTTP status code
        assert response.status_code == 200
        
        # Test response structure and content
        fixtures = response.json()
        
        # Should return exactly 3 fixtures
        assert len(fixtures) == 3
        
        # Test first fixture structure
        first_fixture = fixtures[0]
        assert "id" in first_fixture
        assert "competition_code" in first_fixture
        assert "season_start_year" in first_fixture
        assert "matchday" in first_fixture
        assert "kickoff_at" in first_fixture
        assert "status" in first_fixture
        assert "home_team" in first_fixture
        assert "away_team" in first_fixture
        
        # Test nested team structures
        home_team = first_fixture["home_team"]
        away_team = first_fixture["away_team"]
        assert "id" in home_team
        assert "name" in home_team
        assert "crest_url" in home_team
        
        assert "id" in away_team
        assert "name" in away_team
        assert "crest_url" in away_team

@pytest.mark.asyncio 
async def test_get_fixtures_content():
    """Test that fixtures have expected content"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/fixtures")
        fixtures = response.json()
        
        # Test specific fixture data
        assert fixtures[0]["id"] == 14621
        assert fixtures[0]["competition_code"] == "PL"
        assert fixtures[0]["season_start_year"] == 2026
        assert fixtures[0]["matchday"] == 4
        assert fixtures[0]["status"] == "SCHEDULED"
        
        # Test team data in first fixture  
        home_team = fixtures[0]["home_team"]
        away_team = fixtures[0]["away_team"]
        assert home_team["name"] == "Arsenal"
        assert away_team["name"] == "Chelsea"