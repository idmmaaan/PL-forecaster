import pytest
from datetime import datetime
from app.schemas.team import TeamSummary
from app.schemas.fixture import FixtureResponse

def test_team_summary_schema():
    """Test TeamSummary schema validation"""
    team_data = {
        "id": 1,
        "name": "Arsenal",
        "crest_url": "https://example.com/arsenal.png"
    }
    
    team = TeamSummary(**team_data)
    assert team.id == 1
    assert team.name == "Arsenal"
    assert team.crest_url == "https://example.com/arsenal.png"

def test_fixture_response_schema():
    """Test FixtureResponse schema validation"""
    # Test data with nested teams
    fixture_data = {
        "id": 14621,
        "competition_code": "PL",
        "season_start_year": 2026,
        "matchday": 4,
        "kickoff_at": datetime(2026, 9, 12, 14, 0, 0),
        "status": "SCHEDULED",
        "home_team": {
            "id": 1,
            "name": "Arsenal",
            "crest_url": None
        },
        "away_team": {
            "id": 2,
            "name": "Chelsea", 
            "crest_url": None
        }
    }
    
    fixture = FixtureResponse(**fixture_data)
    assert fixture.id == 14621
    assert fixture.competition_code == "PL"
    assert fixture.season_start_year == 2026
    assert fixture.matchday == 4
    assert fixture.status == "SCHEDULED"
    assert fixture.home_team.name == "Arsenal"
    assert fixture.away_team.name == "Chelsea"

def test_fixture_response_schema_with_optional_crest():
    """Test FixtureResponse with optional crest URLs"""
    fixture_data = {
        "id": 14621,
        "competition_code": "PL",
        "season_start_year": 2026,
        "matchday": 4,
        "kickoff_at": datetime(2026, 9, 12, 14, 0, 0),
        "status": "SCHEDULED",
        "home_team": {
            "id": 1,
            "name": "Arsenal",
            "crest_url": "https://example.com/arsenal.png"
        },
        "away_team": {
            "id": 2,
            "name": "Chelsea", 
            "crest_url": "https://example.com/chelsea.png"
        }
    }
    
    fixture = FixtureResponse(**fixture_data)
    assert fixture.home_team.crest_url == "https://example.com/arsenal.png"
    assert fixture.away_team.crest_url == "https://example.com/chelsea.png"