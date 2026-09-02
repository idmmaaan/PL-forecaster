import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from app.services.fixture_import_service import FixtureImportService
from app.clients.football_data_client import Match, Team as FootballTeam, MatchStatus

@pytest.fixture
def fixture_import_service():
    """Create a fixture import service instance"""
    return FixtureImportService()

def test_convert_football_data_match_to_fixture(fixture_import_service):
    """Test conversion of football-data.org match to our Fixture model"""
    
    # Create mock football data match  
    football_match = Match(
        id=14621,
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at="2026-09-12T14:00:00Z",
        status=MatchStatus.SCHEDULED,
        home_team=FootballTeam(
            id=1,
            name="Arsenal",
            short_name="ARS",
            crest_url="https://example.com/arsenal.png"
        ),
        away_team=FootballTeam(
            id=2, 
            name="Chelsea",
            short_name="CHE",
            crest_url="https://example.com/chelsea.png"
        ),
        home_score=None,
        away_score=None,
        result=None
    )
    
    # Convert to our domain model
    fixture = fixture_import_service.convert_football_data_match_to_fixture(football_match)
    
    # Verify all fields are correctly mapped
    assert fixture.provider == "football-data.org"
    assert fixture.provider_id == "pl-2026-14621"
    assert fixture.competition_code == "PL"
    assert fixture.season_start_year == 2026
    assert fixture.matchday == 4
    assert fixture.kickoff_at == datetime(2026, 9, 12, 14, 0, 0)
    assert fixture.status == MatchStatus.SCHEDULED
    assert fixture.home_score is None
    assert fixture.away_score is None
    assert fixture.result is None

def test_convert_football_data_match_to_team(fixture_import_service):
    """Test conversion of football-data.org team to our Team model"""
    
    # Create mock football data team
    football_team = FootballTeam(
        id=1,
        name="Arsenal",
        short_name="ARS", 
        crest_url="https://example.com/arsenal.png"
    )
    
    # Convert to our domain model
    team = fixture_import_service.convert_football_data_match_to_team(football_team)
    
    # Verify all fields are correctly mapped
    assert team.provider_id == "football-data-1"
    assert team.name == "Arsenal"
    assert team.short_name == "ARS"
    assert team.code is None  # Not provided in football-data.org v4
    assert team.crest_url == "https://example.com/arsenal.png"

def test_convert_football_data_matches_to_domain_models(fixture_import_service):
    """Test conversion of multiple matches to domain models"""
    
    # Create mock football data matches  
    home_team = FootballTeam(
        id=1,
        name="Arsenal",
        short_name="ARS", 
        crest_url="https://example.com/arsenal.png"
    )
    
    away_team = FootballTeam(
        id=2, 
        name="Chelsea",
        short_name="CHE",
        crest_url="https://example.com/chelsea.png"
    )
    
    match1 = Match(
        id=14621,
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at="2026-09-12T14:00:00Z",
        status=MatchStatus.SCHEDULED,
        home_team=home_team,
        away_team=away_team,
        home_score=None,
        away_score=None,
        result=None
    )
    
    match2 = Match(
        id=14622,
        competition_code="PL", 
        season_start_year=2026,
        matchday=4,
        kickoff_at="2026-09-12T16:30:00Z",
        status=MatchStatus.SCHEDULED,
        home_team=away_team,  # Chelsea vs Arsenal (reverse)
        away_team=home_team,
        home_score=None,
        away_score=None,
        result=None
    )
    
    matches = [match1, match2]
    
    # Convert to domain models
    teams, fixtures = fixture_import_service.convert_football_data_matches_to_domain_models(matches)
    
    # Verify we get 2 unique teams and 2 fixtures
    assert len(teams) == 2  # Arsenal and Chelsea (no duplicates)
    assert len(fixtures) == 2
    
    # Check that teams are correctly identified 
    team_names = [team.name for team in teams]
    assert "Arsenal" in team_names
    assert "Chelsea" in team_names
    
    # Check fixtures have correct data
    fixture_ids = [fixture.id for fixture in fixtures]
    assert 14621 in fixture_ids
    assert 14622 in fixture_ids

def test_duplicate_teams_handling(fixture_import_service):
    """Test that duplicate teams are handled correctly"""
    
    # Same team used twice (different matches)
    home_team = FootballTeam(
        id=1,
        name="Arsenal",
        short_name="ARS", 
        crest_url="https://example.com/arsenal.png"
    )
    
    away_team = FootballTeam(
        id=2, 
        name="Chelsea",
        short_name="CHE",
        crest_url="https://example.com/chelsea.png"
    )
    
    match1 = Match(
        id=14621,
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at="2026-09-12T14:00:00Z",
        status=MatchStatus.SCHEDULED,
        home_team=home_team,
        away_team=away_team,
        home_score=None,
        away_score=None,
        result=None
    )
    
    match2 = Match(
        id=14622,
        competition_code="PL", 
        season_start_year=2026,
        matchday=5,
        kickoff_at="2026-09-19T14:00:00Z",
        status=MatchStatus.SCHEDULED,
        home_team=home_team,  # Same team again
        away_team=away_team,
        home_score=None,
        away_score=None,
        result=None
    )
    
    matches = [match1, match2]
    
    teams, fixtures = fixture_import_service.convert_football_data_matches_to_domain_models(matches)
    
    # Should still only have 2 unique teams (no duplicates)  
    assert len(teams) == 2
    assert len(fixtures) == 2

def test_datetime_conversion(fixture_import_service):
    """Test that datetime conversion works correctly"""
    
    football_match = Match(
        id=14621,
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at="2026-09-12T14:00:00Z",  # ISO format with Z
        status=MatchStatus.SCHEDULED,
        home_team=FootballTeam(id=1, name="Arsenal"),
        away_team=FootballTeam(id=2, name="Chelsea"),
        home_score=None,
        away_score=None,
        result=None
    )
    
    fixture = fixture_import_service.convert_football_data_match_to_fixture(football_match)
    
    # Test that datetime is properly parsed
    expected_dt = datetime(2026, 9, 12, 14, 0, 0)
    assert fixture.kickoff_at == expected_dt

def test_partial_team_data(fixture_import_service):
    """Test handling of teams with missing optional fields"""
    
    # Team without short_name
    football_team = FootballTeam(
        id=3,
        name="Manchester City",
        short_name=None,  # Not provided
        crest_url=None    # Not provided  
    )
    
    team = fixture_import_service.convert_football_data_match_to_team(football_team)
    
    assert team.provider_id == "football-data-3"
    assert team.name == "Manchester City"
    assert team.short_name is None
    assert team.crest_url is None