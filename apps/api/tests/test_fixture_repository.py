import pytest
from datetime import datetime, timezone
from app.repositories.fixture_repository import TemporaryFixtureRepository
from app.models.fixture import Fixture

def test_get_fixtures():
    """Test that repository returns all fixtures"""
    repo = TemporaryFixtureRepository()
    fixtures = repo.get_fixtures()
    
    assert len(fixtures) == 3
    assert isinstance(fixtures[0], Fixture)
    assert fixtures[0].id == 14621

def test_get_fixture_by_id():
    """Test getting a specific fixture by ID"""
    repo = TemporaryFixtureRepository()
    
    # Test existing fixture
    fixture = repo.get_fixture_by_id(14621)
    assert fixture is not None
    assert fixture.id == 14621
    assert fixture.home_team_id == 1
    assert fixture.away_team_id == 2
    
    # Test non-existing fixture
    fixture = repo.get_fixture_by_id(99999)
    assert fixture is None

def test_get_upcoming_fixtures():
    """Test getting upcoming fixtures"""
    repo = TemporaryFixtureRepository()
    
    fixtures = repo.get_upcoming_fixtures(limit=5)
    assert len(fixtures) == 3  # Should return all 3 fixtures
    
    # Test limit parameter
    fixtures = repo.get_upcoming_fixtures(limit=2)
    assert len(fixtures) == 2

def test_fixture_attributes():
    """Test fixture attributes are correctly set"""
    repo = TemporaryFixtureRepository()
    fixtures = repo.get_fixtures()
    
    first_fixture = fixtures[0]
    assert first_fixture.provider == "football-data.org"
    assert first_fixture.competition_code == "PL"
    assert first_fixture.season_start_year == 2026
    assert first_fixture.matchday == 4
    
    # Test datetime is properly set with timezone
    assert first_fixture.kickoff_at.tzinfo is not None
    expected_time = datetime(2026, 9, 12, 14, 0, 0, tzinfo=timezone.utc)
    assert first_fixture.kickoff_at == expected_time
    
    assert first_fixture.status.value == "SCHEDULED"