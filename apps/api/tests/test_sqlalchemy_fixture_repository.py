import pytest
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session
from app.repositories.fixture_repository import SQLAlchemyFixtureRepository
from app.models.team import Team
from app.models.fixture import Fixture

@pytest.fixture
def mock_db():
    """Create a mock database session"""
    return Mock(spec=Session)

@pytest.fixture
def fixture_repo(mock_db):
    """Create fixture repository instance"""
    return SQLAlchemyFixtureRepository(mock_db)

def test_get_fixtures(fixture_repo, mock_db):
    """Test get_fixtures method"""
    # Setup mock
    mock_fixture = Mock()
    mock_db.query.return_value.all.return_value = [mock_fixture]
    
    result = fixture_repo.get_fixtures()
    
    mock_db.query.assert_called_once_with(Fixture)
    mock_db.query.return_value.all.assert_called_once()
    assert result == [mock_fixture]

def test_get_fixture_by_id(fixture_repo, mock_db):
    """Test get_fixture_by_id method"""
    # Setup mock
    mock_fixture = Mock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_fixture
    
    result = fixture_repo.get_fixture_by_id(1)
    
    mock_db.query.assert_called_once_with(Fixture)
    mock_db.query.return_value.filter.assert_called_once()
    assert result == mock_fixture

def test_get_upcoming_fixtures(fixture_repo, mock_db):
    """Test get_upcoming_fixtures method"""
    # Setup mock
    mock_fixture = Mock()
    mock_db.query.return_value.limit.return_value.all.return_value = [mock_fixture]
    
    result = fixture_repo.get_upcoming_fixtures(5)
    
    mock_db.query.assert_called_once_with(Fixture)
    mock_db.query.return_value.limit.assert_called_once_with(5)
    assert result == [mock_fixture]

def test_upsert_team_new_team(fixture_repo, mock_db):
    """Test upsert_team with new team"""
    # Setup mock
    team_data = {
        'provider_id': 'football-data-1',
        'name': 'Arsenal',
        'short_name': 'ARS',
        'crest_url': 'https://example.com/arsenal.png'
    }
    
    # Mock that no existing team is found
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    # Mock the Team constructor and commit behavior  
    mock_team_instance = Mock()
    mock_team_instance.id = 1
    with patch('app.repositories.fixture_repository.Team', return_value=mock_team_instance):
        result = fixture_repo.upsert_team(team_data)
    
    # Verify database operations
    mock_db.query.assert_called_once_with(Team)
    mock_db.query.return_value.filter.assert_called_once()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()
    assert result == mock_team_instance

def test_upsert_team_existing_team(fixture_repo, mock_db):
    """Test upsert_team with existing team"""
    # Setup mocks
    team_data = {
        'provider_id': 'football-data-1',
        'name': 'Arsenal Updated',
        'short_name': 'ARS',
        'crest_url': 'https://example.com/arsenal.png'
    }
    
    # Mock that an existing team is found
    mock_existing_team = Mock()
    mock_existing_team.id = 1
    mock_db.query.return_value.filter.return_value.first.return_value = mock_existing_team
    
    result = fixture_repo.upsert_team(team_data)
    
    # Verify database operations
    mock_db.query.assert_called_once_with(Team)
    mock_db.query.return_value.filter.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()
    assert result == mock_existing_team
    # Check that the existing team was updated (not added)
    assert not mock_db.add.called

def test_upsert_fixture_new_fixture(fixture_repo, mock_db):
    """Test upsert_fixture with new fixture"""
    # Setup mock
    fixture_data = {
        'provider': 'football-data.org',
        'provider_id': 'pl-2026-14621',
        'competition_code': 'PL',
        'season_start_year': 2026,
        'matchday': 4,
        'kickoff_at': datetime(2026, 9, 12, 14, 0, 0),
        'status': 'SCHEDULED',
        'home_team_id': 1,
        'away_team_id': 2,
    }
    
    # Mock that no existing fixture is found
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    # Mock the Fixture constructor and commit behavior  
    mock_fixture_instance = Mock()
    mock_fixture_instance.id = 1
    with patch('app.repositories.fixture_repository.Fixture', return_value=mock_fixture_instance):
        result = fixture_repo.upsert_fixture(fixture_data)
    
    # Verify database operations
    mock_db.query.assert_called_once_with(Fixture)
    mock_db.query.return_value.filter.assert_called_once()
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()
    assert result == mock_fixture_instance

def test_upsert_fixture_existing_fixture(fixture_repo, mock_db):
    """Test upsert_fixture with existing fixture"""
    # Setup mocks
    fixture_data = {
        'provider': 'football-data.org',
        'provider_id': 'pl-2026-14621',
        'competition_code': 'PL',
        'season_start_year': 2026,
        'matchday': 5,  # Updated matchday
        'kickoff_at': datetime(2026, 9, 12, 14, 0, 0),
        'status': 'SCHEDULED',
        'home_team_id': 1,
        'away_team_id': 2,
    }
    
    # Mock that an existing fixture is found
    mock_existing_fixture = Mock()
    mock_existing_fixture.id = 1
    mock_db.query.return_value.filter.return_value.first.return_value = mock_existing_fixture
    
    result = fixture_repo.upsert_fixture(fixture_data)
    
    # Verify database operations  
    mock_db.query.assert_called_once_with(Fixture)
    mock_db.query.return_value.filter.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()
    assert result == mock_existing_fixture
    # Check that the existing fixture was updated (not added)
    assert not mock_db.add.called

# Note: We can't test actual database operations without a real DB connection,
# but we've verified the method calls and logic flow are correct.