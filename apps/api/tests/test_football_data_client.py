import pytest
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from app.clients.football_data_client import FootballDataClient

@pytest.fixture
def football_data_client():
    """Create a client instance with mocked environment"""
    # Create a minimal mock for the config to avoid dependency issues
    class MockSettings:
        football_data_base_url = "https://api.football-data.org/v4"
        football_data_api_token = "test-token-12345"
    
    with patch('app.clients.football_data_client.settings', MockSettings()):
        client = FootballDataClient()
        return client

@pytest.mark.asyncio
async def test_client_initialization(football_data_client):
    """Test that the client initializes correctly"""
    assert football_data_client.base_url == "https://api.football-data.org/v4"
    assert football_data_client.api_token == "test-token-12345"

def test_client_requires_api_token():
    """Test that client requires API token"""
    
    class MockSettingsNoToken:
        football_data_base_url = "https://api.football-data.org/v4"
        football_data_api_token = None
    
    with patch('app.clients.football_data_client.settings', MockSettingsNoToken()):
        with pytest.raises(ValueError, match="FOOTBALL_DATA_API_TOKEN must be set"):
            FootballDataClient()

@pytest.mark.asyncio
async def test_fetch_premier_league_matches_requires_token():
    """Test that fetch method raises error without API token"""
    
    class MockSettingsNoToken:
        football_data_base_url = "https://api.football-data.org/v4"
        football_data_api_token = None
    
    with patch('app.clients.football_data_client.settings', MockSettingsNoToken()):
        client = FootballDataClient()
        
        # This should raise an error about missing token
        with pytest.raises(ValueError, match="API token not configured"):
            await client.fetch_premier_league_matches()

@pytest.mark.asyncio  
async def test_fetch_premier_league_matches_http_error():
    """Test that HTTP errors are properly handled"""
    
    # Mock aiohttp session to return error response 
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.json = AsyncMock(return_value={"error": "Not Found"})
    
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Create a client and test it  
        class MockSettings:
            football_data_base_url = "https://api.football-data.org/v4"
            football_data_api_token = "test-token-12345"
        
        with patch('app.clients.football_data_client.settings', MockSettings()):
            client = FootballDataClient()
            
            # Should raise ClientError for HTTP error
            with pytest.raises(aiohttp.ClientError):
                await client.fetch_premier_league_matches("2026")

# Note: Actual integration tests would require mocking the real API calls
# which is complex and not feasible in this environment without more infrastructure