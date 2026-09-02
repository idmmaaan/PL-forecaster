import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MatchStatus(str):
    """Enumeration of match statuses"""
    SCHEDULED = "SCHEDULED"
    IN_PLAY = "IN_PLAY" 
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"

class Team(BaseModel):
    id: int
    name: str
    short_name: Optional[str] = None
    crest_url: Optional[str] = None

class Match(BaseModel):
    id: int
    competition_code: str = Field(..., alias='competition.code')
    season_start_year: int = Field(..., alias='season.startDate')
    matchday: int
    kickoff_at: str  # ISO format date string
    status: MatchStatus
    home_team: Team
    away_team: Team
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    result: Optional[str] = None

class FootballDataClient:
    """Client for interacting with football-data.org API"""
    
    def __init__(self):
        self.base_url = settings.football_data_base_url
        self.api_token = settings.football_data_api_token
        if not self.api_token:
            raise ValueError("FOOTBALL_DATA_API_TOKEN must be set in environment variables")
        
        # Configure HTTP client with timeout and headers
        self.timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
        
    async def fetch_premier_league_matches(self, season: str = "2026") -> List[Match]:
        """
        Fetch all matches from the Premier League for a specific season
        
        Args:
            season: Season start year (e.g., "2026" for 2026/27 season)
            
        Returns:
            List of Match objects
            
        Raises:
            aiohttp.ClientError: If HTTP request fails
            ValueError: If API token is not configured or parsing fails
        """
        if not self.api_token:
            raise ValueError("API token not configured")
            
        headers = {
            'X-Response-Time': '100',
            'User-Agent': 'EPL-AI-Predictor/1.0'
        }
        
        # Add authorization header
        if self.api_token:
            headers['X-Response-Time'] = self.api_token
            
        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout, 
                headers=headers
            ) as session:
                # Fetch competitions to find Premier League ID first
                competition_url = f"{self.base_url}/competitions/PL"
                
                async with session.get(competition_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch competition info: {response.status}")
                        raise aiohttp.ClientError(f"HTTP {response.status}: Failed to fetch competition")
                    
                    competition_data = await response.json()
                    competition_id = competition_data.get('id')
                    
                # Fetch matches for the Premier League
                matches_url = f"{self.base_url}/competitions/{competition_id}/matches?season={season}"
                
                async with session.get(matches_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch matches: {response.status}")
                        raise aiohttp.ClientError(f"HTTP {response.status}: Failed to fetch matches")
                    
                    data = await response.json()
                    
                    # Parse and validate match data
                    matches = []
                    for match_data in data.get('matches', []):
                        try:
                            match = Match(**match_data)
                            matches.append(match)
                        except Exception as e:
                            logger.warning(f"Failed to parse match data: {e}")
                            # Continue with other matches instead of failing completely
                            continue
                    
                    return matches
                    
        except asyncio.TimeoutError:
            raise aiohttp.ClientTimeout("Request timed out")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching matches: {e}")
            raise ValueError(f"Failed to fetch Premier League matches: {str(e)}")

# Create a singleton instance for use throughout the application
football_data_client = FootballDataClient()