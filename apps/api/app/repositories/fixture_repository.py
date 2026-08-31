from typing import List, Optional
from datetime import datetime, timezone
from app.models.fixture import Fixture
from app.schemas.fixture import FixtureResponse
from .fixture_interface import FixtureRepository
from app.models.team import Team

class TemporaryFixtureRepository(FixtureRepository):
    """Temporary fixture repository that returns hardcoded fixtures"""
    
    def __init__(self):
        # Create some dummy teams for the fixtures 
        self.teams = [
            Team(id=1, provider_id="arsenal", name="Arsenal", short_name="ARS", code="ARS", crest_url=None),
            Team(id=2, provider_id="chelsea", name="Chelsea", short_name="CHE", code="CHE", crest_url=None),
            Team(id=3, provider_id="man-city", name="Manchester City", short_name="MCI", code="MCI", crest_url=None),
        ]
        
    def get_fixtures(self) -> List[Fixture]:
        """Get all hardcoded fixtures"""
        return self._create_hardcoded_fixtures()
    
    def get_fixture_by_id(self, fixture_id: int) -> Optional[Fixture]:
        """Get a specific fixture by ID"""
        fixtures = self._create_hardcoded_fixtures()
        for fixture in fixtures:
            if fixture.id == fixture_id:
                return fixture
        return None
    
    def get_upcoming_fixtures(self, limit: int = 10) -> List[Fixture]:
        """Get upcoming fixtures"""
        fixtures = self._create_hardcoded_fixtures()
        # Return all fixtures (they're all upcoming for this temporary implementation)
        return fixtures[:limit]
    
    def _create_hardcoded_fixtures(self) -> List[Fixture]:
        """Create hardcoded Premier League style fixtures"""
        # Create a simple mapping of team IDs to make it easier
        team_map = {team.id: team for team in self.teams}
        
        fixtures = [
            Fixture(
                id=14621,
                provider="football-data.org",
                provider_id="pl-2026-14621",
                competition_code="PL",
                season_start_year=2026,
                matchday=4,
                kickoff_at=datetime(2026, 9, 12, 14, 0, 0, tzinfo=timezone.utc),
                status="SCHEDULED",
                home_team_id=1,  # Arsenal
                away_team_id=2,  # Chelsea
                home_score=None,
                away_score=None,
                result=None
            ),
            Fixture(
                id=14622,
                provider="football-data.org",
                provider_id="pl-2026-14622", 
                competition_code="PL",
                season_start_year=2026,
                matchday=4,
                kickoff_at=datetime(2026, 9, 12, 16, 30, 0, tzinfo=timezone.utc),
                status="SCHEDULED",
                home_team_id=3,  # Manchester City
                away_team_id=1,  # Arsenal
                home_score=None,
                away_score=None,
                result=None
            ),
            Fixture(
                id=14623,
                provider="football-data.org",
                provider_id="pl-2026-14623",
                competition_code="PL",
                season_start_year=2026,
                matchday=5,
                kickoff_at=datetime(2026, 9, 19, 14, 0, 0, tzinfo=timezone.utc),
                status="SCHEDULED",
                home_team_id=2,  # Chelsea
                away_team_id=3,  # Manchester City
                home_score=None,
                away_score=None,
                result=None
            )
        ]
        
        return fixtures