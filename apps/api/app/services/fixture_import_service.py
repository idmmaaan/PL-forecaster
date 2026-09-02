from typing import List, Dict, Any, Optional
from datetime import datetime
from app.models.team import Team
from app.models.fixture import Fixture
from app.clients.football_data_client import Match, Team as FootballTeam
from app.core.config import settings

class FixtureImportService:
    """Service for importing and converting football-data.org fixtures into our domain models"""
    
    def __init__(self):
        self.provider = "football-data.org"
        
    def convert_football_data_match_to_fixture(self, match: Match) -> Fixture:
        """
        Convert a football-data.org Match object to our internal Fixture model
        
        Args:
            match: FootballData API Match object
            
        Returns:
            Our internal Fixture domain model
        """
        return Fixture(
            provider=self.provider,
            provider_id=f"pl-{match.season_start_year}-{match.id}",
            competition_code=match.competition_code,
            season_start_year=match.season_start_year,
            matchday=match.matchday,
            kickoff_at=datetime.fromisoformat(match.kickoff_at.replace('Z', '+00:00')),
            status=match.status,
            home_team_id=None,  # Will be set when teams are created/lookup
            away_team_id=None,  # Will be set when teams are created/lookup
            home_score=match.home_score,
            away_score=match.away_score,
            result=match.result
        )
    
    def convert_football_data_match_to_team(self, football_team: FootballTeam) -> Team:
        """
        Convert a football-data.org Team object to our internal Team model
        
        Args:
            football_team: FootballData API Team object
            
        Returns:
            Our internal Team domain model  
        """
        return Team(
            provider_id=f"football-data-{football_team.id}",
            name=football_team.name,
            short_name=football_team.short_name,
            code=None,  # Not provided in football-data.org v4
            crest_url=football_team.crest_url
        )
    
    def convert_football_data_matches_to_domain_models(self, matches: List[Match]) -> tuple[List[Team], List[Fixture]]:
        """
        Convert a list of football-data.org matches into our domain models (teams and fixtures)
        
        Args:
            matches: List of Match objects from football-data.org
            
        Returns:
            Tuple of (teams_list, fixtures_list) containing our internal domain models
        """
        teams = []
        fixtures = []
        
        # Keep track of unique teams to avoid duplicates  
        team_lookup: Dict[int, Team] = {}
        
        for match in matches:
            # Convert home team if not already seen
            if match.home_team.id not in team_lookup:
                team = self.convert_football_data_match_to_team(match.home_team)
                team_lookup[match.home_team.id] = team
            
            # Convert away team if not already seen  
            if match.away_team.id not in team_lookup:
                team = self.convert_football_data_match_to_team(match.away_team)
                team_lookup[match.away_team.id] = team
                
        # Add all unique teams to the list
        teams.extend(team_lookup.values())
        
        # Convert matches to fixtures
        for match in matches:
            fixture = self.convert_football_data_match_to_fixture(match)
            # Note: We don't set team IDs here since they need to be resolved/assigned later
            fixtures.append(fixture)
            
        return teams, fixtures

# Create a singleton instance for use throughout the application
fixture_import_service = FixtureImportService()