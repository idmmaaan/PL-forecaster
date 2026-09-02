from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.fixture import Fixture
from app.models.team import Team
from app.repositories.fixture_interface import FixtureRepository
from app.core.database import get_db


class SQLAlchemyFixtureRepository(FixtureRepository):
    """SQLAlchemy-backed fixture repository implementation"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_fixtures(self) -> List[Fixture]:
        """Get all fixtures from database"""
        return self.db.query(Fixture).all()
    
    def get_fixture_by_id(self, fixture_id: int) -> Optional[Fixture]:
        """Get a specific fixture by ID"""
        return self.db.query(Fixture).filter(Fixture.id == fixture_id).first()
    
    def get_upcoming_fixtures(self, limit: int = 10) -> List[Fixture]:
        """Get upcoming fixtures"""
        # For now, we'll just return all fixtures - in a real implementation
        # this would filter based on date or status  
        return self.db.query(Fixture).limit(limit).all()
    
    def upsert_team(self, team_data: dict) -> Team:
        """
        Upsert a team into the database
        
        Args:
            team_data: Dictionary containing team information
            
        Returns:
            The persisted Team object
        """
        # Try to find existing team by provider_id
        existing_team = self.db.query(Team).filter(
            Team.provider_id == team_data['provider_id']
        ).first()
        
        if existing_team:
            # Update existing team
            for key, value in team_data.items():
                if hasattr(existing_team, key) and key != 'id':
                    setattr(existing_team, key, value)
        else:
            # Create new team 
            existing_team = Team(**team_data)
            self.db.add(existing_team)
        
        self.db.commit()
        self.db.refresh(existing_team)
        return existing_team
    
    def upsert_fixture(self, fixture_data: dict) -> Fixture:
        """
        Upsert a fixture into the database
        
        Args:
            fixture_data: Dictionary containing fixture information
            
        Returns:
            The persisted Fixture object
        """
        # Try to find existing fixture by provider_id  
        existing_fixture = self.db.query(Fixture).filter(
            Fixture.provider_id == fixture_data['provider_id']
        ).first()
        
        if existing_fixture:
            # Update existing fixture
            for key, value in fixture_data.items():
                if hasattr(existing_fixture, key) and key != 'id':
                    setattr(existing_fixture, key, value)
        else:
            # Create new fixture
            existing_fixture = Fixture(**fixture_data)
            self.db.add(existing_fixture)
        
        self.db.commit()
        self.db.refresh(existing_fixture)
        return existing_fixture

# Global instance for use in dependency injection
# Note: In a real application, this would be properly injected via FastAPI dependencies
_default_db_session = None

def get_fixture_repository(db: Session = None) -> SQLAlchemyFixtureRepository:
    """Factory function to create fixture repository with database session"""
    global _default_db_session
    if db is not None:
        return SQLAlchemyFixtureRepository(db)
    elif _default_db_session is not None:
        return SQLAlchemyFixtureRepository(_default_db_session)
    else:
        # This would be initialized properly in a real FastAPI application
        raise ValueError("No database session provided")