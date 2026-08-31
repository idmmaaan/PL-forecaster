from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.fixture import Fixture
from app.schemas.fixture import FixtureResponse

class FixtureRepository(ABC):
    """Interface for fixture data access operations"""
    
    @abstractmethod
    def get_fixtures(self) -> List[Fixture]:
        """Get all fixtures"""
        pass
    
    @abstractmethod 
    def get_fixture_by_id(self, fixture_id: int) -> Optional[Fixture]:
        """Get a specific fixture by ID"""
        pass
    
    @abstractmethod
    def get_upcoming_fixtures(self, limit: int = 10) -> List[Fixture]:
        """Get upcoming fixtures"""
        pass