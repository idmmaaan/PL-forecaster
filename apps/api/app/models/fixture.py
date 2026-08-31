from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..core.database import Base
from enum import Enum as PyEnum

class FixtureStatus(PyEnum):
    SCHEDULED = "SCHEDULED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"

class Fixture(Base):
    __tablename__ = "fixtures"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False)  # e.g., "football-data.org"
    provider_id = Column(String(100), unique=True, nullable=False, index=True)
    competition_code = Column(String(10), nullable=False)
    season_start_year = Column(Integer, nullable=False)
    matchday = Column(Integer, nullable=False)
    kickoff_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(Enum(FixtureStatus), nullable=False, default=FixtureStatus.SCHEDULED)
    
    # Foreign keys to teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    
    # Optional score fields for completed matches
    home_score = Column(Integer)
    away_score = Column(Integer)
    result = Column(String(10))  # H, D, or A
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])