from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class FixtureResponse(BaseModel):
    id: int
    competition_code: str
    season_start_year: int
    matchday: int
    kickoff_at: datetime
    status: str
    home_team: "TeamSummary"
    away_team: "TeamSummary"
    
    class Config:
        from_attributes = True