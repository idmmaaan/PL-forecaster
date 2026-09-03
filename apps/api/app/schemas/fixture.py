from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.team import TeamSummary


class FixtureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    competition_code: str
    season_start_year: int
    matchday: int
    kickoff_at: datetime
    status: str
    home_team: TeamSummary
    away_team: TeamSummary
