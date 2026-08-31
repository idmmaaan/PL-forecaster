from pydantic import BaseModel, validator
from typing import Optional

class Outcome(str):
    """Enumeration of possible match outcomes"""
    HOME_WIN = "HOME_WIN"
    DRAW = "DRAW" 
    AWAY_WIN = "AWAY_WIN"

class Probabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float
    
    @validator('home_win', 'draw', 'away_win')
    def probability_values_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probabilities must be between 0.0 and 1.0')
        return v

class PredictionResponse(BaseModel):
    prediction_id: int
    fixture_id: int
    home_team: str
    away_team: str
    predicted_outcome: Outcome
    probabilities: Probabilities
    model_name: str
    model_version: str
    feature_schema_version: str
    created_at: str  # ISO format string
    
    class Config:
        use_enum_values = True
        from_attributes = True