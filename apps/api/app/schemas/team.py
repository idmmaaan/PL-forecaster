from pydantic import BaseModel
from typing import Optional

class TeamSummary(BaseModel):
    id: int
    name: str
    crest_url: Optional[str] = None
    
    class Config:
        from_attributes = True