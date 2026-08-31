from fastapi import APIRouter, Depends
from typing import List
from app.repositories.fixture_repository import TemporaryFixtureRepository
from app.schemas.fixture import FixtureResponse

router = APIRouter(prefix="/fixtures", tags=["fixtures"])

@router.get("/", response_model=List[FixtureResponse])
async def get_fixtures(fixture_repo: TemporaryFixtureRepository = Depends()):
    """Get all fixtures"""
    fixtures = fixture_repo.get_fixtures()
    return fixtures