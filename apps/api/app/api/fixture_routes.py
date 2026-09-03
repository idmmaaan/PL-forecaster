from fastapi import APIRouter, Depends

from app.api.deps import get_fixture_repository
from app.core.exceptions import FixtureNotFoundError
from app.repositories.fixture_interface import FixtureRepository
from app.schemas.fixture import FixtureResponse

router = APIRouter(prefix="/fixtures", tags=["fixtures"])


@router.get("", response_model=list[FixtureResponse])
async def list_fixtures(
    fixture_repo: FixtureRepository = Depends(get_fixture_repository),
) -> list[FixtureResponse]:
    """List fixtures, earliest kickoff first."""
    fixtures = fixture_repo.get_fixtures()
    return [FixtureResponse.model_validate(fixture) for fixture in fixtures]


@router.get("/{fixture_id}", response_model=FixtureResponse)
async def get_fixture(
    fixture_id: int,
    fixture_repo: FixtureRepository = Depends(get_fixture_repository),
) -> FixtureResponse:
    """Return one fixture by id."""
    fixture = fixture_repo.get_fixture_by_id(fixture_id)
    if fixture is None:
        raise FixtureNotFoundError(f"Fixture with ID {fixture_id} not found")
    return FixtureResponse.model_validate(fixture)
