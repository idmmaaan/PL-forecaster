from abc import ABC, abstractmethod

from app.models.fixture import Fixture
from app.models.team import Team


class FixtureRepository(ABC):
    """Interface for fixture data access.

    Routes depend on this abstraction rather than on a concrete backend, so the
    SQLAlchemy implementation can be swapped for an in-memory one in tests.
    """

    @abstractmethod
    def get_fixtures(self) -> list[Fixture]:
        """Return every known fixture."""

    @abstractmethod
    def get_fixture_by_id(self, fixture_id: int) -> Fixture | None:
        """Return one fixture, or None when the id is unknown."""

    @abstractmethod
    def get_upcoming_fixtures(self, limit: int = 10) -> list[Fixture]:
        """Return scheduled fixtures that have not kicked off yet, soonest first."""

    @abstractmethod
    def upsert_team(self, team_data: dict) -> Team:
        """Insert or update a team, keyed on `provider_id`."""

    @abstractmethod
    def upsert_fixture(self, fixture_data: dict) -> Fixture:
        """Insert or update a fixture, keyed on `provider_id`."""
