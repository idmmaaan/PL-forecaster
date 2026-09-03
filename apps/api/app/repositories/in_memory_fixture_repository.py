from copy import deepcopy
from typing import Any

from app.db.seed_data import SEED_FIXTURES, SEED_TEAMS
from app.models.fixture import PREDICTABLE_STATUSES, Fixture, FixtureStatus
from app.models.team import Team
from app.repositories.fixture_interface import FixtureRepository


class InMemoryFixtureRepository(FixtureRepository):
    """Fixture repository backed by the seed dataset, with no database.

    Used by the test suite (via FastAPI dependency overrides) so that route and
    prediction behaviour can be verified without a running PostgreSQL instance.
    The production path always uses `SQLAlchemyFixtureRepository`.
    """

    def __init__(self) -> None:
        self._teams: dict[int, Team] = {}
        self._fixtures: dict[int, Fixture] = {}
        self._load_seed_data()

    def _load_seed_data(self) -> None:
        for team_data in deepcopy(SEED_TEAMS):
            team = Team(**team_data)
            self._teams[team.id] = team

        for fixture_data in deepcopy(SEED_FIXTURES):
            fixture = Fixture(status=FixtureStatus.SCHEDULED, **fixture_data)
            # Transient ORM objects have no session to lazy-load from, so the
            # team relationships are attached explicitly.
            fixture.home_team = self._teams[fixture.home_team_id]
            fixture.away_team = self._teams[fixture.away_team_id]
            self._fixtures[fixture.id] = fixture

    def _sorted_fixtures(self) -> list[Fixture]:
        return sorted(self._fixtures.values(), key=lambda f: f.kickoff_at)

    def get_fixtures(self) -> list[Fixture]:
        return self._sorted_fixtures()

    def get_fixture_by_id(self, fixture_id: int) -> Fixture | None:
        return self._fixtures.get(fixture_id)

    def get_upcoming_fixtures(self, limit: int = 10) -> list[Fixture]:
        upcoming = [
            fixture for fixture in self._sorted_fixtures() if fixture.status in PREDICTABLE_STATUSES
        ]
        return upcoming[:limit]

    def upsert_team(self, team_data: dict[str, Any]) -> Team:
        existing = next(
            (t for t in self._teams.values() if t.provider_id == team_data["provider_id"]),
            None,
        )
        if existing is None:
            team = Team(**team_data)
            if team.id is None:
                team.id = max(self._teams, default=0) + 1
            self._teams[team.id] = team
            return team

        for key, value in team_data.items():
            if key != "id" and hasattr(existing, key):
                setattr(existing, key, value)
        return existing

    def upsert_fixture(self, fixture_data: dict[str, Any]) -> Fixture:
        existing = next(
            (f for f in self._fixtures.values() if f.provider_id == fixture_data["provider_id"]),
            None,
        )
        if existing is None:
            fixture = Fixture(**fixture_data)
            if fixture.id is None:
                fixture.id = max(self._fixtures, default=0) + 1
            if fixture.home_team_id in self._teams:
                fixture.home_team = self._teams[fixture.home_team_id]
            if fixture.away_team_id in self._teams:
                fixture.away_team = self._teams[fixture.away_team_id]
            self._fixtures[fixture.id] = fixture
            return fixture

        for key, value in fixture_data.items():
            if key != "id" and hasattr(existing, key):
                setattr(existing, key, value)
        return existing
