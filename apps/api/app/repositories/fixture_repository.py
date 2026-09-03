from datetime import UTC, datetime

from sqlalchemy.orm import Session, joinedload

from app.models.fixture import PREDICTABLE_STATUSES, Fixture
from app.models.team import Team
from app.repositories.fixture_interface import FixtureRepository


class SQLAlchemyFixtureRepository(FixtureRepository):
    """PostgreSQL-backed fixture repository."""

    def __init__(self, db: Session):
        self.db = db

    def get_fixtures(self) -> list[Fixture]:
        return (
            self.db.query(Fixture)
            .options(joinedload(Fixture.home_team), joinedload(Fixture.away_team))
            .order_by(Fixture.kickoff_at.asc())
            .all()
        )

    def get_fixture_by_id(self, fixture_id: int) -> Fixture | None:
        return (
            self.db.query(Fixture)
            .options(joinedload(Fixture.home_team), joinedload(Fixture.away_team))
            .filter(Fixture.id == fixture_id)
            .first()
        )

    def get_upcoming_fixtures(self, limit: int = 10) -> list[Fixture]:
        return (
            self.db.query(Fixture)
            .options(joinedload(Fixture.home_team), joinedload(Fixture.away_team))
            .filter(
                Fixture.status.in_(PREDICTABLE_STATUSES),
                Fixture.kickoff_at >= datetime.now(UTC),
            )
            .order_by(Fixture.kickoff_at.asc())
            .limit(limit)
            .all()
        )

    def upsert_team(self, team_data: dict) -> Team:
        """Insert or update a team, keyed on `provider_id`, and return it.

        Idempotent: importing the same provider payload twice leaves one row.
        """
        team = self.db.query(Team).filter(Team.provider_id == team_data["provider_id"]).first()

        if team is None:
            team = Team(**team_data)
            self.db.add(team)
        else:
            for key, value in team_data.items():
                if key != "id" and hasattr(team, key):
                    setattr(team, key, value)

        self.db.commit()
        self.db.refresh(team)
        return team

    def upsert_fixture(self, fixture_data: dict) -> Fixture:
        """Insert or update a fixture, keyed on `provider_id`, and return it."""
        fixture = (
            self.db.query(Fixture)
            .filter(Fixture.provider_id == fixture_data["provider_id"])
            .first()
        )

        if fixture is None:
            fixture = Fixture(**fixture_data)
            self.db.add(fixture)
        else:
            for key, value in fixture_data.items():
                if key != "id" and hasattr(fixture, key):
                    setattr(fixture, key, value)

        self.db.commit()
        self.db.refresh(fixture)
        return fixture
