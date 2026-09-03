from datetime import datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.models.team import Team


class FixtureStatus(StrEnum):
    """Fixture lifecycle states, mirroring football-data.org match statuses.

    A `StrEnum` member compares equal to the provider's raw string value, so no
    explicit conversion is needed at every boundary.
    """

    SCHEDULED = "SCHEDULED"
    TIMED = "TIMED"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    SUSPENDED = "SUSPENDED"
    CANCELLED = "CANCELLED"


#: Statuses for fixtures that have not started and can still be predicted.
PREDICTABLE_STATUSES = (FixtureStatus.SCHEDULED, FixtureStatus.TIMED)


class Fixture(Base):
    __tablename__ = "fixtures"
    __table_args__ = (CheckConstraint("result IN ('H', 'D', 'A')", name="ck_fixtures_result"),)

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    provider: Mapped[str] = mapped_column(String(50))  # e.g. "football-data.org"
    provider_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    competition_code: Mapped[str] = mapped_column(String(10))
    season_start_year: Mapped[int] = mapped_column()
    matchday: Mapped[int] = mapped_column()
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[FixtureStatus] = mapped_column(
        Enum(FixtureStatus, name="fixture_status"), default=FixtureStatus.SCHEDULED
    )

    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))

    # Populated only once the match has been played.
    home_score: Mapped[int | None] = mapped_column()
    away_score: Mapped[int | None] = mapped_column()
    result: Mapped[str | None] = mapped_column(String(1))  # H, D, A, or NULL

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
