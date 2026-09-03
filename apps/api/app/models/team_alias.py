from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.team import Team


class TeamAlias(Base):
    """Resolves differing club names across data sources to one canonical team.

    Football-Data.co.uk writes "Man City" where football-data.org writes
    "Manchester City FC"; both map to the same `teams` row through this table.
    """

    __tablename__ = "team_aliases"
    __table_args__ = (
        # One alias per source can only ever point at a single team.
        UniqueConstraint("source", "alias", name="uq_team_aliases_source_alias"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(50))  # e.g. "football-data.co.uk"
    alias: Mapped[str] = mapped_column(String(255))

    team: Mapped[Team] = relationship()
