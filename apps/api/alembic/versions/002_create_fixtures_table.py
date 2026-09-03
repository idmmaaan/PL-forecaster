"""Create fixtures table

Revision ID: 002
Revises: 001
Create Date: 2026-08-31 19:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FIXTURE_STATUS_VALUES = (
    "SCHEDULED",
    "TIMED",
    "IN_PLAY",
    "PAUSED",
    "FINISHED",
    "POSTPONED",
    "SUSPENDED",
    "CANCELLED",
)


def upgrade() -> None:
    # The type is created explicitly and then referenced with create_type=False,
    # otherwise create_table would emit a second CREATE TYPE and fail.
    postgresql.ENUM(*FIXTURE_STATUS_VALUES, name="fixture_status").create(
        op.get_bind(), checkfirst=True
    )
    fixture_status = postgresql.ENUM(
        *FIXTURE_STATUS_VALUES, name="fixture_status", create_type=False
    )

    op.create_table(
        "fixtures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("provider_id", sa.String(length=100), nullable=False),
        sa.Column("competition_code", sa.String(length=10), nullable=False),
        sa.Column("season_start_year", sa.Integer(), nullable=False),
        sa.Column("matchday", sa.Integer(), nullable=False),
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", fixture_status, nullable=False),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("result", sa.String(length=1), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"], name="fk_fixtures_home_team_id"),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"], name="fk_fixtures_away_team_id"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("result IN ('H', 'D', 'A')", name="ck_fixtures_result"),
    )
    op.create_index("ix_fixtures_id", "fixtures", ["id"])
    # One canonical row per provider fixture, so re-importing is idempotent.
    op.create_index("ix_fixtures_provider_id", "fixtures", ["provider_id"], unique=True)
    op.create_index("ix_fixtures_kickoff_at", "fixtures", ["kickoff_at"])


def downgrade() -> None:
    op.drop_index("ix_fixtures_kickoff_at", table_name="fixtures")
    op.drop_index("ix_fixtures_provider_id", table_name="fixtures")
    op.drop_index("ix_fixtures_id", table_name="fixtures")
    op.drop_table("fixtures")
    postgresql.ENUM(name="fixture_status").drop(op.get_bind(), checkfirst=True)
