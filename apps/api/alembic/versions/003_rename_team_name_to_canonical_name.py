"""Rename teams.name to teams.canonical_name

Source-specific spellings move to the team_aliases table, so the column on
teams is renamed to say explicitly that it holds the one canonical spelling.

Revision ID: 003
Revises: 002
Create Date: 2026-09-03 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("teams", "name", new_column_name="canonical_name")


def downgrade() -> None:
    op.alter_column("teams", "canonical_name", new_column_name="name")
