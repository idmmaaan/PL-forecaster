"""Create team_aliases, data_imports, feature_snapshots, model_versions, predictions

Completes the schema described in the README: source-name resolution, ingest
auditing, stored feature vectors, the model registry, and persisted predictions.

Revision ID: 004
Revises: 003
Create Date: 2026-09-03 12:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MODEL_STATUS_VALUES = ("CANDIDATE", "ACTIVE", "REJECTED", "ARCHIVED")
OUTCOME_VALUES = ("HOME_WIN", "DRAW", "AWAY_WIN")

TIMESTAMPTZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    postgresql.ENUM(*MODEL_STATUS_VALUES, name="model_status").create(
        op.get_bind(), checkfirst=True
    )
    postgresql.ENUM(*OUTCOME_VALUES, name="outcome").create(op.get_bind(), checkfirst=True)
    model_status = postgresql.ENUM(*MODEL_STATUS_VALUES, name="model_status", create_type=False)
    outcome = postgresql.ENUM(*OUTCOME_VALUES, name="outcome", create_type=False)

    op.create_table(
        "team_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("alias", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name="fk_team_aliases_team_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "alias", name="uq_team_aliases_source_alias"),
    )
    op.create_index("ix_team_aliases_id", "team_aliases", ["id"])
    op.create_index("ix_team_aliases_team_id", "team_aliases", ["team_id"])

    op.create_table(
        "data_imports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("season_start_year", sa.Integer(), nullable=True),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("rows_read", sa.Integer(), nullable=False),
        sa.Column("rows_accepted", sa.Integer(), nullable=False),
        sa.Column("rows_rejected", sa.Integer(), nullable=False),
        sa.Column("imported_at", TIMESTAMPTZ, server_default=sa.text("now()"), nullable=False),
        sa.Column("report_json", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_data_imports_id", "data_imports", ["id"])
    op.create_index("ix_data_imports_season_start_year", "data_imports", ["season_start_year"])

    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=20), nullable=False),
        sa.Column("calculated_at", TIMESTAMPTZ, server_default=sa.text("now()"), nullable=False),
        sa.Column("features_json", postgresql.JSONB(), nullable=False),
        sa.Column("source_cutoff_at", TIMESTAMPTZ, nullable=False),
        sa.ForeignKeyConstraint(
            ["fixture_id"],
            ["fixtures.id"],
            name="fk_feature_snapshots_fixture_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feature_snapshots_id", "feature_snapshots", ["id"])
    op.create_index("ix_feature_snapshots_fixture_id", "feature_snapshots", ["fixture_id"])
    op.create_index(
        "ix_feature_snapshots_feature_schema_version",
        "feature_snapshots",
        ["feature_schema_version"],
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=20), nullable=False),
        sa.Column("adapter_type", sa.String(length=50), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=20), nullable=False),
        sa.Column("trained_from", sa.Date(), nullable=True),
        sa.Column("trained_until", sa.Date(), nullable=True),
        sa.Column("validated_from", sa.Date(), nullable=True),
        sa.Column("validated_until", sa.Date(), nullable=True),
        sa.Column("tested_from", sa.Date(), nullable=True),
        sa.Column("tested_until", sa.Date(), nullable=True),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=False),
        sa.Column("license_summary", sa.Text(), nullable=True),
        sa.Column("status", model_status, nullable=False),
        sa.Column("created_at", TIMESTAMPTZ, server_default=sa.text("now()"), nullable=False),
        sa.Column("activated_at", TIMESTAMPTZ, nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_name", "version", name="uq_model_versions_name_version"),
    )
    op.create_index("ix_model_versions_id", "model_versions", ["id"])
    op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"])
    op.create_index("ix_model_versions_status", "model_versions", ["status"])
    # Promotion moves the ACTIVE pointer; the partial unique index makes it
    # impossible for two models to be active at once.
    op.create_index(
        "uq_model_versions_single_active",
        "model_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'ACTIVE'"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("feature_snapshot_id", sa.Integer(), nullable=False),
        sa.Column("home_win_probability", sa.Double(), nullable=False),
        sa.Column("draw_probability", sa.Double(), nullable=False),
        sa.Column("away_win_probability", sa.Double(), nullable=False),
        sa.Column("predicted_outcome", outcome, nullable=False),
        sa.Column("created_at", TIMESTAMPTZ, server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["fixtures.id"], name="fk_predictions_fixture_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"], ["model_versions.id"], name="fk_predictions_model_version_id"
        ),
        sa.ForeignKeyConstraint(
            ["feature_snapshot_id"],
            ["feature_snapshots.id"],
            name="fk_predictions_feature_snapshot_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "fixture_id", "model_version_id", name="uq_predictions_fixture_model_version"
        ),
        sa.CheckConstraint(
            "home_win_probability BETWEEN 0 AND 1", name="ck_predictions_home_win_probability_range"
        ),
        sa.CheckConstraint(
            "draw_probability BETWEEN 0 AND 1", name="ck_predictions_draw_probability_range"
        ),
        sa.CheckConstraint(
            "away_win_probability BETWEEN 0 AND 1", name="ck_predictions_away_win_probability_range"
        ),
        sa.CheckConstraint(
            "abs(home_win_probability + draw_probability + away_win_probability - 1) < 1e-6",
            name="ck_predictions_probabilities_sum_to_one",
        ),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"])
    op.create_index("ix_predictions_fixture_id", "predictions", ["fixture_id"])
    op.create_index("ix_predictions_model_version_id", "predictions", ["model_version_id"])


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("model_versions")
    op.drop_table("feature_snapshots")
    op.drop_table("data_imports")
    op.drop_table("team_aliases")
    postgresql.ENUM(name="outcome").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="model_status").drop(op.get_bind(), checkfirst=True)
