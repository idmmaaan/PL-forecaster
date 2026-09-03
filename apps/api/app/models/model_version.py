from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Date, DateTime, Enum, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class ModelStatus(StrEnum):
    """Lifecycle of a trained model artifact."""

    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class ModelVersion(Base):
    """Registry row for one trained artifact.

    Promotion only moves the `ACTIVE` status onto a different row; artifacts are
    never overwritten, so a rollback needs no retraining.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_versions_name_version"),
        # At most one ACTIVE model at a time, enforced by the database rather
        # than by convention in the promotion script.
        Index(
            "uq_model_versions_single_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'ACTIVE'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[str] = mapped_column(String(20))
    #: Which `Predictor` implementation can load `artifact_path`.
    adapter_type: Mapped[str] = mapped_column(String(50))
    artifact_path: Mapped[str] = mapped_column(Text)
    feature_schema_version: Mapped[str] = mapped_column(String(20))

    # Chronological split boundaries this model was produced with. Recorded so a
    # later reviewer can confirm the test period was never used for tuning.
    trained_from: Mapped[date | None] = mapped_column(Date)
    trained_until: Mapped[date | None] = mapped_column(Date)
    validated_from: Mapped[date | None] = mapped_column(Date)
    validated_until: Mapped[date | None] = mapped_column(Date)
    tested_from: Mapped[date | None] = mapped_column(Date)
    tested_until: Mapped[date | None] = mapped_column(Date)

    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    license_summary: Mapped[str | None] = mapped_column(Text)
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus, name="model_status"), default=ModelStatus.CANDIDATE, index=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @property
    def full_version(self) -> str:
        """The human-readable identifier, e.g. `catboost-epl-0.1.0`."""
        return f"{self.model_name}-{self.version}"
