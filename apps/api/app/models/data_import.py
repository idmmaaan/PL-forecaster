from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class DataImport(Base):
    """Audit record for one ingest run.

    The checksum plus the accepted/rejected counts make an import reproducible
    and let a later run detect that a source file has changed.
    """

    __tablename__ = "data_imports"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    source: Mapped[str] = mapped_column(String(100))  # e.g. "football-data.co.uk"
    source_uri: Mapped[str] = mapped_column(Text)
    season_start_year: Mapped[int | None] = mapped_column(index=True)
    checksum: Mapped[str] = mapped_column(String(64))  # SHA-256 of the raw payload
    rows_read: Mapped[int] = mapped_column(default=0)
    rows_accepted: Mapped[int] = mapped_column(default=0)
    rows_rejected: Mapped[int] = mapped_column(default=0)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    #: Per-row rejection reasons and column-validation findings.
    report_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
