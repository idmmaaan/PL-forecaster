from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.models.fixture import Fixture


class FeatureSnapshot(Base):
    """The exact feature vector used for one prediction.

    Storing the vector rather than recomputing it later is what makes a
    prediction auditable: `source_cutoff_at` records the latest information
    timestamp that was allowed in, which is how leakage is proven after the fact.
    """

    __tablename__ = "feature_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), index=True
    )
    feature_schema_version: Mapped[str] = mapped_column(String(20), index=True)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    features_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    #: No input to `features_json` may postdate this timestamp.
    source_cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    fixture: Mapped[Fixture] = relationship()
