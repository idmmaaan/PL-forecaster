from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.models.feature_snapshot import FeatureSnapshot
from app.models.fixture import Fixture
from app.models.model_version import ModelVersion
from epl_predictor import Outcome

#: Probabilities are validated in the service layer too; this is the backstop
#: that keeps invalid rows out of the table whatever the caller does.
PROBABILITY_CONSTRAINTS = tuple(
    CheckConstraint(f"{column} BETWEEN 0 AND 1", name=f"ck_predictions_{column}_range")
    for column in ("home_win_probability", "draw_probability", "away_win_probability")
)


class Prediction(Base):
    """One stored prediction, tied to the model and features that produced it."""

    __tablename__ = "predictions"
    __table_args__ = (
        # One canonical prediction per fixture per model version, so repeated
        # clicks on Predict do not accumulate duplicate rows.
        UniqueConstraint(
            "fixture_id", "model_version_id", name="uq_predictions_fixture_model_version"
        ),
        *PROBABILITY_CONSTRAINTS,
        CheckConstraint(
            "abs(home_win_probability + draw_probability + away_win_probability - 1) < 1e-6",
            name="ck_predictions_probabilities_sum_to_one",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    fixture_id: Mapped[int] = mapped_column(
        ForeignKey("fixtures.id", ondelete="CASCADE"), index=True
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    feature_snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("feature_snapshots.id", ondelete="CASCADE")
    )

    home_win_probability: Mapped[float] = mapped_column()
    draw_probability: Mapped[float] = mapped_column()
    away_win_probability: Mapped[float] = mapped_column()
    predicted_outcome: Mapped[Outcome] = mapped_column(Enum(Outcome, name="outcome"))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    fixture: Mapped[Fixture] = relationship()
    model_version: Mapped[ModelVersion] = relationship()
    feature_snapshot: Mapped[FeatureSnapshot] = relationship()

    @property
    def probabilities(self) -> dict[str, float]:
        """The stored probabilities in canonical key order."""
        return {
            "home_win": self.home_win_probability,
            "draw": self.draw_probability,
            "away_win": self.away_win_probability,
        }
