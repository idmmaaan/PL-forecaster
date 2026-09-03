from sqlalchemy.orm import Session, joinedload

from app.models.feature_snapshot import FeatureSnapshot
from app.models.fixture import Fixture
from app.models.model_version import ModelStatus, ModelVersion
from app.models.prediction import Prediction
from app.repositories.prediction_interface import PredictionRecord, PredictionRepository


class SQLAlchemyPredictionRepository(PredictionRepository):
    """PostgreSQL-backed prediction, feature-snapshot, and registry access."""

    def __init__(self, db: Session):
        self.db = db

    def get_active_model_version(self) -> ModelVersion | None:
        return (
            self.db.query(ModelVersion)
            .filter(ModelVersion.status == ModelStatus.ACTIVE)
            .one_or_none()
        )

    def record_prediction(self, record: PredictionRecord) -> Prediction:
        snapshot = self._upsert_feature_snapshot(record)

        prediction = (
            self.db.query(Prediction)
            .filter(
                Prediction.fixture_id == record.fixture_id,
                Prediction.model_version_id == record.model_version_id,
            )
            .one_or_none()
        )

        if prediction is None:
            prediction = Prediction(
                fixture_id=record.fixture_id,
                model_version_id=record.model_version_id,
                feature_snapshot_id=snapshot.id,
            )
            self.db.add(prediction)
        else:
            # Re-predicting points the row at the newer snapshot rather than
            # accumulating a second row for the same fixture and model.
            prediction.feature_snapshot_id = snapshot.id

        prediction.home_win_probability = record.probabilities["home_win"]
        prediction.draw_probability = record.probabilities["draw"]
        prediction.away_win_probability = record.probabilities["away_win"]
        prediction.predicted_outcome = record.predicted_outcome

        self.db.commit()
        self.db.refresh(prediction)
        return prediction

    def _upsert_feature_snapshot(self, record: PredictionRecord) -> FeatureSnapshot:
        """Return the snapshot for this record, reusing an identical one.

        Re-predicting an unchanged fixture must not leave a trail of duplicate
        vectors; a genuinely different vector still gets its own row so the
        audit history stays complete.
        """
        existing = (
            self.db.query(FeatureSnapshot)
            .filter(
                FeatureSnapshot.fixture_id == record.fixture_id,
                FeatureSnapshot.feature_schema_version == record.feature_schema_version,
                FeatureSnapshot.features_json == record.features,
            )
            .order_by(FeatureSnapshot.id.desc())
            .first()
        )
        if existing is not None:
            return existing

        snapshot = FeatureSnapshot(
            fixture_id=record.fixture_id,
            feature_schema_version=record.feature_schema_version,
            features_json=record.features,
            source_cutoff_at=record.source_cutoff_at,
        )
        self.db.add(snapshot)
        self.db.flush()
        return snapshot

    def get_prediction_by_id(self, prediction_id: int) -> Prediction | None:
        return (
            self.db.query(Prediction)
            .options(
                joinedload(Prediction.fixture).joinedload(Fixture.home_team),
                joinedload(Prediction.fixture).joinedload(Fixture.away_team),
                joinedload(Prediction.model_version),
                joinedload(Prediction.feature_snapshot),
            )
            .filter(Prediction.id == prediction_id)
            .one_or_none()
        )
