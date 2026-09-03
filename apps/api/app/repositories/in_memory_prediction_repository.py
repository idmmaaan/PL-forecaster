from datetime import UTC, datetime

from app.models.feature_snapshot import FeatureSnapshot
from app.models.model_version import ModelStatus, ModelVersion
from app.models.prediction import Prediction
from app.repositories.fixture_interface import FixtureRepository
from app.repositories.prediction_interface import PredictionRecord, PredictionRepository

#: The registry row the in-memory repository presents as ACTIVE. It mirrors what
#: `app.db.seed` writes to PostgreSQL, so tests exercise the same code path.
BOOTSTRAP_MODEL_VERSION = {
    "id": 1,
    "model_name": "dummy-epl",
    "version": "0.1.0",
    "adapter_type": "dummy",
    "artifact_path": "ml/artifacts/dummy-epl-0.1.0",
    "feature_schema_version": "1.0.0",
    "license_summary": "Project-internal stub predictor; no third-party weights.",
}


class InMemoryPredictionRepository(PredictionRepository):
    """Prediction storage with no database, for route and service tests."""

    def __init__(self, fixture_repo: FixtureRepository, active: bool = True):
        self._fixture_repo = fixture_repo
        self._predictions: dict[int, Prediction] = {}
        self._next_id = 1
        self._active_model_version: ModelVersion | None = None
        if active:
            self._active_model_version = ModelVersion(
                status=ModelStatus.ACTIVE,
                activated_at=datetime.now(UTC),
                metrics_json={},
                **BOOTSTRAP_MODEL_VERSION,
            )

    def get_active_model_version(self) -> ModelVersion | None:
        return self._active_model_version

    def _require_model_version(self, model_version_id: int) -> ModelVersion:
        """Stand in for the foreign key that PostgreSQL would enforce."""
        active = self._active_model_version
        if active is None or active.id != model_version_id:
            raise ValueError(f"No registered model version with id {model_version_id}")
        return active

    def record_prediction(self, record: PredictionRecord) -> Prediction:
        model_version = self._require_model_version(record.model_version_id)
        snapshot = FeatureSnapshot(
            id=self._next_id,
            fixture_id=record.fixture_id,
            feature_schema_version=record.feature_schema_version,
            features_json=record.features,
            source_cutoff_at=record.source_cutoff_at,
            calculated_at=datetime.now(UTC),
        )

        existing = next(
            (
                prediction
                for prediction in self._predictions.values()
                if prediction.fixture_id == record.fixture_id
                and prediction.model_version_id == record.model_version_id
            ),
            None,
        )
        prediction = existing or Prediction(
            id=self._next_id,
            fixture_id=record.fixture_id,
            model_version_id=record.model_version_id,
            created_at=datetime.now(UTC),
        )
        if existing is None:
            self._predictions[prediction.id] = prediction
            self._next_id += 1

        prediction.feature_snapshot_id = snapshot.id
        prediction.feature_snapshot = snapshot
        prediction.home_win_probability = record.probabilities["home_win"]
        prediction.draw_probability = record.probabilities["draw"]
        prediction.away_win_probability = record.probabilities["away_win"]
        prediction.predicted_outcome = record.predicted_outcome
        prediction.model_version = model_version
        # Transient rows have no session, so the fixture relationship is attached.
        fixture = self._fixture_repo.get_fixture_by_id(record.fixture_id)
        if fixture is not None:
            prediction.fixture = fixture

        return prediction

    def get_prediction_by_id(self, prediction_id: int) -> Prediction | None:
        return self._predictions.get(prediction_id)
