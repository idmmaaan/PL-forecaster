from typing import Any

from app.core.exceptions import FixtureNotFoundError, InvalidPredictionError
from app.models.model_version import ModelVersion
from app.repositories.fixture_interface import FixtureRepository
from app.repositories.prediction_interface import PredictionRecord, PredictionRepository
from app.schemas.prediction import PredictionResponse
from app.services.feature_service import build_feature_vector
from epl_predictor import PROBABILITY_KEY_TO_OUTCOME, PROBABILITY_KEYS, Outcome
from epl_predictor.predictors.base import Predictor

#: Tolerance for the "probabilities sum to approximately 1.0" invariant.
PROBABILITY_SUM_TOLERANCE = 1e-6


class PredictorService:
    """Produces and persists predictions using the promoted model."""

    def __init__(
        self,
        fixture_repo: FixtureRepository,
        prediction_repo: PredictionRepository,
        predictor: Predictor,
        model_version: ModelVersion,
    ):
        self.fixture_repo = fixture_repo
        self.prediction_repo = prediction_repo
        self.predictor = predictor
        self.model_version = model_version

    def predict(self, fixture_id: int, features: dict[str, Any]) -> PredictionResponse:
        """Predict one fixture, storing the feature snapshot and the result.

        Raises:
            FixtureNotFoundError: the fixture id is unknown.
            InvalidPredictionError: the predictor broke the output contract.
        """
        fixture = self.fixture_repo.get_fixture_by_id(fixture_id)
        if fixture is None:
            raise FixtureNotFoundError(f"Fixture with ID {fixture_id} not found")

        feature_vector = build_feature_vector(fixture, features)
        probabilities = validate_probabilities(
            self.predictor.predict_proba(feature_vector.features)
        )

        prediction = self.prediction_repo.record_prediction(
            PredictionRecord(
                fixture_id=fixture.id,
                model_version_id=self.model_version.id,
                probabilities=probabilities,
                predicted_outcome=argmax_outcome(probabilities),
                features=feature_vector.features,
                feature_schema_version=feature_vector.feature_schema_version,
                source_cutoff_at=feature_vector.source_cutoff_at,
            )
        )

        return PredictionResponse.from_prediction(prediction)


def validate_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Check a predictor's output against the README's output invariants.

    Returns the probabilities unchanged when valid; never repairs them.
    """
    missing = set(PROBABILITY_KEYS) - probabilities.keys()
    if missing:
        raise InvalidPredictionError(f"Predictor omitted required probabilities: {sorted(missing)}")

    unexpected = probabilities.keys() - set(PROBABILITY_KEYS)
    if unexpected:
        raise InvalidPredictionError(
            f"Predictor returned unexpected probability keys: {sorted(unexpected)}"
        )

    for key in PROBABILITY_KEYS:
        value = probabilities[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise InvalidPredictionError(f"Probability '{key}' is not numeric: {value!r}")
        if not 0.0 <= value <= 1.0:
            raise InvalidPredictionError(f"Probability '{key}' is outside [0, 1]: {value}")

    total = sum(probabilities[key] for key in PROBABILITY_KEYS)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise InvalidPredictionError(f"Probabilities must sum to 1.0, got {total}")

    return probabilities


def argmax_outcome(probabilities: dict[str, float]) -> Outcome:
    """Return the outcome with the largest probability.

    Iterating in canonical class order makes an exact tie resolve
    deterministically to the earlier class rather than by dict ordering.
    """
    best_key = max(PROBABILITY_KEY_TO_OUTCOME, key=lambda key: probabilities[key])
    return PROBABILITY_KEY_TO_OUTCOME[best_key]
