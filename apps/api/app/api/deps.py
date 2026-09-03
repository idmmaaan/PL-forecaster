"""Shared FastAPI dependencies.

Routes depend on these providers rather than constructing repositories or
predictors themselves, which is what lets the test suite swap in in-memory
repositories through `app.dependency_overrides`.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import PredictorUnavailableError
from app.models.model_version import ModelVersion
from app.repositories.fixture_interface import FixtureRepository
from app.repositories.fixture_repository import SQLAlchemyFixtureRepository
from app.repositories.prediction_interface import PredictionRepository
from app.repositories.prediction_repository import SQLAlchemyPredictionRepository
from app.services.model_loader import LoadedModel, load_model
from app.services.predictor_service import PredictorService
from epl_predictor.predictors.base import Predictor


def get_fixture_repository(db: Session = Depends(get_db)) -> FixtureRepository:
    return SQLAlchemyFixtureRepository(db)


def get_prediction_repository(db: Session = Depends(get_db)) -> PredictionRepository:
    return SQLAlchemyPredictionRepository(db)


def get_active_model_version(
    prediction_repo: PredictionRepository = Depends(get_prediction_repository),
) -> ModelVersion:
    """Return the promoted model, or fail loudly when none is registered."""
    model_version = prediction_repo.get_active_model_version()
    if model_version is None:
        raise PredictorUnavailableError(
            "No model version is promoted to ACTIVE. Register one with `make seed` "
            "for the stub predictor, or promote a trained model with `make promote`."
        )
    return model_version


def get_loaded_model(
    model_version: ModelVersion = Depends(get_active_model_version),
) -> LoadedModel:
    """Load the artifact belonging to the promoted model version."""
    return load_model(model_version)


def get_predictor(loaded: LoadedModel = Depends(get_loaded_model)) -> Predictor:
    """Return just the active predictor, for callers that need nothing else."""
    return loaded.predictor


def get_predictor_service(
    fixture_repo: FixtureRepository = Depends(get_fixture_repository),
    prediction_repo: PredictionRepository = Depends(get_prediction_repository),
    loaded: LoadedModel = Depends(get_loaded_model),
) -> PredictorService:
    return PredictorService(
        fixture_repo,
        prediction_repo,
        loaded.predictor,
        loaded.model_version,
        loaded.feature_builder,
    )
