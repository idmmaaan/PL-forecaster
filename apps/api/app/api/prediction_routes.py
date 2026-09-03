from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from app.api.deps import get_prediction_repository, get_predictor_service
from app.repositories.prediction_interface import PredictionRepository
from app.schemas.prediction import PredictionResponse
from app.services.predictor_service import PredictorService

router = APIRouter(tags=["predictions"])


@router.post("/fixtures/{fixture_id}/predict", response_model=PredictionResponse)
async def predict_fixture(
    fixture_id: int,
    features: dict[str, Any] | None = Body(default=None),
    predictor_service: PredictorService = Depends(get_predictor_service),
) -> PredictionResponse:
    """Predict home win, draw, and away win probabilities for a fixture.

    The feature snapshot and the resulting prediction are both stored, so the
    returned `prediction_id` can be fetched again later. An optional request
    body supplies feature overrides for experimentation.
    """
    return predictor_service.predict(fixture_id, features or {})


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: int,
    prediction_repo: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionResponse:
    """Return a previously stored prediction."""
    prediction = prediction_repo.get_prediction_by_id(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Prediction with ID {prediction_id} not found")
    return PredictionResponse.from_prediction(prediction)
