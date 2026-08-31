from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from app.repositories.fixture_repository import TemporaryFixtureRepository
from app.services.predictor_service import PredictorService
from app.schemas.prediction import PredictionResponse
from ml.src/epl_predictor.predictors.dummy import DummyPredictor

router = APIRouter(prefix="/fixtures", tags=["predictions"])

# Initialize services 
dummy_predictor = DummyPredictor()
fixture_repo = TemporaryFixtureRepository()
predictor_service = PredictorService(fixture_repo, dummy_predictor)

@router.post("/{fixture_id}/predict", response_model=PredictionResponse)
async def predict_fixture(fixture_id: int, features: Dict[str, Any] = None):
    """
    Generate prediction for a specific fixture
    
    Args:
        fixture_id: ID of the fixture to predict
        features: Optional dictionary of input features
        
    Returns:
        PredictionResponse object with probabilities and outcome
    """
    try:
        if features is None:
            features = {}
            
        prediction = predictor_service.predict(fixture_id, features)
        return prediction
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))