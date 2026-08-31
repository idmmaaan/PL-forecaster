from typing import Dict, Any
from app.repositories.fixture_interface import FixtureRepository
from ml.src/epl_predictor.predictors.dummy import DummyPredictor
from app.schemas.prediction import PredictionResponse
from datetime import datetime

class PredictorService:
    """Service class for handling prediction operations"""
    
    def __init__(self, fixture_repo: FixtureRepository, predictor: DummyPredictor):
        self.fixture_repo = fixture_repo
        self.predictor = predictor
    
    def predict(self, fixture_id: int, features: Dict[str, Any]) -> PredictionResponse:
        """
        Generate a prediction for a specific fixture
        
        Args:
            fixture_id: ID of the fixture to predict
            features: Dictionary of input features
            
        Returns:
            PredictionResponse object with probabilities and outcome
            
        Raises:
            ValueError: If fixture is not found or invalid
        """
        # Get the fixture 
        fixture = self.fixture_repo.get_fixture_by_id(fixture_id)
        if not fixture:
            raise ValueError(f"Fixture with ID {fixture_id} not found")
        
        # Generate prediction using dummy predictor
        probabilities = self.predictor.predict_proba(features)
        
        # Determine predicted outcome (class with highest probability)
        predicted_outcome = max(probabilities.keys(), key=lambda k: probabilities[k])
        
        # Create and return prediction response
        prediction_response = PredictionResponse(
            prediction_id=1,  # Placeholder ID - would be generated in real implementation  
            fixture_id=fixture.id,
            home_team=fixture.home_team.name if hasattr(fixture, 'home_team') else "Unknown",
            away_team=fixture.away_team.name if hasattr(fixture, 'away_team') else "Unknown",
            predicted_outcome=predicted_outcome,
            probabilities=probabilities,
            model_name=self.predictor.model_name,
            model_version=self.predictor.model_version,
            feature_schema_version="1.0.0",  # Placeholder version
            created_at=datetime.now().isoformat()
        )
        
        return prediction_response