import pytest
from pydantic import ValidationError
from app.schemas.prediction import PredictionResponse, Probabilities, Outcome

def test_probabilities_validation():
    """Test that probabilities are validated to be between 0 and 1"""
    
    # Valid probabilities
    valid_probs = Probabilities(home_win=0.5, draw=0.3, away_win=0.2)
    assert valid_probs.home_win == 0.5
    assert valid_probs.draw == 0.3  
    assert valid_probs.away_win == 0.2
    
    # Test boundary values
    boundary_probs = Probabilities(home_win=0.0, draw=1.0, away_win=0.5)
    assert boundary_probs.home_win == 0.0
    assert boundary_probs.draw == 1.0
    assert boundary_probs.away_win == 0.5
    
def test_probabilities_invalid_values():
    """Test that invalid probability values raise ValidationError"""
    
    # Test values below 0
    with pytest.raises(ValidationError):
        Probabilities(home_win=-0.1, draw=0.5, away_win=0.5)
        
    # Test values above 1  
    with pytest.raises(ValidationError):
        Probabilities(home_win=0.5, draw=1.5, away_win=0.5)
        
    # Test exact boundary values (should work)
    try:
        probs = Probabilities(home_win=0.0, draw=1.0, away_win=0.0)
        assert True  # If we get here without exception, it's valid
    except ValidationError:
        pytest.fail("Boundary values should be accepted")

def test_prediction_response_schema():
    """Test PredictionResponse schema validation"""
    
    prediction_data = {
        "prediction_id": 1042,
        "fixture_id": 14621,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "predicted_outcome": "HOME_WIN",  # This should be the enum value
        "probabilities": {
            "home_win": 0.51,
            "draw": 0.27,
            "away_win": 0.22
        },
        "model_name": "tabicl-epl",
        "model_version": "0.1.0",
        "feature_schema_version": "1.0.0",
        "created_at": "2026-08-31T15:00:00Z"
    }
    
    prediction = PredictionResponse(**prediction_data)
    assert prediction.prediction_id == 1042
    assert prediction.fixture_id == 14621
    assert prediction.home_team == "Arsenal"
    assert prediction.away_team == "Chelsea"
    assert prediction.predicted_outcome.value == "HOME_WIN"
    assert prediction.probabilities.home_win == 0.51
    assert prediction.probabilities.draw == 0.27
    assert prediction.probabilities.away_win == 0.22
    assert prediction.model_name == "tabicl-epl"
    assert prediction.model_version == "0.1.0"
    assert prediction.feature_schema_version == "1.0.0"
    assert prediction.created_at == "2026-08-31T15:00:00Z"

def test_prediction_response_with_boundary_probabilities():
    """Test PredictionResponse with boundary probability values"""
    
    prediction_data = {
        "prediction_id": 1042,
        "fixture_id": 14621,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "predicted_outcome": "DRAW",
        "probabilities": {
            "home_win": 0.0,
            "draw": 1.0,
            "away_win": 0.0
        },
        "model_name": "tabicl-epl",
        "model_version": "0.1.0",
        "feature_schema_version": "1.0.0", 
        "created_at": "2026-08-31T15:00:00Z"
    }
    
    prediction = PredictionResponse(**prediction_data)
    assert prediction.predicted_outcome.value == "DRAW"
    assert prediction.probabilities.home_win == 0.0
    assert prediction.probabilities.draw == 1.0
    assert prediction.probabilities.away_win == 0.0

def test_prediction_response_sum_validation():
    """Test that prediction response validates probability sums (not required by schema but good to check)"""
    
    # Test that probabilities sum correctly
    prediction_data = {
        "prediction_id": 1042,
        "fixture_id": 14621,
        "home_team": "Arsenal",
        "away_team": "Chelsea", 
        "predicted_outcome": "HOME_WIN",
        "probabilities": {
            "home_win": 0.51,
            "draw": 0.27,
            "away_win": 0.22
        },
        "model_name": "tabicl-epl",
        "model_version": "0.1.0",
        "feature_schema_version": "1.0.0",
        "created_at": "2026-08-31T15:00:00Z"
    }
    
    prediction = PredictionResponse(**prediction_data)
    total_prob = (prediction.probabilities.home_win + 
                  prediction.probabilities.draw + 
                  prediction.probabilities.away_win)
    # Note: We don't validate sum == 1 in schema, but this is good for testing
    assert abs(total_prob - 1.0) < 0.0001  # Allow for floating point precision