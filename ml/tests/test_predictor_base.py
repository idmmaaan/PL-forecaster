import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from ml.src/epl_predictor.predictors.base import Predictor

class MockPredictor(Predictor):
    """Concrete implementation for testing the base class"""
    
    def __init__(self, name: str = "test-model", version: str = "0.1.0"):
        self._model_name = name
        self._model_version = version
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property 
    def model_version(self) -> str:
        return self._model_version
    
    def predict_proba(self, features: dict) -> dict:
        # Mock implementation - just returns fixed probabilities
        return {
            "home_win": 0.4,
            "draw": 0.3,
            "away_win": 0.3
        }
    
    def save(self, path: Path) -> None:
        # Mock implementation - just creates a marker file
        (path / "model_saved").touch()
        
    @classmethod
    def load(cls, path: Path) -> "Predictor":
        # Mock implementation - returns new instance 
        return cls()

def test_predictor_base_class_interface():
    """Test that the base class defines all required methods and properties"""
    
    # Test that we can't instantiate the abstract class directly
    with pytest.raises(TypeError):
        Predictor()
        
    # Test that concrete implementations work correctly
    mock_predictor = MockPredictor("test-model", "1.0.0")
    
    assert mock_predictor.model_name == "test-model"
    assert mock_predictor.model_version == "1.0.0"
    
    # Test predict_proba method
    features = {"some_feature": 1.0}
    probabilities = mock_predictor.predict_proba(features)
    assert isinstance(probabilities, dict)
    assert "home_win" in probabilities
    assert "draw" in probabilities  
    assert "away_win" in probabilities
    
    # Test save and load methods (they should not raise errors)
    with patch('pathlib.Path') as mock_path:
        mock_path.return_value.__truediv__.return_value = Mock()
        mock_path.return_value.exists.return_value = True
        mock_predictor.save(Path("/tmp/test"))
        
    # Test that the class method works  
    loaded_predictor = MockPredictor.load(Path("/tmp/test"))
    assert isinstance(loaded_predictor, MockPredictor)

def test_predictor_properties():
    """Test predictor property access"""
    
    predictor = MockPredictor("sample-model", "2.5.1")
    
    assert predictor.model_name == "sample-model"
    assert predictor.model_version == "2.5.1"

def test_predictor_predict_proba_signature():
    """Test that predict_proba accepts features and returns proper structure"""
    
    predictor = MockPredictor()
    features = {
        "home_team": "Arsenal",
        "away_team": "Chelsea", 
        "elo_difference": 150.0
    }
    
    result = predictor.predict_proba(features)
    
    assert isinstance(result, dict)
    assert "home_win" in result
    assert "draw" in result
    assert "away_win" in result
    assert all(isinstance(prob, (int, float)) for prob in result.values())