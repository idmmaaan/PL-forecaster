from pathlib import Path
from typing import Dict, Any
from .base import Predictor

class DummyPredictor(Predictor):
    """A dummy predictor that returns fixed probabilities for testing and development"""
    
    def __init__(self, name: str = "dummy-epl", version: str = "0.1.0"):
        self._model_name = name
        self._model_version = version
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property 
    def model_version(self) -> str:
        return self._model_version
    
    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Return fixed probabilities for all fixtures.
        
        Returns:
            Dictionary with keys 'home_win', 'draw', 'away_win' mapping to probabilities
        """
        # Fixed deterministic probabilities - no randomness
        return {
            "home_win": 0.40,
            "draw": 0.30,
            "away_win": 0.30
        }
    
    def save(self, path: Path) -> None:
        """Save the dummy predictor (no-op for this simple implementation)"""
        # Dummy predictor doesn't need to be saved as it's stateless
        pass
        
    @classmethod
    def load(cls, path: Path) -> "Predictor":
        """
        Load a dummy predictor (returns new instance since it's stateless)
        
        Args:
            path: Directory or file path from which to load the model
            
        Returns:
            New DummyPredictor instance
        """
        return cls()