from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

class Predictor(ABC):
    """Abstract base class for all predictors in the EPL AI Predictor system"""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of this predictor model"""
        pass
    
    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the version of this predictor model"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict probabilities for home win, draw, and away win
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Dictionary with keys 'home_win', 'draw', 'away_win' mapping to probabilities
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the predictor model to disk
        
        Args:
            path: Directory or file path where the model should be saved
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Predictor":
        """
        Load a predictor model from disk
        
        Args:
            path: Directory or file path from which to load the model
            
        Returns:
            Loaded predictor instance
        """
        pass