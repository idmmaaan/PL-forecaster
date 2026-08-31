import pytest
from pathlib import Path
from ml.src/epl_predictor.predictors.dummy import DummyPredictor

def test_dummy_predictor_initialization():
    """Test that DummyPredictor can be initialized with default and custom parameters"""
    
    # Test default initialization
    predictor = DummyPredictor()
    assert predictor.model_name == "dummy-epl"
    assert predictor.model_version == "0.1.0"
    
    # Test custom initialization  
    predictor = DummyPredictor("custom-dummy", "2.0.0")
    assert predictor.model_name == "custom-dummy"
    assert predictor.model_version == "2.0.0"

def test_dummy_predictor_deterministic_output():
    """Test that the dummy predictor returns deterministic probabilities"""
    
    predictor = DummyPredictor()
    
    # Test multiple calls return same results
    features1 = {"some_feature": 1.0}
    features2 = {"another_feature": 2.0}
    features3 = {"home_team": "Arsenal", "away_team": "Chelsea"}
    
    result1 = predictor.predict_proba(features1)
    result2 = predictor.predict_proba(features2) 
    result3 = predictor.predict_proba(features3)
    
    # All should return the same fixed probabilities
    expected = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
    
    assert result1 == expected
    assert result2 == expected
    assert result3 == expected

def test_dummy_predictor_probability_sum():
    """Test that probabilities sum to 1.0"""
    
    predictor = DummyPredictor()
    features = {"test_feature": 1.0}
    probabilities = predictor.predict_proba(features)
    
    total = probabilities["home_win"] + probabilities["draw"] + probabilities["away_win"]
    assert abs(total - 1.0) < 0.0001  # Allow for floating point precision

def test_dummy_predictor_all_classes_present():
    """Test that all three outcome classes are present in the output"""
    
    predictor = DummyPredictor()
    features = {"test": 1.0}
    probabilities = predictor.predict_proba(features)
    
    assert "home_win" in probabilities
    assert "draw" in probabilities
    assert "away_win" in probabilities
    
    # Test that they have numeric values
    assert isinstance(probabilities["home_win"], (int, float))
    assert isinstance(probabilities["draw"], (int, float))  
    assert isinstance(probabilities["away_win"], (int, float))

def test_dummy_predictor_save_load():
    """Test save and load methods work without errors"""
    
    predictor = DummyPredictor()
    
    # Test save method (should not raise error)
    predictor.save(Path("/tmp/test_dummy"))
    
    # Test load method returns proper instance
    loaded_predictor = DummyPredictor.load(Path("/tmp/test_dummy"))
    assert isinstance(loaded_predictor, DummyPredictor)
    assert loaded_predictor.model_name == "dummy-epl"
    assert loaded_predictor.model_version == "0.1.0"

def test_dummy_predictor_with_various_features():
    """Test that predictor works with various feature inputs"""
    
    predictor = DummyPredictor()
    
    # Test with different types of features
    test_cases = [
        {"empty": {}},
        {"simple": 1},
        {"complex": {"home_team": "Arsenal", "away_team": "Chelsea"}},
        {"numeric": [1, 2, 3]},
        {"string": "test_feature"}
    ]
    
    for case_name, features in test_cases.items():
        probabilities = predictor.predict_proba(features)
        
        # Should always return the same fixed values
        expected = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}
        assert probabilities == expected

def test_dummy_predictor_probability_ranges():
    """Test that all probability values are within valid range [0, 1]"""
    
    predictor = DummyPredictor()
    features = {"test": 1.0}
    probabilities = predictor.predict_proba(features)
    
    for key, value in probabilities.items():
        assert 0.0 <= value <= 1.0, f"Probability {key} ({value}) must be between 0 and 1"