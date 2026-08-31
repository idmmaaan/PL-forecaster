import pytest
from httpx import AsyncClient
from app.main import app
from datetime import datetime

@pytest.mark.asyncio
async def test_predict_valid_fixture():
    """Test POST /api/v1/fixtures/{fixture_id}/predict with valid fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test with a known fixture ID (should be 14621 from our temporary repo)
        response = await ac.post("/api/v1/fixtures/14621/predict")
        
        # Should return HTTP 200
        assert response.status_code == 200
        
        # Parse the response
        prediction = response.json()
        
        # Check all required fields are present
        assert "prediction_id" in prediction
        assert "fixture_id" in prediction
        assert "home_team" in prediction
        assert "away_team" in prediction
        assert "predicted_outcome" in prediction
        assert "probabilities" in prediction
        assert "model_name" in prediction
        assert "model_version" in prediction
        assert "feature_schema_version" in prediction
        assert "created_at" in prediction
        
        # Check specific values
        assert prediction["fixture_id"] == 14621
        assert prediction["home_team"] == "Arsenal"
        assert prediction["away_team"] == "Chelsea"
        
        # Check probabilities are valid (0.0 to 1.0)
        probs = prediction["probabilities"]
        assert probs["home_win"] == 0.40
        assert probs["draw"] == 0.30
        assert probs["away_win"] == 0.30
        
        # Check probabilities sum to 1.0
        total_prob = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert abs(total_prob - 1.0) < 0.0001
        
        # Check predicted outcome is correct (highest probability)
        assert prediction["predicted_outcome"] == "HOME_WIN"  # Since home_win=0.40 > draw=0.30 and away_win=0.30

@pytest.mark.asyncio
async def test_predict_unknown_fixture():
    """Test POST /api/v1/fixtures/{fixture_id}/predict with unknown fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test with a non-existent fixture ID
        response = await ac.post("/api/v1/fixtures/99999/predict")
        
        # Should return HTTP 404
        assert response.status_code == 404
        
        # Check error message is present
        error_data = response.json()
        assert "detail" in error_data

@pytest.mark.asyncio
async def test_predict_with_features():
    """Test POST /api/v1/fixtures/{fixture_id}/predict with features parameter"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test with features provided
        response = await ac.post("/api/v1/fixtures/14621/predict", json={"some_feature": 1.0})
        
        # Should return HTTP 200 
        assert response.status_code == 200
        
        # Parse the response
        prediction = response.json()
        
        # Check all required fields are present (same as before)
        assert "prediction_id" in prediction
        assert "fixture_id" in prediction
        assert "home_team" in prediction
        assert "away_team" in prediction
        assert "predicted_outcome" in prediction
        assert "probabilities" in prediction
        assert "model_name" in prediction
        assert "model_version" in prediction
        assert "feature_schema_version" in prediction
        assert "created_at" in prediction
        
        # Check that probabilities are still the fixed values (dummy predictor ignores features)
        probs = prediction["probabilities"]
        assert probs["home_win"] == 0.40
        assert probs["draw"] == 0.30
        assert probs["away_win"] == 0.30

@pytest.mark.asyncio
async def test_predict_probability_values():
    """Test that prediction returns correct probability values"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/v1/fixtures/14621/predict")
        prediction = response.json()
        
        probs = prediction["probabilities"]
        
        # Check all probabilities are within valid range
        assert 0.0 <= probs["home_win"] <= 1.0
        assert 0.0 <= probs["draw"] <= 1.0  
        assert 0.0 <= probs["away_win"] <= 1.0
        
        # Check specific values match requirements
        assert probs["home_win"] == 0.40
        assert probs["draw"] == 0.30
        assert probs["away_win"] == 0.30

@pytest.mark.asyncio 
async def test_predict_predicted_class():
    """Test that predicted outcome is correctly determined"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/v1/fixtures/14621/predict")
        prediction = response.json()
        
        # With fixed probabilities 0.40, 0.30, 0.30, HOME_WIN should be predicted
        assert prediction["predicted_outcome"] == "HOME_WIN"
        
        # Test with different probabilities (if we had a more complex implementation)
        # For now we know the dummy returns 0.40 for home_win and 0.30 for others

@pytest.mark.asyncio
async def test_predict_response_structure():
    """Test that prediction response has correct structure"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/v1/fixtures/14621/predict")
        
        assert response.status_code == 200
        prediction = response.json()
        
        # Check that all expected fields are present with correct types
        assert isinstance(prediction["prediction_id"], int)
        assert isinstance(prediction["fixture_id"], int)
        assert isinstance(prediction["home_team"], str)
        assert isinstance(prediction["away_team"], str)
        assert isinstance(prediction["predicted_outcome"], str)
        assert isinstance(prediction["probabilities"], dict)
        assert isinstance(prediction["model_name"], str)  
        assert isinstance(prediction["model_version"], str)
        assert isinstance(prediction["feature_schema_version"], str)
        assert isinstance(prediction["created_at"], str)