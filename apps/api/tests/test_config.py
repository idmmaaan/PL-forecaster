import os
from app.core.config import settings

def test_default_values():
    """Test that configuration has appropriate default values"""
    assert settings.app_env == "development"
    assert settings.database_url == "postgresql+psycopg://epl:epl@localhost:5432/epl_predictor"
    assert settings.football_data_base_url == "https://api.football-data.org/v4"
    assert settings.model_artifact_path == "./ml/artifacts"

def test_environment_variable_override():
    """Test that environment variables can override defaults"""
    # Set environment variables
    os.environ["APP_ENV"] = "test"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"
    
    # Reload settings
    from app.core.config import Settings
    test_settings = Settings()
    
    assert test_settings.app_env == "test"
    assert test_settings.database_url == "postgresql://test:test@localhost:5432/test_db"
    
    # Clean up
    del os.environ["APP_ENV"]
    del os.environ["DATABASE_URL"]