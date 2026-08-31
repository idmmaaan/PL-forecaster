from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    app_env: str = Field(default="development", env="APP_ENV")
    database_url: str = Field(default="postgresql+psycopg://epl:epl@localhost:5432/epl_predictor", env="DATABASE_URL")
    football_data_api_token: Optional[str] = Field(default=None, env="FOOTBALL_DATA_API_TOKEN")
    football_data_base_url: str = Field(default="https://api.football-data.org/v4", env="FOOTBALL_DATA_BASE_URL")
    model_artifact_path: str = Field(default="./ml/artifacts", env="MODEL_ARTIFACT_PATH")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create a global settings instance
settings = Settings()