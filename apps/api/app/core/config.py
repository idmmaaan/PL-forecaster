from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/app/core/config.py -> repo root is five levels up.
REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Application settings, populated from the environment and an optional .env file.

    Field names are lower-case and matched case-insensitively, so `DATABASE_URL`
    in the environment populates `database_url`.
    """

    model_config = SettingsConfigDict(
        # The repo-root .env is read first so the API behaves the same regardless
        # of the directory uvicorn was launched from; a local .env still wins.
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # `model_artifact_path` would otherwise collide with pydantic's
        # reserved `model_` namespace.
        protected_namespaces=(),
    )

    app_env: str = "development"
    database_url: str = "postgresql+psycopg://epl:epl@localhost:5432/epl_predictor"
    football_data_api_token: str | None = None
    football_data_base_url: str = "https://api.football-data.org/v4"
    model_artifact_path: str = "./ml/artifacts"

    api_v1_prefix: str = "/api/v1"

    #: Comma-separated browser origins allowed to call the API. Kept as a string
    #: rather than a list so plain env values do not need JSON quoting.
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @cached_property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @cached_property
    def artifact_root(self) -> Path:
        path = Path(self.model_artifact_path)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()


# Global settings instance used by the application at runtime. Tests that need
# deterministic values should construct `Settings(_env_file=None)` instead.
settings = Settings()
