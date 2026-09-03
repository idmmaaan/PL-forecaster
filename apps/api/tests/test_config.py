import pytest

from app.core.config import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove settings variables so defaults are observable."""
    for name in (
        "APP_ENV",
        "DATABASE_URL",
        "FOOTBALL_DATA_API_TOKEN",
        "FOOTBALL_DATA_BASE_URL",
        "MODEL_ARTIFACT_PATH",
        "CORS_ALLOW_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_values(clean_env: None) -> None:
    """Defaults are read without a .env file so the test does not depend on one."""
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.database_url == "postgresql+psycopg://epl:epl@localhost:5432/epl_predictor"
    assert settings.football_data_base_url == "https://api.football-data.org/v4"
    assert settings.model_artifact_path == "./ml/artifacts"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.football_data_api_token is None


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upper-case environment variables populate the lower-case fields."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test_db")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.database_url == "postgresql+psycopg://test:test@localhost:5432/test_db"


def test_cors_origins_are_split_on_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://a.test, http://b.test ,")

    settings = Settings(_env_file=None)

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_artifact_root_is_absolute(clean_env: None) -> None:
    """A relative artifact path resolves against the repo root, not the cwd."""
    settings = Settings(_env_file=None)

    assert settings.artifact_root.is_absolute()
    assert settings.artifact_root.name == "artifacts"
