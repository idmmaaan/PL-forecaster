from httpx import AsyncClient

from app.services.feature_service import FIXTURE_METADATA_SCHEMA_VERSION

# The stub predictor is deterministic, so these values are exact.
EXPECTED_PROBABILITIES = {"home_win": 0.40, "draw": 0.30, "away_win": 0.30}

REQUIRED_FIELDS = (
    "prediction_id",
    "fixture_id",
    "home_team",
    "away_team",
    "predicted_outcome",
    "probabilities",
    "model_name",
    "model_version",
    "feature_schema_version",
    "created_at",
)


async def test_predict_valid_fixture(client: AsyncClient) -> None:
    """POST /api/v1/fixtures/{id}/predict returns the full prediction contract."""
    response = await client.post("/api/v1/fixtures/14621/predict")

    assert response.status_code == 200
    prediction = response.json()

    for field in REQUIRED_FIELDS:
        assert field in prediction

    assert prediction["fixture_id"] == 14621
    assert prediction["home_team"] == "Arsenal"
    assert prediction["away_team"] == "Chelsea"
    assert prediction["probabilities"] == EXPECTED_PROBABILITIES
    assert prediction["predicted_outcome"] == "HOME_WIN"


async def test_predict_probabilities_sum_to_one(client: AsyncClient) -> None:
    response = await client.post("/api/v1/fixtures/14621/predict")
    probabilities = response.json()["probabilities"]

    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())


async def test_predict_unknown_fixture(client: AsyncClient) -> None:
    response = await client.post("/api/v1/fixtures/99999/predict")

    assert response.status_code == 404
    assert "detail" in response.json()


async def test_predict_with_features(client: AsyncClient) -> None:
    """A feature body is accepted; the stub predictor ignores its contents."""
    response = await client.post("/api/v1/fixtures/14621/predict", json={"elo_difference": 120.0})

    assert response.status_code == 200
    prediction = response.json()
    for field in REQUIRED_FIELDS:
        assert field in prediction
    assert prediction["probabilities"] == EXPECTED_PROBABILITIES


async def test_predict_reports_model_and_feature_versions(client: AsyncClient) -> None:
    """Every prediction must be traceable to a model and a feature schema."""
    response = await client.post("/api/v1/fixtures/14621/predict")
    prediction = response.json()

    assert prediction["model_name"] == "dummy-epl"
    assert prediction["model_version"] == "0.1.0"
    # The stub predictor is fed fixture metadata, not the v1 feature set, and
    # the snapshot must say so rather than claim a schema it never used.
    assert prediction["feature_schema_version"] == FIXTURE_METADATA_SCHEMA_VERSION


async def test_predict_response_types(client: AsyncClient) -> None:
    response = await client.post("/api/v1/fixtures/14621/predict")
    prediction = response.json()

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
