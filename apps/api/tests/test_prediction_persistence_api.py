"""End-to-end behaviour of prediction persistence and retrieval."""

from httpx import ASGITransport, AsyncClient

from app.api.deps import get_fixture_repository, get_prediction_repository
from app.main import app as fastapi_app
from app.repositories.in_memory_fixture_repository import InMemoryFixtureRepository
from app.repositories.in_memory_prediction_repository import InMemoryPredictionRepository


async def test_prediction_can_be_fetched_by_its_returned_id(client: AsyncClient) -> None:
    created = (await client.post("/api/v1/fixtures/14621/predict")).json()

    fetched = await client.get(f"/api/v1/predictions/{created['prediction_id']}")

    assert fetched.status_code == 200
    assert fetched.json() == created


async def test_prediction_id_is_not_the_fixture_id(client: AsyncClient) -> None:
    """The placeholder id is gone: predictions have their own identity."""
    prediction = (await client.post("/api/v1/fixtures/14621/predict")).json()

    assert prediction["fixture_id"] == 14621
    assert prediction["prediction_id"] != 14621


async def test_distinct_fixtures_get_distinct_prediction_ids(client: AsyncClient) -> None:
    first = (await client.post("/api/v1/fixtures/14621/predict")).json()
    second = (await client.post("/api/v1/fixtures/14622/predict")).json()

    assert first["prediction_id"] != second["prediction_id"]
    assert second["home_team"] == "Liverpool"


async def test_repredicting_reuses_the_same_prediction_row(client: AsyncClient) -> None:
    first = (await client.post("/api/v1/fixtures/14621/predict")).json()
    second = (await client.post("/api/v1/fixtures/14621/predict")).json()

    assert first["prediction_id"] == second["prediction_id"]


async def test_unknown_prediction_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/v1/predictions/424242")

    assert response.status_code == 404
    assert "detail" in response.json()


async def test_feature_overrides_are_recorded_in_the_snapshot(
    client: AsyncClient, prediction_repo: InMemoryPredictionRepository
) -> None:
    """A prediction made with overrides must be distinguishable afterwards."""
    created = (
        await client.post("/api/v1/fixtures/14621/predict", json={"elo_difference": 120.0})
    ).json()

    stored = prediction_repo.get_prediction_by_id(created["prediction_id"])

    assert stored is not None
    assert stored.feature_snapshot.features_json["overrides"] == {"elo_difference": 120.0}


async def test_prediction_fails_clearly_when_no_model_is_active() -> None:
    """With no promoted model the API must refuse, not invent probabilities."""
    fixture_repo = InMemoryFixtureRepository()
    fastapi_app.dependency_overrides[get_fixture_repository] = lambda: fixture_repo
    fastapi_app.dependency_overrides[get_prediction_repository] = lambda: (
        InMemoryPredictionRepository(fixture_repo, active=False)
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://test"
        ) as unpromoted_client:
            response = await unpromoted_client.post("/api/v1/fixtures/14621/predict")
    finally:
        fastapi_app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "ACTIVE" in response.json()["detail"]
