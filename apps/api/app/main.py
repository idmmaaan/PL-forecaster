import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.fixture_routes import router as fixture_router
from app.api.prediction_routes import router as prediction_router
from app.core.config import settings
from app.core.exceptions import (
    FeaturesUnavailableError,
    FixtureNotFoundError,
    FixtureNotPredictableError,
    InvalidPredictionError,
    PredictorUnavailableError,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EPL AI Match Predictor",
    version="0.1.0",
    description=(
        "Predicts English Premier League match outcomes as home win, draw, and "
        "away win probabilities. Predictions are probabilistic and are not betting advice."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The database schema is owned by Alembic; the app never issues DDL. Run
# `alembic upgrade head` before starting the API.
app.include_router(fixture_router, prefix=settings.api_v1_prefix)
app.include_router(prediction_router, prefix=settings.api_v1_prefix)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(FixtureNotFoundError)
async def handle_fixture_not_found(_: Request, exc: FixtureNotFoundError) -> JSONResponse:
    return _error_response(404, str(exc))


@app.exception_handler(FixtureNotPredictableError)
async def handle_fixture_not_predictable(
    _: Request, exc: FixtureNotPredictableError
) -> JSONResponse:
    return _error_response(409, str(exc))


@app.exception_handler(PredictorUnavailableError)
async def handle_predictor_unavailable(_: Request, exc: PredictorUnavailableError) -> JSONResponse:
    logger.error("Active predictor unavailable: %s", exc)
    return _error_response(503, f"Prediction model unavailable: {exc}")


@app.exception_handler(FeaturesUnavailableError)
async def handle_features_unavailable(_: Request, exc: FeaturesUnavailableError) -> JSONResponse:
    logger.error("Could not build features: %s", exc)
    return _error_response(422, f"Fixture cannot be described for this model: {exc}")


@app.exception_handler(InvalidPredictionError)
async def handle_invalid_prediction(_: Request, exc: InvalidPredictionError) -> JSONResponse:
    # A model that breaks the probability contract is reported, never patched up.
    logger.error("Predictor violated the output contract: %s", exc)
    return _error_response(502, f"Prediction rejected by output validation: {exc}")


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
