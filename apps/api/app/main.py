from fastapi import FastAPI
from app.core.config import settings
from app.core.database import Base, engine
from app.api.fixture_routes import router as fixture_router
from app.api.prediction_routes import router as prediction_router
from app.repositories.fixture_repository import TemporaryFixtureRepository

# Create database tables (for development only)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include routers
app.include_router(fixture_router)
app.include_router(prediction_router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}