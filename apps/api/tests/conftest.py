"""Shared test fixtures for the API test suite.

Route and service tests run against an in-memory repository so they need no
database. Repository tests need real SQL and therefore a live PostgreSQL; they
are skipped with a clear message when one is not reachable.
"""

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Importing the models package registers every table on Base.metadata.
import app.models  # noqa: F401
from app.api.deps import get_fixture_repository, get_prediction_repository
from app.core.config import settings
from app.core.database import Base
from app.main import app as fastapi_app
from app.repositories.in_memory_fixture_repository import InMemoryFixtureRepository
from app.repositories.in_memory_prediction_repository import InMemoryPredictionRepository


@pytest.fixture
def fixture_repo() -> InMemoryFixtureRepository:
    return InMemoryFixtureRepository()


@pytest.fixture
def prediction_repo(
    fixture_repo: InMemoryFixtureRepository,
) -> InMemoryPredictionRepository:
    return InMemoryPredictionRepository(fixture_repo)


@pytest.fixture
def api_app(
    fixture_repo: InMemoryFixtureRepository,
    prediction_repo: InMemoryPredictionRepository,
) -> Iterator[FastAPI]:
    """The real application with only its persistence dependencies overridden.

    Model resolution is left intact, so the tests exercise the same
    active-model lookup and artifact loading that production uses.
    """
    fastapi_app.dependency_overrides[get_fixture_repository] = lambda: fixture_repo
    fastapi_app.dependency_overrides[get_prediction_repository] = lambda: prediction_repo
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
async def client(api_app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=api_app), base_url="http://test"
    ) as async_client:
        yield async_client


@pytest.fixture(scope="session")
def database_url() -> str:
    """Connection string for a dedicated test database.

    Defaults to the development database name with a `_test` suffix, so the
    suite never reads or writes rows that `make seed` created.
    """
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit

    url = make_url(settings.database_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def database_engine(database_url: str) -> Iterator[Engine]:
    """Engine for the test database, created on first use.

    Skips the whole suite of database tests when PostgreSQL is unreachable, so
    `make test` still works without Docker running.
    """
    url = make_url(database_url)
    admin_engine = create_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True
    )
    try:
        with admin_engine.connect() as connection:
            already_exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database}
            ).scalar()
            if not already_exists:
                connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    except SQLAlchemyError as exc:
        pytest.skip(f"PostgreSQL not reachable at {url.set(password=None)}: {exc}")
    finally:
        admin_engine.dispose()

    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine, checkfirst=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(database_engine: Engine) -> Iterator[Session]:
    """A session wrapped in a transaction that is always rolled back."""
    connection = database_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
