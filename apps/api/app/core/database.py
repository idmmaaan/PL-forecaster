from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model and by Alembic autogenerate."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine, created on first use.

    Building the engine lazily keeps `import app.main` free of database side
    effects, so the app and its tests can be imported without a running server.
    """
    return create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped database session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
