"""ORM models.

Importing this package registers every table on `Base.metadata`, which is what
Alembic autogenerate and the test schema helpers rely on.
"""

from app.models.data_import import DataImport
from app.models.feature_snapshot import FeatureSnapshot
from app.models.fixture import PREDICTABLE_STATUSES, Fixture, FixtureStatus
from app.models.model_version import ModelStatus, ModelVersion
from app.models.prediction import Prediction
from app.models.team import Team
from app.models.team_alias import TeamAlias

__all__ = [
    "PREDICTABLE_STATUSES",
    "DataImport",
    "FeatureSnapshot",
    "Fixture",
    "FixtureStatus",
    "ModelStatus",
    "ModelVersion",
    "Prediction",
    "Team",
    "TeamAlias",
]
