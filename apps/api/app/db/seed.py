"""Load development data into PostgreSQL.

Run with `make seed` (or `python -m app.db.seed`). Idempotent: repeated runs
converge on the same rows instead of duplicating them, so it is safe to re-run
after a migration.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.database import get_session_factory
from app.db.seed_data import SEED_FIXTURES, SEED_TEAMS
from app.models.fixture import FixtureStatus
from app.models.model_version import ModelStatus, ModelVersion
from app.models.team import Team
from app.models.team_alias import TeamAlias
from app.repositories.fixture_repository import SQLAlchemyFixtureRepository
from app.services.feature_service import FIXTURE_METADATA_SCHEMA_VERSION
from app.services.model_loader import resolve_artifact_path

logger = logging.getLogger(__name__)

STUB_MODEL = {
    "model_name": "dummy-epl",
    "version": "0.1.0",
    "adapter_type": "dummy",
    "artifact_path": "ml/artifacts/dummy-epl-0.1.0",
    # The stub ignores its input, so it is registered against the metadata
    # vector rather than claiming the v1 feature schema it never saw.
    "feature_schema_version": FIXTURE_METADATA_SCHEMA_VERSION,
    "license_summary": "Project-internal stub predictor; no third-party weights.",
}

#: Football-Data.co.uk spellings for the seeded clubs, so historical CSV rows
#: resolve to the same `teams` rows the API serves.
HISTORICAL_ALIASES = {
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
}
HISTORICAL_SOURCE = "football-data.co.uk"


def seed_teams_and_fixtures(session: Session) -> tuple[int, int]:
    """Upsert the seeded teams and fixtures, returning how many of each."""
    repo = SQLAlchemyFixtureRepository(session)

    # Provider ids are the natural keys; the hardcoded ids in the seed data are
    # only used to link fixtures to teams, not written to the database.
    team_ids: dict[int, int] = {}
    for team_data in SEED_TEAMS:
        seed_id = team_data["id"]
        team = repo.upsert_team({key: value for key, value in team_data.items() if key != "id"})
        team_ids[seed_id] = team.id

    for fixture_data in SEED_FIXTURES:
        payload = {key: value for key, value in fixture_data.items() if key != "id"}
        payload["home_team_id"] = team_ids[fixture_data["home_team_id"]]
        payload["away_team_id"] = team_ids[fixture_data["away_team_id"]]
        payload["status"] = FixtureStatus.SCHEDULED
        repo.upsert_fixture(payload)

    return len(SEED_TEAMS), len(SEED_FIXTURES)


def seed_team_aliases(session: Session) -> int:
    """Register the historical-source spelling for each seeded team."""
    created = 0
    for canonical_name, alias in HISTORICAL_ALIASES.items():
        team = session.query(Team).filter(Team.canonical_name == canonical_name).one_or_none()
        if team is None:
            continue

        exists = (
            session.query(TeamAlias)
            .filter(
                TeamAlias.source == HISTORICAL_SOURCE,
                TeamAlias.alias == alias,
            )
            .one_or_none()
        )
        if exists is None:
            session.add(TeamAlias(team_id=team.id, source=HISTORICAL_SOURCE, alias=alias))
            created += 1
        elif exists.team_id != team.id:
            exists.team_id = team.id

    session.commit()
    return created


def seed_stub_model_version(session: Session) -> ModelVersion:
    """Register and activate the stub predictor if no model is active yet.

    An existing ACTIVE model is left alone: seeding must never silently demote
    a trained model that somebody promoted on purpose.
    """
    existing = (
        session.query(ModelVersion)
        .filter(
            ModelVersion.model_name == STUB_MODEL["model_name"],
            ModelVersion.version == STUB_MODEL["version"],
        )
        .one_or_none()
    )
    if existing is None:
        existing = ModelVersion(**STUB_MODEL, status=ModelStatus.CANDIDATE, metrics_json={})
        session.add(existing)
        session.flush()

    active = (
        session.query(ModelVersion).filter(ModelVersion.status == ModelStatus.ACTIVE).one_or_none()
    )
    if active is None:
        existing.status = ModelStatus.ACTIVE
        existing.activated_at = datetime.now(UTC)
    elif active.id != existing.id:
        logger.info("Leaving %s active; not demoting it to seed the stub.", active.full_version)

    session.commit()
    session.refresh(existing)
    return existing


def write_stub_artifact() -> None:
    """Write the stub predictor's artifact so the registry path resolves."""
    from epl_predictor.predictors.dummy import DummyPredictor

    artifact_path = resolve_artifact_path(str(STUB_MODEL["artifact_path"]))
    DummyPredictor(str(STUB_MODEL["model_name"]), str(STUB_MODEL["version"])).save(artifact_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Seed fixtures only, leaving the model registry untouched.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with get_session_factory()() as session:
        teams, fixtures = seed_teams_and_fixtures(session)
        aliases = seed_team_aliases(session)
        logger.info("Seeded %d teams, %d fixtures, %d new aliases.", teams, fixtures, aliases)

        if not args.skip_model:
            write_stub_artifact()
            model_version = seed_stub_model_version(session)
            logger.info("Model %s is %s.", model_version.full_version, model_version.status.value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
