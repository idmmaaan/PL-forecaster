"""Database-level guarantees for the registry and prediction tables.

These invariants are the reason the constraints exist: the application must not
be the only thing standing between a bug and a corrupt row.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    DataImport,
    FeatureSnapshot,
    Fixture,
    FixtureStatus,
    ModelStatus,
    ModelVersion,
    Prediction,
    Team,
    TeamAlias,
)
from epl_predictor import Outcome

NOW = datetime.now(UTC)


@pytest.fixture
def team(db_session: Session) -> Team:
    team = Team(provider_id="football-data-57", canonical_name="Arsenal")
    db_session.add(team)
    db_session.flush()
    return team


@pytest.fixture
def fixture(db_session: Session, team: Team) -> Fixture:
    away = Team(provider_id="football-data-61", canonical_name="Chelsea")
    db_session.add(away)
    db_session.flush()

    fixture = Fixture(
        provider="football-data.org",
        provider_id="pl-2026-14621",
        competition_code="PL",
        season_start_year=2026,
        matchday=4,
        kickoff_at=NOW + timedelta(days=1),
        status=FixtureStatus.SCHEDULED,
        home_team_id=team.id,
        away_team_id=away.id,
    )
    db_session.add(fixture)
    db_session.flush()
    return fixture


@pytest.fixture
def model_version(db_session: Session) -> ModelVersion:
    version = ModelVersion(
        model_name="dummy-epl",
        version="0.1.0",
        adapter_type="dummy",
        artifact_path="ml/artifacts/dummy-epl-0.1.0",
        feature_schema_version="1.0.0",
        status=ModelStatus.ACTIVE,
        metrics_json={"log_loss": 1.0986},
    )
    db_session.add(version)
    db_session.flush()
    return version


@pytest.fixture
def feature_snapshot(db_session: Session, fixture: Fixture) -> FeatureSnapshot:
    snapshot = FeatureSnapshot(
        fixture_id=fixture.id,
        feature_schema_version="1.0.0",
        features_json={"elo_difference": 42.0},
        source_cutoff_at=fixture.kickoff_at,
    )
    db_session.add(snapshot)
    db_session.flush()
    return snapshot


def make_prediction(
    fixture: Fixture,
    model_version: ModelVersion,
    feature_snapshot: FeatureSnapshot,
    home: float = 0.4,
    draw: float = 0.3,
    away: float = 0.3,
) -> Prediction:
    return Prediction(
        fixture_id=fixture.id,
        model_version_id=model_version.id,
        feature_snapshot_id=feature_snapshot.id,
        home_win_probability=home,
        draw_probability=draw,
        away_win_probability=away,
        predicted_outcome=Outcome.HOME_WIN,
    )


# --- team_aliases ---


def test_team_alias_resolves_a_source_spelling(db_session: Session, team: Team) -> None:
    db_session.add(TeamAlias(team_id=team.id, source="football-data.co.uk", alias="Arsenal"))
    db_session.flush()

    alias = db_session.query(TeamAlias).one()

    assert alias.team.canonical_name == "Arsenal"


def test_same_alias_may_exist_for_different_sources(db_session: Session, team: Team) -> None:
    db_session.add_all(
        [
            TeamAlias(team_id=team.id, source="football-data.co.uk", alias="Arsenal"),
            TeamAlias(team_id=team.id, source="football-data.org", alias="Arsenal"),
        ]
    )
    db_session.flush()

    assert db_session.query(TeamAlias).count() == 2


def test_one_alias_per_source_cannot_map_to_two_teams(db_session: Session, team: Team) -> None:
    """An ambiguous alias would silently corrupt every historical feature."""
    other = Team(provider_id="football-data-61", canonical_name="Chelsea")
    db_session.add(other)
    db_session.flush()

    db_session.add(TeamAlias(team_id=team.id, source="football-data.co.uk", alias="Arsenal"))
    db_session.flush()
    db_session.add(TeamAlias(team_id=other.id, source="football-data.co.uk", alias="Arsenal"))

    with pytest.raises(IntegrityError):
        db_session.flush()


# --- data_imports ---


def test_data_import_records_an_audit_trail(db_session: Session) -> None:
    db_session.add(
        DataImport(
            source="football-data.co.uk",
            source_uri="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
            season_start_year=2023,
            checksum="a" * 64,
            rows_read=380,
            rows_accepted=380,
            rows_rejected=0,
            report_json={"rejections": []},
        )
    )
    db_session.flush()

    record = db_session.query(DataImport).one()

    assert record.rows_read == record.rows_accepted
    assert record.imported_at is not None
    assert record.report_json == {"rejections": []}


# --- feature_snapshots ---


def test_feature_snapshot_stores_the_vector_and_its_cutoff(
    db_session: Session, feature_snapshot: FeatureSnapshot
) -> None:
    """The stored vector is what makes a past prediction auditable."""
    assert feature_snapshot.features_json == {"elo_difference": 42.0}
    assert feature_snapshot.source_cutoff_at is not None
    assert feature_snapshot.feature_schema_version == "1.0.0"


# --- model_versions ---


def test_model_name_and_version_are_unique(
    db_session: Session, model_version: ModelVersion
) -> None:
    db_session.add(
        ModelVersion(
            model_name=model_version.model_name,
            version=model_version.version,
            adapter_type="dummy",
            artifact_path="elsewhere",
            feature_schema_version="1.0.0",
            status=ModelStatus.CANDIDATE,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_only_one_model_can_be_active(db_session: Session, model_version: ModelVersion) -> None:
    """Two active models would make served predictions non-reproducible."""
    db_session.add(
        ModelVersion(
            model_name="catboost-epl",
            version="0.1.0",
            adapter_type="catboost",
            artifact_path="ml/artifacts/catboost-epl-0.1.0",
            feature_schema_version="1.0.0",
            status=ModelStatus.ACTIVE,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_many_candidates_may_coexist(db_session: Session, model_version: ModelVersion) -> None:
    db_session.add_all(
        [
            ModelVersion(
                model_name="catboost-epl",
                version="0.1.0",
                adapter_type="catboost",
                artifact_path="a",
                feature_schema_version="1.0.0",
                status=ModelStatus.CANDIDATE,
            ),
            ModelVersion(
                model_name="tabicl-epl",
                version="0.1.0",
                adapter_type="tabicl",
                artifact_path="b",
                feature_schema_version="1.0.0",
                status=ModelStatus.CANDIDATE,
            ),
        ]
    )
    db_session.flush()

    assert db_session.query(ModelVersion).filter(ModelVersion.status == "CANDIDATE").count() == 2


def test_model_version_full_version_label(model_version: ModelVersion) -> None:
    assert model_version.full_version == "dummy-epl-0.1.0"


# --- predictions ---


def test_prediction_round_trips_with_its_lineage(
    db_session: Session,
    fixture: Fixture,
    model_version: ModelVersion,
    feature_snapshot: FeatureSnapshot,
) -> None:
    db_session.add(make_prediction(fixture, model_version, feature_snapshot))
    db_session.flush()

    prediction = db_session.query(Prediction).one()

    assert prediction.probabilities == {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}
    assert prediction.predicted_outcome is Outcome.HOME_WIN
    assert prediction.model_version.full_version == "dummy-epl-0.1.0"
    assert prediction.feature_snapshot.features_json == {"elo_difference": 42.0}
    assert prediction.fixture.provider_id == "pl-2026-14621"


def test_one_prediction_per_fixture_and_model_version(
    db_session: Session,
    fixture: Fixture,
    model_version: ModelVersion,
    feature_snapshot: FeatureSnapshot,
) -> None:
    db_session.add(make_prediction(fixture, model_version, feature_snapshot))
    db_session.flush()
    db_session.add(make_prediction(fixture, model_version, feature_snapshot))

    with pytest.raises(IntegrityError):
        db_session.flush()


@pytest.mark.parametrize(
    ("home", "draw", "away"),
    [
        pytest.param(1.5, -0.3, -0.2, id="outside-unit-interval"),
        pytest.param(-0.1, 0.6, 0.5, id="negative-probability"),
    ],
)
def test_probabilities_must_lie_in_the_unit_interval(
    db_session: Session,
    fixture: Fixture,
    model_version: ModelVersion,
    feature_snapshot: FeatureSnapshot,
    home: float,
    draw: float,
    away: float,
) -> None:
    db_session.add(
        make_prediction(fixture, model_version, feature_snapshot, home=home, draw=draw, away=away)
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


@pytest.mark.parametrize(
    ("home", "draw", "away"),
    [
        pytest.param(0.4, 0.4, 0.4, id="sum-above-one"),
        pytest.param(0.1, 0.1, 0.1, id="sum-below-one"),
    ],
)
def test_probabilities_must_sum_to_one(
    db_session: Session,
    fixture: Fixture,
    model_version: ModelVersion,
    feature_snapshot: FeatureSnapshot,
    home: float,
    draw: float,
    away: float,
) -> None:
    db_session.add(
        make_prediction(fixture, model_version, feature_snapshot, home=home, draw=draw, away=away)
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_fixture_result_accepts_only_h_d_a_or_null(db_session: Session, fixture: Fixture) -> None:
    fixture.result = None
    db_session.flush()

    fixture.result = "D"
    db_session.flush()

    fixture.result = "W"
    with pytest.raises(IntegrityError):
        db_session.flush()
