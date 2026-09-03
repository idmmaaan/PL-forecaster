"""Tests for resolving a registry row into a live predictor.

The behaviour that matters here is refusal. A registry row pointing at a
missing artifact, an unknown adapter, or a feature schema that disagrees with
the artifact must produce an error, because every alternative ends with the
API serving numbers that look like predictions but are not.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.core.exceptions import FeaturesUnavailableError, PredictorUnavailableError
from app.models.fixture import Fixture
from app.models.model_version import ModelStatus, ModelVersion
from app.models.team import Team
from app.services.feature_service import (
    FIXTURE_METADATA_SCHEMA_VERSION,
    ArtifactFeatureBuilder,
    FixtureMetadataFeatureBuilder,
)
from app.services.model_loader import (
    ADAPTERS,
    clear_cache,
    load_model,
    resolve_artifact_path,
)
from epl_predictor import FEATURE_SCHEMA_VERSION
from epl_predictor.features.builder import build_training_table
from epl_predictor.features.context import FeatureContext
from epl_predictor.predictors.class_frequency import ClassFrequencyPredictor
from epl_predictor.predictors.dummy import DummyPredictor


@pytest.fixture(autouse=True)
def _isolate_artifact_cache() -> Any:
    """Keep one test's loaded artifact out of the next test's cache."""
    clear_cache()
    yield
    clear_cache()


def synthetic_matches(rows: int = 60) -> pd.DataFrame:
    """A small chronological match table, enough to fit a frequency baseline."""
    clubs = ["Arsenal", "Chelsea", "Liverpool", "Manchester City"]
    records = []
    for index in range(rows):
        home = clubs[index % len(clubs)]
        away = clubs[(index + 1 + index // len(clubs)) % len(clubs)]
        if home == away:
            away = clubs[(index + 2) % len(clubs)]
        home_goals, away_goals = (2, 0) if index % 3 else (1, 1)
        records.append(
            {
                "match_id": f"m{index}",
                "kickoff_date": date(2024, 1, 1) + timedelta(days=index * 3),
                "season_start_year": 2023,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": "H" if home_goals > away_goals else "D",
                "home_shots_on_target": 5,
                "away_shots_on_target": 3,
            }
        )
    frame = pd.DataFrame(records)
    frame["kickoff_date"] = pd.to_datetime(frame["kickoff_date"]).dt.date
    return frame


@pytest.fixture
def trained_artifact(tmp_path: Path) -> Path:
    """A complete artifact: fitted model, plus the league state to feed it."""
    matches = synthetic_matches()
    table, state = build_training_table(matches)

    predictor = ClassFrequencyPredictor()
    predictor.fit(table)

    artifact_path = tmp_path / "class-frequency-epl-0.0.1"
    predictor.save(artifact_path)
    FeatureContext.from_matches(state, matches).save(artifact_path)
    return artifact_path


def registry_row(
    artifact_path: Path,
    adapter_type: str = "class-frequency",
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> ModelVersion:
    return ModelVersion(
        id=1,
        model_name="class-frequency-epl",
        version="0.0.1",
        adapter_type=adapter_type,
        artifact_path=str(artifact_path),
        feature_schema_version=feature_schema_version,
        status=ModelStatus.ACTIVE,
    )


def fixture_for(home: str, away: str, kickoff: datetime | None = None) -> Fixture:
    """A transient fixture with its clubs attached, as the repository returns."""
    fixture = Fixture(
        id=1,
        provider="football-data.org",
        provider_id="pl-test-1",
        competition_code="PL",
        season_start_year=2024,
        matchday=5,
        kickoff_at=kickoff or datetime(2024, 9, 14, 14, 0, tzinfo=UTC),
        home_team_id=1,
        away_team_id=2,
    )
    fixture.home_team = Team(id=1, provider_id="t1", canonical_name=home)
    fixture.away_team = Team(id=2, provider_id="t2", canonical_name=away)
    return fixture


class TestAdapterRegistry:
    def test_every_mandatory_baseline_has_an_adapter(self) -> None:
        """The README's baselines must all be loadable by the API."""
        assert {"dummy", "class-frequency", "logistic-regression", "catboost"} <= set(ADAPTERS)

    def test_an_unknown_adapter_type_is_refused(self, trained_artifact: Path) -> None:
        with pytest.raises(PredictorUnavailableError, match="Unknown adapter type"):
            load_model(registry_row(trained_artifact, adapter_type="tabicl-v9"))

    def test_a_missing_artifact_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(PredictorUnavailableError, match="Could not load artifact"):
            load_model(registry_row(tmp_path / "was-never-trained"))

    def test_relative_artifact_paths_resolve_against_the_repo(self) -> None:
        resolved = resolve_artifact_path("ml/artifacts/example")
        assert resolved.is_absolute()
        assert resolved.parts[-3:] == ("ml", "artifacts", "example")

    def test_absolute_artifact_paths_are_left_alone(self, tmp_path: Path) -> None:
        assert resolve_artifact_path(str(tmp_path)) == tmp_path


class TestFeatureBuilderSelection:
    def test_a_trained_model_gets_the_league_state_from_its_artifact(
        self, trained_artifact: Path
    ) -> None:
        loaded = load_model(registry_row(trained_artifact))

        assert isinstance(loaded.feature_builder, ArtifactFeatureBuilder)
        assert loaded.feature_builder.feature_schema_version == FEATURE_SCHEMA_VERSION

    def test_a_feature_free_model_needs_no_league_state(self, tmp_path: Path) -> None:
        """The stub predictor has no feature context and must still load."""
        artifact_path = tmp_path / "dummy-epl-0.1.0"
        DummyPredictor().save(artifact_path)

        loaded = load_model(
            ModelVersion(
                id=2,
                model_name="dummy-epl",
                version="0.1.0",
                adapter_type="dummy",
                artifact_path=str(artifact_path),
                feature_schema_version=FIXTURE_METADATA_SCHEMA_VERSION,
                status=ModelStatus.ACTIVE,
            )
        )

        assert isinstance(loaded.feature_builder, FixtureMetadataFeatureBuilder)

    def test_a_trained_model_without_league_state_is_refused(self, tmp_path: Path) -> None:
        """A model cannot be served features it was not fitted on."""
        artifact_path = tmp_path / "class-frequency-epl-0.0.1"
        predictor = ClassFrequencyPredictor()
        predictor.fit(build_training_table(synthetic_matches())[0])
        predictor.save(artifact_path)
        # Deliberately no FeatureContext saved alongside it.

        with pytest.raises(PredictorUnavailableError, match="no usable feature"):
            load_model(registry_row(artifact_path))

    def test_a_schema_disagreement_between_row_and_artifact_is_refused(
        self, trained_artifact: Path
    ) -> None:
        with pytest.raises(PredictorUnavailableError, match="feature schema"):
            load_model(registry_row(trained_artifact, feature_schema_version="0.9.0"))


class TestArtifactCaching:
    def test_a_second_load_reuses_the_deserialised_model(self, trained_artifact: Path) -> None:
        """Deserialising per request would dominate prediction latency."""
        first = load_model(registry_row(trained_artifact))
        second = load_model(registry_row(trained_artifact))

        assert first.predictor is second.predictor
        assert first.feature_builder is second.feature_builder

    def test_clearing_the_cache_forces_a_reload(self, trained_artifact: Path) -> None:
        first = load_model(registry_row(trained_artifact))
        clear_cache()
        second = load_model(registry_row(trained_artifact))

        assert first.predictor is not second.predictor

    def test_a_new_version_of_the_same_model_is_loaded_separately(
        self, trained_artifact: Path
    ) -> None:
        """Promotion writes a new row, which must not hit the old row's entry."""
        first = load_model(registry_row(trained_artifact))
        newer = registry_row(trained_artifact)
        newer.version = "0.0.2"

        assert load_model(newer).predictor is not first.predictor


class TestArtifactFeatureBuilder:
    def test_features_match_the_schema_the_model_was_trained_on(
        self, trained_artifact: Path
    ) -> None:
        loaded = load_model(registry_row(trained_artifact))

        vector = loaded.feature_builder.build(fixture_for("Arsenal", "Chelsea"))

        assert vector.feature_schema_version == FEATURE_SCHEMA_VERSION
        assert vector.features["home_team"] == "Arsenal"
        assert vector.features["away_team"] == "Chelsea"
        assert vector.features["home_elo"] is not None

    def test_the_snapshot_records_how_stale_the_league_state_was(
        self, trained_artifact: Path
    ) -> None:
        """Serving a fixture months after the last result is legitimate but
        worth recording, so a later reviewer can see it."""
        loaded = load_model(registry_row(trained_artifact))

        vector = loaded.feature_builder.build(fixture_for("Arsenal", "Chelsea"))

        assert vector.features["feature_context_staleness_days"] > 0
        assert vector.features["feature_context_last_match_date"] is not None

    def test_provider_style_club_names_are_normalised(self, trained_artifact: Path) -> None:
        """The database stores provider spellings; the model knows canonical ones."""
        loaded = load_model(registry_row(trained_artifact))

        vector = loaded.feature_builder.build(fixture_for("Arsenal FC", "Chelsea FC"))

        assert vector.features["home_team"] == "Arsenal"
        assert vector.features["away_team"] == "Chelsea"

    def test_an_unmapped_club_is_refused_rather_than_guessed(self, trained_artifact: Path) -> None:
        loaded = load_model(registry_row(trained_artifact))

        with pytest.raises(FeaturesUnavailableError, match="alias table"):
            loaded.feature_builder.build(fixture_for("Real Madrid", "Chelsea"))

    def test_the_cutoff_is_the_kickoff_time(self, trained_artifact: Path) -> None:
        """Recording the cutoff is what makes a later leakage audit possible."""
        kickoff = datetime(2024, 9, 14, 14, 0, tzinfo=UTC)
        loaded = load_model(registry_row(trained_artifact))

        vector = loaded.feature_builder.build(fixture_for("Arsenal", "Chelsea", kickoff))

        assert vector.source_cutoff_at == kickoff

    def test_overrides_are_recorded_alongside_the_values_they_replace(
        self, trained_artifact: Path
    ) -> None:
        loaded = load_model(registry_row(trained_artifact))

        vector = loaded.feature_builder.build(
            fixture_for("Arsenal", "Chelsea"), {"home_elo": 1900.0}
        )

        assert vector.features["home_elo"] == 1900.0
        assert vector.features["overrides"] == {"home_elo": 1900.0}

    def test_a_prediction_can_be_produced_from_the_built_features(
        self, trained_artifact: Path
    ) -> None:
        """The whole point of pairing the two: the model accepts what the
        builder produces, with no adaptation in between."""
        loaded = load_model(registry_row(trained_artifact))

        vector = loaded.feature_builder.build(fixture_for("Arsenal", "Chelsea"))
        probabilities = loaded.predictor.predict_proba(vector.features)

        assert set(probabilities) == {"home_win", "draw", "away_win"}
        assert sum(probabilities.values()) == pytest.approx(1.0)


class TestFixtureMetadataFeatureBuilder:
    def test_only_pre_kickoff_metadata_is_described(self) -> None:
        builder = FixtureMetadataFeatureBuilder()

        vector = builder.build(fixture_for("Arsenal", "Chelsea"))

        assert vector.features == {
            "season_start_year": 2024,
            "matchday": 5,
            "home_team_id": 1,
            "away_team_id": 2,
            "kickoff_hour_utc": 14,
            "kickoff_day_of_week": 6,
        }

    def test_a_completed_fixture_yields_its_pre_kickoff_vector(self) -> None:
        """Scores must never reach the feature vector, even once known."""
        fixture = fixture_for("Arsenal", "Chelsea")
        before = FixtureMetadataFeatureBuilder().build(fixture)

        fixture.home_goals = 4
        fixture.away_goals = 0
        fixture.result = "H"

        assert FixtureMetadataFeatureBuilder().build(fixture).features == before.features

    def test_a_naive_kickoff_is_read_as_utc(self) -> None:
        """Feature values must not depend on the server's locale."""
        fixture = fixture_for("Arsenal", "Chelsea", datetime(2024, 9, 14, 14, 0))

        vector = FixtureMetadataFeatureBuilder().build(fixture)

        assert vector.features["kickoff_hour_utc"] == 14
        assert vector.source_cutoff_at.tzinfo is UTC
