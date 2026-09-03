"""Builds the feature vector for a fixture that has not kicked off.

Two builders exist because two kinds of model exist. A feature-free model such
as the stub predictor needs only fixture metadata, while a trained model needs
the exact v1 feature schema it was fitted on, which can only come from the
league state saved inside its artifact.

Both record `source_cutoff_at`, the kickoff time, alongside the vector. Storing
the cutoff with every snapshot is what later makes a leakage audit possible:
any value in the vector must have been knowable before that moment.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from app.core.exceptions import FeaturesUnavailableError
from app.models.fixture import Fixture
from epl_predictor.data.teams import try_normalise_team_name
from epl_predictor.features.context import FeatureContext

#: Schema identifier for the metadata-only vector, which is not the v1 feature
#: set and must never be confused with it in a stored snapshot. Kept within the
#: 20 characters the `feature_schema_version` columns allow.
FIXTURE_METADATA_SCHEMA_VERSION = "fixture-meta-1.0.0"


@dataclass(frozen=True)
class FeatureVector:
    """A feature vector plus the provenance needed to audit it."""

    features: dict[str, Any]
    feature_schema_version: str
    source_cutoff_at: datetime


@runtime_checkable
class FeatureBuilder(Protocol):
    """Produces the feature vector a particular model expects."""

    feature_schema_version: str

    def build(self, fixture: Fixture, overrides: dict[str, Any] | None = None) -> FeatureVector:
        """Describe `fixture` using only pre-kickoff information."""


class FixtureMetadataFeatureBuilder:
    """Fixture metadata only, for models that ignore features.

    Used by the stub predictor so the full request path can be exercised
    before any model is trained.
    """

    feature_schema_version = FIXTURE_METADATA_SCHEMA_VERSION

    def build(self, fixture: Fixture, overrides: dict[str, Any] | None = None) -> FeatureVector:
        """Return metadata about the fixture.

        The fixture's score and result are deliberately never read, so a
        completed match yields the vector it would have yielded before kickoff.
        """
        kickoff_at = as_utc(fixture.kickoff_at)
        features: dict[str, Any] = {
            "season_start_year": fixture.season_start_year,
            "matchday": fixture.matchday,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "kickoff_hour_utc": kickoff_at.hour,
            "kickoff_day_of_week": kickoff_at.isoweekday(),
        }
        return _apply_overrides(features, overrides, self.feature_schema_version, kickoff_at)


class ArtifactFeatureBuilder:
    """The v1 feature schema, built from the league state in a model artifact.

    Training walked the entire match history to produce these features. The
    API reuses the state that walk left behind, through the same builder code,
    which is what keeps served features identical to trained ones.
    """

    def __init__(self, context: FeatureContext):
        self.context = context
        self.feature_schema_version = context.feature_schema_version

    def build(self, fixture: Fixture, overrides: dict[str, Any] | None = None) -> FeatureVector:
        """Describe `fixture` from the artifact's league state.

        Raises:
            FeaturesUnavailableError: A club's stored name is not in the alias
                table, so its history cannot be identified.
        """
        kickoff_at = as_utc(fixture.kickoff_at)
        home = self._canonical_name(fixture, "home")
        away = self._canonical_name(fixture, "away")

        features = self.context.build_features(
            home_team=home,
            away_team=away,
            kickoff_date=kickoff_at.date(),
            season_start_year=fixture.season_start_year,
        )
        features["feature_context_last_match_date"] = (
            self.context.last_match_date.isoformat() if self.context.last_match_date else None
        )
        features["feature_context_staleness_days"] = self.context.staleness_days(kickoff_at.date())

        return _apply_overrides(features, overrides, self.feature_schema_version, kickoff_at)

    def _canonical_name(self, fixture: Fixture, side: str) -> str:
        team = fixture.home_team if side == "home" else fixture.away_team
        if team is None:
            raise FeaturesUnavailableError(
                f"Fixture {fixture.id} has no {side} team loaded, so it cannot be described."
            )

        canonical = try_normalise_team_name(team.canonical_name)
        if canonical is None:
            raise FeaturesUnavailableError(
                f"Club {team.canonical_name!r} is not in the team alias table, so its "
                "match history cannot be identified. Add it to "
                "epl_predictor.data.teams rather than predicting without history."
            )
        return canonical


def _apply_overrides(
    features: dict[str, Any],
    overrides: dict[str, Any] | None,
    schema_version: str,
    kickoff_at: datetime,
) -> FeatureVector:
    """Merge caller overrides, recording that they were used.

    The overrides are stored under their own key as well as merged, so a
    prediction made with them can never be mistaken for one made purely from
    stored data.
    """
    if overrides:
        features = {**features, "overrides": dict(overrides), **overrides}

    return FeatureVector(
        features=features,
        feature_schema_version=schema_version,
        source_cutoff_at=kickoff_at,
    )


def as_utc(value: datetime) -> datetime:
    """Treat a naive timestamp as UTC so feature values never depend on locale."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
