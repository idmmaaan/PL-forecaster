"""Builds the feature vector for a fixture that has not kicked off.

Only facts that are known before kickoff may appear here. `source_cutoff_at` is
set to the kickoff time and recorded alongside the vector, which is what later
makes a leakage audit possible.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.models.fixture import Fixture
from epl_predictor import FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class FeatureVector:
    """A feature vector plus the provenance needed to audit it."""

    features: dict[str, Any]
    feature_schema_version: str
    source_cutoff_at: datetime


def build_feature_vector(
    fixture: Fixture, overrides: dict[str, Any] | None = None
) -> FeatureVector:
    """Return the feature vector for `fixture`.

    Args:
        fixture: The fixture to describe. Its score and result are deliberately
            never read, so a completed match yields the same vector it would
            have yielded before kickoff.
        overrides: Caller-supplied values merged on top, for experimentation.
            Recorded in the snapshot so a prediction made with overrides is
            distinguishable from one made from stored data alone.
    """
    kickoff_at = _as_utc(fixture.kickoff_at)

    features: dict[str, Any] = {
        "season_start_year": fixture.season_start_year,
        "matchday": fixture.matchday,
        "home_team_id": fixture.home_team_id,
        "away_team_id": fixture.away_team_id,
        "kickoff_hour_utc": kickoff_at.hour,
        "kickoff_day_of_week": kickoff_at.isoweekday(),
    }

    if overrides:
        features["overrides"] = dict(overrides)
        features.update(overrides)

    return FeatureVector(
        features=features,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        source_cutoff_at=kickoff_at,
    )


def _as_utc(value: datetime) -> datetime:
    """Treat a naive timestamp as UTC so feature values never depend on locale."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
