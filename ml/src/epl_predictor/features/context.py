"""League state packaged for inference.

Training walks the whole match history to build features. The API cannot do
that per request, and must not: rebuilding state from whatever happens to be
in the database would make a prediction depend on how complete that database
was at the time.

So the state left standing at the end of training is saved into the model
artifact. The API loads it and describes an upcoming fixture through exactly
the same code path the training rows went through, which is what keeps
training and serving features identical.

The context records the date of the last match it saw. A fixture far beyond
that date is still predictable, but its form and Elo are stale, so the
staleness is reported rather than hidden and the retraining runbook uses it.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from epl_predictor.features.builder import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    build_fixture_features,
)
from epl_predictor.features.state import LeagueState

CONTEXT_FILENAME = "feature_context.joblib"


@dataclass
class FeatureContext:
    """A `LeagueState` plus the provenance needed to trust features from it."""

    state: LeagueState
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    last_match_date: date | None = None
    last_season_start_year: int | None = None
    matches_seen: int = 0

    @classmethod
    def from_matches(cls, state: LeagueState, matches: pd.DataFrame) -> "FeatureContext":
        """Capture the state produced by building features over `matches`."""
        if matches.empty:
            return cls(state=state)

        last = pd.to_datetime(matches["kickoff_date"]).max()
        return cls(
            state=state,
            last_match_date=last.date(),
            last_season_start_year=int(matches["season_start_year"].max()),
            matches_seen=len(matches),
        )

    def staleness_days(self, kickoff_date: date) -> int | None:
        """Days between the last match seen and the fixture being predicted."""
        if self.last_match_date is None:
            return None
        return (kickoff_date - self.last_match_date).days

    def build_features(
        self,
        home_team: str,
        away_team: str,
        kickoff_date: Any,
        season_start_year: int | None = None,
    ) -> dict[str, Any]:
        """Describe an upcoming fixture, without recording it.

        Raises:
            ValueError: The season cannot be determined.
        """
        kickoff = pd.Timestamp(kickoff_date).date()
        season = season_start_year if season_start_year is not None else _infer_season(kickoff)

        features = build_fixture_features(self.state, home_team, away_team, kickoff, season)
        missing = set(FEATURE_COLUMNS) - set(features)
        if missing:
            raise ValueError(f"Feature vector is missing {sorted(missing)}")
        return features

    def metadata(self) -> dict[str, Any]:
        """Provenance to record in the artifact and in a feature snapshot."""
        return {
            "feature_schema_version": self.feature_schema_version,
            "last_match_date": self.last_match_date.isoformat() if self.last_match_date else None,
            "last_season_start_year": self.last_season_start_year,
            "matches_seen": self.matches_seen,
            "clubs_known": len(self.state.elo.as_dict()),
        }

    def save(self, artifact_path: Path) -> Path:
        """Write the context into a model artifact directory."""
        artifact_path = Path(artifact_path)
        artifact_path.mkdir(parents=True, exist_ok=True)
        target = artifact_path / CONTEXT_FILENAME
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, artifact_path: Path) -> "FeatureContext":
        """Read the context from a model artifact directory.

        Raises:
            FileNotFoundError: The artifact has no feature context.
        """
        target = Path(artifact_path) / CONTEXT_FILENAME
        if not target.exists():
            raise FileNotFoundError(
                f"No {CONTEXT_FILENAME} in {artifact_path}. The artifact cannot "
                "produce features, so it must not be used to serve predictions."
            )
        context: FeatureContext = joblib.load(target)
        return context


def _infer_season(kickoff: date) -> int:
    """Season start year for a date, using the usual July season boundary.

    A Premier League season runs August to May, so anything from July onwards
    belongs to the season starting that calendar year.
    """
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1
