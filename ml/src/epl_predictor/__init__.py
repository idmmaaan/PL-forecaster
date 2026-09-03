"""EPL match-outcome prediction: data ingest, features, predictors, evaluation, registry.

This package is the offline/ML half of the project. It is imported by the FastAPI
application only through the :class:`~epl_predictor.predictors.base.Predictor`
interface, so the application never depends on a specific model library.
"""

from enum import StrEnum

__version__ = "0.1.0"

FEATURE_SCHEMA_VERSION = "1.0.0"


class Outcome(StrEnum):
    """The three match outcomes. Single source of truth for the API and the DB."""

    HOME_WIN = "HOME_WIN"
    DRAW = "DRAW"
    AWAY_WIN = "AWAY_WIN"


#: Canonical class order. Every predictor, artifact, and metric report must use
#: this order so that probability vectors are never silently permuted.
OUTCOME_CLASSES: tuple[str, ...] = tuple(outcome.value for outcome in Outcome)

#: Mapping from historical source labels (Football-Data.co.uk `FTR`) to application labels.
RESULT_TO_OUTCOME = {"H": Outcome.HOME_WIN, "D": Outcome.DRAW, "A": Outcome.AWAY_WIN}

#: Probability keys used in the prediction contract, in canonical class order.
PROBABILITY_KEYS = ("home_win", "draw", "away_win")

#: Probability key for each outcome, so the two representations cannot drift.
OUTCOME_TO_PROBABILITY_KEY = dict(zip(Outcome, PROBABILITY_KEYS, strict=True))
PROBABILITY_KEY_TO_OUTCOME = {key: outcome for outcome, key in OUTCOME_TO_PROBABILITY_KEY.items()}

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "OUTCOME_CLASSES",
    "OUTCOME_TO_PROBABILITY_KEY",
    "PROBABILITY_KEYS",
    "PROBABILITY_KEY_TO_OUTCOME",
    "RESULT_TO_OUTCOME",
    "Outcome",
    "__version__",
]
