"""Domain errors raised by services and translated to HTTP status codes in routes."""


class FixtureNotFoundError(ValueError):
    """The requested fixture id does not exist."""


class FixtureNotPredictableError(ValueError):
    """The fixture exists but product rules forbid predicting it (e.g. already played)."""


class PredictorUnavailableError(RuntimeError):
    """No active model artifact could be loaded.

    Surfaced as a clear API error: the application must never fall back to
    invented probabilities when the model is missing.
    """


class FeaturesUnavailableError(RuntimeError):
    """A fixture could not be described in the schema the active model expects.

    Usually an unmapped club name. Raised instead of substituting defaults,
    because a model fed neutral features returns a confident-looking answer
    based on nothing.
    """


class InvalidPredictionError(RuntimeError):
    """A predictor returned probabilities that violate the output contract.

    Raised instead of silently renormalising, so a broken model is visible
    rather than hidden behind plausible-looking numbers.
    """
