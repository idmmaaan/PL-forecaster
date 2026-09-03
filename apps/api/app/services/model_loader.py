"""Resolves a registry row into a live `Predictor` instance.

The mapping from `adapter_type` to an implementation is the only place the
application knows about concrete model libraries. Adding a candidate model means
adding one entry here plus its optional dependency group.
"""

from pathlib import Path

from app.core.config import REPO_ROOT
from app.core.exceptions import PredictorUnavailableError
from app.models.model_version import ModelVersion
from epl_predictor.predictors.base import Predictor
from epl_predictor.predictors.dummy import DummyPredictor

ADAPTERS: dict[str, type[Predictor]] = {
    "dummy": DummyPredictor,
}


def resolve_artifact_path(artifact_path: str) -> Path:
    """Resolve a registry artifact path, treating relative paths as repo-relative."""
    path = Path(artifact_path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_predictor(model_version: ModelVersion) -> Predictor:
    """Load the artifact named by a registry row.

    Raises:
        PredictorUnavailableError: the adapter type is unknown or the artifact
            is missing. The API reports this rather than serving invented
            probabilities from a fallback model.
    """
    adapter = ADAPTERS.get(model_version.adapter_type)
    if adapter is None:
        raise PredictorUnavailableError(
            f"Unknown adapter type '{model_version.adapter_type}' for "
            f"{model_version.full_version}. Known adapters: {sorted(ADAPTERS)}"
        )

    artifact_path = resolve_artifact_path(model_version.artifact_path)
    try:
        return adapter.load(artifact_path)
    except (OSError, ValueError) as exc:
        raise PredictorUnavailableError(
            f"Could not load artifact for {model_version.full_version} from {artifact_path}: {exc}"
        ) from exc
