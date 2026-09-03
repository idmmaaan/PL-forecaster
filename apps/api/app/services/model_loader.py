"""Resolves a registry row into a live predictor and its feature builder.

The mapping from `adapter_type` to an implementation is the only place the
application knows about concrete model libraries. Adding a candidate model
means adding one entry here plus its optional dependency group.

A predictor is always returned together with the feature builder that matches
it. Pairing them here rather than at the call site removes the possibility of
serving a model features from a different schema, which would produce
confident nonsense with no error anywhere.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import REPO_ROOT
from app.core.exceptions import PredictorUnavailableError
from app.models.model_version import ModelVersion
from app.services.feature_service import (
    ArtifactFeatureBuilder,
    FeatureBuilder,
    FixtureMetadataFeatureBuilder,
)
from epl_predictor.features.context import FeatureContext
from epl_predictor.predictors.base import Predictor
from epl_predictor.predictors.catboost import CatBoostPredictor
from epl_predictor.predictors.class_frequency import ClassFrequencyPredictor
from epl_predictor.predictors.dummy import DummyPredictor
from epl_predictor.predictors.logistic_regression import LogisticRegressionPredictor

logger = logging.getLogger(__name__)

ADAPTERS: dict[str, type[Predictor]] = {
    "dummy": DummyPredictor,
    "class-frequency": ClassFrequencyPredictor,
    "logistic-regression": LogisticRegressionPredictor,
    "catboost": CatBoostPredictor,
}

#: Adapters that ignore their input entirely, so they need no league state.
FEATURE_FREE_ADAPTERS = frozenset({"dummy"})


@dataclass(frozen=True)
class LoadedModel:
    """A predictor together with the feature builder it was trained against."""

    predictor: Predictor
    feature_builder: FeatureBuilder
    model_version: ModelVersion


#: Loaded artifacts, keyed by the registry identity of the row they came from.
_CACHE: dict[tuple[str, str, str], LoadedModel] = {}


def clear_cache() -> None:
    """Drop cached artifacts, so the next request reloads them from disk."""
    _CACHE.clear()


def resolve_artifact_path(artifact_path: str) -> Path:
    """Resolve a registry artifact path, treating relative paths as repo-relative."""
    path = Path(artifact_path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_model(model_version: ModelVersion) -> LoadedModel:
    """Load the artifact named by a registry row, with its feature builder.

    Results are cached per registry row, because deserialising a gradient
    boosted model and a full league state on every request would dominate
    prediction latency. Promotion writes a new row, so the cache cannot serve
    a stale model; `clear_cache` covers artifacts edited in place.

    Raises:
        PredictorUnavailableError: The adapter type is unknown, the artifact is
            missing, or a trained model's artifact has no feature context. The
            API reports this rather than serving invented probabilities from a
            fallback model.
    """
    cache_key = (model_version.model_name, model_version.version, model_version.artifact_path)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return LoadedModel(
            predictor=cached.predictor,
            feature_builder=cached.feature_builder,
            model_version=model_version,
        )

    adapter = ADAPTERS.get(model_version.adapter_type)
    if adapter is None:
        raise PredictorUnavailableError(
            f"Unknown adapter type '{model_version.adapter_type}' for "
            f"{model_version.full_version}. Known adapters: {sorted(ADAPTERS)}"
        )

    artifact_path = resolve_artifact_path(model_version.artifact_path)
    try:
        predictor = adapter.load(artifact_path)
    except (OSError, ValueError, KeyError) as exc:
        raise PredictorUnavailableError(
            f"Could not load artifact for {model_version.full_version} from {artifact_path}: {exc}"
        ) from exc

    loaded = LoadedModel(
        predictor=predictor,
        feature_builder=_load_feature_builder(model_version, artifact_path),
        model_version=model_version,
    )
    logger.info(
        "Loaded %s from %s (features %s)",
        model_version.full_version,
        artifact_path,
        loaded.feature_builder.feature_schema_version,
    )
    _CACHE[cache_key] = loaded
    return loaded


def _load_feature_builder(model_version: ModelVersion, artifact_path: Path) -> FeatureBuilder:
    """Choose the feature builder for an artifact.

    A trained model whose artifact has no league state cannot be served: it
    would receive a feature vector of a different shape than it was fitted on.
    """
    if model_version.adapter_type in FEATURE_FREE_ADAPTERS:
        return FixtureMetadataFeatureBuilder()

    try:
        context = FeatureContext.load(artifact_path)
    except (OSError, ValueError) as exc:
        raise PredictorUnavailableError(
            f"Artifact for {model_version.full_version} has no usable feature "
            f"context, so it cannot produce features: {exc}"
        ) from exc

    if context.feature_schema_version != model_version.feature_schema_version:
        raise PredictorUnavailableError(
            f"{model_version.full_version} is registered against feature schema "
            f"{model_version.feature_schema_version} but its artifact carries "
            f"{context.feature_schema_version}."
        )

    return ArtifactFeatureBuilder(context)


def load_predictor(model_version: ModelVersion) -> Predictor:
    """Load only the predictor, for callers that build features themselves."""
    return load_model(model_version).predictor
