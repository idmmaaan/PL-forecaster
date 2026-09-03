"""Predictor adapters, all behind the common `Predictor` interface.

Foundation-model adapters are imported lazily through `ADAPTERS` so that the
default environment does not need torch or multi-gigabyte pretrained weights
just to load this package.
"""

from epl_predictor.predictors.base import Predictor, normalise_probabilities
from epl_predictor.predictors.class_frequency import ClassFrequencyPredictor
from epl_predictor.predictors.dummy import DummyPredictor
from epl_predictor.predictors.logistic_regression import LogisticRegressionPredictor
from epl_predictor.predictors.tabular import NotFittedError, TabularPredictor

__all__ = [
    "ClassFrequencyPredictor",
    "DummyPredictor",
    "LogisticRegressionPredictor",
    "NotFittedError",
    "Predictor",
    "TabularPredictor",
    "normalise_probabilities",
]
