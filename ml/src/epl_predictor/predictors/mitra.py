"""Mitra adapter: the README's main challenger to TabICL.

Mitra is a ~72M-parameter tabular foundation model published by the AutoGluon
team and reached through AutoGluon's tabular predictor, so the adapter is a
thin specialisation of the shared AutoGluon host.

Its model card documents both a zero-shot mode and a fine-tuned mode. This
adapter fine-tunes by default, because the README's question is whether a
foundation model beats CatBoost *after adaptation to this dataset*, and the
zero-shot number would answer a weaker one. Zero-shot is still reachable with
`fine_tune=False` for comparison.

Licence: Apache-2.0, permissive for the intended use.
"""

from epl_predictor.predictors.autogluon_host import AutoGluonPredictor
from epl_predictor.predictors.foundation import CANDIDATES


class MitraPredictor(AutoGluonPredictor):
    """Mitra Classifier over the v1 feature schema."""

    adapter_type = "mitra"
    spec = CANDIDATES["mitra"]
    autogluon_key = "MITRA"
