"""TabPFNMix adapter: the README's lightweight challenger.

TabPFNMix is a ~39M-parameter prior-fitted network, roughly half Mitra's size,
also published through AutoGluon. It shares the host and therefore almost all
of the adapter; the interesting question it answers is whether the extra
parameters in the larger candidates buy anything on a dataset of a few
thousand matches.

Its model card describes training with configurable epochs rather than a
fine-tuning step count, so the epoch budget is exposed here and passed
through alongside the shared fine-tuning options.

Licence: Apache-2.0, permissive for the intended use.
"""

from typing import Any

from epl_predictor.predictors.autogluon_host import AutoGluonPredictor
from epl_predictor.predictors.foundation import CANDIDATES

DEFAULT_EPOCHS = 30


class TabPFNMixPredictor(AutoGluonPredictor):
    """TabPFNMix Classifier over the v1 feature schema."""

    adapter_type = "tabpfn-mix"
    spec = CANDIDATES["tabpfn-mix"]
    autogluon_key = "TABPFNMIX"

    def __init__(self, *args: Any, epochs: int = DEFAULT_EPOCHS, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.epochs = epochs

    def _hyperparameters(self) -> dict[str, list[dict[str, Any]]]:
        return {
            self.autogluon_key: [
                {
                    "n_ensembles": 1,
                    "max_epochs": self.epochs,
                }
            ]
        }

    def _extra_metadata(self) -> dict[str, Any]:
        return {**super()._extra_metadata(), "epochs": self.epochs}
