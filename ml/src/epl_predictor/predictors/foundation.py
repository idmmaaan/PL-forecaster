"""Shared plumbing for tabular foundation-model candidates.

Every candidate here is a pretrained transformer over tables rather than a
model fitted from scratch on this dataset. They differ enormously in size,
hardware requirements, and licence, but they share three problems that this
module solves once:

**They are optional.** Each one pulls in torch and multi-gigabyte pretrained
weights, so none is installed by default. `require_library` turns a missing
import into a message naming the exact command to install it, instead of an
`ImportError` from three frames deep in an adapter.

**They want numbers.** The v1 schema carries two club-name columns and 26
numeric ones, many with honest nulls. `encode_frame` produces the dense
numeric matrix these models expect, and the encoder is fitted once and saved
with the artifact so inference cannot depend on whatever data is loaded.

**They keep their training set.** In-context models carry the training rows
into inference, so a decade of matches makes prediction slow and memory-hungry
for no gain. `context_rows` takes the *most recent* rows up to a cap, which is
both the cheaper and the more relevant half of the history.

Licence is tracked as data, in `CANDIDATES`, because the README's selection
gate treats it as a hard criterion and one candidate (TabPFN-3) may not be
used commercially at all.
"""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from epl_predictor.predictors.tabular import FloatMatrix, TabularPredictor

#: Integer labels, aligned with `OUTCOME_CLASSES` positions.
LabelVector = npt.NDArray[np.int64]

ENCODER_FILENAME = "encoder.joblib"
MODEL_FILENAME = "model.joblib"

#: Unknown clubs encode to this rather than raising: a promoted team appears in
#: the fixture list before it appears in the training table.
UNKNOWN_CATEGORY = -1

#: Default ceiling on in-context training rows. Roughly four Premier League
#: seasons, which keeps inference on a laptop tractable.
DEFAULT_MAX_CONTEXT_ROWS = 1520


class MissingDependencyError(RuntimeError):
    """A candidate's library is not installed in this environment.

    Carries the install command, because the alternative — a bare ImportError
    from inside an adapter — tells the reader nothing about which optional
    group they were supposed to sync.
    """


@dataclass(frozen=True)
class CandidateSpec:
    """What the project needs to know about a candidate before running it.

    Recorded as data so the licence and hardware gates can be applied and
    tested without importing the candidate's library, which is the whole point
    of keeping these behind optional dependency groups.
    """

    key: str
    model_name: str
    hf_repo: str
    extra: str
    license_summary: str
    #: False when the licence forbids production or commercial use. Such a
    #: candidate may be benchmarked but must never be promoted.
    production_use_allowed: bool
    role: str
    #: True when the candidate's own documentation requires hardware this
    #: project does not target, so a failure to run is expected, not a bug.
    requires_cuda: bool = False
    notes: str = ""

    @property
    def install_command(self) -> str:
        return f"uv sync --inexact --package epl-predictor --extra {self.extra}"


#: The five candidates named in the README, in its priority order. Licence
#: text is a summary for triage only; the README requires rechecking the model
#: card before any deployment.
CANDIDATES: dict[str, CandidateSpec] = {
    "tabicl": CandidateSpec(
        key="tabicl",
        model_name="tabicl-epl",
        hf_repo="jingang/TabICL",
        extra="tabicl",
        license_summary="BSD-3-Clause (model and code)",
        production_use_allowed=True,
        role="Primary candidate",
    ),
    "mitra": CandidateSpec(
        key="mitra",
        model_name="mitra-epl",
        hf_repo="autogluon/mitra-classifier",
        extra="mitra",
        license_summary="Apache-2.0",
        production_use_allowed=True,
        role="Main challenger",
        notes="AutoGluon adds a large dependency surface; ~72M parameters.",
    ),
    "tabstar": CandidateSpec(
        key="tabstar",
        model_name="tabstar-epl",
        hf_repo="alana89/TabSTAR",
        extra="tabstar",
        license_summary="Model card CC-BY-4.0; repository code MIT (verify separately)",
        production_use_allowed=True,
        role="Semantic-feature challenger",
        notes=("Semantic advantage is limited on the v1 schema, which is almost entirely numeric."),
    ),
    "tabpfn-mix": CandidateSpec(
        key="tabpfn-mix",
        model_name="tabpfn-mix-epl",
        hf_repo="autogluon/tabpfn-mix-1.0-classifier",
        extra="tabpfn-mix",
        license_summary="Apache-2.0",
        production_use_allowed=True,
        role="Lightweight challenger",
        notes="~39M parameters, reached through the AutoGluon predictor interface.",
    ),
    "tabpfn3": CandidateSpec(
        key="tabpfn3",
        model_name="tabpfn3-epl",
        hf_repo="Prior-Labs/tabpfn_3",
        extra="tabpfn3",
        license_summary="Prior Labs non-commercial model licence",
        production_use_allowed=False,
        role="Research benchmark only",
        requires_cuda=True,
        notes=(
            "The licence restricts commercial and production use, and the "
            "published fine-tuning example requires an 80 GB CUDA GPU. Run in "
            "inference mode as a research benchmark; never promote."
        ),
    ),
}


def require_library(module: str, spec: CandidateSpec) -> Any:
    """Import a candidate's library or explain how to install it.

    Raises:
        MissingDependencyError: The module is not importable.
    """
    from importlib import import_module

    try:
        return import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"{spec.model_name} needs the '{module}' package, which is not installed. "
            f"Install it with `{spec.install_command}`."
        ) from exc


def build_encoder() -> Pipeline:
    """A fitted-once encoder turning the v1 schema into a dense float matrix.

    Clubs are ordinal-encoded rather than one-hot encoded: these models see
    only a few thousand in-context rows, and 40 extra sparse columns would
    crowd out the 26 numeric features that carry the signal. Nulls are imputed
    with the training median, so the feature table keeps its honest nulls and
    only this encoder decides what to substitute.
    """
    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        (
                            "clubs",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=UNKNOWN_CATEGORY,
                                encoded_missing_value=UNKNOWN_CATEGORY,
                            ),
                            list(CATEGORICAL_FEATURES),
                        ),
                        (
                            "numbers",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            list(NUMERIC_FEATURES),
                        ),
                    ],
                    remainder="drop",
                ),
            )
        ]
    )


def encoder_input(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce a feature frame into the dtypes the encoder can consume.

    The v1 schema uses pandas nullable integers, so a missing league position
    arrives as `pd.NA`. Scikit-learn's imputer cannot read that: it reaches
    `float(pd.NA)` and raises. Converting the numeric columns to float turns
    every flavour of null into `np.nan`, which the imputer does understand,
    and leaves the honest nulls in the feature table untouched.

    Club names are cast to string for the same reason: a `None` home team
    must reach the ordinal encoder as a missing category, not as an object it
    cannot compare.
    """
    numeric = features[list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce")
    clubs = (
        features[list(CATEGORICAL_FEATURES)]
        .astype("object")
        .where(features[list(CATEGORICAL_FEATURES)].notna(), None)
    )
    return pd.concat([clubs, numeric.astype("float64")], axis=1)[
        list(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    ]


def context_rows(train: pd.DataFrame, limit: int) -> pd.DataFrame:
    """The most recent `limit` rows of a chronologically ordered frame.

    Truncating from the front keeps the rows closest to what is being
    predicted. It is not a random subsample: dropping recent seasons to keep
    2010 would be both slower to learn from and less relevant.
    """
    if limit <= 0 or len(train) <= limit:
        return train
    return train.iloc[-limit:].reset_index(drop=True)


class FoundationPredictor(TabularPredictor):
    """Base class for the pretrained tabular candidates.

    Subclasses implement `_fit_encoded` and `_predict_encoded`, working on
    float matrices and integer labels; encoding, class order, calibration, and
    the artifact layout are inherited.
    """

    #: Filled in by each subclass from `CANDIDATES`.
    spec: CandidateSpec

    def __init__(
        self,
        name: str | None = None,
        version: str = "0.1.0",
        calibrator: Any = None,
        max_context_rows: int = DEFAULT_MAX_CONTEXT_ROWS,
        device: str | None = None,
        random_state: int = 20260903,
    ):
        super().__init__(name=name or self.spec.model_name, version=version, calibrator=calibrator)
        self.max_context_rows = max_context_rows
        self.device = device
        self.random_state = random_state
        self.encoder: Pipeline | None = None
        self.context_size = 0

    # --- Subclass responsibilities ------------------------------------------

    @abstractmethod
    def _fit_encoded(self, features: FloatMatrix, labels: LabelVector) -> None:
        """Fit or adapt the pretrained model on encoded training rows."""

    @abstractmethod
    def _predict_encoded(self, features: FloatMatrix) -> FloatMatrix:
        """Return (n, 3) probabilities, columns ordered as this model reports them."""

    def _class_order(self) -> list[str]:
        """The class order the underlying model's probability columns follow.

        Defaults to the canonical order; subclasses whose library sorts labels
        differently override this so `_predict_matrix` can reorder.
        """
        return list(OUTCOME_CLASSES)

    # --- Shared behaviour ---------------------------------------------------

    def _fit_frame(self, train: pd.DataFrame, validation: pd.DataFrame | None) -> None:
        """Encode the most recent training rows and hand them to the model.

        The validation frame is deliberately unused for fitting: these
        candidates are adapted in-context or fine-tuned on the training period
        only, and the validation period is reserved for calibration and
        measurement.
        """
        context = context_rows(train, self.max_context_rows)
        self.context_size = len(context)

        self.encoder = build_encoder()
        features = np.asarray(self.encoder.fit_transform(encoder_input(context)), dtype=np.float64)
        self._fit_encoded(features, encode_labels(context[TARGET_COLUMN]))

    def _predict_matrix(self, features: pd.DataFrame) -> FloatMatrix:
        """Encode a frame and reorder the model's output to canonical order."""
        if self.encoder is None:
            raise RuntimeError(f"{type(self).__name__} has no fitted encoder.")

        encoded = np.asarray(self.encoder.transform(encoder_input(features)), dtype=np.float64)
        return reorder_columns(self._predict_encoded(encoded), self._class_order())

    def _extra_metadata(self) -> dict[str, Any]:
        return {
            "hf_repo": self.spec.hf_repo,
            "license_summary": self.spec.license_summary,
            "production_use_allowed": self.spec.production_use_allowed,
            "role": self.spec.role,
            "max_context_rows": self.max_context_rows,
            "context_size": self.context_size,
            "device": self.device,
            "random_state": self.random_state,
        }

    def _save_model(self, artifact_path: Path) -> None:
        joblib.dump(self.encoder, artifact_path / ENCODER_FILENAME)
        self._save_estimator(artifact_path)

    def _load_model(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        self.encoder = joblib.load(artifact_path / ENCODER_FILENAME)
        self.max_context_rows = metadata.get("max_context_rows", self.max_context_rows)
        self.context_size = metadata.get("context_size", 0)
        self.random_state = metadata.get("random_state", self.random_state)
        self._load_estimator(artifact_path, metadata)

    @abstractmethod
    def _save_estimator(self, artifact_path: Path) -> None:
        """Persist whatever the candidate needs to predict again."""

    @abstractmethod
    def _load_estimator(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        """Restore the candidate from an artifact directory."""


def encode_labels(labels: pd.Series) -> LabelVector:
    """Map outcome strings onto their canonical class indices.

    Raises:
        ValueError: A label is not one of the three outcomes.
    """
    lookup = {outcome: index for index, outcome in enumerate(OUTCOME_CLASSES)}
    unknown = sorted(set(labels.astype(str)) - lookup.keys())
    if unknown:
        raise ValueError(f"Unexpected outcome label(s): {', '.join(unknown)}")
    return np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)


def reorder_columns(matrix: FloatMatrix, class_order: list[str]) -> FloatMatrix:
    """Rearrange probability columns from `class_order` into canonical order.

    Libraries that sort labels alphabetically return columns in an order this
    project does not use. A permuted column would not raise anywhere; it would
    just silently swap home wins for away wins, so the reordering is explicit.

    Raises:
        ValueError: `class_order` does not describe the three outcomes, or the
            matrix has the wrong width.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(OUTCOME_CLASSES):
        raise ValueError(
            f"Expected an (n, {len(OUTCOME_CLASSES)}) probability matrix, got {matrix.shape}"
        )
    if sorted(class_order) != sorted(OUTCOME_CLASSES):
        raise ValueError(f"Class order {class_order} does not match {list(OUTCOME_CLASSES)}")

    if list(class_order) == list(OUTCOME_CLASSES):
        return matrix
    positions = [class_order.index(outcome) for outcome in OUTCOME_CLASSES]
    return matrix[:, positions]
