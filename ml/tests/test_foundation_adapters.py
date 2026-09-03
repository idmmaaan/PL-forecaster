"""Tests for the foundation-model adapter scaffolding.

None of these tests need torch or a pretrained checkpoint. They cover the
parts of the candidate machinery that are project code rather than library
code: the licence catalogue the selection gate reads, the encoding every
candidate shares, the class-order reordering that stops a silent outcome
swap, and the message a missing optional group produces.

A fake candidate stands in for the real models, so the shared base class is
exercised end to end — fit, predict, save, reload — in milliseconds.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from epl_predictor import OUTCOME_CLASSES
from epl_predictor.features.builder import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_training_table,
)
from epl_predictor.predictors.foundation import (
    CANDIDATES,
    UNKNOWN_CATEGORY,
    CandidateSpec,
    FoundationPredictor,
    LabelVector,
    MissingDependencyError,
    build_encoder,
    context_rows,
    encode_labels,
    encoder_input,
    reorder_columns,
    require_library,
)
from epl_predictor.predictors.tabular import FloatMatrix


def synthetic_matches(rows: int = 80) -> pd.DataFrame:
    """A chronological match table large enough to fill rolling windows."""
    clubs = ["Arsenal", "Chelsea", "Liverpool", "Manchester City"]
    records = []
    for index in range(rows):
        home = clubs[index % len(clubs)]
        away = clubs[(index + 1 + index // len(clubs)) % len(clubs)]
        if home == away:
            away = clubs[(index + 2) % len(clubs)]
        home_goals, away_goals = [(2, 0), (1, 1), (0, 2)][index % 3]
        records.append(
            {
                "match_id": f"m{index}",
                "kickoff_date": date(2020, 1, 1) + timedelta(days=index * 4),
                "season_start_year": 2019 + index // 40,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": "H"
                if home_goals > away_goals
                else ("D" if home_goals == away_goals else "A"),
                "home_shots_on_target": 5,
                "away_shots_on_target": 4,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def feature_table() -> pd.DataFrame:
    table, _ = build_training_table(synthetic_matches())
    return table


class FakeSpec:
    """A candidate spec for a model that does not exist."""


FAKE_SPEC = CandidateSpec(
    key="fake",
    model_name="fake-epl",
    hf_repo="example/fake",
    extra="fake",
    license_summary="MIT",
    production_use_allowed=True,
    role="Test double",
)


class FakePredictor(FoundationPredictor):
    """A foundation predictor whose 'model' is a class-frequency count.

    Stands in for the real candidates so the shared base class can be tested
    without downloading weights. Reports its columns in reverse canonical
    order, which exercises the reordering path that the real adapters rely on.
    """

    adapter_type = "fake"
    spec = FAKE_SPEC

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.frequencies = np.full(len(OUTCOME_CLASSES), 1 / len(OUTCOME_CLASSES))
        self.seen_features: FloatMatrix | None = None

    def _fit_encoded(self, features: FloatMatrix, labels: LabelVector) -> None:
        self.seen_features = features
        counts = np.bincount(labels, minlength=len(OUTCOME_CLASSES)).astype(float)
        canonical = counts / counts.sum()
        # Stored reversed, matching the order `_class_order` advertises.
        self.frequencies = canonical[::-1]

    def _predict_encoded(self, features: FloatMatrix) -> FloatMatrix:
        return np.tile(self.frequencies, (len(features), 1))

    def _class_order(self) -> list[str]:
        return list(reversed(OUTCOME_CLASSES))

    def _save_estimator(self, artifact_path: Path) -> None:
        np.save(artifact_path / "frequencies.npy", self.frequencies)

    def _load_estimator(self, artifact_path: Path, metadata: dict[str, Any]) -> None:
        self.frequencies = np.load(artifact_path / "frequencies.npy")


class TestCandidateCatalogue:
    def test_every_readme_candidate_is_registered(self) -> None:
        assert set(CANDIDATES) == {"tabicl", "mitra", "tabstar", "tabpfn-mix", "tabpfn3"}

    def test_every_candidate_records_a_licence_and_a_repository(self) -> None:
        """The gate treats licence as a hard criterion, so none may be blank."""
        for spec in CANDIDATES.values():
            assert spec.license_summary, f"{spec.key} has no licence summary"
            assert spec.hf_repo, f"{spec.key} has no Hugging Face repository"

    def test_tabpfn3_is_marked_as_not_usable_in_production(self) -> None:
        """Its licence forbids commercial use, so the gate must refuse it."""
        assert CANDIDATES["tabpfn3"].production_use_allowed is False
        assert CANDIDATES["tabpfn3"].requires_cuda is True

    def test_the_permissively_licensed_candidates_are_promotable(self) -> None:
        for key in ("tabicl", "mitra", "tabstar", "tabpfn-mix"):
            assert CANDIDATES[key].production_use_allowed is True

    def test_the_install_command_names_the_candidates_own_extra(self) -> None:
        command = CANDIDATES["tabicl"].install_command
        assert "--extra tabicl" in command
        assert "epl-predictor" in command


class TestMissingDependencyReporting:
    def test_a_missing_library_names_the_install_command(self) -> None:
        """A bare ImportError would not say which group to sync."""
        with pytest.raises(MissingDependencyError) as caught:
            require_library("a_module_that_does_not_exist", CANDIDATES["mitra"])

        message = str(caught.value)
        assert "a_module_that_does_not_exist" in message
        assert CANDIDATES["mitra"].install_command in message

    def test_an_installed_library_is_returned(self) -> None:
        assert require_library("json", CANDIDATES["tabicl"]).dumps([1]) == "[1]"


class TestLabelEncoding:
    def test_outcomes_map_to_their_canonical_indices(self) -> None:
        encoded = encode_labels(pd.Series(list(OUTCOME_CLASSES)))

        assert encoded.tolist() == [0, 1, 2]

    def test_an_unknown_label_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unexpected outcome label"):
            encode_labels(pd.Series(["HOME_WIN", "PENALTIES"]))


class TestColumnReordering:
    def test_a_reversed_class_order_is_corrected(self) -> None:
        """The bug this prevents: home wins served as away wins, silently."""
        matrix = np.array([[0.1, 0.2, 0.7]])

        reordered = reorder_columns(matrix, list(reversed(OUTCOME_CLASSES)))

        assert reordered.tolist() == [[0.7, 0.2, 0.1]]

    def test_canonical_order_passes_through_unchanged(self) -> None:
        matrix = np.array([[0.1, 0.2, 0.7]])

        assert reorder_columns(matrix, list(OUTCOME_CLASSES)).tolist() == matrix.tolist()

    def test_alphabetical_order_is_corrected(self) -> None:
        """Several libraries sort labels alphabetically rather than canonically."""
        alphabetical = sorted(OUTCOME_CLASSES)
        values = {outcome: 0.1 * (index + 1) for index, outcome in enumerate(OUTCOME_CLASSES)}
        matrix = np.array([[values[outcome] for outcome in alphabetical]])

        reordered = reorder_columns(matrix, alphabetical)

        assert reordered[0].tolist() == pytest.approx([values[o] for o in OUTCOME_CLASSES])

    def test_a_class_order_that_is_not_the_three_outcomes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            reorder_columns(np.zeros((1, 3)), ["HOME_WIN", "DRAW", "SHOOTOUT"])

    def test_a_matrix_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="probability matrix"):
            reorder_columns(np.zeros((1, 2)), list(OUTCOME_CLASSES))


class TestContextTruncation:
    def test_the_most_recent_rows_are_kept(self) -> None:
        """Recent matches are both cheaper to carry and more relevant."""
        frame = pd.DataFrame({"n": range(100)})

        assert context_rows(frame, 10)["n"].tolist() == list(range(90, 100))

    def test_a_frame_within_the_limit_is_untouched(self) -> None:
        frame = pd.DataFrame({"n": range(5)})

        assert context_rows(frame, 10) is frame

    def test_a_zero_limit_means_no_truncation(self) -> None:
        frame = pd.DataFrame({"n": range(5)})

        assert len(context_rows(frame, 0)) == 5


class TestSharedEncoder:
    def test_the_encoded_matrix_is_finite_and_has_one_column_per_feature(
        self, feature_table: pd.DataFrame
    ) -> None:
        """Foundation models need dense numbers; the schema's nulls must go."""
        columns = list(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
        encoded = build_encoder().fit_transform(encoder_input(feature_table))

        assert encoded.shape == (len(feature_table), len(columns))
        assert np.all(np.isfinite(encoded))

    def test_an_unseen_club_encodes_to_the_unknown_category(
        self, feature_table: pd.DataFrame
    ) -> None:
        """A promoted club appears in the fixture list before the training table."""
        columns = list(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
        encoder = build_encoder().fit(encoder_input(feature_table))

        unseen = feature_table[columns].head(1).copy()
        unseen.loc[:, "home_team"] = "Luton Town"

        assert encoder.transform(encoder_input(unseen))[0][0] == UNKNOWN_CATEGORY

    def test_pandas_nullable_integer_nulls_are_accepted(self, feature_table: pd.DataFrame) -> None:
        """League position is genuinely null on an opening weekend, and the v1
        schema stores it as a nullable integer. Feeding `pd.NA` straight to
        scikit-learn's imputer raises `float(pd.NA)`, so it must be coerced."""
        columns = list(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
        encoder = build_encoder().fit(encoder_input(feature_table))

        row = feature_table[columns].head(1).copy()
        row.loc[:, "home_league_position_before_match"] = pd.NA

        encoded = encoder.transform(encoder_input(row))

        assert np.all(np.isfinite(encoded))

    def test_the_encoder_does_not_refit_at_prediction_time(
        self, feature_table: pd.DataFrame
    ) -> None:
        """Imputation medians must come from training, not from the request."""
        encoder = build_encoder().fit(encoder_input(feature_table))

        first = encoder.transform(encoder_input(feature_table.head(1)))
        encoder.transform(encoder_input(feature_table.tail(20)))
        again = encoder.transform(encoder_input(feature_table.head(1)))

        assert first.tolist() == again.tolist()


class TestFoundationPredictorBase:
    def test_probabilities_are_returned_in_canonical_order(
        self, feature_table: pd.DataFrame
    ) -> None:
        """The fake reports reversed columns; the base must correct them."""
        predictor = FakePredictor()
        predictor.fit(feature_table)

        probabilities = predictor.predict_proba(
            {column: feature_table.iloc[0][column] for column in FEATURE_COLUMNS}
        )

        counts = feature_table["outcome"].value_counts(normalize=True)
        assert probabilities["home_win"] == pytest.approx(counts["HOME_WIN"], abs=1e-9)
        assert probabilities["away_win"] == pytest.approx(counts["AWAY_WIN"], abs=1e-9)

    def test_output_is_a_valid_distribution(self, feature_table: pd.DataFrame) -> None:
        predictor = FakePredictor()
        predictor.fit(feature_table)

        matrix = predictor.predict_matrix(feature_table)

        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert matrix.min() >= 0.0

    def test_the_context_is_capped_and_recorded(self, feature_table: pd.DataFrame) -> None:
        predictor = FakePredictor(max_context_rows=20)
        predictor.fit(feature_table)

        assert predictor.context_size == 20
        assert predictor.seen_features is not None
        assert len(predictor.seen_features) == 20

    def test_the_artifact_round_trips_to_identical_probabilities(
        self, feature_table: pd.DataFrame, tmp_path: Path
    ) -> None:
        """A gate criterion: an artifact that loads but predicts differently
        is worse than one that fails to load."""
        predictor = FakePredictor(version="2.0.0")
        predictor.fit(feature_table)
        before = predictor.predict_matrix(feature_table)

        predictor.save(tmp_path / "artifact")
        reloaded = FakePredictor.load(tmp_path / "artifact")

        assert np.allclose(before, reloaded.predict_matrix(feature_table))
        assert reloaded.model_version == "2.0.0"

    def test_the_encoder_is_saved_with_the_model(
        self, feature_table: pd.DataFrame, tmp_path: Path
    ) -> None:
        predictor = FakePredictor()
        predictor.fit(feature_table)
        predictor.save(tmp_path / "artifact")

        assert (tmp_path / "artifact" / "encoder.joblib").exists()

    def test_metadata_records_the_licence_and_provenance(self, feature_table: pd.DataFrame) -> None:
        """The registry stores a licence summary with every model version."""
        predictor = FakePredictor()
        predictor.fit(feature_table)

        metadata = predictor.metadata()

        assert metadata["license_summary"] == "MIT"
        assert metadata["hf_repo"] == "example/fake"
        assert metadata["production_use_allowed"] is True
        assert metadata["class_order"] == list(OUTCOME_CLASSES)

    def test_predicting_before_fitting_is_refused(self, feature_table: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            FakePredictor().predict_matrix(feature_table)


class TestAdapterDeclarations:
    """Each adapter's declared identity, checked without importing its library."""

    @pytest.mark.parametrize(
        ("module_name", "class_name", "key"),
        [
            ("tabicl", "TabICLPredictor", "tabicl"),
            ("mitra", "MitraPredictor", "mitra"),
            ("tabstar", "TabSTARPredictor", "tabstar"),
            ("tabpfn_mix", "TabPFNMixPredictor", "tabpfn-mix"),
            ("tabpfn3", "TabPFN3Predictor", "tabpfn3"),
        ],
    )
    def test_each_adapter_declares_its_candidate_spec(
        self, module_name: str, class_name: str, key: str
    ) -> None:
        from importlib import import_module

        module = import_module(f"epl_predictor.predictors.{module_name}")
        adapter = getattr(module, class_name)

        assert adapter.spec is CANDIDATES[key]
        assert adapter.adapter_type == key
        assert issubclass(adapter, FoundationPredictor)
