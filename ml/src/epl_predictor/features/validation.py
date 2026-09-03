"""Runtime guards against leaked features.

These checks are cheap enough to run on every training and inference build, so
a leak introduced by a future change fails immediately rather than quietly
inflating a metric.
"""

import pandas as pd

from epl_predictor.features.builder import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_SOURCES,
)


def assert_no_forbidden_columns(features: pd.DataFrame) -> None:
    """Reject a feature frame containing any current-match quantity.

    Raises:
        ValueError: A forbidden column is present.
    """
    present = [column for column in FORBIDDEN_FEATURE_SOURCES if column in features.columns]
    if present:
        raise ValueError(
            "Feature frame contains current-match column(s) that must never be "
            f"used as input: {', '.join(present)}."
        )


def assert_schema(features: pd.DataFrame) -> None:
    """Reject a feature frame that does not match the published schema.

    Raises:
        ValueError: Columns are missing, extra, or out of order.
    """
    assert_no_forbidden_columns(features)

    actual = tuple(features.columns)
    if actual == FEATURE_COLUMNS:
        return

    missing = [column for column in FEATURE_COLUMNS if column not in actual]
    extra = [column for column in actual if column not in FEATURE_COLUMNS]
    details = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if extra:
        details.append(f"unexpected {', '.join(extra)}")
    if not details:
        details.append("columns are out of schema order")

    raise ValueError(f"Feature frame does not match the v1 schema: {'; '.join(details)}.")
