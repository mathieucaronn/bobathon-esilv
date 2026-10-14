"""
Shared data-loading and CV-split helpers used by Day 2 steps 3-7.

Import this module in any experiment script that needs the same
patient-grouped cross-validation splits, so all steps compare on
identical folds.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold


FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]


def load_visits() -> pd.DataFrame:
    """Return the merged train DataFrame (X_train joined with y_train on Index)."""
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    return X_train.merge(y_train, on="Index")


def make_grouped_cv_splits(
    visits: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple]:
    """
    Precompute GroupKFold splits keyed on patient_id.

    Each split is a (train_indices, test_indices) pair where no patient_id
    appears on both sides — a faithful dress rehearsal of the Kaggle holdout.

    Parameters
    ----------
    visits : merged DataFrame containing patient_id, FEATURE_COLS, and target.
    n_splits : number of folds (default 5, matching docs/DAY2.md step 3).

    Returns
    -------
    List of (train_idx, test_idx) integer-array pairs.
    """
    X = visits[FEATURE_COLS]
    y = visits["target"]
    groups = visits["patient_id"]
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
