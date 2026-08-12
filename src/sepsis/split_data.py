from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, FEATURE_COLUMNS, PATIENT_ID_COLUMN, TARGET, TIME_COLUMNS, RANDOM_STATE, TEST_SIZE


def load_raw_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path, low_memory=False)


def compute_patient_level_label(df: pd.DataFrame) -> pd.Series:
    """Compute a patient-level label: 1 if any observation for patient has SepsisLabel==1."""
    grp = df.groupby(PATIENT_ID_COLUMN)[TARGET].max()
    return grp


def patient_level_train_test_split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
    """Create a patient-level train/test split.

    Returns: X_train, X_test, y_train, y_test, train_patient_ids, test_patient_ids
    All X_* are DataFrames containing FEATURE_COLUMNS only (Patient_ID and time columns excluded).
    y_* are Series aligned to rows in X_* containing SepsisLabel.
    """
    if PATIENT_ID_COLUMN not in df.columns:
        raise ValueError(f"{PATIENT_ID_COLUMN} not found in dataframe")
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in dataframe")

    # Compute patient-level labels for stratification
    patient_label = compute_patient_level_label(df)

    patient_ids = patient_label.index.to_numpy()
    patient_labels = patient_label.values

    # Stratified split by patient-level label to preserve prevalence
    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_labels,
    )

    train_ids = set(train_ids.tolist())
    test_ids = set(test_ids.tolist())

    # Ensure zero overlap
    if train_ids & test_ids:
        raise RuntimeError("Patient ID overlap between train and test")

    # Filter rows
    train_mask = df[PATIENT_ID_COLUMN].isin(train_ids)
    test_mask = df[PATIENT_ID_COLUMN].isin(test_ids)

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    # Separate features and targets; preserve Patient_ID and time columns outside X
    X_train = df_train[FEATURE_COLUMNS].copy()
    y_train = df_train[TARGET].copy()

    X_test = df_test[FEATURE_COLUMNS].copy()
    y_test = df_test[TARGET].copy()

    return X_train, X_test, y_train, y_test, pd.Index(sorted(list(train_ids))), pd.Index(sorted(list(test_ids)))


def summarize_split(df: pd.DataFrame, train_ids: pd.Index, test_ids: pd.Index) -> dict:
    total_patients = df[PATIENT_ID_COLUMN].nunique()
    train_patients = len(train_ids)
    test_patients = len(test_ids)
    train_obs = df[df[PATIENT_ID_COLUMN].isin(train_ids)].shape[0]
    test_obs = df[df[PATIENT_ID_COLUMN].isin(test_ids)].shape[0]

    def prevalence(mask):
        s = df[mask][TARGET]
        return float(s.mean()), int(s.sum()), int(s.shape[0])

    train_mask = df[PATIENT_ID_COLUMN].isin(train_ids)
    test_mask = df[PATIENT_ID_COLUMN].isin(test_ids)

    train_prev, train_pos, train_total = prevalence(train_mask)
    test_prev, test_pos, test_total = prevalence(test_mask)

    return {
        "total_patients": total_patients,
        "train_patients": train_patients,
        "test_patients": test_patients,
        "train_observations": train_obs,
        "test_observations": test_obs,
        "train_positive_count": train_pos,
        "train_total": train_total,
        "train_prevalence": train_prev,
        "test_positive_count": test_pos,
        "test_total": test_total,
        "test_prevalence": test_prev,
        "patient_overlap": len(set(train_ids) & set(test_ids)),
    }
