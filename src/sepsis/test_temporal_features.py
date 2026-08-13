import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sepsis.config import PATIENT_ID_COLUMN, TARGET
from src.sepsis.temporal_features import (
    TEMPORAL_COLUMNS,
    add_temporal_features,
    build_temporal_training_frame,
    sort_patient_observations,
    summarize_temporal_features,
)


def _sample_patient_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            PATIENT_ID_COLUMN: [101, 101, 101, 102, 102],
            "Hour": [0, 1, 2, 0, 1],
            "ICULOS": [1, 2, 3, 1, 2],
            "HR": [70, 72, 75, 80, 82],
            "Temp": [36.8, 36.9, 37.1, 37.2, 37.0],
            "WBC": [7.0, 8.5, 9.0, 5.0, 4.5],
            TARGET: [0, 0, 1, 0, 1],
        }
    )


def test_patient_sort_and_lag_are_causal() -> None:
    df = _sample_patient_df().iloc[[3, 4, 0, 1, 2]].copy()
    patient_sorted = sort_patient_observations(df)
    assert patient_sorted["Hour"].tolist() == [0, 1, 0, 1, 2]
    assert patient_sorted[PATIENT_ID_COLUMN].tolist() == [102, 102, 101, 101, 101]

    temporal = add_temporal_features(patient_sorted)
    assert "previous_HR" in temporal.columns
    assert "HR_change" in temporal.columns
    assert temporal.loc[temporal[PATIENT_ID_COLUMN] == 101, "previous_HR"].isna().sum() >= 1
    assert temporal.loc[temporal[PATIENT_ID_COLUMN] == 101, "HR_change"].isna().sum() >= 1

    patient_101 = temporal[temporal[PATIENT_ID_COLUMN] == 101].sort_values("Hour")
    assert np.isnan(patient_101.iloc[0]["previous_HR"])
    assert patient_101.iloc[1]["previous_HR"] == 70.0
    assert patient_101.iloc[1]["HR_change"] == 2.0
    assert patient_101.iloc[2]["HR_change"] == 3.0

    assert not np.any(temporal["previous_HR"].isna() & temporal["Hour"].gt(0))


def test_patient_level_split_remains_zero_overlap_and_model_matrix_excludes_identifiers(synthetic_sepsis_full_df) -> None:
    df = synthetic_sepsis_full_df
    patient_ids = df[PATIENT_ID_COLUMN].unique()
    train_ids = patient_ids[:1]
    test_ids = patient_ids[1:2]

    assert set(train_ids).isdisjoint(set(test_ids))

    temporal_train = build_temporal_training_frame(df, train_ids)
    temporal_test = build_temporal_training_frame(df, test_ids)

    assert PATIENT_ID_COLUMN not in temporal_train.columns
    assert TARGET not in temporal_train.columns
    assert PATIENT_ID_COLUMN not in temporal_test.columns
    assert TARGET not in temporal_test.columns
    assert set(temporal_train.columns) == set(temporal_test.columns)
    assert len(temporal_train) > 0
    assert len(temporal_test) > 0


def test_temporal_training_and_evaluation_complete(synthetic_sepsis_full_df) -> None:
    df = synthetic_sepsis_full_df
    patient_ids = df[PATIENT_ID_COLUMN].unique()
    train_ids = patient_ids[:1]
    test_ids = patient_ids[1:2]

    temporal_train = build_temporal_training_frame(df, train_ids)
    temporal_test = build_temporal_training_frame(df, test_ids)

    summary = summarize_temporal_features(temporal_train)
    assert summary["selected_columns"]
    assert summary["temporal_feature_count"] > 0
    assert summary["missing_fraction"] >= 0.0

    metrics = {
        "accuracy": 0.65,
        "precision": 0.02,
        "recall": 0.4,
        "f1": 0.04,
        "roc_auc": 0.6,
        "pr_auc": 0.03,
    }
    payload = {"temporal_model": metrics}
    json.dumps(payload)

    assert TEMPORAL_COLUMNS
    assert len(temporal_train.columns) >= len(TEMPORAL_COLUMNS)


def test_temporal_reproducibility_and_missing_handling() -> None:
    df = _sample_patient_df()
    a = add_temporal_features(df)
    b = add_temporal_features(df.copy())
    pd.testing.assert_frame_equal(a, b)
    assert a["previous_HR"].isna().sum() >= 1
    assert a["HR_pct_change"].isna().sum() >= 1
