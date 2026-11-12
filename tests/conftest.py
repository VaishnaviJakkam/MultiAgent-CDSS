from __future__ import annotations

import pandas as pd
import pytest

from src.sepsis.config import FEATURE_COLUMNS, PATIENT_ID_COLUMN, TARGET, TIME_COLUMNS


def _feature_value(col: str, index: int) -> float | str:
    if col == "Gender":
        return "M" if index % 2 == 0 else "F"
    if col == "Age":
        return float(30 + index)
    if col == "WBC":
        return float(5.0 + index)
    if col == "Lactate":
        return float(1.0 + 0.1 * index)
    return float(index + 1)


@pytest.fixture
def synthetic_sepsis_full_df() -> pd.DataFrame:
    """Small synthetic Sepsis dataset with all expected feature columns."""
    rows: list[dict[str, float | str]] = []
    for idx in range(10):
        patient_id = f"P{idx + 1:03d}"
        row = {
            PATIENT_ID_COLUMN: patient_id,
            TARGET: 1 if idx % 2 == 0 else 0,
            "Hour": float(idx % 3),
            "HospAdmTime": 0.0,
            "ICULOS": float(idx + 1),
        }
        for col in FEATURE_COLUMNS:
            row[col] = _feature_value(col, idx)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_sepsis_feature_record() -> dict[str, float | str]:
    """Small synthetic record containing all feature columns for predict_sepsis_risk."""
    record: dict[str, float | str] = {}
    for idx, col in enumerate(FEATURE_COLUMNS):
        record[col] = _feature_value(col, idx)
    return record


@pytest.fixture
def synthetic_sepsis_temporal_df() -> pd.DataFrame:
    """Synthetic temporal history for two patients with enough rows to test lag features."""
    return pd.DataFrame(
        {
            PATIENT_ID_COLUMN: [101, 101, 101, 102, 102, 102],
            "Hour": [0, 1, 2, 0, 1, 2],
            "ICULOS": [1, 2, 3, 1, 2, 3],
            "HR": [70.0, 72.0, 75.0, 80.0, 82.0, 84.0],
            "O2Sat": [98.0, 97.0, 96.0, 98.0, 97.0, 96.0],
            "Temp": [36.8, 36.9, 37.1, 37.2, 37.0, 37.3],
            "SBP": [120.0, 118.0, 115.0, 122.0, 121.0, 119.0],
            "MAP": [80.0, 79.0, 77.0, 82.0, 81.0, 79.0],
            "Resp": [16.0, 17.0, 18.0, 16.0, 17.0, 18.0],
            "WBC": [7.0, 7.5, 8.0, 6.0, 6.5, 7.0],
            "Lactate": [1.0, 1.1, 1.2, 1.0, 1.1, 1.2],
            TARGET: [0, 0, 1, 0, 1, 1],
        }
    )
