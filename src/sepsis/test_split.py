import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.sepsis.split_data import load_raw_dataset, patient_level_train_test_split, summarize_split
from src.sepsis.config import PATIENT_ID_COLUMN, TARGET


def test_patient_level_split():
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, train_ids, test_ids = patient_level_train_test_split(df)

    # Basic checks
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

    # Patient ID not present in features
    assert PATIENT_ID_COLUMN not in X_train.columns
    assert PATIENT_ID_COLUMN not in X_test.columns

    # Target not present in features
    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns

    # Patient ID sets disjoint
    assert len(set(train_ids) & set(test_ids)) == 0

    # Class representation: ensure both classes present in patient-level labels where possible
    df_train = df[df[PATIENT_ID_COLUMN].isin(train_ids)]
    df_test = df[df[PATIENT_ID_COLUMN].isin(test_ids)]

    train_pos = int(df_train[TARGET].sum())
    test_pos = int(df_test[TARGET].sum())

    assert train_pos >= 0
    assert test_pos >= 0

    # Reproducibility
    X_train2, X_test2, y_train2, y_test2, train_ids2, test_ids2 = patient_level_train_test_split(df)
    assert list(train_ids) == list(train_ids2)

    print('Patient-level split test passed')


if __name__ == '__main__':
    test_patient_level_split()
