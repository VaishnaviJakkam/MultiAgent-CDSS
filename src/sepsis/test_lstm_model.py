import numpy as np
import pandas as pd
import pytest

from src.sepsis.config import PATIENT_ID_COLUMN, TARGET
from src.sepsis.lstm_model import (
    LSTM_FEATURES,
    SEQUENCE_LENGTH,
    build_preprocessor,
    build_sequences,
    calculate_class_weights,
    fit_training_preprocessor,
    sort_patient_sequences,
    validate_experiment_columns,
)


def _sequence_frame() -> pd.DataFrame:
    rows = []
    for patient_id, offset in [("A", 0), ("B", 100)]:
        for hour in range(3):
            row = {PATIENT_ID_COLUMN: patient_id, "Hour": hour, TARGET: int(patient_id == "A" and hour == 2)}
            row.update({feature: float(offset + hour + 1) for feature in LSTM_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_sequences_are_chronological_and_patient_isolated() -> None:
    frame = _sequence_frame()
    preprocessor = fit_training_preprocessor(frame)
    sequences, labels, sequence_patients = build_sequences(frame, ["A", "B"], preprocessor, sequence_length=3)

    assert sequences.shape == (6, 3, len(LSTM_FEATURES))
    assert labels.tolist() == [0, 0, 1, 0, 0, 0]
    assert sequence_patients.tolist() == ["A", "A", "A", "B", "B", "B"]
    # The final A window contains only A's scaled observations, never B's values.
    assert not np.array_equal(sequences[2, -1], sequences[5, -1])


def test_short_patient_sequence_is_left_padded_and_has_current_label() -> None:
    frame = _sequence_frame().query("Patient_ID == 'A'").iloc[:2].copy()
    preprocessor = fit_training_preprocessor(frame)
    sequences, labels, _ = build_sequences(frame, ["A"], preprocessor, sequence_length=4)

    assert sequences.shape == (2, 4, len(LSTM_FEATURES))
    assert np.all(sequences[0, :3] == 0.0)
    expected_labels = frame.sort_values("Hour")[TARGET].tolist()
    assert labels.tolist() == expected_labels


def test_sorting_is_by_patient_then_hour() -> None:
    sorted_frame = sort_patient_sequences(_sequence_frame())
    assert sorted_frame[PATIENT_ID_COLUMN].tolist() == ["A", "A", "A", "B", "B", "B"]
    assert sorted_frame["Hour"].tolist() == [0, 1, 2, 0, 1, 2]


def test_forbidden_model_features_are_rejected() -> None:
    frame = _sequence_frame()
    with pytest.raises(ValueError):
        validate_experiment_columns(frame, ["HR", PATIENT_ID_COLUMN])
    with pytest.raises(ValueError):
        build_preprocessor(["HR", TARGET])


def test_preprocessor_is_fit_on_training_data_only() -> None:
    frame = _sequence_frame()
    train_frame = frame[frame[PATIENT_ID_COLUMN] == "A"]
    preprocessor = fit_training_preprocessor(train_frame)
    assert np.isclose(preprocessor.named_steps["imputer"].statistics_[0], 2.0)
    transformed_test = preprocessor.transform(frame[frame[PATIENT_ID_COLUMN] == "B"][LSTM_FEATURES])
    assert transformed_test.shape == (3, len(LSTM_FEATURES))


def test_class_weights_use_only_supplied_training_labels() -> None:
    weights = calculate_class_weights(np.array([0, 0, 0, 1], dtype=np.int32))
    assert weights[0] == pytest.approx(4 / 6)
    assert weights[1] == pytest.approx(2.0)


def test_default_sequence_contract() -> None:
    assert SEQUENCE_LENGTH == 12
    assert PATIENT_ID_COLUMN not in LSTM_FEATURES
    assert TARGET not in LSTM_FEATURES