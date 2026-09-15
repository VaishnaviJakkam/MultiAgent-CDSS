from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.mongodb import MongoDatabase
from src.database.repositories import PatientRepository


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sepsis"
    / "Dataset.csv"
)


PATIENTS = [
    {
        "dataset_id": 3373,
        "group": "SEPSIS",
    },
    {
        "dataset_id": 1446,
        "group": "SEPSIS",
    },
    {
        "dataset_id": 1185,
        "group": "SEPSIS",
    },
    {
        "dataset_id": 18823,
        "group": "NON-SEPSIS",
    },
    {
        "dataset_id": 19217,
        "group": "NON-SEPSIS",
    },
    {
        "dataset_id": 100559,
        "group": "NON-SEPSIS",
    },
]


SEPSIS_FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
    "WBC",
    "Lactate",
]


# ============================================================
# HELPERS
# ============================================================

def python_value(value: Any):
    """
    Convert pandas/numpy values into Mongo-friendly Python values.

    Missing measurements are stored as None.

    IMPORTANT:
    We are NOT inventing values for missing clinical measurements.
    """

    if pd.isna(value):
        return None

    return float(value)


def choose_positive_window(
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find:
        12 observations before first SepsisLabel = 1
        +
        the first SepsisLabel = 1 observation

    We return all 13 here, but later ONLY the first 12
    are inserted into MongoDB.
    """

    patient_df = (
        patient_df
        .sort_values("Hour")
        .reset_index(drop=True)
    )

    positive_indices = patient_df.index[
        patient_df["SepsisLabel"] == 1
    ].tolist()

    if not positive_indices:
        raise ValueError(
            "No SepsisLabel=1 found."
        )

    first_positive = positive_indices[0]

    if first_positive < 12:
        raise ValueError(
            "Not enough history before sepsis onset."
        )

    return patient_df.iloc[
        first_positive - 12:
        first_positive + 1
    ].copy()


def choose_negative_window(
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pick the most complete contiguous 13-observation window
    for a non-sepsis patient.

    Again, only observations 1-12 will actually be stored.
    """

    patient_df = (
        patient_df
        .sort_values("Hour")
        .reset_index(drop=True)
    )

    if len(patient_df) < 13:
        raise ValueError(
            "Patient has fewer than 13 observations."
        )

    best_window = None
    best_score = -1

    for start in range(
        0,
        len(patient_df) - 12,
    ):
        window = patient_df.iloc[
            start:start + 13
        ]

        score = int(
            window[
                SEPSIS_FEATURES
            ]
            .notna()
            .sum()
            .sum()
        )

        if score > best_score:
            best_score = score
            best_window = window.copy()

    if best_window is None:
        raise RuntimeError(
            "Unable to choose patient window."
        )

    return best_window


def row_to_clinical_parameters(
    row: pd.Series,
) -> dict:
    """
    Store all eight Sepsis model features.

    Missing values remain None.

    Having the keys present is useful because the Sepsis
    preprocessing pipeline can later represent them as
    missing values for imputation.
    """

    return {
        feature: python_value(
            row[feature]
        )
        for feature in SEPSIS_FEATURES
    }


def delete_previous_demo_patient(
    database: MongoDatabase,
    patient_id: str,
):
    """
    Make script safe to rerun.

    This removes ONLY these demo patient IDs.
    """

    collections = [
        "prioritizations",
        "trends",
        "assessments",
        "observations",
        "admissions",
        "patients",
    ]

    for collection_name in collections:
        database[
            collection_name
        ].delete_many(
            {
                "patient_id": patient_id
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("LOAD HISTORICAL OBSERVATIONS")
    print("=" * 80)

    print(
        "\nPurpose:"
        "\n• Create 6 demo patients"
        "\n• Store observations 1-12"
        "\n• DO NOT store observation 13"
        "\n• DO NOT run Agent 1"
        "\n• DO NOT run Agent 2"
        "\n• DO NOT create assessments/trends/priorities"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    print("\nReading Dataset.csv...")

    columns = [
        "Patient_ID",
        "Hour",
        "SepsisLabel",
        "Age",
        "Gender",
        *SEPSIS_FEATURES,
    ]

    dataset = pd.read_csv(
        DATASET_PATH,
        usecols=columns,
    )

    wanted_ids = [
        patient["dataset_id"]
        for patient in PATIENTS
    ]

    dataset = dataset[
        dataset["Patient_ID"].isin(
            wanted_ids
        )
    ].copy()

    print(
        f"✓ Found "
        f"{dataset['Patient_ID'].nunique()} "
        f"selected patients"
    )

    # --------------------------------------------------------
    # MONGODB
    # --------------------------------------------------------

    print("\nConnecting to MongoDB...")

    database = MongoDatabase()

    database.ping()

    repository = PatientRepository(
        database
    )

    print("✓ MongoDB connected")

    # --------------------------------------------------------
    # PATIENTS
    # --------------------------------------------------------

    for number, config in enumerate(
        PATIENTS,
        start=1,
    ):

        dataset_id = config[
            "dataset_id"
        ]

        patient_id = (
            f"P{dataset_id}"
        )

        admission_id = (
            f"ADM-{dataset_id}"
        )

        print()
        print("-" * 80)
        print(
            f"[{number}/6] {patient_id}"
        )
        print("-" * 80)

        patient_df = dataset[
            dataset["Patient_ID"]
            == dataset_id
        ].copy()

        if patient_df.empty:
            print(
                "✗ Patient not found"
            )
            continue

        is_positive = (
            patient_df[
                "SepsisLabel"
            ].max()
            == 1
        )

        # ----------------------------------------------------
        # FIND SAME 13-ROW WINDOW AS READ-ONLY SCRIPT
        # ----------------------------------------------------

        if is_positive:
            full_window = (
                choose_positive_window(
                    patient_df
                )
            )
        else:
            full_window = (
                choose_negative_window(
                    patient_df
                )
            )

        full_window = (
            full_window
            .sort_values("Hour")
            .reset_index(drop=True)
        )

        # THIS IS THE IMPORTANT LINE.
        #
        # Observation 13 stays OUT of MongoDB.
        #
        historical_rows = (
            full_window.iloc[:12].copy()
        )

        future_ui_row = (
            full_window.iloc[12]
        )

        # ----------------------------------------------------
        # CLEAN PREVIOUS DEMO COPY
        # ----------------------------------------------------

        delete_previous_demo_patient(
            database,
            patient_id,
        )

        # ----------------------------------------------------
        # CREATE PATIENT
        # ----------------------------------------------------

        first_patient_row = (
            patient_df.iloc[0]
        )

        demographics = {}

        if not pd.isna(
            first_patient_row["Age"]
        ):
            demographics[
                "age"
            ] = float(
                first_patient_row["Age"]
            )

        if not pd.isna(
            first_patient_row["Gender"]
        ):
            demographics[
                "gender"
            ] = int(
                first_patient_row["Gender"]
            )

        repository.create_patient(
            patient_id=patient_id,
            demographics=demographics,
            status="active",
        )

        # ----------------------------------------------------
        # CREATE ADMISSION
        # ----------------------------------------------------

        admission_time = (
            datetime.now(
                timezone.utc
            )
            - timedelta(
                hours=12
            )
        )

        repository.create_admission(
            patient_id=patient_id,
            admission_id=admission_id,
            admission_time=admission_time,
            status="active",
        )

        print(
            "✓ Patient created"
        )

        print(
            "✓ Admission created"
        )

        # ----------------------------------------------------
        # INSERT OBSERVATIONS 1-12
        # ----------------------------------------------------

        for index, row in (
            historical_rows.iterrows()
        ):

            observation_time = (
                admission_time
                + timedelta(
                    hours=index
                )
            )

            clinical_parameters = (
                row_to_clinical_parameters(
                    row
                )
            )

            repository.add_observation(
                patient_id=patient_id,
                admission_id=admission_id,
                observation_time=observation_time,
                clinical_parameters=clinical_parameters,
                source="historical_record",
            )

        print(
            "✓ 12 historical observations stored"
        )

        # ----------------------------------------------------
        # SHOW WHAT WAS NOT STORED
        # ----------------------------------------------------

        print()
        print(
            "Observation #13 RESERVED FOR UI:"
        )

        for feature in SEPSIS_FEATURES:

            value = future_ui_row[
                feature
            ]

            if pd.isna(value):
                display = "-"
            else:
                display = round(
                    float(value),
                    2,
                )

            print(
                f"  {feature:<8} = {display}"
            )

        print(
            f"  Dataset label = "
            f"{int(future_ui_row['SepsisLabel'])}"
        )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        history = (
            repository.get_patient_history(
                patient_id,
                admission_id,
            )
        )

        observations = (
            history.get(
                "observations",
                []
            )
        )

        assessments = (
            history.get(
                "assessments",
                []
            )
        )

        trends = (
            history.get(
                "trends",
                []
            )
        )

        prioritization = (
            history.get(
                "latest_prioritization"
            )
        )

        print()
        print(
            f"Mongo observations : "
            f"{len(observations)}"
        )

        print(
            f"Assessments        : "
            f"{len(assessments)}"
        )

        print(
            f"Trends             : "
            f"{len(trends)}"
        )

        print(
            f"Prioritization     : "
            f"{'present' if prioritization else 'none'}"
        )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("HISTORY LOADING COMPLETE")
    print("=" * 80)

    print(
        "\nExpected state for EACH patient:"
    )

    print(
        "12 observations"
    )

    print(
        "0 assessments"
    )

    print(
        "0 trends"
    )

    print(
        "0 prioritizations"
    )

    print(
        "\nObservation #13 must now come "
        "through the Nurse UI."
    )

    database.close()


if __name__ == "__main__":
    main()