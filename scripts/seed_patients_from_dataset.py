from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sepsis"
    / "Dataset.csv"
)


# Patients already selected for our manual system test.
PATIENTS = [
    {
        "dataset_id": 3373,
        "expected_group": "SEPSIS",
    },
    {
        "dataset_id": 1446,
        "expected_group": "SEPSIS",
    },
    {
        "dataset_id": 1185,
        "expected_group": "SEPSIS",
    },
    {
        "dataset_id": 18823,
        "expected_group": "NON-SEPSIS",
    },
    {
        "dataset_id": 19217,
        "expected_group": "NON-SEPSIS",
    },
    {
        "dataset_id": 100559,
        "expected_group": "NON-SEPSIS",
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

def choose_positive_rows(
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a sepsis-positive patient:

    Show the 12 observations immediately before
    the first SepsisLabel=1 observation, plus
    the first positive observation.

    Total = 13 observations.
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
            "Patient does not contain a positive SepsisLabel."
        )

    first_positive_index = positive_indices[0]

    if first_positive_index < 12:
        raise ValueError(
            "Patient does not have 12 observations "
            "before the first positive label."
        )

    return patient_df.iloc[
        first_positive_index - 12:
        first_positive_index + 1
    ].copy()


def choose_negative_rows(
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a non-sepsis patient:

    Find a 13-observation window containing the
    greatest amount of actually recorded data.
    """

    patient_df = (
        patient_df
        .sort_values("Hour")
        .reset_index(drop=True)
    )

    if len(patient_df) < 13:
        raise ValueError(
            "Patient does not contain 13 observations."
        )

    best_window = None
    best_score = -1

    for start in range(
        len(patient_df) - 12
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
            "Could not select patient window."
        )

    return best_window


def display_value(value):
    """
    Keep missing measurements visible instead of
    inventing/filling values.
    """

    if pd.isna(value):
        return "-"

    if isinstance(value, float):
        return round(value, 2)

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("PATIENTS FOR MANUAL NURSE / UI TESTING")
    print("=" * 100)

    print(
        "\nREAD-ONLY MODE:"
        "\nThis script DOES NOT connect to MongoDB."
        "\nThis script DOES NOT run any agent."
        "\nThis script ONLY reads Dataset.csv.\n"
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    # --------------------------------------------------------
    # Load only columns we actually need.
    # --------------------------------------------------------

    columns = [
        "Patient_ID",
        "Hour",
        "SepsisLabel",
        "Age",
        "Gender",
        *SEPSIS_FEATURES,
    ]

    print("Reading Dataset.csv...")

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
        f"✓ Loaded records for "
        f"{dataset['Patient_ID'].nunique()} patients."
    )

    # --------------------------------------------------------
    # Process each selected patient.
    # --------------------------------------------------------

    for patient_number, config in enumerate(
        PATIENTS,
        start=1,
    ):

        patient_id = config["dataset_id"]

        patient_df = dataset[
            dataset["Patient_ID"]
            == patient_id
        ].copy()

        if patient_df.empty:
            print()
            print(
                f"Patient {patient_id} "
                f"was not found."
            )
            continue

        actual_positive = (
            patient_df["SepsisLabel"].max()
            == 1
        )

        if actual_positive:
            window = choose_positive_rows(
                patient_df
            )
        else:
            window = choose_negative_rows(
                patient_df
            )

        window = (
            window
            .sort_values("Hour")
            .reset_index(drop=True)
        )

        first_row = patient_df.iloc[0]

        print()
        print()
        print("=" * 100)

        print(
            f"PATIENT {patient_number}/6"
        )

        print("=" * 100)

        print(
            f"Patient ID to use in UI : P{patient_id}"
        )

        print(
            f"Admission ID to use     : ADM-{patient_id}"
        )

        print(
            f"Dataset group           : "
            f"{'SEPSIS' if actual_positive else 'NON-SEPSIS'}"
        )

        if not pd.isna(
            first_row["Age"]
        ):
            print(
                f"Age                     : "
                f"{display_value(first_row['Age'])}"
            )

        if not pd.isna(
            first_row["Gender"]
        ):
            print(
                f"Gender value            : "
                f"{display_value(first_row['Gender'])}"
            )

        print()
        print(
            "OBSERVATIONS TO ENTER"
        )

        print("-" * 100)

        display_rows = []

        for index, row in window.iterrows():

            display_rows.append(
                {
                    "#": index + 1,

                    "Hour":
                        display_value(
                            row["Hour"]
                        ),

                    "HR":
                        display_value(
                            row["HR"]
                        ),

                    "O2Sat":
                        display_value(
                            row["O2Sat"]
                        ),

                    "Temp":
                        display_value(
                            row["Temp"]
                        ),

                    "SBP":
                        display_value(
                            row["SBP"]
                        ),

                    "MAP":
                        display_value(
                            row["MAP"]
                        ),

                    "Resp":
                        display_value(
                            row["Resp"]
                        ),

                    "WBC":
                        display_value(
                            row["WBC"]
                        ),

                    "Lactate":
                        display_value(
                            row["Lactate"]
                        ),

                    "Label":
                        int(
                            row["SepsisLabel"]
                        ),
                }
            )

        display_df = pd.DataFrame(
            display_rows
        )

        print(
            display_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Highlight final observation.
        # ----------------------------------------------------

        final_row = window.iloc[-1]

        print()
        print(
            "LATEST OBSERVATION"
        )

        print("-" * 50)

        for feature in SEPSIS_FEATURES:
            print(
                f"{feature:<10}: "
                f"{display_value(final_row[feature])}"
            )

        print(
            f"{'Label':<10}: "
            f"{int(final_row['SepsisLabel'])}"
        )

        print()

        if actual_positive:
            print(
                "NOTE: Observation 13 is the "
                "first SepsisLabel = 1 observation."
            )
        else:
            print(
                "NOTE: This patient's selected "
                "window remains SepsisLabel = 0."
            )

    print()
    print()
    print("=" * 100)

    print(
        "FINISHED — NOTHING WAS WRITTEN TO MONGODB"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()