from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd


# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.database.mongodb import MongoDatabase
from src.database.repositories import PatientRepository


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sepsis"
    / "Dataset.csv"
)


DEMO_PATIENTS = [
    {
        "patient_id": "P1446",
        "admission_id": "ADM-1446",
        "dataset_patient_id": 1446,
    },
    {
        "patient_id": "P1185",
        "admission_id": "ADM-1185",
        "dataset_patient_id": 1185,
    },
    {
        "patient_id": "P19217",
        "admission_id": "ADM-19217",
        "dataset_patient_id": 19217,
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


COLLECTIONS_TO_CLEAR = [
    "patients",
    "admissions",
    "observations",
    "assessments",
    "trends",
    "prioritizations",
]


# =========================================================
# HELPERS
# =========================================================

def clean_number(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}"
        )

    print("\nLoading Sepsis dataset...")

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "Patient_ID",
        "Hour",
        *SEPSIS_FEATURES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    print(f"Loaded {len(df):,} rows.")

    return df


def get_patient_rows(
    df: pd.DataFrame,
    dataset_patient_id: int,
) -> pd.DataFrame:

    rows = df[
        df["Patient_ID"] == dataset_patient_id
    ].copy()

    rows = (
        rows
        .sort_values("Hour")
        .reset_index(drop=True)
    )

    return rows


def clear_database(database: MongoDatabase):
    print("\nClearing demo collections...")

    for name in COLLECTIONS_TO_CLEAR:
        collection = database[name]

        count_before = collection.count_documents({})

        result = collection.delete_many({})

        print(
            f"{name:<18} "
            f"{count_before:>5} -> 0 "
            f"(deleted {result.deleted_count})"
        )


def build_clinical_parameters(row: pd.Series):
    return {
        feature: clean_number(row[feature])
        for feature in SEPSIS_FEATURES
    }


# =========================================================
# LOAD ONE PATIENT
# =========================================================

def seed_patient(
    repository: PatientRepository,
    df: pd.DataFrame,
    config: dict,
):
    patient_id = config["patient_id"]
    admission_id = config["admission_id"]
    dataset_patient_id = config["dataset_patient_id"]

    print("\n" + "=" * 60)
    print(f"Seeding {patient_id} / {admission_id}")
    print("=" * 60)

    rows = get_patient_rows(
        df,
        dataset_patient_id,
    )

    if len(rows) < 12:
        raise ValueError(
            f"{patient_id} has only {len(rows)} rows."
        )

    historical_rows = rows.iloc[:12].copy()

    # -----------------------------------------------------
    # Create patient using repository contract
    # -----------------------------------------------------

    repository.create_patient(
        patient_id=patient_id,
        demographics={
            "dataset_patient_id": dataset_patient_id,
            "demo_patient": True,
        },
        status="active",
    )

    # -----------------------------------------------------
    # Create admission using repository contract
    # -----------------------------------------------------

    admission_time = (
        datetime.now(timezone.utc)
        - timedelta(hours=13)
    )

    repository.create_admission(
        patient_id=patient_id,
        admission_id=admission_id,
        admission_time=admission_time,
        status="active",
    )

    # -----------------------------------------------------
    # Add first 12 observations using repository contract
    # -----------------------------------------------------

    for index, row in historical_rows.iterrows():

        observation_time = (
            admission_time
            + timedelta(hours=index + 1)
        )

        clinical_parameters = (
            build_clinical_parameters(row)
        )

        repository.add_observation(
            patient_id=patient_id,
            admission_id=admission_id,
            observation_time=observation_time,
            clinical_parameters=clinical_parameters,
            source="historical_dataset",
        )

        print(
            f"Obs {index + 1:>2} "
            f"| HR={clinical_parameters['HR']} "
            f"| SpO2={clinical_parameters['O2Sat']} "
            f"| Temp={clinical_parameters['Temp']} "
            f"| SBP={clinical_parameters['SBP']} "
            f"| MAP={clinical_parameters['MAP']} "
            f"| Resp={clinical_parameters['Resp']} "
            f"| WBC={clinical_parameters['WBC']} "
            f"| Lactate={clinical_parameters['Lactate']}"
        )

    print(f"\n{patient_id} seeded with 12 observations.")


# =========================================================
# VERIFY
# =========================================================

def verify(
    database: MongoDatabase,
    repository: PatientRepository,
):

    print("\n" + "=" * 68)
    print("FINAL VERIFICATION")
    print("=" * 68)

    totals = {
        "patients":
            database["patients"].count_documents({}),

        "admissions":
            database["admissions"].count_documents({}),

        "observations":
            database["observations"].count_documents({}),

        "assessments":
            database["assessments"].count_documents({}),

        "trends":
            database["trends"].count_documents({}),

        "prioritizations":
            database["prioritizations"].count_documents({}),
    }

    for name, count in totals.items():
        print(f"{name:<18}: {count}")

    print("\nPer-patient check:")

    for config in DEMO_PATIENTS:
        patient_id = config["patient_id"]
        admission_id = config["admission_id"]

        admission = repository.get_admission(
            patient_id,
            admission_id,
        )

        observations = (
            repository.get_patient_observations(
                patient_id,
                admission_id,
            )
        )

        assessments = (
            repository.get_patient_assessments(
                patient_id,
                admission_id,
            )
        )

        trends = (
            repository.get_patient_trends(
                patient_id,
                admission_id,
            )
        )

        priority = (
            repository.get_latest_prioritization(
                patient_id,
                admission_id,
            )
        )

        print(
            f"{patient_id:<8} "
            f"admission={'OK' if admission else 'MISSING'} "
            f"| observations={len(observations)} "
            f"| assessments={len(assessments)} "
            f"| trends={len(trends)} "
            f"| priority={'YES' if priority else 'NO'}"
        )

    valid = (
        totals["patients"] == 3
        and totals["admissions"] == 3
        and totals["observations"] == 36
        and totals["assessments"] == 0
        and totals["trends"] == 0
        and totals["prioritizations"] == 0
    )

    print()

    if valid:
        print("DEMO DATABASE READY")
        print()
        print("Expected state:")
        print("P1446  -> 12 historical observations")
        print("P1185  -> 12 historical observations")
        print("P19217 -> 12 historical observations")
        print()
        print("Next step:")
        print(
            "Submit Observation 13 through Nurse Dashboard."
        )
    else:
        print(
            "WARNING: database counts are not as expected."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 68)
    print("MULTI-AGENT CDSS — CLEAN DEMO RESET")
    print("=" * 68)

    print(
        "\nThis will DELETE all documents from:"
    )

    for name in COLLECTIONS_TO_CLEAR:
        print(f"  - {name}")

    print(
        "\nThen it will create ONLY:"
    )

    for config in DEMO_PATIENTS:
        print(
            f"  - {config['patient_id']} "
            f"/ {config['admission_id']}"
        )

    confirmation = input(
        '\nType "RESET DEMO" to continue: '
    )

    if confirmation != "RESET DEMO":
        print("\nReset cancelled.")
        return

    database = MongoDatabase()

    try:
        print("\nConnecting to MongoDB...")

        if not database.ping():
            raise ConnectionError(
                "MongoDB ping failed."
            )

        print(
            f"Connected to database: "
            f"{database.database_name}"
        )

        repository = PatientRepository(
            database
        )

        df = load_dataset()

        # -------------------------------------------------
        # Validate BEFORE deleting existing data
        # -------------------------------------------------

        print("\nValidating patients before reset...")

        for config in DEMO_PATIENTS:

            rows = get_patient_rows(
                df,
                config["dataset_patient_id"],
            )

            print(
                f"{config['patient_id']}: "
                f"{len(rows)} dataset rows"
            )

            if len(rows) < 12:
                raise ValueError(
                    f"{config['patient_id']} "
                    "does not have 12 observations."
                )

        print(
            "\nAll three patients validated."
        )

        # -------------------------------------------------
        # Clear current database
        # -------------------------------------------------

        clear_database(
            database
        )

        # -------------------------------------------------
        # Seed patients correctly
        # -------------------------------------------------

        for config in DEMO_PATIENTS:

            seed_patient(
                repository=repository,
                df=df,
                config=config,
            )

        # -------------------------------------------------
        # Verify
        # -------------------------------------------------

        verify(
            database=database,
            repository=repository,
        )

    finally:
        database.close()


if __name__ == "__main__":
    main()