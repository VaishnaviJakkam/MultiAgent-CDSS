from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.aki_agent import AKIDetectionAgent
from src.agents.gemini_planner import GeminiPlanner
from src.agents.llm_disease_assessment_agent import (
    LLMDiseaseAssessmentAgent,
)
from src.agents.sepsis_agent import SepsisDetectionAgent
from src.database.repositories import (
    InMemoryDatabase,
    PatientRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEPSIS_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sepsis"
    / "Dataset.csv"
)

AKI_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aki"
    / "Raw_aki_patient_data.csv"
)

AKI_METADATA = (
    PROJECT_ROOT
    / "models"
    / "aki_xgboost"
    / "metadata.json"
)

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


def python_value(value):
    """
    Convert numpy values into normal Python values.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, np.generic):
        return value.item()

    return value


def load_sepsis_test_sequence():
    """
    Pick one real Sepsis patient having at least 12 observations.
    """

    df = pd.read_csv(SEPSIS_DATASET)

    required = [
        "Patient_ID",
        *SEPSIS_FEATURES,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing Sepsis columns: {missing}"
        )

    patient_counts = (
        df.groupby("Patient_ID")
        .size()
        .sort_values(ascending=False)
    )

    eligible = patient_counts[
        patient_counts >= 12
    ]

    if eligible.empty:
        raise RuntimeError(
            "No Sepsis patient has at least "
            "12 observations."
        )

    patient_id = eligible.index[0]

    patient_df = (
        df[
            df["Patient_ID"] == patient_id
        ]
        .head(12)
        .copy()
    )

    return patient_df


def load_aki_test_row():
    """
    Find one real AKI dataset row containing every feature
    required by the trained XGBoost model.
    """

    metadata = json.loads(
        AKI_METADATA.read_text(
            encoding="utf-8"
        )
    )

    feature_names = metadata[
        "feature_columns"
    ]

    df = pd.read_csv(
        AKI_DATASET
    )

    missing_columns = [
        column
        for column in feature_names
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "AKI raw dataset is missing "
            f"model features: {missing_columns}"
        )

    candidate_df = (
        df[feature_names]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if candidate_df.empty:
        raise RuntimeError(
            "Could not find an AKI dataset row "
            "with complete finite values for all "
            "trained model features."
        )

    row = candidate_df.iloc[0]

    return {
        feature: python_value(
            row[feature]
        )
        for feature in feature_names
    }


def main():

    print(
        "\nLoading real datasets..."
    )

    sepsis_sequence = (
        load_sepsis_test_sequence()
    )

    aki_values = (
        load_aki_test_row()
    )

    print(
        "Sepsis observations:",
        len(sepsis_sequence),
    )

    print(
        "AKI model features:",
        len(aki_values),
    )

    # ---------------------------------------------------------
    # In-memory database
    # ---------------------------------------------------------

    database = InMemoryDatabase()

    repository = PatientRepository(
        database
    )

    patient_id = "INTEGRATION_PATIENT_001"

    admission_id = "INTEGRATION_ADMISSION_001"

    start_time = datetime.now(
        timezone.utc
    ) - timedelta(hours=11)

    repository.create_patient(
        patient_id=patient_id,
        demographics={
            "source": (
                "real-model integration test"
            ),
        },
    )

    repository.create_admission(
        patient_id=patient_id,
        admission_id=admission_id,
        admission_time=start_time,
    )

    # ---------------------------------------------------------
    # Add 12 real Sepsis observations
    # ---------------------------------------------------------

    for index, (_, row) in enumerate(
        sepsis_sequence.iterrows()
    ):

        clinical_parameters = {
            feature: python_value(
                row[feature]
            )
            for feature
            in SEPSIS_FEATURES
        }

        # AKI agent uses the latest observation.
        # Therefore add the real AKI feature row only
        # to observation number 12.
        if index == 11:

            clinical_parameters.update(
                aki_values
            )

        repository.add_observation(
            patient_id=patient_id,
            admission_id=admission_id,
            observation_time=(
                start_time
                + timedelta(
                    hours=index
                )
            ),
            clinical_parameters=(
                clinical_parameters
            ),
            source="integration_test",
        )

    # ---------------------------------------------------------
    # Real tools
    # ---------------------------------------------------------

    print(
        "\nLoading real ML models..."
    )

    sepsis_tool = (
        SepsisDetectionAgent(
            model="advanced"
        )
    )

    aki_tool = (
        AKIDetectionAgent(
            model="advanced"
        )
    )

    print(
        "Loading Gemini planner..."
    )

    gemini = GeminiPlanner()

    # ---------------------------------------------------------
    # Agent 1
    # ---------------------------------------------------------

    agent = (
        LLMDiseaseAssessmentAgent(
            repository=repository,
            sepsis_tool=sepsis_tool,
            aki_tool=aki_tool,
            llm=gemini,
        )
    )

    print(
        "\nRunning Agent 1...\n"
    )

    result = agent.analyze(
        patient_id=patient_id,
        admission_id=admission_id,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print(
        "=" * 60
    )

    print(
        "AGENT 1 RESULT"
    )

    print(
        "=" * 60
    )

    print(
        "\nOverall status:",
        result["status"],
    )

    print(
        "\nSepsis result:"
    )

    print(
        json.dumps(
            result["sepsis"],
            indent=2,
            default=str,
        )
    )

    print(
        "\nAKI result:"
    )

    print(
        json.dumps(
            result["aki"],
            indent=2,
            default=str,
        )
    )

    print(
        "\nReAct trace:"
    )

    for entry in result["trace"]:

        print(
            json.dumps(
                entry,
                indent=2,
                default=str,
            )
        )

    # ---------------------------------------------------------
    # Verify stored assessment
    # ---------------------------------------------------------

    assessments = (
        repository
        .get_patient_assessments(
            patient_id,
            admission_id,
        )
    )

    print(
        "\nStored assessments:",
        len(assessments),
    )

    if assessments:

        print(
            "\nStored assessment:"
        )

        print(
            json.dumps(
                assessments[-1],
                indent=2,
                default=str,
            )
        )

    # ---------------------------------------------------------
    # Show current in-memory patient history
    # ---------------------------------------------------------

    print(
        "\n========== CURRENT IN-MEMORY DATABASE =========="
    )

    history = repository.get_patient_history(
        patient_id,
        admission_id,
    )

    print(
        json.dumps(
            history,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()