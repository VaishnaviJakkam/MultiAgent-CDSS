import json
from datetime import datetime, timezone

from src.database.mongodb import MongoDatabase
from src.database.repositories import PatientRepository


def main():

    print("\nConnecting to MongoDB...")

    database = MongoDatabase()

    try:

        if database.ping():
            print("MongoDB connected successfully.")

        repository = PatientRepository(database)

        patient_id = "TEST-PATIENT-MONGO-001"
        admission_id = "ADM-MONGO-001"

        # --------------------------------------------------
        # PATIENT
        # --------------------------------------------------

        existing_patient = repository.get_patient(
            patient_id
        )

        if existing_patient is None:

            patient = repository.create_patient(
                patient_id=patient_id,
                demographics={
                    "name": "Test Patient",
                    "age": 55,
                    "sex": "female",
                },
            )

            print("\nPatient created.")

        else:

            patient = existing_patient

            print("\nPatient already exists.")

        print(
            json.dumps(
                patient,
                indent=2,
                default=str,
            )
        )

        # --------------------------------------------------
        # ADMISSION
        # --------------------------------------------------

        existing_admission = (
            repository.get_admission(
                patient_id,
                admission_id,
            )
        )

        if existing_admission is None:

            admission = (
                repository.create_admission(
                    patient_id=patient_id,
                    admission_id=admission_id,
                    admission_time=datetime.now(
                        timezone.utc
                    ),
                )
            )

            print("\nAdmission created.")

        else:

            admission = existing_admission

            print("\nAdmission already exists.")

        print(
            json.dumps(
                admission,
                indent=2,
                default=str,
            )
        )

        # --------------------------------------------------
        # OBSERVATION
        # --------------------------------------------------

        observation = repository.add_observation(
            patient_id=patient_id,
            admission_id=admission_id,
            observation_time=datetime.now(
                timezone.utc
            ),
            clinical_parameters={
                "HR": 108.0,
                "O2Sat": 93.0,
                "Temp": 38.4,
                "SBP": 102.0,
                "DBP": 64.0,
                "MAP": 76.67,
                "Resp": 24.0,
                "WBC": 9.0,
                "Lactate": 2.8,
            },
            source="input_pipeline",
        )

        print(
            "\n========== STORED OBSERVATION =========="
        )

        print(
            json.dumps(
                observation,
                indent=2,
                default=str,
            )
        )

        # --------------------------------------------------
        # READ HISTORY
        # --------------------------------------------------

        history = repository.get_patient_history(
            patient_id,
            admission_id,
        )

        print(
            "\n========== PATIENT HISTORY =========="
        )

        print(
            json.dumps(
                history,
                indent=2,
                default=str,
            )
        )

        print(
            "\nSUCCESS: MongoDB repository is working."
        )

    finally:

        database.close()


if __name__ == "__main__":
    main()