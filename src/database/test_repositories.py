from datetime import datetime, timedelta, timezone

import pytest

from src.database.repositories import InMemoryDatabase, PatientRepository


@pytest.fixture
def repository() -> PatientRepository:
    return PatientRepository(InMemoryDatabase())


@pytest.fixture
def admission(repository: PatientRepository) -> PatientRepository:
    repository.create_patient("P001", {"age": 42})
    repository.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    return repository


def test_create_and_retrieve_patient(repository: PatientRepository) -> None:
    created = repository.create_patient("P001", {"sex": "F"})
    assert repository.get_patient("P001")["patient_id"] == created["patient_id"]


def test_create_and_retrieve_admission(admission: PatientRepository) -> None:
    assert admission.get_admission("P001", "A001")["admission_id"] == "A001"


def test_multiple_observations_are_preserved_and_chronological(admission: PatientRepository) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    admission.add_observation("P001", "A001", start + timedelta(hours=4), {"HR": 90})
    admission.add_observation("P001", "A001", start, {"HR": 80})
    observations = admission.get_patient_observations("P001", "A001")
    assert len(observations) == 2
    assert [item["clinical_parameters"]["HR"] for item in observations] == [80, 90]


def test_multiple_assessments_are_preserved(admission: PatientRepository) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    admission.add_assessment("P001", "A001", start, sepsis_result={"probability": 0.2})
    admission.add_assessment("P001", "A001", start + timedelta(hours=1), sepsis_result={"probability": 0.6})
    assessments = admission.get_patient_assessments("P001", "A001")
    assert len(assessments) == 2
    assert assessments[0]["sepsis"]["probability"] == 0.2
    assert assessments[1]["sepsis"]["probability"] == 0.6


def test_trend_and_latest_prioritization(admission: PatientRepository) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    admission.add_trend("P001", "A001", "Sepsis", timestamp, "WORSENING", 0.2, 0.6, 0.4, 2)
    admission.add_prioritization("P001", "A001", "HIGH", "Sepsis", 0.6, ["Sepsis"], "Worsening", timestamp)
    assert admission.get_patient_trends("P001", "A001")[0]["trend"] == "WORSENING"
    assert admission.get_latest_prioritization("P001", "A001")["priority_level"] == "HIGH"


def test_complete_history_is_scoped_and_chronological(admission: PatientRepository) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    admission.add_observation("P001", "A001", timestamp, {"Temp": 37.0})
    admission.add_assessment("P001", "A001", timestamp, aki_result={"probability": 0.2})
    history = admission.get_patient_history("P001", "A001")
    assert history["patient"]["patient_id"] == "P001"
    assert history["admission"]["admission_id"] == "A001"
    assert len(history["observations"]) == 1
    assert len(history["assessments"]) == 1


def test_cross_patient_and_admission_history_cannot_mix(repository: PatientRepository) -> None:
    repository.create_patient("P001")
    repository.create_patient("P002")
    repository.create_admission("P001", "A001", datetime.now(timezone.utc))
    repository.create_admission("P002", "A002", datetime.now(timezone.utc))
    repository.add_observation("P001", "A001", datetime.now(timezone.utc), {"HR": 80})
    with pytest.raises(ValueError):
        repository.get_patient_history("P001", "A002")
    with pytest.raises(ValueError):
        repository.add_observation("P001", "A002", datetime.now(timezone.utc), {"HR": 80})


def test_same_patient_can_have_multiple_admissions(repository: PatientRepository) -> None:
    repository.create_patient("P001")
    repository.create_admission("P001", "A001", datetime.now(timezone.utc))
    repository.create_admission("P001", "A002", datetime.now(timezone.utc))
    assert repository.get_admission("P001", "A001")
    assert repository.get_admission("P001", "A002")


def test_boundary_validation(repository: PatientRepository) -> None:
    with pytest.raises(ValueError, match="patient_id cannot be empty"):
        repository.create_patient(" ")
    with pytest.raises(ValueError, match="admission_id cannot be empty"):
        repository.create_patient("P001")
        repository.create_admission("P001", " ", datetime.now(timezone.utc))
    with pytest.raises(ValueError, match="must be a datetime"):
        repository.create_patient("P002")
        repository.create_admission("P002", "A002", "2026-01-01")
