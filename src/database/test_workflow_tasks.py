from datetime import datetime, timezone

import pytest

from src.database.repositories import InMemoryDatabase, PatientRepository, WorkflowTaskRepository
from src.database.workflow_tasks import TaskStatus, TaskType


@pytest.fixture
def repository() -> WorkflowTaskRepository:
    database = InMemoryDatabase()
    patient_repository = PatientRepository(database)
    patient_repository.create_patient("P001")
    patient_repository.create_patient("P002")
    patient_repository.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    patient_repository.create_admission("P002", "A002", datetime(2026, 1, 1, tzinfo=timezone.utc))
    return WorkflowTaskRepository(database)


def create_task(repository: WorkflowTaskRepository, task_id: str = "T001") -> dict:
    return repository.create_task(
        task_id=task_id,
        task_type=TaskType.NURSE_INPUT,
        patient_id="P001",
        admission_id="A001",
        required_fields=["HR", "Temp"],
        metadata={"assigned_role": "NURSE"},
    )


def test_create_and_retrieve_task(repository: WorkflowTaskRepository) -> None:
    task = create_task(repository)

    assert task["task_id"] == "T001"
    assert task["task_type"] == TaskType.NURSE_INPUT.value
    assert task["status"] == TaskStatus.PENDING.value
    assert task["required_fields"] == ["HR", "Temp"]
    assert task["provided_fields"] == []
    assert task["completed_at"] is None
    assert repository.get_task("T001") == task


def test_pending_tasks_are_filtered_and_patient_admission_scoped(repository: WorkflowTaskRepository) -> None:
    create_task(repository, "T001")
    repository.create_task(TaskType.NURSE_INPUT, "P002", "A002", task_id="T002")

    assert [task["task_id"] for task in repository.get_pending_tasks()] == ["T001", "T002"]
    assert [task["task_id"] for task in repository.get_patient_tasks("P001", "A001")] == ["T001"]
    assert repository.get_patient_tasks("P001", "A002") == []
    assert [task["task_id"] for task in repository.get_patient_tasks("P002", "A002")] == ["T002"]


def test_update_task_changes_fields_and_updated_at(repository: WorkflowTaskRepository) -> None:
    task = create_task(repository)
    original_created_at = task["created_at"]
    original_updated_at = task["updated_at"]

    updated = repository.update_task(
        "T001",
        status=TaskStatus.IN_PROGRESS,
        provided_fields=["HR"],
        metadata={"assigned_role": "NURSE", "attempt": 1},
    )

    assert updated["status"] == TaskStatus.IN_PROGRESS.value
    assert updated["provided_fields"] == ["HR"]
    assert updated["metadata"]["attempt"] == 1
    assert updated["created_at"] == original_created_at
    assert updated["updated_at"] >= original_updated_at
    assert updated["completed_at"] is None


def test_complete_task_sets_completed_at_and_updates_status(repository: WorkflowTaskRepository) -> None:
    task = create_task(repository)

    completed = repository.complete_task(task["task_id"])

    assert completed["status"] == TaskStatus.COMPLETED.value
    assert completed["completed_at"] is not None
    assert completed["updated_at"] >= completed["created_at"]
    assert repository.get_pending_tasks() == []


def test_non_completed_updates_clear_completed_at(repository: WorkflowTaskRepository) -> None:
    completed = repository.complete_task(create_task(repository)["task_id"])

    reopened = repository.update_task(completed["task_id"], status=TaskStatus.IN_PROGRESS)

    assert reopened["status"] == TaskStatus.IN_PROGRESS.value
    assert reopened["completed_at"] is None


def test_duplicate_task_id_is_rejected(repository: WorkflowTaskRepository) -> None:
    create_task(repository)

    with pytest.raises(ValueError, match="Task already exists"):
        create_task(repository)


def test_invalid_task_type_and_status_are_rejected(repository: WorkflowTaskRepository) -> None:
    with pytest.raises(ValueError, match="invalid task_type"):
        repository.create_task("UNKNOWN", "P001", "A001")

    create_task(repository)
    with pytest.raises(ValueError, match="invalid status"):
        repository.update_task("T001", status="UNKNOWN")


def test_unknown_and_cross_scope_tasks_are_rejected_or_empty(repository: WorkflowTaskRepository) -> None:
    with pytest.raises(ValueError, match="Task not found"):
        repository.update_task("missing", status=TaskStatus.IN_PROGRESS)

    assert repository.get_patient_tasks("P001", "A002") == []
