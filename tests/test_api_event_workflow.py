from __future__ import annotations

from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
sys.modules.setdefault("faster_whisper", SimpleNamespace(WhisperModel=object))
import src.input.nurse_audio_agent as nurse_audio_module
from src.database.repositories import InMemoryDatabase
from src.events import EventType


class FakeReportAgent:
    def analyze(self, storage_reference: str) -> dict:
        return {
            "lab_results": {
                "WBC": {"value": 8.5, "unit": "thou/mm3"},
                "Lactate": {"value": 1.2, "unit": "mmol/L"},
            }
        }


class FakeSepsisAgent:
    def analyze(self, history: list[dict]) -> dict:
        return {
            "patient_id": history[-1]["Patient_ID"],
            "disease": "Sepsis",
            "model": "test",
            "prediction": 0,
            "probability": 0.2,
            "threshold": 0.5,
            "risk_level": "LOWER",
            "observation_count": len(history),
            "assessment_time": history[-1]["Hour"],
            "status": "success",
            "message": "test",
        }


class FakeAkiAgent:
    def analyze(self, patient_data: dict) -> dict:
        return {
            "patient_id": patient_data["patient_reference"],
            "disease": "AKI",
            "model": "test",
            "prediction": 0,
            "probability": 0.1,
            "threshold": 0.5,
            "risk_level": "LOWER",
            "observation_count": 1,
            "assessment_time": None,
            "status": "success",
            "message": "test",
        }


@pytest.fixture
def client() -> TestClient:
    database = InMemoryDatabase()
    database["patients"].insert_one({"_id": "patient-P001", "patient_id": "P001"})
    database["admissions"].insert_one(
        {
            "_id": "admission-A001",
            "patient_id": "P001",
            "admission_id": "A001",
            "admission_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "status": "active",
        }
    )
    api.configure_workflow(
        database,
        FakeSepsisAgent(),
        FakeAkiAgent(),
        report_agent_instance=FakeReportAgent(),
    )
    return TestClient(api.app)


def upload_report(client: TestClient) -> dict:
    response = client.post(
        "/api/lab/upload-report",
        data={"patient_id": "P001", "admission_id": "A001"},
        files={"report_file": ("labs.png", b"report", "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_persists_report_and_starts_event_workflow(client: TestClient) -> None:
    result = upload_report(client)

    assert result["report_id"].startswith("report_")
    assert result["workflow_started"] is True
    assert result["status"] == "PROCESSED"
    assert result["task_id"]
    assert api.lab_report_repository.get_report(result["report_id"]) is not None
    assert api.event_repository.list_events(event_type="REPORT_UPLOADED")
    assert api.event_repository.list_events(event_type="REPORT_PROCESSED")
    assert api.event_repository.list_events(event_type="GENERATE_RESULT_REQUESTED")
    assert api.event_repository.list_events(event_type="NURSE_INPUT_REQUIRED")


def test_nurse_task_retrieval_audio_progression_and_completion(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = upload_report(client)
    task_id = result["task_id"]
    task = client.get(f"/api/nurse/tasks/{task_id}").json()
    assert task["metadata"]["report_id"] == result["report_id"]
    assert task["metadata"]["current_question"]

    current_field = {"name": None}

    class FakeAudioAgent:
        def analyze(self, audio_path: str) -> dict:
            return {
                "extracted_parameters": {
                    "HR": 80.0,
                    "O2Sat": 98.0,
                    "Temp": 37.0,
                    "BP": {"SBP": 120.0, "DBP": 80.0},
                    "Resp": 16.0,
                }
            }

    monkeypatch.setattr(nurse_audio_module, "NurseAudioAgent", FakeAudioAgent)
    completed = None
    for _ in range(10):
        task = client.get(f"/api/nurse/tasks/{task_id}").json()
        fields = task["metadata"].get("question_fields", task["required_fields"])
        index = task["metadata"].get("question_index", 0)
        if index >= len(fields):
            break
        current_field["name"] = fields[index]
        response = client.post(
            f"/api/nurse/tasks/{task_id}/audio",
            files={"audio_file": ("answer.wav", b"audio", "audio/wav")},
        )
        assert response.status_code == 200, response.text
        completed = response.json()

    assert completed is not None
    assert completed["completed"] is True
    assert completed["observation_id"]
    assert completed["assessment_started"] is True
    assert api.repository.get_patient_assessments("P001", "A001")


def test_doctor_history_and_dashboard_summary(client: TestClient) -> None:
    result = upload_report(client)

    doctor = client.get(
        "/api/doctor/patient/P001",
        params={"admission_id": "A001"},
    )
    assert doctor.status_code == 200, doctor.text
    payload = doctor.json()
    assert set(payload) == {
        "patient",
        "admission",
        "observations",
        "assessments",
        "trends",
        "prioritizations",
        "reports",
        "tasks",
    }
    assert payload["reports"][0]["report_id"] == result["report_id"]

    dashboard = client.get("/api/dashboard/patients")
    assert dashboard.status_code == 200, dashboard.text
    patient = dashboard.json()["patients"][0]
    assert "latest_report" in patient
    assert "pending_task_count" in patient
    assert "latest_assessment" in patient
    assert "latest_trend" in patient
    assert "latest_prioritization" in patient


def test_end_to_end_workflow_reaches_completion_and_notifications(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_handlers = {
        EventType.REPORT_UPLOADED,
        EventType.REPORT_PROCESSED,
        EventType.GENERATE_RESULT_REQUESTED,
        EventType.NURSE_INPUT_REQUIRED,
        EventType.NURSE_INPUT_RECEIVED,
        EventType.OBSERVATION_READY,
        EventType.ASSESSMENT_COMPLETED,
        EventType.TREND_ANALYSIS_COMPLETED,
    }
    assert expected_handlers <= set(api.event_bus._handlers)

    current_field = {"name": None}

    class FakeAudioAgent:
        def analyze(self, audio_path: str) -> dict:
            return {
                "extracted_parameters": {
                    "HR": 80.0,
                    "O2Sat": 98.0,
                    "Temp": 37.0,
                    "BP": {"SBP": 120.0, "DBP": 80.0},
                    "Resp": 16.0,
                }
            }

    monkeypatch.setattr(nurse_audio_module, "NurseAudioAgent", FakeAudioAgent)

    def complete_task(task_id: str) -> None:
        for _ in range(10):
            task = client.get(f"/api/nurse/tasks/{task_id}").json()
            fields = task["metadata"].get("question_fields", task["required_fields"])
            index = task["metadata"].get("question_index", 0)
            if index >= len(fields):
                return
            current_field["name"] = fields[index]
            response = client.post(
                f"/api/nurse/tasks/{task_id}/audio",
                files={"audio_file": ("answer.wav", b"audio", "audio/wav")},
            )
            assert response.status_code == 200, response.text
        raise AssertionError("nurse task did not complete")

    first = upload_report(client)
    complete_task(first["task_id"])
    second = upload_report(client)
    complete_task(second["task_id"])

    event_types = [event["event_type"] for event in api.event_repository.list_events()]
    assert "REPORT_UPLOADED" in event_types
    assert "REPORT_PROCESSED" in event_types
    assert "GENERATE_RESULT_REQUESTED" in event_types
    assert "NURSE_INPUT_REQUIRED" in event_types
    assert "OBSERVATION_READY" in event_types
    assert "ASSESSMENT_COMPLETED" in event_types
    assert "TREND_ANALYSIS_COMPLETED" in event_types
    assert "RISK_PRIORITIZATION_COMPLETED" in event_types
    assert "WORKFLOW_COMPLETED" in event_types
    assert all(event["status"] == "COMPLETED" for event in api.event_repository.list_events())

    completed = api.event_repository.list_events(event_type="WORKFLOW_COMPLETED")
    assert completed[0]["payload"]["workflow_status"] == "COMPLETED"
    assert client.get("/api/notifications/NURSE").status_code == 200
    doctor_notifications = client.get("/api/notifications/DOCTOR").json()
    assert {item["notification_type"] for item in doctor_notifications} >= {
        "ASSESSMENT_COMPLETED",
        "RISK_PRIORITIZATION_COMPLETED",
    }

    for collection_name in (
        "lab_reports",
        "workflow_events",
        "workflow_tasks",
        "observations",
        "assessments",
        "trends",
        "prioritizations",
    ):
        assert api.database[collection_name].find({}), collection_name

    nurse_tasks = client.get("/api/nurse/tasks", params={"patient_id": "P001", "admission_id": "A001"})
    assert nurse_tasks.status_code == 200
    assert len(nurse_tasks.json()) >= 2

    doctor_history = client.get("/api/doctor/patient/P001", params={"admission_id": "A001"})
    assert doctor_history.status_code == 200
    assert doctor_history.json()["prioritizations"]

    dashboard = client.get("/api/dashboard/patients")
    assert dashboard.status_code == 200
    assert any(item["patient_id"] == "P001" for item in dashboard.json()["patients"])
