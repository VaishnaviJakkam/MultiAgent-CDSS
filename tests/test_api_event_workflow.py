from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.assessment_event_workflow import ObservationReadyAssessmentHandler
from src.agents.risk_prioritization_event_workflow import TrendCompletedRiskPrioritizationHandler
from src.agents.trend_event_workflow import AssessmentCompletedTrendHandler
from src.database.lab_reports import ReportProcessingStatus
from src.database.repositories import (
    InMemoryDatabase,
    LabReportRepository,
    PatientRepository,
    WorkflowEventRepository,
    WorkflowTaskRepository,
)
from src.events import EventBus, EventType
from src.input.nurse_input_workflow import NurseInputReceivedEventHandler, NurseInputSubmissionService
from src.input.report_event_workflow import (
    GenerateResultRequestService,
    GenerateResultRequestedEventHandler,
    LabReportSubmissionService,
    ReportProcessingEventHandler,
)
from src.api.workflow_routes import WorkflowAPIContext, create_workflow_router


FIELDS = {"HR": 90, "O2Sat": 97, "Temp": 37.1, "SBP": 120, "MAP": 83.33, "Resp": 18, "WBC": 8.5, "Lactate": 1.2}


class FakeReportAgent:
    def analyze(self, _: str) -> dict:
        return {"lab_results": FIELDS}


class FakeAudioAgent:
    def analyze(self, _: str) -> dict:
        return {"extracted_parameters": {"HR": 90}}


class FakeAssessmentAgent:
    def analyze(self, **kwargs: object) -> dict:
        return {
            "status": "completed",
            "sepsis": {"status": "success", "probability": 0.2},
            "aki": {"status": "success", "probability": 0.1},
            **kwargs,
        }


class FakeTrendAgent:
    def analyze(self, _: dict) -> dict:
        trend = {"trend": "STABLE", "first": 0.2, "latest": 0.2, "change": 0.0, "count": 1}
        return {"status": "success", "disease_trends": {"Sepsis": trend, "AKI": trend}}


class FakePriorityAgent:
    def prioritize_history(self, *_: object) -> dict:
        return {"status": "success", "priority": "LOW", "highest_risk_disease": "Sepsis", "highest_probability": 0.2, "worsening_diseases": [], "reasons": ["test"]}


def build_client() -> tuple[TestClient, dict[str, object]]:
    database = InMemoryDatabase()
    repository = PatientRepository(database)
    repository.create_patient("P001")
    repository.create_patient("P002")
    repository.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    repository.create_admission("P002", "A002", datetime(2026, 1, 1, tzinfo=timezone.utc))
    reports = LabReportRepository(database)
    tasks = WorkflowTaskRepository(database)
    events = WorkflowEventRepository(database)
    bus = EventBus(events)

    ReportProcessingEventHandler(reports, bus, FakeReportAgent())
    GenerateResultRequestedEventHandler(reports, tasks, bus, patient_repository=repository)
    NurseInputReceivedEventHandler(tasks, reports, repository, bus)
    ObservationReadyAssessmentHandler(repository, bus, FakeAssessmentAgent())
    AssessmentCompletedTrendHandler(repository, bus, FakeTrendAgent())
    TrendCompletedRiskPrioritizationHandler(repository, bus, FakePriorityAgent())

    from src.input.report_event_workflow import LabReportSubmissionService

    context = WorkflowAPIContext(
        repository=repository,
        report_repository=reports,
        task_repository=tasks,
        report_submission_service=LabReportSubmissionService(reports, bus),
        generate_result_service=GenerateResultRequestService(reports, bus),
        nurse_input_service=NurseInputSubmissionService(tasks, reports, bus),
    )
    app = FastAPI()
    app.include_router(create_workflow_router(context))
    return TestClient(app), {"repository": repository, "reports": reports, "tasks": tasks, "events": events, "context": context}


def upload(client: TestClient, patient_id: str = "P001", admission_id: str = "A001"):
    return client.post(
        "/api/reports",
        data={"patient_id": patient_id, "admission_id": admission_id},
        files={"report": ("labs.png", BytesIO(b"report"), "image/png")},
    )


def test_upload_processes_and_waits_for_explicit_generate():
    client, state = build_client()
    response = upload(client)
    assert response.status_code == 200
    payload = response.json()
    assert payload["report_id"]
    assert payload["processing_status"] == ReportProcessingStatus.PROCESSED.value
    assert state["tasks"].get_pending_tasks() == []
    assert len(state["events"].list_events(event_type=EventType.REPORT_UPLOADED)) == 1
    assert len(state["events"].list_events(event_type=EventType.REPORT_PROCESSED)) == 1

    status = client.get(f"/api/reports/{payload['report_id']}")
    assert status.status_code == 200
    assert status.json()["extracted_data"] == FIELDS


def test_generate_complete_data_persists_observation_before_ready_and_runs_downstream():
    client, state = build_client()
    report_id = upload(client).json()["report_id"]
    response = client.post(f"/api/reports/{report_id}/generate-result?patient_id=P001&admission_id=A001")
    assert response.status_code == 200
    assert response.json()["status"] == "OBSERVATION_READY"
    observations = state["repository"].get_patient_observations("P001", "A001")
    assert len(observations) == 1
    ready = state["events"].list_events(event_type=EventType.OBSERVATION_READY)
    assert ready[0]["payload"]["observation_id"] == observations[0]["observation_id"]
    assert len(state["repository"].get_patient_assessments("P001", "A001")) == 1
    assert len(state["repository"].get_patient_trends("P001", "A001")) == 2
    assert state["repository"].get_latest_prioritization("P001", "A001") is not None


def test_generate_missing_data_creates_persistent_task_and_current_question():
    client, state = build_client()
    report = state["reports"].create_report(
        report_id="R-MISSING", patient_id="P001", admission_id="A001", report_date=None,
        filename="labs.png", content_type="image/png", processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data={"WBC": 8.5},
    )
    response = client.post(f"/api/reports/{report['report_id']}/generate-result?patient_id=P001&admission_id=A001")
    assert response.status_code == 200
    task = response.json()["task"]
    assert task["current_question"]["field"] == "HR"
    assert state["tasks"].get_task(task["task_id"]) is not None
    assert state["events"].list_events(event_type=EventType.OBSERVATION_READY) == []

    question = client.get(f"/api/nurse/tasks/{task['task_id']}/question?patient_id=P001&admission_id=A001")
    assert question.status_code == 200
    assert question.json()["question_index"] == 0


def test_report_and_task_scoping_and_duplicate_generate_are_safe():
    client, state = build_client()
    report_id = upload(client).json()["report_id"]
    assert client.get(f"/api/reports/{report_id}").status_code == 200
    assert client.post(f"/api/reports/{report_id}/generate-result?patient_id=P002&admission_id=A002").status_code == 404

    first = client.post(f"/api/reports/{report_id}/generate-result?patient_id=P001&admission_id=A001")
    second = client.post(f"/api/reports/{report_id}/generate-result?patient_id=P001&admission_id=A001")
    assert first.status_code == second.status_code == 200
    assert len(state["repository"].get_patient_observations("P001", "A001")) == 1
    assert len(state["events"].list_events(event_type=EventType.OBSERVATION_READY)) == 1


def test_nurse_audio_service_advances_one_question_and_keeps_state_persistent():
    client, state = build_client()
    reports = state["reports"]
    reports.create_report(
        report_id="R-AUDIO", patient_id="P001", admission_id="A001", report_date=None,
        filename="labs.png", content_type="image/png", processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data={"WBC": 8.5},
    )
    generated = client.post("/api/reports/R-AUDIO/generate-result?patient_id=P001&admission_id=A001").json()
    task_id = generated["task"]["task_id"]
    context = state["context"]
    context.nurse_input_service.submit_nurse_audio = lambda *args: state["tasks"].update_task(task_id, metadata={**state["tasks"].get_task(task_id)["metadata"], "question_index": 1, "answers": {"HR": 90}}, status="IN_PROGRESS")
    response = client.post(
        f"/api/nurse/tasks/{task_id}/audio",
        data={"patient_id": "P001", "admission_id": "A001"},
        files={"audio": ("answer.wav", BytesIO(b"audio"), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["question_index"] == 1
    assert state["tasks"].get_task(task_id)["metadata"]["answers"] == {"HR": 90}
