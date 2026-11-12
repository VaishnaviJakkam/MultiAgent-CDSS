from datetime import datetime, timezone

from src.agents.nurse_agent import NurseAgentDecision
from src.database.lab_reports import ReportProcessingStatus
from src.database.repositories import InMemoryDatabase, LabReportRepository, PatientRepository, WorkflowEventRepository, WorkflowTaskRepository
from src.events import EventBus
from src.input.nurse_input_workflow import NurseInputSubmissionService


class StubNurseAgent:
    def decide(self, **state: object) -> NurseAgentDecision:
        nurse_values = state["nurse_values"]
        if isinstance(nurse_values, dict) and "HR" in nurse_values:
            return NurseAgentDecision("ASK_NURSE", "O2Sat", "Please provide oxygen saturation.", "O2Sat remains unresolved.")
        return NurseAgentDecision("ASK_NURSE", "HR", "Please provide heart rate.", "HR remains unresolved.")


def test_submission_uses_new_agent_decision_instead_of_incrementing_sequence() -> None:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P001")
    patients.create_admission("P001", "A001", datetime.now(timezone.utc))
    reports = LabReportRepository(database)
    tasks = WorkflowTaskRepository(database)
    events = WorkflowEventRepository(database)
    bus = EventBus(events)
    reports.create_report(
        report_id="R001",
        patient_id="P001",
        admission_id="A001",
        report_date=datetime.now(timezone.utc),
        filename="labs.png",
        content_type="image/png",
        processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data={"WBC": 8.5},
    )
    tasks.create_task(
        task_id="T001",
        task_type="NURSE_INPUT",
        patient_id="P001",
        admission_id="A001",
        required_fields=["HR", "O2Sat"],
        provided_fields=["WBC"],
        metadata={
            "report_id": "R001",
            "question_fields": ["HR"],
            "question_index": 0,
            "answers": {},
        },
    )

    service = NurseInputSubmissionService(
        tasks,
        reports,
        bus,
        nurse_agent=StubNurseAgent(),
    )
    service.submit_nurse_input("T001", "P001", "A001", {"HR": 90})

    task = tasks.get_task("T001")
    assert task is not None
    assert task["metadata"]["question_fields"] == ["O2Sat"]
    assert task["metadata"]["question_index"] == 0
    assert task["metadata"]["nurse_agent_decision"]["parameter"] == "O2Sat"
