from datetime import datetime, timezone

from src.agents.nurse_agent import NurseAgentDecision
from src.database.lab_reports import ReportProcessingStatus
from src.database.repositories import InMemoryDatabase, LabReportRepository, PatientRepository, WorkflowEventRepository, WorkflowTaskRepository
from src.events import EventBus, EventType, WorkflowEvent
from src.input.report_event_workflow import GenerateResultRequestedEventHandler, ReportProcessingEventHandler
from src.input.report_input_agent import ReportInputAgent


def test_report_input_agent_preserves_all_detected_clinical_values() -> None:
    agent = ReportInputAgent()
    result = agent.extract_lab_results(
        "Parameter | Value | Unit | Reference Range\n"
        "BaseExcess | 6.0 | mmol/L | -2 to 2\n"
        "HCO3 | 26.0 | mmol/L | 22 to 29\n"
        "FiO2 | 0.8 | fraction | -\n"
        "pH | 7.35 | - | 7.35 to 7.45\n"
        "PaCO2 | 40.0 | mmHg | 35 to 45\n"
        "SaO2 | 98.0 | % | 95 to 100\n"
        "AST | 30.0 | U/L | 10 to 40\n"
        "BUN | 23 | mg/dL | 7 to 20\n"
        "Alkalinephos | 90.0 | U/L | 44 to 147\n"
        "Calcium | 9.2 | mg/dL | 8.5 to 10.5\n"
        "Chloride | 102.0 | mmol/L | 96 to 106\n"
        "Creatinine | 0.8 | mg/dL | 0.6 to 1.3\n"
        "Bilirubin_direct | 0.2 | mg/dL | 0 to 0.3\n"
        "Glucose | 110.0 | mg/dL | 70 to 140\n"
        "Lactate | 1.9 | mmol/L | 0.5 to 2.2\n"
        "Magnesium | 2.0 | mg/dL | 1.7 to 2.4\n"
        "Phosphate | 3.5 | mg/dL | 2.5 to 4.5\n"
        "Sodium | 138 | mmol/L | 135 to 145\n"
        "Potassium | 4.2 | mmol/L | 3.5 to 5.1\n"
        "Bilirubin_total | 0.8 | mg/dL | 0.1 to 1.2\n"
        "TroponinI | 0.01 | ng/mL | 0 to 0.04\n"
        "Hct | 30.0 | % | 36 to 46\n"
        "Hgb | 9.5 | g/dL | 12 to 16\n"
        "PTT | 30.0 | sec | 25 to 35\n"
        "WBC | 11.3 | thou/mm3 | 4 to 11\n"
        "Fibrinogen | 330.0 | mg/dL | 200 to 400\n"
        "Temp | 35.78 | C | 36.1 to 37.2\n"
        "SBP | 142.0 | mmHg | 90 to 140\n"
        "MAP | 93.5 | mmHg | 70 to 100\n"
        "DBP | 78.0 | mmHg | 60 to 90\n"
        "Resp | 16.0 | /min | 12 to 20\n"
        "ETCO2 | 35.0 | mmHg | 30 to  forty\n"
        "HR | 79 | bpm | 60 to 100\n"
        "O2Sat | 99 | % | 95 to 100\n"
        "Platelets | 330.0 | thou/mm3 | 150 to 400\n"
    )

    assert result["HR"]["value"] == 79
    assert result["O2Sat"]["value"] == 99
    assert result["Temp"]["value"] == 35.78
    assert result["SBP"]["value"] == 142.0
    assert result["MAP"]["value"] == 93.5
    assert result["DBP"]["value"] == 78.0
    assert result["Resp"]["value"] == 16.0
    assert result["WBC"]["value"] == 11.3
    assert result["Lactate"]["value"] == 1.9
    assert result["Creatinine"]["value"] == 0.8
    assert result["Hgb"]["value"] == 9.5
    assert result["Platelets"]["value"] == 330.0
    assert result["BaseExcess"]["value"] == 6.0
    assert result["HCO3"]["value"] == 26.0
    assert result["FiO2"]["value"] == 0.8
    assert result["pH"]["value"] == 7.35
    assert result["AST"]["value"] == 30.0
    assert result["Calcium"]["value"] == 9.2
    assert result["Potassium"]["value"] == 4.2
    assert result["Magnesium"]["value"] == 2.0
    assert result["Phosphate"]["value"] == 3.5
    assert result["HR"]["source_name"] == "HR"
    assert result["HR"]["unit"] == "bpm"
    assert result["Sodium"]["value"] == 138
    assert result["Potassium"]["value"] == 4.2


def test_report_processing_persists_complete_values_and_raw_text() -> None:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P001")
    patients.create_admission("P001", "A001", datetime.now(timezone.utc))
    reports = LabReportRepository(database)
    events = WorkflowEventRepository(database)
    bus = EventBus(events)
    raw_text = "Heart Rate 112\nCreatinine 1.8"

    class FakeReportAgent:
        def analyze(self, _: str) -> dict:
            return {
                "raw_text": raw_text,
                "lab_results": {
                    "HR": {"value": 112, "raw_line": "Heart Rate 112"},
                    "Creatinine": {"value": 1.8, "raw_line": "Creatinine 1.8"},
                },
                "extracted_data": {
                    "HR": {"value": 112, "raw_line": "Heart Rate 112"},
                    "Creatinine": {"value": 1.8, "raw_line": "Creatinine 1.8"},
                    "Hgb": {"value": 9.5, "raw_line": "Hgb 9.5"},
                },
            }

    report = reports.create_report(
        report_id="R001",
        patient_id="P001",
        admission_id="A001",
        report_date=None,
        filename="report.png",
        content_type="image/png",
        storage_reference="report.png",
    )
    ReportProcessingEventHandler(reports, bus, FakeReportAgent()).handle(
        WorkflowEvent(
            event_type=EventType.REPORT_UPLOADED,
            patient_id="P001",
            admission_id="A001",
            related_report_id=report["report_id"],
            payload={"report_id": report["report_id"]},
        )
    )

    persisted = reports.get_report("R001")
    assert persisted is not None
    assert persisted["raw_text"] == raw_text
    assert set(persisted["extracted_data"]) == {"HR", "Creatinine", "Hgb"}


def test_generate_result_passes_only_missing_parameters_to_nurse_agent() -> None:
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
        report_date=None,
        filename="report.png",
        content_type="image/png",
        storage_reference="report.png",
        processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data={
            "HR": {"value": 112},
            "Temp": {"value": 38.7},
            "WBC": {"value": 14.2},
            "Lactate": {"value": 3.8},
            "O2Sat": {"value": 94},
            "Creatinine": {"value": 1.8},
        },
    )

    class RecordingNurseAgent:
        def __init__(self) -> None:
            self.states: list[dict] = []

        def decide(self, **state: object) -> NurseAgentDecision:
            self.states.append(state)
            return NurseAgentDecision("ASK_NURSE", "BP", "Provide BP", "BP is missing")

    nurse_agent = RecordingNurseAgent()
    handler = GenerateResultRequestedEventHandler(
        reports,
        tasks,
        bus,
        patient_repository=patients,
        nurse_agent=nurse_agent,
    )
    handler.handle(
        WorkflowEvent(
            event_type=EventType.GENERATE_RESULT_REQUESTED,
            patient_id="P001",
            admission_id="A001",
            related_report_id="R001",
            payload={"report_id": "R001"},
        )
    )

    state = nurse_agent.states[0]
    assert state["lab_values"]["HR"] == 112
    assert state["lab_values"]["Lactate"] == 3.8
    assert state["lab_values"]["Creatinine"] == 1.8
    assert tasks.get_pending_tasks()[0]["metadata"]["question_fields"] == ["BP"]


def test_generate_result_does_not_create_nurse_task_when_report_is_complete() -> None:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P002")
    patients.create_admission("P002", "A002", datetime.now(timezone.utc))
    reports = LabReportRepository(database)
    tasks = WorkflowTaskRepository(database)
    events = WorkflowEventRepository(database)
    bus = EventBus(events)
    complete_values = {
        "HR": {"value": 79},
        "O2Sat": {"value": 99},
        "Temp": {"value": 35.78},
        "SBP": {"value": 142},
        "MAP": {"value": 93.5},
        "DBP": {"value": 78},
        "Resp": {"value": 16},
        "WBC": {"value": 11.3},
        "Lactate": {"value": 1.9},
        "Creatinine": {"value": 0.8},
        "BUN": {"value": 23},
        "Hgb": {"value": 9.5},
        "Platelets": {"value": 330},
    }
    reports.create_report(
        report_id="R002",
        patient_id="P002",
        admission_id="A002",
        report_date=None,
        filename="complete-report.png",
        content_type="image/png",
        storage_reference="complete-report.png",
        processing_status=ReportProcessingStatus.PROCESSED,
        extracted_data=complete_values,
    )

    class CompleteNurseAgent:
        def decide(self, **_: object) -> NurseAgentDecision:
            return NurseAgentDecision("COMPLETE", None, None, "All observation values are present in the report.")

    GenerateResultRequestedEventHandler(
        reports,
        tasks,
        bus,
        patient_repository=patients,
        nurse_agent=CompleteNurseAgent(),
    ).handle(
        WorkflowEvent(
            event_type=EventType.GENERATE_RESULT_REQUESTED,
            patient_id="P002",
            admission_id="A002",
            related_report_id="R002",
            payload={"report_id": "R002"},
        )
    )

    assert tasks.get_patient_tasks("P002", "A002") == []
    persisted = reports.get_report("R002")
    assert persisted is not None
    assert set(persisted["extracted_data"]) == set(complete_values)
