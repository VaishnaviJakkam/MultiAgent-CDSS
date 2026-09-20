from datetime import datetime, timezone

import pytest

from src.database.lab_reports import ReportProcessingStatus, ReportSource
from src.database.repositories import InMemoryDatabase, LabReportRepository, PatientRepository


@pytest.fixture
def repository() -> LabReportRepository:
    database = InMemoryDatabase()
    patients = PatientRepository(database)
    patients.create_patient("P001")
    patients.create_patient("P002")
    patients.create_admission("P001", "A001", datetime(2026, 1, 1, tzinfo=timezone.utc))
    patients.create_admission("P001", "A002", datetime(2026, 1, 2, tzinfo=timezone.utc))
    patients.create_admission("P002", "A003", datetime(2026, 1, 3, tzinfo=timezone.utc))
    return LabReportRepository(database)


def create_report(repository: LabReportRepository, report_id: str = "R001", admission_id: str = "A001") -> dict:
    return repository.create_report(
        report_id=report_id,
        patient_id="P001",
        admission_id=admission_id,
        report_date=datetime(2026, 1, 4, tzinfo=timezone.utc),
        uploaded_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        source=ReportSource.LAB_UPLOAD,
        filename="labs.pdf",
        content_type="application/pdf",
        storage_reference="reports/labs.pdf",
        extracted_data={"WBC": 8.5, "Lactate": 1.4},
        metadata={"department": "laboratory"},
    )


def test_create_and_retrieve_report(repository: LabReportRepository) -> None:
    report = create_report(repository)

    assert report["report_id"] == "R001"
    assert report["source"] == ReportSource.LAB_UPLOAD.value
    assert report["processing_status"] == ReportProcessingStatus.UPLOADED.value
    assert report["extracted_data"] == {"WBC": 8.5, "Lactate": 1.4}
    assert repository.get_report("R001") == report


def test_patient_admission_report_filtering_is_isolated(repository: LabReportRepository) -> None:
    create_report(repository, "R001", "A001")
    create_report(repository, "R002", "A002")
    repository.create_report(
        report_id="R003",
        patient_id="P002",
        admission_id="A003",
        report_date=None,
        filename="other.pdf",
        content_type="application/pdf",
    )

    assert [item["report_id"] for item in repository.get_patient_reports("P001", "A001")] == ["R001"]
    assert [item["report_id"] for item in repository.get_patient_reports("P001", "A002")] == ["R002"]
    assert repository.get_patient_reports("P002", "A001") == []


def test_update_report_metadata_and_extracted_data_preserves_uploaded_at(repository: LabReportRepository) -> None:
    report = create_report(repository)
    uploaded_at = report["uploaded_at"]

    updated = repository.update_report(
        "R001",
        metadata={"department": "laboratory", "reviewed": True},
        extracted_data={"WBC": 9.2},
        storage_reference="reports/revised-labs.pdf",
    )

    assert updated["uploaded_at"] == uploaded_at
    assert updated["updated_at"] >= report["updated_at"]
    assert updated["metadata"]["reviewed"] is True
    assert updated["extracted_data"] == {"WBC": 9.2}
    assert updated["storage_reference"] == "reports/revised-labs.pdf"


def test_processing_status_updates_and_status_filtering(repository: LabReportRepository) -> None:
    create_report(repository, "R001")
    create_report(repository, "R002", "A002")

    processing = repository.update_report("R001", processing_status=ReportProcessingStatus.PROCESSING)
    processed = repository.update_report("R001", processing_status=ReportProcessingStatus.PROCESSED)
    failed = repository.update_report("R002", processing_status=ReportProcessingStatus.FAILED)

    assert processing["processing_status"] == "PROCESSING"
    assert processed["processing_status"] == "PROCESSED"
    assert failed["processing_status"] == "FAILED"
    assert [item["report_id"] for item in repository.get_reports_by_status(ReportProcessingStatus.PROCESSED)] == ["R001"]
    assert [item["report_id"] for item in repository.get_reports_by_status("FAILED", "P001", "A002")] == ["R002"]
    assert repository.get_reports_by_status(ReportProcessingStatus.PROCESSED, "P001", "A002") == []


def test_optional_report_date_and_extracted_fields_are_supported(repository: LabReportRepository) -> None:
    report = repository.create_report(
        patient_id="P001",
        admission_id="A001",
        report_date=None,
        filename="empty.pdf",
        content_type="application/pdf",
    )

    assert report["report_date"] is None
    assert report["extracted_data"] == {}
    assert report["metadata"] == {}


def test_duplicate_report_id_is_rejected(repository: LabReportRepository) -> None:
    create_report(repository)

    with pytest.raises(ValueError, match="Report already exists"):
        create_report(repository)


def test_invalid_status_source_and_scope_are_rejected(repository: LabReportRepository) -> None:
    with pytest.raises(ValueError, match="invalid processing_status"):
        repository.create_report("P001", "A001", None, "labs.pdf", "application/pdf", processing_status="UNKNOWN")
    with pytest.raises(ValueError, match="invalid source"):
        repository.create_report("P001", "A001", None, "labs.pdf", "application/pdf", source="UNKNOWN")
    with pytest.raises(ValueError, match="existing admission"):
        repository.create_report("P001", "A999", None, "labs.pdf", "application/pdf")


def test_failed_report_status_is_persisted(repository: LabReportRepository) -> None:
    report = repository.create_report(
        patient_id="P001",
        admission_id="A001",
        report_date=None,
        filename="failed.pdf",
        content_type="application/pdf",
        processing_status=ReportProcessingStatus.FAILED,
        metadata={"error": "unreadable report"},
    )

    assert report["processing_status"] == ReportProcessingStatus.FAILED.value
    assert report["metadata"]["error"] == "unreadable report"
