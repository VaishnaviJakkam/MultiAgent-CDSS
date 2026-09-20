from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from .models import COLLECTIONS, require_identifier, require_timestamp, utc_now, with_created_updated


class _InsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class _UpdateResult:
    def __init__(self, matched_count: int, modified_count: int):
        self.matched_count = matched_count
        self.modified_count = modified_count


class _InMemoryCollection:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    def insert_one(self, document: dict[str, Any]) -> _InsertResult:
        stored = deepcopy(document)
        self.documents.append(stored)
        return _InsertResult(stored["_id"])

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return deepcopy(document)
        return None

    def find(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return [deepcopy(document) for document in self.documents if all(document.get(key) == value for key, value in query.items())]

    def update_one(self, query: dict[str, Any], update: dict[str, Any]) -> _UpdateResult:
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                changes = update.get("$set", {})
                document.update(deepcopy(changes))
                return _UpdateResult(1, 1 if changes else 0)
        return _UpdateResult(0, 0)


class InMemoryDatabase:
    """Small Mongo-like adapter used by tests without requiring a server."""

    def __init__(self) -> None:
        self.collections = {name: _InMemoryCollection() for name in COLLECTIONS}

    def __getitem__(self, name: str) -> _InMemoryCollection:
        return self.collections[name]


class WorkflowEventRepository:
    """Persistence operations for the auditable workflow event stream."""

    def __init__(self, database: Any):
        self.database = database
        self.events = database["workflow_events"]

    @staticmethod
    def _event_id() -> str:
        return f"event_{uuid4().hex}"

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Persist an event once and return the stored document.

        Event IDs are the first idempotency boundary. Re-delivery of the same
        event returns the original document instead of appending a duplicate.
        """
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")

        event_id = require_identifier(event.get("event_id", self._event_id()), "event_id")
        existing = self.events.find_one({"event_id": event_id})
        if existing is not None:
            return existing

        stored = with_created_updated({
            **deepcopy(event),
            "_id": event_id,
            "event_id": event_id,
            "status": event.get("status", "PENDING"),
            "processing_at": None,
            "completed_at": None,
            "failed_at": None,
            "error": None,
        })
        self.events.insert_one(stored)
        return stored

    def update_status(self, event_id: str, status: str, error: str | None = None) -> dict[str, Any]:
        normalized_event_id = require_identifier(event_id, "event_id")
        event = self.get(normalized_event_id)
        if event is None:
            raise ValueError(f"Event not found: {normalized_event_id}")
        timestamp = utc_now()
        changes: dict[str, Any] = {"status": status, "updated_at": timestamp, "error": error}
        if status == "PROCESSING":
            changes["processing_at"] = timestamp
        elif status == "COMPLETED":
            changes["completed_at"] = timestamp
        elif status == "FAILED":
            changes["failed_at"] = timestamp
        self.events.update_one({"event_id": normalized_event_id}, {"$set": changes})
        return self.get(normalized_event_id)  # type: ignore[return-value]

    def mark_processing(self, event_id: str) -> dict[str, Any]:
        return self.update_status(event_id, "PROCESSING")

    def mark_completed(self, event_id: str) -> dict[str, Any]:
        return self.update_status(event_id, "COMPLETED")

    def mark_failed(self, event_id: str, error: str) -> dict[str, Any]:
        return self.update_status(event_id, "FAILED", error)

    def get(self, event_id: str) -> dict[str, Any] | None:
        return self.events.find_one({"event_id": require_identifier(event_id, "event_id")})

    def list_events(
        self,
        patient_id: str | None = None,
        admission_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, str] = {}
        if patient_id is not None:
            query["patient_id"] = require_identifier(patient_id, "patient_id")
        if admission_id is not None:
            query["admission_id"] = require_identifier(admission_id, "admission_id")
        if event_type is not None:
            query["event_type"] = require_identifier(event_type, "event_type")

        records = self.events.find(query)
        return sorted(records, key=lambda item: item["timestamp"])


class WorkflowTaskRepository:
    """Persistence operations for human-action workflow tasks."""

    def __init__(self, database: Any):
        self.database = database
        self.patients = database["patients"]
        self.admissions = database["admissions"]
        self.tasks = database["workflow_tasks"]

    @staticmethod
    def _task_id() -> str:
        return f"task_{uuid4().hex}"

    def _scope(self, patient_id: str, admission_id: str) -> dict[str, str]:
        return {
            "patient_id": require_identifier(patient_id, "patient_id"),
            "admission_id": require_identifier(admission_id, "admission_id"),
        }

    def _require_admission(self, patient_id: str, admission_id: str) -> dict[str, str]:
        scope = self._scope(patient_id, admission_id)
        if self.patients.find_one({"patient_id": scope["patient_id"]}) is None:
            raise ValueError(f"Patient does not exist: {scope['patient_id']}")
        if self.admissions.find_one(scope) is None:
            raise ValueError("patient_id and admission_id must reference an existing admission")
        return scope

    @staticmethod
    def _enum_value(value: Any, enum_type: Any, field_name: str) -> str:
        try:
            return enum_type(value).value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field_name}: {value!r}") from exc

    def create_task(
        self,
        task_type: Any,
        patient_id: str,
        admission_id: str,
        required_fields: list[str] | None = None,
        provided_fields: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        from .workflow_tasks import TaskStatus, TaskType

        scope = self._require_admission(patient_id, admission_id)
        normalized_task_id = require_identifier(task_id, "task_id") if task_id is not None else self._task_id()
        if self.tasks.find_one({"task_id": normalized_task_id}) is not None:
            raise ValueError(f"Task already exists: {normalized_task_id}")

        now = utc_now()
        document = {
            "_id": normalized_task_id,
            "task_id": normalized_task_id,
            "task_type": self._enum_value(task_type, TaskType, "task_type"),
            **scope,
            "status": TaskStatus.PENDING.value,
            "required_fields": list(required_fields or []),
            "provided_fields": list(provided_fields or []),
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "metadata": deepcopy(metadata or {}),
        }
        self.tasks.insert_one(document)
        return document

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.tasks.find_one({"task_id": require_identifier(task_id, "task_id")})

    def update_task(self, task_id: str, updates: dict[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
        from .workflow_tasks import TaskStatus

        normalized_task_id = require_identifier(task_id, "task_id")
        task = self.get_task(normalized_task_id)
        if task is None:
            raise ValueError(f"Task not found: {normalized_task_id}")
        if updates is not None and not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")

        pending_changes = {**(updates or {}), **changes}
        allowed = {"status", "required_fields", "provided_fields", "metadata"}
        unexpected = set(pending_changes) - allowed
        if unexpected:
            raise ValueError(f"unsupported task fields: {sorted(unexpected)}")
        if "status" in pending_changes:
            pending_changes["status"] = self._enum_value(pending_changes["status"], TaskStatus, "status")
        if "required_fields" in pending_changes:
            pending_changes["required_fields"] = list(pending_changes["required_fields"])
        if "provided_fields" in pending_changes:
            pending_changes["provided_fields"] = list(pending_changes["provided_fields"])
        if "metadata" in pending_changes:
            if not isinstance(pending_changes["metadata"], dict):
                raise ValueError("metadata must be a dictionary")
            pending_changes["metadata"] = deepcopy(pending_changes["metadata"])

        if pending_changes.get("status") == TaskStatus.COMPLETED.value:
            pending_changes["completed_at"] = utc_now()
        elif "status" in pending_changes:
            pending_changes["completed_at"] = None
        pending_changes["updated_at"] = utc_now()

        self.tasks.update_one({"task_id": normalized_task_id}, {"$set": pending_changes})
        return self.get_task(normalized_task_id)  # type: ignore[return-value]

    def get_pending_tasks(self) -> list[dict[str, Any]]:
        from .workflow_tasks import TaskStatus

        records = self.tasks.find({"status": TaskStatus.PENDING.value})
        return sorted(records, key=lambda item: item["created_at"])

    def get_patient_tasks(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        records = self.tasks.find(scope)
        return sorted(records, key=lambda item: item["created_at"])

    def list_tasks(
        self,
        patient_id: str | None = None,
        admission_id: str | None = None,
        status: Any | None = None,
    ) -> list[dict[str, Any]]:
        from .workflow_tasks import TaskStatus

        if admission_id is not None and patient_id is None:
            raise ValueError("patient_id is required when admission_id is provided")
        query: dict[str, str] = {}
        if patient_id is not None:
            query["patient_id"] = require_identifier(patient_id, "patient_id")
        if admission_id is not None:
            query["admission_id"] = require_identifier(admission_id, "admission_id")
        if status is not None:
            query["status"] = self._enum_value(status, TaskStatus, "status")
        records = self.tasks.find(query)
        return sorted(records, key=lambda item: item["created_at"])

    def complete_task(self, task_id: str) -> dict[str, Any]:
        from .workflow_tasks import TaskStatus

        return self.update_task(task_id, status=TaskStatus.COMPLETED)


class NotificationRepository:
    """Persistence operations for role-targeted workflow notifications."""

    def __init__(self, database: Any):
        self.database = database
        self.notifications = database["notifications"]

    @staticmethod
    def _notification_id() -> str:
        return f"notification_{uuid4().hex}"

    def create_notification(
        self,
        role: str,
        notification_type: str,
        message: str,
        patient_id: str | None = None,
        admission_id: str | None = None,
        event_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role cannot be empty")
        if not isinstance(notification_type, str) or not notification_type.strip():
            raise ValueError("notification_type cannot be empty")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message cannot be empty")
        if event_id is not None and self.notifications.find_one({"event_id": event_id}) is not None:
            return self.notifications.find_one({"event_id": event_id})  # type: ignore[return-value]
        document = with_created_updated({
            "_id": self._notification_id(),
            "notification_id": self._notification_id(),
            "role": role.strip().upper(),
            "notification_type": notification_type.strip(),
            "message": message.strip(),
            "patient_id": patient_id,
            "admission_id": admission_id,
            "event_id": event_id,
            "payload": deepcopy(payload or {}),
            "read": False,
        })
        self.notifications.insert_one(document)
        return document

    def list_notifications(self, role: str) -> list[dict[str, Any]]:
        normalized_role = require_identifier(role, "role").upper()
        records = self.notifications.find({"role": normalized_role})
        return sorted(records, key=lambda item: item["created_at"], reverse=True)


class LabReportRepository:
    """Persistence operations for uploaded laboratory reports."""

    def __init__(self, database: Any):
        self.database = database
        self.patients = database["patients"]
        self.admissions = database["admissions"]
        self.reports = database["lab_reports"]

    @staticmethod
    def _report_id() -> str:
        return f"report_{uuid4().hex}"

    @staticmethod
    def _enum_value(value: Any, enum_type: Any, field_name: str) -> str:
        try:
            return enum_type(value).value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field_name}: {value!r}") from exc

    def _scope(self, patient_id: str, admission_id: str) -> dict[str, str]:
        return {
            "patient_id": require_identifier(patient_id, "patient_id"),
            "admission_id": require_identifier(admission_id, "admission_id"),
        }

    def _require_admission(self, patient_id: str, admission_id: str) -> dict[str, str]:
        scope = self._scope(patient_id, admission_id)
        if self.patients.find_one({"patient_id": scope["patient_id"]}) is None:
            raise ValueError(f"Patient does not exist: {scope['patient_id']}")
        if self.admissions.find_one(scope) is None:
            raise ValueError("patient_id and admission_id must reference an existing admission")
        return scope

    def create_report(
        self,
        patient_id: str,
        admission_id: str,
        report_date: datetime | None,
        filename: str,
        content_type: str,
        storage_reference: str | None = None,
        source: Any = "LAB_UPLOAD",
        processing_status: Any = "UPLOADED",
        extracted_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        report_id: str | None = None,
        uploaded_at: datetime | None = None,
    ) -> dict[str, Any]:
        from .lab_reports import ReportProcessingStatus, ReportSource

        scope = self._require_admission(patient_id, admission_id)
        normalized_report_id = require_identifier(report_id, "report_id") if report_id is not None else self._report_id()
        if self.reports.find_one({"report_id": normalized_report_id}) is not None:
            raise ValueError(f"Report already exists: {normalized_report_id}")
        if report_date is not None:
            require_timestamp(report_date, "report_date")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename cannot be empty")
        if not isinstance(content_type, str) or not content_type.strip():
            raise ValueError("content_type cannot be empty")
        if uploaded_at is None:
            uploaded_at = utc_now()
        require_timestamp(uploaded_at, "uploaded_at")
        if extracted_data is not None and not isinstance(extracted_data, dict):
            raise ValueError("extracted_data must be a dictionary")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

        now = utc_now()
        document = {
            "_id": normalized_report_id,
            "report_id": normalized_report_id,
            **scope,
            "report_date": report_date,
            "uploaded_at": uploaded_at,
            "source": self._enum_value(source, ReportSource, "source"),
            "filename": filename.strip(),
            "content_type": content_type.strip(),
            "storage_reference": storage_reference,
            "processing_status": self._enum_value(processing_status, ReportProcessingStatus, "processing_status"),
            "extracted_data": deepcopy(extracted_data or {}),
            "metadata": deepcopy(metadata or {}),
            "created_at": now,
            "updated_at": now,
        }
        self.reports.insert_one(document)
        return document

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        return self.reports.find_one({"report_id": require_identifier(report_id, "report_id")})

    def update_report(self, report_id: str, updates: dict[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
        from .lab_reports import ReportProcessingStatus, ReportSource

        normalized_report_id = require_identifier(report_id, "report_id")
        if self.get_report(normalized_report_id) is None:
            raise ValueError(f"Report not found: {normalized_report_id}")
        if updates is not None and not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")
        pending_changes = {**(updates or {}), **changes}
        allowed = {"report_date", "source", "filename", "content_type", "storage_reference", "processing_status", "extracted_data", "metadata"}
        unexpected = set(pending_changes) - allowed
        if unexpected:
            raise ValueError(f"unsupported report fields: {sorted(unexpected)}")
        if "report_date" in pending_changes and pending_changes["report_date"] is not None:
            require_timestamp(pending_changes["report_date"], "report_date")
        if "source" in pending_changes:
            pending_changes["source"] = self._enum_value(pending_changes["source"], ReportSource, "source")
        if "processing_status" in pending_changes:
            pending_changes["processing_status"] = self._enum_value(pending_changes["processing_status"], ReportProcessingStatus, "processing_status")
        for field_name in ("extracted_data", "metadata"):
            if field_name in pending_changes:
                if not isinstance(pending_changes[field_name], dict):
                    raise ValueError(f"{field_name} must be a dictionary")
                pending_changes[field_name] = deepcopy(pending_changes[field_name])
        for field_name in ("filename", "content_type"):
            if field_name in pending_changes:
                if not isinstance(pending_changes[field_name], str) or not pending_changes[field_name].strip():
                    raise ValueError(f"{field_name} cannot be empty")
                pending_changes[field_name] = pending_changes[field_name].strip()
        pending_changes["updated_at"] = utc_now()
        self.reports.update_one({"report_id": normalized_report_id}, {"$set": pending_changes})
        return self.get_report(normalized_report_id)  # type: ignore[return-value]

    def get_patient_reports(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        records = self.reports.find(self._scope(patient_id, admission_id))
        return sorted(records, key=lambda item: item["uploaded_at"])

    def get_reports_by_status(self, status: Any, patient_id: str | None = None, admission_id: str | None = None) -> list[dict[str, Any]]:
        from .lab_reports import ReportProcessingStatus

        query: dict[str, str] = {"processing_status": self._enum_value(status, ReportProcessingStatus, "processing_status")}
        if patient_id is not None or admission_id is not None:
            if patient_id is None or admission_id is None:
                raise ValueError("patient_id and admission_id must be provided together")
            query.update(self._scope(patient_id, admission_id))
        records = self.reports.find(query)
        return sorted(records, key=lambda item: item["uploaded_at"])


class PatientRepository:
    """Persistence operations for patients and one-admission longitudinal histories."""

    def __init__(self, database: Any):
        self.database = database
        self.patients = database["patients"]
        self.admissions = database["admissions"]
        self.observations = database["observations"]
        self.assessments = database["assessments"]
        self.trends = database["trends"]
        self.prioritizations = database["prioritizations"]

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"

    @staticmethod
    def _scope(patient_id: str, admission_id: str) -> dict[str, str]:
        return {"patient_id": require_identifier(patient_id, "patient_id"), "admission_id": require_identifier(admission_id, "admission_id")}

    def create_patient(self, patient_id: str, demographics: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
        patient_id = require_identifier(patient_id, "patient_id")
        if self.patients.find_one({"patient_id": patient_id}):
            raise ValueError(f"Patient already exists: {patient_id}")
        document = with_created_updated({"_id": self._id("patient"), "patient_id": patient_id, "demographics": demographics or {}, "status": status})
        self.patients.insert_one(document)
        return document

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        patient_id = require_identifier(patient_id, "patient_id")
        return self.patients.find_one({"patient_id": patient_id})

    def create_admission(self, patient_id: str, admission_id: str, admission_time: datetime, status: str = "active") -> dict[str, Any]:
        scope = self._scope(patient_id, admission_id)
        require_timestamp(admission_time, "admission_time")
        if not self.get_patient(scope["patient_id"]):
            raise ValueError(f"Patient does not exist: {scope['patient_id']}")
        if self.admissions.find_one(scope):
            raise ValueError(f"Admission already exists: {scope['admission_id']}")
        document = with_created_updated({"_id": self._id("admission"), **scope, "admission_time": admission_time, "status": status})
        self.admissions.insert_one(document)
        return document

    def get_admission(self, patient_id: str, admission_id: str) -> dict[str, Any] | None:
        return self.admissions.find_one(self._scope(patient_id, admission_id))

    def _require_admission(self, patient_id: str, admission_id: str) -> dict[str, str]:
        scope = self._scope(patient_id, admission_id)
        if not self.get_admission(**scope):
            raise ValueError("patient_id and admission_id must reference an existing admission")
        return scope

    def add_observation(
        self,
        patient_id: str,
        admission_id: str,
        observation_time: datetime,
        clinical_parameters: dict[str, Any],
        source: str = "user",
        task_id: str | None = None,
        report_id: str | None = None,
        parameter_sources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(observation_time, "observation_time")
        if not isinstance(clinical_parameters, dict):
            raise ValueError("clinical_parameters must be a dictionary")
        if task_id is not None:
            require_identifier(task_id, "task_id")
        if report_id is not None:
            require_identifier(report_id, "report_id")
        if parameter_sources is not None and not isinstance(parameter_sources, dict):
            raise ValueError("parameter_sources must be a dictionary")
        observation_id = self._id("observation")
        document = with_created_updated({
            "_id": observation_id,
            "observation_id": observation_id,
            **scope,
            "observation_time": observation_time,
            "clinical_parameters": deepcopy(clinical_parameters),
            "source": source,
        })
        if task_id is not None:
            document["task_id"] = task_id
        if report_id is not None:
            document["report_id"] = report_id
        if parameter_sources is not None:
            document["parameter_sources"] = deepcopy(parameter_sources)
        self.observations.insert_one(document)
        return document

    def get_patient_observations(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        records = self.observations.find(scope)
        return sorted(records, key=lambda item: item["observation_time"])

    def get_observation_by_task(self, patient_id: str, admission_id: str, task_id: str) -> dict[str, Any] | None:
        scope = self._scope(patient_id, admission_id)
        scope["task_id"] = require_identifier(task_id, "task_id")
        return self.observations.find_one(scope)

    def get_observation(self, patient_id: str, admission_id: str, observation_id: str) -> dict[str, Any] | None:
        scope = self._scope(patient_id, admission_id)
        scope["observation_id"] = require_identifier(observation_id, "observation_id")
        return self.observations.find_one(scope)

    def add_assessment(
        self,
        patient_id: str,
        admission_id: str,
        assessment_time: datetime,
        sepsis_result: dict[str, Any] | None = None,
        aki_result: dict[str, Any] | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(assessment_time, "assessment_time")
        if sepsis_result is None and aki_result is None:
            raise ValueError("At least one disease assessment result is required")
        if observation_id is not None:
            require_identifier(observation_id, "observation_id")
        assessment_id = self._id("assessment")
        document = with_created_updated({"_id": assessment_id, "assessment_id": assessment_id, **scope, "assessment_time": assessment_time, "sepsis": deepcopy(sepsis_result), "aki": deepcopy(aki_result)})
        if observation_id is not None:
            document["observation_id"] = observation_id
        self.assessments.insert_one(document)
        return document

    def get_patient_assessments(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        records = self.assessments.find(scope)
        return sorted(records, key=lambda item: item["assessment_time"])

    def get_assessment_by_observation(self, patient_id: str, admission_id: str, observation_id: str) -> dict[str, Any] | None:
        scope = self._scope(patient_id, admission_id)
        scope["observation_id"] = require_identifier(observation_id, "observation_id")
        return self.assessments.find_one(scope)

    def get_assessment(self, patient_id: str, admission_id: str, assessment_id: str) -> dict[str, Any] | None:
        scope = self._scope(patient_id, admission_id)
        scope["assessment_id"] = require_identifier(assessment_id, "assessment_id")
        return self.assessments.find_one(scope)

    def add_trend(
        self,
        patient_id: str,
        admission_id: str,
        disease: str,
        assessment_time: datetime,
        trend: str,
        first_probability: float,
        latest_probability: float,
        probability_change: float,
        observation_count: int,
        assessment_id: str | None = None,
        trend_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(assessment_time, "assessment_time")
        if assessment_id is not None:
            require_identifier(assessment_id, "assessment_id")
        document = with_created_updated({"_id": self._id("trend"), **scope, "disease": disease, "assessment_time": assessment_time, "trend": trend, "first_probability": first_probability, "latest_probability": latest_probability, "probability_change": probability_change, "observation_count": observation_count})
        if assessment_id is not None:
            document["assessment_id"] = assessment_id
        if trend_result is not None:
            if not isinstance(trend_result, dict):
                raise ValueError("trend_result must be a dictionary")
            document["trend_result"] = deepcopy(trend_result)
        self.trends.insert_one(document)
        return document

    def get_patient_trends(self, patient_id: str, admission_id: str, disease: str | None = None) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        if disease is not None:
            scope["disease"] = disease
        records = self.trends.find(scope)
        return sorted(records, key=lambda item: item["assessment_time"])

    def get_trends_by_assessment(self, patient_id: str, admission_id: str, assessment_id: str) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        scope["assessment_id"] = require_identifier(assessment_id, "assessment_id")
        records = self.trends.find(scope)
        return sorted(records, key=lambda item: item["assessment_time"])

    def get_trend(self, patient_id: str, admission_id: str, trend_id: str) -> dict[str, Any] | None:
        scope = self._scope(patient_id, admission_id)
        scope["_id"] = require_identifier(trend_id, "trend_id")
        return self.trends.find_one(scope)

    def get_prioritization_by_source_trend(self, patient_id: str, admission_id: str, source_trend_id: str) -> dict[str, Any] | None:
        scope = self._scope(patient_id, admission_id)
        scope["source_trend_id"] = require_identifier(source_trend_id, "source_trend_id")
        return self.prioritizations.find_one(scope)

    def add_prioritization(self, patient_id: str, admission_id: str, priority_level: str, highest_risk_disease: str, highest_probability: float, worsening_diseases: list[str], reason: str, assessment_time: datetime, source_trend_id: str | None = None, assessment_ids: list[str] | None = None, trend_ids: list[str] | None = None) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(assessment_time, "assessment_time")
        document = with_created_updated({"_id": self._id("prioritization"), **scope, "priority_level": priority_level, "highest_risk_disease": highest_risk_disease, "highest_probability": highest_probability, "worsening_diseases": list(worsening_diseases), "reason": reason, "assessment_time": assessment_time})
        if source_trend_id is not None:
            document["source_trend_id"] = require_identifier(source_trend_id, "source_trend_id")
        if assessment_ids is not None:
            document["assessment_ids"] = list(assessment_ids)
        if trend_ids is not None:
            document["trend_ids"] = list(trend_ids)
        self.prioritizations.insert_one(document)
        return document

    def get_latest_prioritization(self, patient_id: str, admission_id: str) -> dict[str, Any] | None:
        records = self.prioritizations.find(self._scope(patient_id, admission_id))
        return max(records, key=lambda item: item["assessment_time"], default=None)

    def get_patient_prioritizations(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        records = self.prioritizations.find(self._scope(patient_id, admission_id))
        return sorted(records, key=lambda item: item["assessment_time"])

    def get_patient_history(self, patient_id: str, admission_id: str) -> dict[str, Any]:
        scope = self._scope(patient_id, admission_id)
        admission = self.get_admission(**scope)
        if admission is None:
            raise ValueError("patient_id and admission_id must reference an existing admission")
        return {"patient": self.get_patient(scope["patient_id"]), "admission": admission, "observations": self.get_patient_observations(**scope), "assessments": self.get_patient_assessments(**scope), "trends": self.get_patient_trends(**scope), "latest_prioritization": self.get_latest_prioritization(**scope)}
