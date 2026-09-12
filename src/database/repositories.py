from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from .models import COLLECTIONS, require_identifier, require_timestamp, utc_now, with_created_updated


class _InsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


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


class InMemoryDatabase:
    """Small Mongo-like adapter used by tests without requiring a server."""

    def __init__(self) -> None:
        self.collections = {name: _InMemoryCollection() for name in COLLECTIONS}

    def __getitem__(self, name: str) -> _InMemoryCollection:
        return self.collections[name]


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

    def add_observation(self, patient_id: str, admission_id: str, observation_time: datetime, clinical_parameters: dict[str, Any], source: str = "user") -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(observation_time, "observation_time")
        if not isinstance(clinical_parameters, dict):
            raise ValueError("clinical_parameters must be a dictionary")
        document = with_created_updated({"_id": self._id("observation"), "observation_id": self._id("observation"), **scope, "observation_time": observation_time, "clinical_parameters": deepcopy(clinical_parameters), "source": source})
        self.observations.insert_one(document)
        return document

    def get_patient_observations(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        records = self.observations.find(scope)
        return sorted(records, key=lambda item: item["observation_time"])

    def add_assessment(self, patient_id: str, admission_id: str, assessment_time: datetime, sepsis_result: dict[str, Any] | None = None, aki_result: dict[str, Any] | None = None) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(assessment_time, "assessment_time")
        if sepsis_result is None and aki_result is None:
            raise ValueError("At least one disease assessment result is required")
        document = with_created_updated({"_id": self._id("assessment"), "assessment_id": self._id("assessment"), **scope, "assessment_time": assessment_time, "sepsis": deepcopy(sepsis_result), "aki": deepcopy(aki_result)})
        self.assessments.insert_one(document)
        return document

    def get_patient_assessments(self, patient_id: str, admission_id: str) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        records = self.assessments.find(scope)
        return sorted(records, key=lambda item: item["assessment_time"])

    def add_trend(self, patient_id: str, admission_id: str, disease: str, assessment_time: datetime, trend: str, first_probability: float, latest_probability: float, probability_change: float, observation_count: int) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(assessment_time, "assessment_time")
        document = with_created_updated({"_id": self._id("trend"), **scope, "disease": disease, "assessment_time": assessment_time, "trend": trend, "first_probability": first_probability, "latest_probability": latest_probability, "probability_change": probability_change, "observation_count": observation_count})
        self.trends.insert_one(document)
        return document

    def get_patient_trends(self, patient_id: str, admission_id: str, disease: str | None = None) -> list[dict[str, Any]]:
        scope = self._scope(patient_id, admission_id)
        if disease is not None:
            scope["disease"] = disease
        records = self.trends.find(scope)
        return sorted(records, key=lambda item: item["assessment_time"])

    def add_prioritization(self, patient_id: str, admission_id: str, priority_level: str, highest_risk_disease: str, highest_probability: float, worsening_diseases: list[str], reason: str, assessment_time: datetime) -> dict[str, Any]:
        scope = self._require_admission(patient_id, admission_id)
        require_timestamp(assessment_time, "assessment_time")
        document = with_created_updated({"_id": self._id("prioritization"), **scope, "priority_level": priority_level, "highest_risk_disease": highest_risk_disease, "highest_probability": highest_probability, "worsening_diseases": list(worsening_diseases), "reason": reason, "assessment_time": assessment_time})
        self.prioritizations.insert_one(document)
        return document

    def get_latest_prioritization(self, patient_id: str, admission_id: str) -> dict[str, Any] | None:
        records = self.prioritizations.find(self._scope(patient_id, admission_id))
        return max(records, key=lambda item: item["assessment_time"], default=None)

    def get_patient_history(self, patient_id: str, admission_id: str) -> dict[str, Any]:
        scope = self._scope(patient_id, admission_id)
        admission = self.get_admission(**scope)
        if admission is None:
            raise ValueError("patient_id and admission_id must reference an existing admission")
        return {"patient": self.get_patient(scope["patient_id"]), "admission": admission, "observations": self.get_patient_observations(**scope), "assessments": self.get_patient_assessments(**scope), "trends": self.get_patient_trends(**scope), "latest_prioritization": self.get_latest_prioritization(**scope)}
