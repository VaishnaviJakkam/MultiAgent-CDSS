const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export type ReportResponse = {
  report_id: string;
  patient_id: string;
  admission_id: string;
  processing_status: string;
  extracted_data: Record<string, unknown>;
  task_status?: string;
};

export type NurseTask = {
  task_id: string;
  patient_id: string;
  admission_id: string;
  status: string;
  questionnaire_status: string;
  current_question: {
    field: string;
    text: string;
  } | null;
  question_index: number;
  total_questions: number;
  collected_fields: string[];
  required_fields: string[];
  next_action: string;
};

export type Admission = {
  patient_id: string;
  admission_id: string;
  admission_time?: string;
  status?: string;
};

export type PatientSummary = {
  patient_id: string;
  admission_id: string;
  observation_count?: number;
  assessment_count?: number;
  sepsis?: { status?: string; probability?: number | null; risk_level?: string | null; trend?: string | null };
  aki?: { status?: string; probability?: number | null; risk_level?: string | null; trend?: string | null };
  priority?: { level?: string; highest_risk_disease?: string | null; reason?: string | null; worsening_diseases?: string[] };
  latest_parameters?: Record<string, number | null>;
};

async function requestJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (body.detail && typeof body.detail.message === "string") {
        detail = body.detail.message;
      }
    } catch {
      // Keep the HTTP status when the server does not return JSON.
    }
    const error = new Error(detail) as Error & { status?: number };
    error.status = response.status;
    throw error;
  }
  return response.json() as Promise<T>;
}

export async function uploadReport(
  patientId: string,
  admissionId: string,
  reportFile: File
) : Promise<ReportResponse> {
  const formData = new FormData();

  formData.append("patient_id", patientId);
  formData.append("admission_id", admissionId);
  formData.append("report", reportFile);

  return requestJson<ReportResponse>(`${API_BASE}/api/reports`, {
    method: "POST",
    body: formData,
  });
}

export async function generateResult(
  reportId: string,
  patientId: string,
  admissionId: string
) {
  const params = new URLSearchParams({
    patient_id: patientId,
    admission_id: admissionId,
  });
  return requestJson<{
    report_id: string;
    status: string;
    task: NurseTask | null;
    observation_id: string | null;
  }>(`${API_BASE}/api/reports/${reportId}/generate-result?${params}`, {
    method: "POST",
  });
}

export async function getReportStatus(reportId: string) {
  return requestJson<ReportResponse>(`${API_BASE}/api/reports/${reportId}`);
}

export async function getNurseTasks(
  patientId?: string,
  admissionId?: string
) {
  const params = new URLSearchParams();
  if (patientId && admissionId) {
    params.set("patient_id", patientId);
    params.set("admission_id", admissionId);
  }
  return requestJson<{ tasks: NurseTask[] }>(
    `${API_BASE}/api/nurse/tasks${params.size ? `?${params}` : ""}`
  );
}

export async function getNurseTask(
  taskId: string,
  patientId: string,
  admissionId: string
) {
  const params = new URLSearchParams({
    patient_id: patientId,
    admission_id: admissionId,
  });
  return requestJson<NurseTask>(
    `${API_BASE}/api/nurse/tasks/${taskId}?${params}`
  );
}

export async function getCurrentNurseQuestion(
  taskId: string,
  patientId: string,
  admissionId: string
) {
  const params = new URLSearchParams({
    patient_id: patientId,
    admission_id: admissionId,
  });
  return requestJson<NurseTask>(
    `${API_BASE}/api/nurse/tasks/${taskId}/question?${params}`
  );
}

export async function uploadNurseAudio(
  taskId: string,
  patientId: string,
  admissionId: string,
  audioFile: File
) : Promise<NurseTask> {
  const formData = new FormData();

  formData.append("patient_id", patientId);
  formData.append("admission_id", admissionId);
  formData.append("audio", audioFile);

  return requestJson<NurseTask>(
    `${API_BASE}/api/nurse/tasks/${taskId}/audio`,
    {
      method: "POST",
      body: formData,
    }
  );
}

export async function getPatientHistory(
  patientId: string,
  admissionId: string
) {
  return requestJson<Record<string, unknown>>(
    `${API_BASE}/api/patients/${patientId}/admissions/${admissionId}/history`
  );
}

export async function getPatientAdmissions(patientId: string) {
  return requestJson<{ patient_id: string; admissions: Admission[] }>(
    `${API_BASE}/api/patients/${encodeURIComponent(patientId)}/admissions`
  );
}

export async function createPatientAndAdmission(input: {
  patient_id: string;
  admission_id: string;
  admission_time?: string;
  demographics?: Record<string, string>;
}) {
  return requestJson<{ patient: Record<string, unknown>; admission: Admission }>(
    `${API_BASE}/api/patients`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }
  );
}

export async function startNurseTask(
  taskId: string,
  patientId: string,
  admissionId: string
) {
  const params = new URLSearchParams({
    patient_id: patientId,
    admission_id: admissionId,
  });
  return requestJson<NurseTask>(
    `${API_BASE}/api/nurse/tasks/${taskId}/start?${params}`,
    { method: "POST" }
  );
}

export async function submitStructuredNurseAnswer(
  taskId: string,
  patientId: string,
  admissionId: string,
  providedValues: Record<string, unknown>
) {
  const params = new URLSearchParams({
    patient_id: patientId,
    admission_id: admissionId,
  });
  return requestJson<NurseTask>(
    `${API_BASE}/api/nurse/tasks/${taskId}/answer?${params}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(providedValues),
    }
  );
}

export async function getDashboardPatients() {
  return requestJson<{ patients: PatientSummary[] }>(`${API_BASE}/api/dashboard/patients`);
}