export type WorkflowTaskMetadata = {
  report_id?: string;
  missing_fields?: string[];
  current_question?: string | null;
  question_index?: number;
  question_fields?: string[];
  [key: string]: unknown;
};

export type WorkflowTask = {
  task_id: string;
  task_type: string;
  patient_id: string;
  admission_id: string;
  status: string;
  required_fields: string[];
  provided_fields: string[];
  created_at?: string | null;
  updated_at?: string | null;
  completed_at?: string | null;
  metadata: WorkflowTaskMetadata;
};

export type LabUploadResponse = {
  report_id: string;
  status: string;
  workflow_started: boolean;
  task_id?: string | null;
};

export type NurseAudioResponse = {
  completed: boolean;
  next_question?: string | null;
  question_index?: number | null;
  total_questions?: number | null;
  observation_id?: string | null;
  assessment_started: boolean;
};

export type PatientRecord = {
  patient_id?: string;
  demographics?: Record<string, unknown>;
  [key: string]: unknown;
};

export type ObservationRecord = {
  clinical_parameters?: Record<string, number | null>;
  observation_time?: string;
  [key: string]: unknown;
};

export type DiseaseAssessmentRecord = {
  probability?: number;
  status?: string;
  risk_level?: string;
  [key: string]: unknown;
};

export type AssessmentRecord = {
  sepsis?: DiseaseAssessmentRecord;
  aki?: DiseaseAssessmentRecord;
  sepsis_result?: DiseaseAssessmentRecord;
  aki_result?: DiseaseAssessmentRecord;
  assessment_time?: string;
  [key: string]: unknown;
};

export type TrendRecord = {
  disease?: string;
  trend?: string;
  first_probability?: number | null;
  latest_probability?: number | null;
  probability_change?: number | null;
  [key: string]: unknown;
};

export type PrioritizationRecord = {
  priority_level?: string;
  reason?: string;
  highest_risk_disease?: string;
  [key: string]: unknown;
};

export type DoctorPatientResponse = {
  patient: PatientRecord;
  admission: Record<string, unknown>;
  observations: ObservationRecord[];
  assessments: AssessmentRecord[];
  trends: TrendRecord[];
  prioritizations: PrioritizationRecord[];
  reports: Record<string, unknown>[];
  tasks: WorkflowTask[];
};

export type DashboardPatient = {
  patient_id: string;
  admission_id: string;
  observation_count?: number;
  assessment_count?: number;
  sepsis?: {
    status?: string;
    probability?: number | null;
    risk_level?: string | null;
    trend?: string | null;
    probability_change?: number | null;
    first_probability?: number | null;
    latest_probability?: number | null;
  };
  aki?: {
    status?: string;
    probability?: number | null;
    risk_level?: string | null;
    trend?: string | null;
  };
  priority?: {
    level?: string;
    reason?: string | null;
    highest_risk_disease?: string | null;
    worsening_diseases?: string[];
  };
  latest_parameters?: Record<string, number | null>;
  latest_report?: Record<string, unknown> | null;
  pending_task_count?: number;
  latest_assessment?: Record<string, unknown> | null;
  latest_trend?: Record<string, unknown> | null;
  latest_prioritization?: Record<string, unknown> | null;
};

export type DashboardResponse = {
  status: string;
  patient_count: number;
  patients: DashboardPatient[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail || detail;
    } catch {
      // Keep the status-based error when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function uploadLabReport(patientId: string, admissionId: string, reportFile: File): Promise<LabUploadResponse> {
  const formData = new FormData();
  formData.append("patient_id", patientId);
  formData.append("admission_id", admissionId);
  formData.append("report_file", reportFile);
  return request<LabUploadResponse>("/api/lab/upload-report", { method: "POST", body: formData });
}

export function getNurseTasks(filters?: { patientId?: string; admissionId?: string; status?: string }): Promise<WorkflowTask[]> {
  const params = new URLSearchParams();
  if (filters?.patientId) params.set("patient_id", filters.patientId);
  if (filters?.admissionId) params.set("admission_id", filters.admissionId);
  if (filters?.status) params.set("status", filters.status);
  const query = params.toString() ? `?${params.toString()}` : "";
  return request<WorkflowTask[]>(`/api/nurse/tasks${query}`);
}

export function getNurseTask(taskId: string): Promise<WorkflowTask> {
  return request<WorkflowTask>(`/api/nurse/tasks/${encodeURIComponent(taskId)}`);
}

export function submitNurseAudio(taskId: string, audioFile: File): Promise<NurseAudioResponse> {
  const formData = new FormData();
  formData.append("audio_file", audioFile);
  return request<NurseAudioResponse>(`/api/nurse/tasks/${encodeURIComponent(taskId)}/audio`, { method: "POST", body: formData });
}

export function getDoctorPatient(patientId: string, admissionId: string): Promise<DoctorPatientResponse> {
  return request<DoctorPatientResponse>(`/api/doctor/patient/${encodeURIComponent(patientId)}?admission_id=${encodeURIComponent(admissionId)}`);
}

export function getDashboardPatients(): Promise<DashboardResponse> {
  return request<DashboardResponse>("/api/dashboard/patients");
}
