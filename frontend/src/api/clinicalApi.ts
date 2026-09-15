const API_BASE = "http://127.0.0.1:8000";

export async function uploadReport(
  patientId: string,
  admissionId: string,
  reportFile: File
) {
  const formData = new FormData();

  formData.append("patient_id", patientId);
  formData.append("admission_id", admissionId);
  formData.append("report", reportFile);

  const response = await fetch(
    `${API_BASE}/api/input/report`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error("Report upload failed.");
  }

  return response.json();
}

export async function uploadNurseAudio(
  inputId: string,
  audioFile: File
) {
  const formData = new FormData();

  formData.append("audio", audioFile);

  const response = await fetch(
    `${API_BASE}/api/input/${inputId}/audio`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error("Audio upload failed.");
  }

  return response.json();
}

export async function confirmObservation(
  inputId: string,
  confirmedParameters: Record<string, number>
) {
  const response = await fetch(
    `${API_BASE}/api/input/${inputId}/confirm`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        confirmed_parameters: confirmedParameters,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Observation confirmation failed.");
  }

  return response.json();
}

export async function getPatientHistory(
  patientId: string,
  admissionId: string
) {
  const response = await fetch(
    `${API_BASE}/api/patients/${patientId}/admissions/${admissionId}/history`
  );

  if (!response.ok) {
    throw new Error("Could not load patient history.");
  }

  return response.json();
}