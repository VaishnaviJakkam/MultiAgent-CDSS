import { useRef, useState } from "react";
import { CheckCircle2, FileText, Upload } from "lucide-react";

import BackButton from "../components/BackButton";
import {
  createPatientAndAdmission,
  generateResult,
  getPatientAdmissions,
  uploadReport,
} from "../api/clinicalApi";
import type { Admission } from "../api/clinicalApi";

type PatientMode = "new" | "existing" | null;

export default function LabDashboard() {
  const [mode, setMode] = useState<PatientMode>(null);
  const [patientId, setPatientId] = useState("");
  const [admissionId, setAdmissionId] = useState("");
  const [admissions, setAdmissions] = useState<Admission[]>([]);
  const [reportFile, setReportFile] = useState<File | null>(null);
  const [reportId, setReportId] = useState("");
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("");
  const reportInputRef = useRef<HTMLInputElement>(null);

  function resetPatientSelection() {
    setMode(null);
    setPatientId("");
    setAdmissionId("");
    setAdmissions([]);
    setReportFile(null);
    setReportId("");
    setStatus("idle");
    setMessage("");
    if (reportInputRef.current) {
      reportInputRef.current.value = "";
    }
  }

  async function preparePatient() {
    setMessage("");
    try {
      if (mode === "new") {
        await createPatientAndAdmission({ patient_id: patientId.trim(), admission_id: admissionId.trim() });
        setStatus("ready");
        return;
      }
      const result = await getPatientAdmissions(patientId.trim());
      const active = result.admissions.filter((item) => item.status === "active");
      setAdmissions(result.admissions);
      if (active.length === 1) {
        setAdmissionId(active[0].admission_id);
        setStatus("ready");
      } else if (active.length === 0) {
        setMessage("Patient not found or has no active admission. Please verify the Patient ID.");
      } else {
        setMessage("Select the active admission before continuing.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not prepare patient.");
    }
  }

  async function submitReport() {
    if (!reportFile || !patientId || !admissionId) return;
    setStatus("uploading");
    setMessage("Uploading report...");
    try {
      const result = await uploadReport(patientId, admissionId, reportFile);
      setReportId(result.report_id);
      if (result.processing_status === "PROCESSED") {
        setStatus("processed");
        setMessage("Report processed successfully.");
      } else {
        setStatus("failed");
        setMessage(`Report processing ${result.processing_status.toLowerCase()}. Retry the upload.`);
      }
    } catch (error) {
      setStatus("failed");
      setMessage(error instanceof Error ? error.message : "Report processing failed.");
    }
  }

  async function generateReport() {
    if (!reportId) return;
    setStatus("generating");
    setMessage("Generating report...");
    try {
      const result = await generateResult(reportId, patientId, admissionId);
      setStatus("complete");
      setMessage(result.task ? "Additional clinical information has been requested from the nurse." : "Report generated successfully.");
    } catch (error) {
      setStatus("failed");
      setMessage(error instanceof Error ? error.message : "Could not generate report.");
    }
  }

  return (
    <div className="page">
      <div className="page-header"><div><span className="eyebrow">Lab Workspace</span><h1>Laboratory intake</h1><p>Prepare and generate a report for the clinical team.</p></div><div className="live-chip"><span /> Lab workflow</div></div>
      {message && <div className="confirmation-note">{message}</div>}
      <section className="workflow-card">
        <div className="step-header"><div className="step-number">1</div><div><h2>Identify patient</h2><p>Choose the patient context before uploading.</p></div>{mode && <BackButton label="Back to patient options" onClick={resetPatientSelection} disabled={status === "uploading" || status === "generating"} />}</div>
        {!mode ? <div className="request-grid"><button className="primary-button" onClick={() => setMode("new")}>New Patient</button><button className="primary-button" onClick={() => setMode("existing")}>Existing Patient</button></div> : <>
          <div className="editable-grid"><label>Patient ID *<input value={patientId} onChange={(event) => setPatientId(event.target.value)} /></label>{mode === "new" && <label>Admission ID *<input value={admissionId} onChange={(event) => setAdmissionId(event.target.value)} /></label>}{mode === "existing" && admissions.length > 1 && <label>Active admission *<select value={admissionId} onChange={(event) => setAdmissionId(event.target.value)}><option value="">Select admission</option>{admissions.filter((item) => item.status === "active").map((item) => <option key={item.admission_id} value={item.admission_id}>{item.admission_id}</option>)}</select></label>}</div>
          <button className="primary-button" onClick={() => { void preparePatient(); }} disabled={!patientId.trim() || (mode === "new" && !admissionId.trim())}>{mode === "new" ? "Create Patient & Continue" : "Resolve Patient"}</button>
        </>}
      </section>
      <section className="workflow-card"><div className="step-header"><div className="step-number">2</div><div><h2>Upload laboratory report</h2><p>Processing completes before generation.</p></div></div><label className="upload-zone"><FileText size={34} /><strong>{reportFile?.name || "Choose laboratory report"}</strong><span>PDF, DOCX, PNG, JPG or JPEG</span><input ref={reportInputRef} type="file" accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/png,image/jpeg" hidden disabled={status !== "ready"} onChange={(event) => setReportFile(event.target.files?.[0] || null)} /></label><button className="primary-button" onClick={() => { void submitReport(); }} disabled={status !== "ready" || !reportFile}><Upload size={18} /> Upload Report</button></section>
      {status === "processed" && <section className="result-panel"><div className="result-heading"><div><span className="eyebrow">Report processed</span><h2>Ready to generate</h2></div><CheckCircle2 size={28} className="success-icon" /></div><button className="primary-button" onClick={() => { void generateReport(); }}>Generate Report</button></section>}
      {status === "complete" && <section className="result-panel"><div className="result-heading"><div><span className="eyebrow">Lab task complete</span><h2>Report generated successfully</h2></div><CheckCircle2 size={28} className="success-icon" /></div><div className="assessment-note">The nurse can complete the persistent clinical assessment separately.</div></section>}
    </div>
  );
}
