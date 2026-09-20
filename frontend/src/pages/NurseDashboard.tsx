import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, FileText, Mic, RefreshCw, Upload } from "lucide-react";

import {
  getNurseTask,
  getNurseTasks,
  submitNurseAudio,
  uploadLabReport,
  type LabUploadResponse,
  type WorkflowTask,
} from "../api/clinicalApi";

function getQuestion(task: WorkflowTask): string | null {
  if (task.metadata.current_question) return task.metadata.current_question;
  const fields = task.metadata.question_fields || task.required_fields;
  const index = task.metadata.question_index || 0;
  return index < fields.length ? `Please provide ${fields[index]}.` : null;
}

export default function NurseDashboard() {
  const [patientId, setPatientId] = useState("P1446");
  const [admissionId, setAdmissionId] = useState("ADM-1446");
  const [reportFile, setReportFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [tasks, setTasks] = useState<WorkflowTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<WorkflowTask | null>(null);
  const [uploadResult, setUploadResult] = useState<LabUploadResponse | null>(null);
  const [status, setStatus] = useState("idle");
  const [loadingTasks, setLoadingTasks] = useState(true);
  const [error, setError] = useState("");

  const refreshTasks = useCallback(async () => {
    try {
      setLoadingTasks(true);
      const result = await getNurseTasks({ patientId, admissionId });
      setTasks(result);
      if (selectedTask) setSelectedTask(await getNurseTask(selectedTask.task_id));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load tasks.");
    } finally {
      setLoadingTasks(false);
    }
  }, [patientId, admissionId, selectedTask]);

  async function selectTask(taskId: string) {
    try {
      setError("");
      setSelectedTask(await getNurseTask(taskId));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load task.");
    }
  }

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void refreshTasks(), 0);
    const interval = window.setInterval(() => void refreshTasks(), 15000);
    return () => {
      window.clearTimeout(initialLoad);
      window.clearInterval(interval);
    };
  }, [refreshTasks]);

  async function handleReportUpload() {
    if (!reportFile) return;
    try {
      setError("");
      setStatus("processing-report");
      const result = await uploadLabReport(patientId, admissionId, reportFile);
      setUploadResult(result);
      setStatus("workflow-started");
      await refreshTasks();
      if (result.task_id) await selectTask(result.task_id);
    } catch (requestError) {
      setStatus("error");
      setError(requestError instanceof Error ? requestError.message : "Upload failed.");
    }
  }

  async function handleAudioUpload() {
    if (!audioFile || !selectedTask) return;
    try {
      setError("");
      setStatus("processing-audio");
      const result = await submitNurseAudio(selectedTask.task_id, audioFile);
      setSelectedTask(await getNurseTask(selectedTask.task_id));
      await refreshTasks();
      setAudioFile(null);
      setStatus(result.completed ? "completed" : "question-ready");
    } catch (requestError) {
      setStatus("error");
      setError(requestError instanceof Error ? requestError.message : "Audio processing failed.");
    }
  }

  const selectedFields = selectedTask?.metadata.question_fields || selectedTask?.required_fields || [];
  const selectedIndex = selectedTask?.metadata.question_index || 0;

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">Nurse Workspace</span>
          <h1>Event-Driven Clinical Tasks</h1>
          <p>Upload laboratory reports and complete assigned bedside questionnaires.</p>
        </div>
        <div className="live-chip"><span /> Live Monitoring</div>
      </div>

      <div className="patient-strip">
        <div><label>Patient ID</label><input value={patientId} onChange={(event) => setPatientId(event.target.value)} /></div>
        <div><label>Admission ID</label><input value={admissionId} onChange={(event) => setAdmissionId(event.target.value)} /></div>
        <button className="primary-button" onClick={() => void refreshTasks()} title="Refresh tasks"><RefreshCw size={17} /> Refresh</button>
      </div>

      {error && <div className="empty-state">{error}</div>}

      <div className="workflow-grid">
        <section className="workflow-card">
          <div className="step-header"><div className="step-number">1</div><div><h2>Upload laboratory report</h2><p>Start a durable event-driven workflow.</p></div></div>
          <label className="upload-zone">
            <FileText size={34} />
            <strong>{reportFile ? reportFile.name : "Choose laboratory report"}</strong>
            <span>PDF, DOCX, PNG or JPG</span>
            <input type="file" accept=".pdf,.docx,image/*" hidden onChange={(event) => setReportFile(event.target.files?.[0] || null)} />
          </label>
          <button className="primary-button" onClick={() => void handleReportUpload()} disabled={!reportFile || status === "processing-report"}>
            <Upload size={18} /> {status === "processing-report" ? "Uploading..." : "Upload Report"}
          </button>
          {uploadResult && <div className="result-panel"><strong>Report uploaded</strong><span>Workflow started</span><small>Report ID: {uploadResult.report_id}</small></div>}
        </section>

        <section className="workflow-card">
          <div className="step-header"><div className="step-number">2</div><div><h2>Nurse task queue</h2><p>Tasks refresh every 15 seconds.</p></div></div>
          {loadingTasks ? <div className="empty-state">Loading...</div> : tasks.length === 0 ? <div className="empty-state">No tasks</div> : (
            <div className="request-grid">
              {tasks.map((task) => <button className="request-chip" key={task.task_id} onClick={() => void selectTask(task.task_id)}><CheckCircle2 size={16} /><span>{task.task_id}<br />{task.patient_id} / {task.admission_id}<br />{task.status} - {getQuestion(task) || "Complete"}</span></button>)}
            </div>
          )}
        </section>

        <section className="workflow-card">
          <div className="step-header"><div className="step-number">3</div><div><h2>Voice questionnaire</h2><p>Answer the current question to advance the task.</p></div></div>
          {!selectedTask ? <div className="empty-state">Select a task to begin.</div> : (
            <>
              <div className="confirmation-note"><strong>Current question</strong><p>{getQuestion(selectedTask) || "Questionnaire completed"}</p><span>Question {Math.min(selectedIndex + 1, selectedFields.length)} of {selectedFields.length}</span></div>
              <label className="audio-zone">
                <div className="mic-circle"><Mic size={28} /></div>
                <strong>{audioFile ? audioFile.name : "Upload nurse recording"}</strong>
                <span>Audio is transcribed and submitted automatically.</span>
                <input type="file" accept="audio/*" hidden onChange={(event) => setAudioFile(event.target.files?.[0] || null)} />
              </label>
              <button className="primary-button" onClick={() => void handleAudioUpload()} disabled={!audioFile || status === "processing-audio" || selectedTask.status === "COMPLETED"}><Mic size={18} /> {status === "processing-audio" ? "Processing..." : "Submit Answer"}</button>
            </>
          )}
        </section>
      </div>

      {status === "question-ready" && <section className="result-panel"><strong>Answer recorded</strong><span>Next question loaded.</span></section>}
      {status === "completed" && <section className="result-panel"><div className="result-heading"><div><span className="eyebrow">Questionnaire Complete</span><h2>Assessment started</h2></div><CheckCircle2 size={28} className="success-icon" /></div></section>}
    </div>
  );
}
