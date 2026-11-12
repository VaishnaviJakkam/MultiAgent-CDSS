import { useEffect, useRef, useState } from "react";
import { Bell, CheckCircle2, Mic, Radio, Volume2 } from "lucide-react";

import BackButton from "../components/BackButton";
import { getCurrentNurseQuestion, getNurseTasks, getPatientHistory, startNurseTask, submitStructuredNurseAnswer, uploadNurseAudio } from "../api/clinicalApi";
import type { NurseTask } from "../api/clinicalApi";

type VoiceState = "IDLE" | "AGENT_SPEAKING" | "LISTENING" | "PROCESSING" | "ANSWER_ACCEPTED" | "RETRY" | "COMPLETED";
type AnswerMode = "voice" | "text";

const CLINICAL_VALUE_RANGES: Record<string, { min: number; max: number }> = {
  HR: { min: 30, max: 250 },
  O2Sat: { min: 50, max: 100 },
  Temp: { min: 25, max: 45 },
  SBP: { min: 40, max: 250 },
  DBP: { min: 20, max: 150 },
  Resp: { min: 4, max: 60 },
  WBC: { min: 0.1, max: 200 },
  Lactate: { min: 0.1, max: 30 },
};

export default function NurseDashboard() {
  const [tasks, setTasks] = useState<NurseTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<NurseTask | null>(null);
  const [voiceState, setVoiceState] = useState<VoiceState>("IDLE");
  const [answerMode, setAnswerMode] = useState<AnswerMode>("voice");
  const [message, setMessage] = useState("");
  const [patientResult, setPatientResult] = useState<Record<string, unknown> | null>(null);
  const [completedTask, setCompletedTask] = useState<NurseTask | null>(null);
  const [systolic, setSystolic] = useState("");
  const [diastolic, setDiastolic] = useState("");
  const [textAnswer, setTextAnswer] = useState("");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function loadTasks() {
    try {
      const result = await getNurseTasks();
      setTasks(result.tasks.filter((task) => task.status !== "COMPLETED" && task.status !== "CANCELLED"));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load clinical tasks.");
    }
  }

  function backToTasks() {
    window.speechSynthesis.cancel();
    const recorder = recorderRef.current;
    if (recorder) {
      recorder.onstop = null;
      if (recorder.state === "recording") {
        recorder.stop();
      }
      recorder.stream.getTracks().forEach((track) => track.stop());
      recorderRef.current = null;
    }
    chunksRef.current = [];
    setSelectedTask(null);
    setCompletedTask(null);
    setPatientResult(null);
    setVoiceState("IDLE");
    setAnswerMode("voice");
    setSystolic("");
    setDiastolic("");
    setTextAnswer("");
    setMessage("");
    void loadTasks();
  }

  useEffect(() => {
    const initialLoad = window.setTimeout(() => { void loadTasks(); }, 0);
    const timer = window.setInterval(() => { void loadTasks(); }, 15000);
    return () => {
      window.clearTimeout(initialLoad);
      window.clearInterval(timer);
    };
  }, []);

  function speak(text: string, onEnd?: () => void) {
    window.speechSynthesis.cancel();
    setVoiceState("AGENT_SPEAKING");
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => onEnd?.();
    window.speechSynthesis.speak(utterance);
  }

  function spokenQuestion(text: string) {
    return `${text} If no value is available, say none.`;
  }

  async function completeAssessment(task: NurseTask) {
    setSelectedTask(null);
    setVoiceState("COMPLETED");
    setCompletedTask(task);
    speak("Clinical information collected successfully. Your assessment has been submitted.");
  }

  async function showResult() {
    if (!completedTask) return;
    try {
      setPatientResult(await getPatientHistory(completedTask.patient_id, completedTask.admission_id));
    } catch {
      setMessage("The assessment is complete, but the patient result could not be loaded.");
    }
  }

  function buildTextAnswerPayload(field: string, rawValue: string): Record<string, unknown> {
    const trimmed = rawValue.trim();
    if (!trimmed) {
      throw new Error("Enter a value before submitting.");
    }

    const normalized = trimmed.toLowerCase();
    const unavailableAnswers = ["none", "n/a", "na"];
    if (field === "BP") {
      if (unavailableAnswers.includes(normalized)) {
        return { BP: { SBP: null, DBP: null } };
      }
      const match = trimmed.match(/^(\d+(?:\.\d+)?)\s*[/,]\s*(\d+(?:\.\d+)?)$/);
      if (!match) {
        throw new Error("Enter blood pressure as positive numbers in the format '120/80'.");
      }
      const systolic = Number(match[1]);
      const diastolic = Number(match[2]);
      const systolicRange = CLINICAL_VALUE_RANGES.SBP;
      const diastolicRange = CLINICAL_VALUE_RANGES.DBP;
      if (!Number.isFinite(systolic) || systolic < systolicRange.min || systolic > systolicRange.max) {
        throw new Error(`Systolic blood pressure must be between ${systolicRange.min} and ${systolicRange.max} mmHg.`);
      }
      if (!Number.isFinite(diastolic) || diastolic < diastolicRange.min || diastolic > diastolicRange.max) {
        throw new Error(`Diastolic blood pressure must be between ${diastolicRange.min} and ${diastolicRange.max} mmHg.`);
      }
      if (systolic <= diastolic) {
        throw new Error("Systolic pressure must be greater than diastolic pressure.");
      }
      return { BP: { SBP: systolic, DBP: diastolic } };
    }

    if (unavailableAnswers.includes(normalized)) {
      return { [field]: null };
    }

    const numericText = field === "O2Sat" ? trimmed.replace(/%$/, "").trim() : trimmed;
    if (!/^(?:\d+(?:\.\d+)?|\.\d+)$/.test(numericText)) {
      throw new Error(`Enter a positive numeric value for ${field}.`);
    }

    const value = Number(numericText);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`Enter a positive numeric value for ${field}.`);
    }
    const range = CLINICAL_VALUE_RANGES[field];
    if (range && (value < range.min || value > range.max)) {
      const unit = field === "O2Sat" ? "%" : field === "Temp" ? " °C" : field === "HR" ? " bpm" : field === "Resp" ? " breaths/min" : "";
      throw new Error(`${field} must be between ${range.min} and ${range.max}${unit}.`);
    }
    return { [field]: value };
  }

  async function submitTextAnswer() {
    if (!selectedTask || !selectedTask.current_question) return;

    const field = selectedTask.current_question.field;
    const typedAnswer = textAnswer.trim();
    if (!typedAnswer) {
      setMessage(`Enter a value for ${field}.`);
      return;
    }

    setVoiceState("PROCESSING");
    setMessage("");
    try {
      const next = await submitStructuredNurseAnswer(
        selectedTask.task_id,
        selectedTask.patient_id,
        selectedTask.admission_id,
        buildTextAnswerPayload(field, typedAnswer)
      );
      setTextAnswer("");
      if (next.status === "COMPLETED") {
        await completeAssessment(selectedTask);
        return;
      }
      setSelectedTask(next);
      setVoiceState("ANSWER_ACCEPTED");
      speak(spokenQuestion(next.current_question?.text || "What is the next clinical value?"), () => setVoiceState("IDLE"));
    } catch (error) {
      setVoiceState("RETRY");
      setMessage(error instanceof Error ? error.message : "Could not save that answer. Please try again.");
    }
  }

  async function beginAssessment(task: NurseTask) {
    setMessage("");
    setTextAnswer("");
    setAnswerMode("voice");
    try {
      const started = await startNurseTask(task.task_id, task.patient_id, task.admission_id);
      const current = await getCurrentNurseQuestion(started.task_id, started.patient_id, started.admission_id);
      setSelectedTask(current);
      const question = current.current_question?.text || "I need additional clinical information.";
      speak(`Hello. I need some additional information about patient ${current.patient_id}. ${spokenQuestion(question)}`, () => setVoiceState("IDLE"));
    } catch (error) {
      const status = error instanceof Error && "status" in error
        ? (error as Error & { status?: number }).status
        : undefined;
      setMessage(
        status === 404
          ? "Assessment task could not be found. Please refresh the task list."
          : error instanceof Error
            ? error.message
            : "Could not open this task."
      );
    }
  }

  async function startListening() {
    if (!selectedTask) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => { if (event.data.size > 0) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        const audio = new File([blob], "nurse-answer.webm", { type: blob.type });
        void submitAudio(audio);
      };
      recorderRef.current = recorder;
      recorder.start();
      setVoiceState("LISTENING");
    } catch {
      setVoiceState("RETRY");
      setMessage("Microphone access is required to begin the voice assessment.");
    }
  }

  function stopListening() {
    if (recorderRef.current?.state === "recording") {
      setVoiceState("PROCESSING");
      recorderRef.current.stop();
    }
  }

  async function submitAudio(audio: File) {
    if (!selectedTask) return;
    setVoiceState("PROCESSING");
    try {
      const next = await uploadNurseAudio(selectedTask.task_id, selectedTask.patient_id, selectedTask.admission_id, audio);
      setTasks((current) => current.filter((task) => task.task_id !== selectedTask.task_id || next.status !== "COMPLETED"));
      if (next.status === "COMPLETED") {
        await completeAssessment(selectedTask);
        return;
      }
      setSelectedTask(next);
      setVoiceState("ANSWER_ACCEPTED");
      speak(spokenQuestion(next.current_question?.text || "What is the next clinical value?"), () => setVoiceState("IDLE"));
    } catch (error) {
      setVoiceState("RETRY");
      setMessage(error instanceof Error ? error.message : "I couldn't confidently understand that answer. Please try again.");
    }
  }

  async function submitBloodPressure() {
    if (!selectedTask || !systolic || !diastolic) {
      setMessage("Enter both systolic and diastolic blood pressure values.");
      return;
    }

    const systolicValue = Number(systolic);
    const diastolicValue = Number(diastolic);
    if (!Number.isFinite(systolicValue) || !Number.isFinite(diastolicValue) || systolicValue <= 0 || diastolicValue <= 0) {
      setMessage("Enter valid positive blood pressure values.");
      return;
    }

    setVoiceState("PROCESSING");
    setMessage("");
    try {
      const next = await submitStructuredNurseAnswer(
        selectedTask.task_id,
        selectedTask.patient_id,
        selectedTask.admission_id,
        { BP: { SBP: systolicValue, DBP: diastolicValue } }
      );
      setSystolic("");
      setDiastolic("");
      if (next.status === "COMPLETED") {
        await completeAssessment(selectedTask);
        return;
      }
      setSelectedTask(next);
      setVoiceState("ANSWER_ACCEPTED");
      speak(spokenQuestion(next.current_question?.text || "What is the next clinical value?"), () => setVoiceState("IDLE"));
    } catch (error) {
      setVoiceState("RETRY");
      setMessage(error instanceof Error ? error.message : "Could not save blood pressure. Please try again.");
    }
  }

  const question = selectedTask?.current_question?.text || "";
  const stateLabel = {
    IDLE: "Ready when you are.",
    AGENT_SPEAKING: "Clinical Agent is speaking...",
    LISTENING: "I'm listening...",
    PROCESSING: "Processing your response...",
    ANSWER_ACCEPTED: "Recorded successfully.",
    RETRY: "I couldn't confidently understand that answer. Please try again.",
    COMPLETED: "Clinical information collected successfully.",
  }[voiceState];

  return (
    <div className="page">
      <div className="page-header"><div><span className="eyebrow">Nurse Workspace</span><h1>Clinical Tasks</h1><p>Persistent assessments assigned by the laboratory workflow.</p></div><div className="page-header-actions">{(selectedTask || voiceState === "COMPLETED") && <BackButton label="Back to tasks" onClick={backToTasks} disabled={voiceState === "PROCESSING"} />}<div className="live-chip"><span /> Task center</div></div></div>
      {message && <div className="confirmation-note">{message}</div>}
      {!selectedTask && voiceState !== "COMPLETED" && <section className="workflow-card"><div className="step-header"><Bell size={22} /><div><h2>New Clinical Task</h2><p>Tasks remain available after refresh and across workstations.</p></div></div>{tasks.length === 0 ? <div className="empty-state">No new clinical tasks.</div> : <div className="request-grid">{tasks.map((task) => { const inProgress = task.status === "IN_PROGRESS"; return <article className="request-chip" key={task.task_id}><div><strong>{inProgress ? "Assessment Pending" : "New Clinical Assessment"}</strong><br />Patient: {task.patient_id}<br />{inProgress ? `Question ${task.question_index + 1} of ${task.total_questions}` : "Additional clinical information required."}<br />Requested: {task.required_fields.join(", ")}</div><button className="primary-button" onClick={() => { void beginAssessment(task); }}>{inProgress ? "Continue Assessment" : "Start Assessment"}</button></article>; })}</div>}</section>}
      {selectedTask && <section className="workflow-card"><div className="step-header"><div className="mic-circle"><Radio size={24} /></div><div><span className="eyebrow">Clinical Voice Agent</span><h2>Patient {selectedTask.patient_id}</h2><p>Question {selectedTask.question_index + 1} of {selectedTask.total_questions}</p></div></div><div className="confirmation-note"><Volume2 size={18} /> {question}</div><div className="editable-grid" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))", marginBottom: "0.75rem" }}><div style={{ display: "flex", gap: "0.5rem", gridColumn: "1 / -1" }}><button className={answerMode === "voice" ? "primary-button" : "secondary-button"} onClick={() => setAnswerMode("voice")} type="button">Voice</button><button className={answerMode === "text" ? "primary-button" : "secondary-button"} onClick={() => setAnswerMode("text")} type="button">Text</button></div></div>{answerMode === "voice" ? (selectedTask.current_question?.field === "BP" ? <div className="editable-grid"><label>Systolic BP<input inputMode="decimal" type="number" min={CLINICAL_VALUE_RANGES.SBP.min} max={CLINICAL_VALUE_RANGES.SBP.max} value={systolic} onChange={(event) => setSystolic(event.target.value)} /></label><label>Diastolic BP<input inputMode="decimal" type="number" min={CLINICAL_VALUE_RANGES.DBP.min} max={CLINICAL_VALUE_RANGES.DBP.max} value={diastolic} onChange={(event) => setDiastolic(event.target.value)} /></label><button className="primary-button" onClick={() => { void submitBloodPressure(); }} disabled={voiceState === "PROCESSING"}>Submit Blood Pressure</button></div> : <><div className={`voice-orb voice-${voiceState.toLowerCase()}`}><Mic size={42} /><strong>{stateLabel}</strong><span>{voiceState === "LISTENING" ? "Speak naturally, then stop when finished." : ""}</span></div>{voiceState === "IDLE" || voiceState === "RETRY" || voiceState === "ANSWER_ACCEPTED" ? <button className="primary-button" onClick={() => { void startListening(); }}><Mic size={18} /> Start Listening</button> : voiceState === "LISTENING" ? <button className="primary-button" onClick={stopListening}><Mic size={18} /> Stop Recording</button> : null}</>) : <div className="editable-grid"><label>{selectedTask.current_question?.field === "BP" ? "Blood pressure" : selectedTask.current_question?.field || "Answer"}<input type="text" inputMode="decimal" min={CLINICAL_VALUE_RANGES[selectedTask.current_question?.field || ""]?.min} max={CLINICAL_VALUE_RANGES[selectedTask.current_question?.field || ""]?.max} value={textAnswer} onChange={(event) => setTextAnswer(event.target.value)} placeholder={selectedTask.current_question?.field === "BP" ? "120/80" : "Enter a value within the allowed range"} /></label><button className="primary-button" onClick={() => { void submitTextAnswer(); }} disabled={voiceState === "PROCESSING"}>Submit Answer</button></div>}</section>}
      {voiceState === "COMPLETED" && completedTask && <section className="result-panel"><div className="result-heading"><div><span className="eyebrow">Assessment Completed</span><h2>Clinical information collected successfully.</h2><p>Patient: {completedTask.patient_id}</p></div><CheckCircle2 size={28} className="success-icon" /></div><button className="primary-button" onClick={() => { void showResult(); }}>Show Result</button><div className="assessment-note">The downstream clinical workflow continues automatically.</div>{patientResult && <pre>{JSON.stringify({ assessments: patientResult.assessments, trends: patientResult.trends, prioritization: patientResult.latest_prioritization }, null, 2)}</pre>}</section>}
    </div>
  );
}
