import { useEffect, useRef, useState } from "react";
import { Bell, CheckCircle2, Mic, Radio, Volume2 } from "lucide-react";

import { getCurrentNurseQuestion, getNurseTasks, getPatientHistory, startNurseTask, submitStructuredNurseAnswer, uploadNurseAudio } from "../api/clinicalApi";
import type { NurseTask } from "../api/clinicalApi";

type VoiceState = "IDLE" | "AGENT_SPEAKING" | "LISTENING" | "PROCESSING" | "ANSWER_ACCEPTED" | "RETRY" | "COMPLETED";

export default function NurseDashboard() {
  const [tasks, setTasks] = useState<NurseTask[]>([]);
  const [selectedTask, setSelectedTask] = useState<NurseTask | null>(null);
  const [voiceState, setVoiceState] = useState<VoiceState>("IDLE");
  const [message, setMessage] = useState("");
  const [patientResult, setPatientResult] = useState<Record<string, unknown> | null>(null);
  const [completedTask, setCompletedTask] = useState<NurseTask | null>(null);
  const [systolic, setSystolic] = useState("");
  const [diastolic, setDiastolic] = useState("");
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

  async function beginAssessment(task: NurseTask) {
    setMessage("");
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
      <div className="page-header"><div><span className="eyebrow">Nurse Workspace</span><h1>Clinical Tasks</h1><p>Persistent assessments assigned by the laboratory workflow.</p></div><div className="live-chip"><span /> Task center</div></div>
      {message && <div className="confirmation-note">{message}</div>}
      {!selectedTask && voiceState !== "COMPLETED" && <section className="workflow-card"><div className="step-header"><Bell size={22} /><div><h2>New Clinical Task</h2><p>Tasks remain available after refresh and across workstations.</p></div></div>{tasks.length === 0 ? <div className="empty-state">No new clinical tasks.</div> : <div className="request-grid">{tasks.map((task) => { const inProgress = task.status === "IN_PROGRESS"; return <article className="request-chip" key={task.task_id}><div><strong>{inProgress ? "Assessment Pending" : "New Clinical Assessment"}</strong><br />Patient: {task.patient_id}<br />{inProgress ? `Question ${task.question_index + 1} of ${task.total_questions}` : "Additional clinical information required."}<br />Requested: {task.required_fields.join(", ")}</div><button className="primary-button" onClick={() => { void beginAssessment(task); }}>{inProgress ? "Continue Assessment" : "Start Assessment"}</button></article>; })}</div>}</section>}
      {selectedTask && <section className="workflow-card"><div className="step-header"><div className="mic-circle"><Radio size={24} /></div><div><span className="eyebrow">Clinical Voice Agent</span><h2>Patient {selectedTask.patient_id}</h2><p>Question {selectedTask.question_index + 1} of {selectedTask.total_questions}</p></div></div><div className="confirmation-note"><Volume2 size={18} /> {question}</div>{selectedTask.current_question?.field === "BP" ? <div className="editable-grid"><label>Systolic BP<input inputMode="numeric" type="number" min="1" value={systolic} onChange={(event) => setSystolic(event.target.value)} /></label><label>Diastolic BP<input inputMode="numeric" type="number" min="1" value={diastolic} onChange={(event) => setDiastolic(event.target.value)} /></label><button className="primary-button" onClick={() => { void submitBloodPressure(); }} disabled={voiceState === "PROCESSING"}>Submit Blood Pressure</button></div> : <><div className={`voice-orb voice-${voiceState.toLowerCase()}`}><Mic size={42} /><strong>{stateLabel}</strong><span>{voiceState === "LISTENING" ? "Speak naturally, then stop when finished." : ""}</span></div>{voiceState === "IDLE" || voiceState === "RETRY" || voiceState === "ANSWER_ACCEPTED" ? <button className="primary-button" onClick={() => { void startListening(); }}><Mic size={18} /> Start Listening</button> : voiceState === "LISTENING" ? <button className="primary-button" onClick={stopListening}><Mic size={18} /> Stop Recording</button> : null}</>}</section>}
      {voiceState === "COMPLETED" && completedTask && <section className="result-panel"><div className="result-heading"><div><span className="eyebrow">Assessment Completed</span><h2>Clinical information collected successfully.</h2><p>Patient: {completedTask.patient_id}</p></div><CheckCircle2 size={28} className="success-icon" /></div><button className="primary-button" onClick={() => { void showResult(); }}>Show Result</button><div className="assessment-note">The downstream clinical workflow continues automatically.</div>{patientResult && <pre>{JSON.stringify({ assessments: patientResult.assessments, trends: patientResult.trends, prioritization: patientResult.latest_prioritization }, null, 2)}</pre>}</section>}
    </div>
  );
}
