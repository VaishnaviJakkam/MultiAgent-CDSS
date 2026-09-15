import { useState } from "react";
import {
  CheckCircle2,
  FileText,
  Mic,
  Upload,
} from "lucide-react";

import {
  confirmObservation,
  uploadNurseAudio,
  uploadReport,
} from "../api/clinicalApi";

import ParameterCard from "../components/ParameterCard";

export default function NurseDashboard() {
  const [patientId, setPatientId] = useState(
  "P1446"
);

const [admissionId, setAdmissionId] = useState(
  "ADM-1446"
);

  const [reportFile, setReportFile] =
    useState<File | null>(null);

  const [audioFile, setAudioFile] =
    useState<File | null>(null);

  const [inputId, setInputId] =
    useState("");

  const [nurseRequest, setNurseRequest] =
    useState<string[]>([]);

  const [extracted, setExtracted] =
    useState<Record<string, number>>({});

  const [transcript, setTranscript] =
    useState("");

  const [status, setStatus] =
    useState("idle");

  const [finalResult, setFinalResult] =
    useState<any>(null);

  async function handleReportUpload() {
    if (!reportFile) return;

    setStatus("processing-report");

    try {
      const result = await uploadReport(
        patientId,
        admissionId,
        reportFile
      );

      setInputId(result.input_id);

      setNurseRequest(
        result.nurse_request
          ?.requested_nurse_fields || []
      );

      setStatus("report-complete");
    } catch {
      setStatus("error");
    }
  }

  async function handleAudioUpload() {
    if (!audioFile || !inputId) return;

    setStatus("processing-audio");

    try {
      const result = await uploadNurseAudio(
        inputId,
        audioFile
      );

      setExtracted(
        result.extracted_parameters || {}
      );

      setTranscript(
        result.transcript || ""
      );

      setStatus("awaiting-confirmation");
    } catch {
      setStatus("error");
    }
  }

  function updateParameter(
    key: string,
    value: string
  ) {
    setExtracted((current) => ({
      ...current,
      [key]: Number(value),
    }));
  }

  async function handleConfirm() {
    if (!inputId) return;

    setStatus("saving");

    try {
      const result =
        await confirmObservation(
          inputId,
          extracted
        );

      setFinalResult(result);
      setStatus("completed");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">
            Nurse Workspace
          </span>

          <h1>New Clinical Observation</h1>

          <p>
            Upload a report, provide missing
            bedside parameters and confirm the
            observation before assessment.
          </p>
        </div>

        <div className="live-chip">
          <span />
          Live Monitoring
        </div>
      </div>

      <div className="patient-strip">
        <div>
          <label>Patient ID</label>
          <input
            value={patientId}
            onChange={(e) =>
              setPatientId(e.target.value)
            }
          />
        </div>

        <div>
          <label>Admission ID</label>
          <input
            value={admissionId}
            onChange={(e) =>
              setAdmissionId(e.target.value)
            }
          />
        </div>

        <div className="patient-status">
          <span className="status-dot" />
          Active Admission
        </div>
      </div>

      <div className="workflow-grid">
        <section className="workflow-card">
          <div className="step-header">
            <div className="step-number">
              1
            </div>

            <div>
              <h2>Upload laboratory report</h2>
              <p>
                OCR extracts available lab values.
              </p>
            </div>
          </div>

          <label className="upload-zone">
            <FileText size={34} />

            <strong>
              {reportFile
                ? reportFile.name
                : "Choose laboratory report"}
            </strong>

            <span>
              PNG, JPG or other supported image
            </span>

            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(e) =>
                setReportFile(
                  e.target.files?.[0] || null
                )
              }
            />
          </label>

          <button
            className="primary-button"
            onClick={handleReportUpload}
            disabled={!reportFile}
          >
            <Upload size={18} />
            Analyze Report
          </button>
        </section>

        <section className="workflow-card">
          <div className="step-header">
            <div className="step-number">
              2
            </div>

            <div>
              <h2>Required bedside data</h2>
              <p>
                Requested based on missing model
                inputs.
              </p>
            </div>
          </div>

          {nurseRequest.length === 0 ? (
            <div className="empty-state">
              Upload a report to determine
              additional parameters.
            </div>
          ) : (
            <div className="request-grid">
              {nurseRequest.map((item) => (
                <div
                  className="request-chip"
                  key={item}
                >
                  <CheckCircle2 size={16} />
                  {item}
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="workflow-card">
          <div className="step-header">
            <div className="step-number">
              3
            </div>

            <div>
              <h2>Nurse audio input</h2>
              <p>
                Speak or upload a recorded bedside
                observation.
              </p>
            </div>
          </div>

          <label className="audio-zone">
            <div className="mic-circle">
              <Mic size={28} />
            </div>

            <strong>
              {audioFile
                ? audioFile.name
                : "Upload nurse recording"}
            </strong>

            <span>
              Speech is converted into structured
              parameters.
            </span>

            <input
              type="file"
              accept="audio/*"
              hidden
              onChange={(e) =>
                setAudioFile(
                  e.target.files?.[0] || null
                )
              }
            />
          </label>

          <button
            className="primary-button"
            onClick={handleAudioUpload}
            disabled={!audioFile || !inputId}
          >
            <Mic size={18} />
            Extract Parameters
          </button>
        </section>

        <section className="workflow-card">
          <div className="step-header">
            <div className="step-number">
              4
            </div>

            <div>
              <h2>Review extracted values</h2>
              <p>
                Confirm or edit before storing.
              </p>
            </div>
          </div>

          {Object.keys(extracted).length === 0 ? (
            <div className="empty-state">
              Audio-extracted values will appear
              here.
            </div>
          ) : (
            <>
              <div className="editable-grid">
                {Object.entries(extracted).map(
                  ([key, value]) => (
                    <div
                      className="editable-field"
                      key={key}
                    >
                      <label>{key}</label>

                      <input
                        type="number"
                        step="any"
                        value={value}
                        onChange={(e) =>
                          updateParameter(
                            key,
                            e.target.value
                          )
                        }
                      />
                    </div>
                  )
                )}
              </div>

              {transcript && (
                <div className="transcript-box">
                  <span>AI transcript</span>
                  <p>{transcript}</p>
                </div>
              )}

              <div className="confirmation-note">
                Review all values before confirming
                the observation.
              </div>

              <button
                className="confirm-button"
                onClick={handleConfirm}
              >
                <CheckCircle2 size={18} />
                Confirm Observation
              </button>
            </>
          )}
        </section>
      </div>

      {status === "completed" &&
        finalResult && (
          <section className="result-panel">
            <div className="result-heading">
              <div>
                <span className="eyebrow">
                  Assessment Complete
                </span>

                <h2>
                  Observation successfully stored
                </h2>
              </div>

              <CheckCircle2
                size={28}
                className="success-icon"
              />
            </div>

            <div className="parameter-display-grid">
              {Object.entries(
                finalResult.observation
                  ?.clinical_parameters || {}
              ).map(([key, value]) => (
                <ParameterCard
                  key={key}
                  label={key}
                  value={String(value)}
                />
              ))}
            </div>

            <div className="assessment-note">
              Sepsis and AKI assessment agents have
              processed the updated longitudinal
              history. Trend and prioritization
              analysis has also been triggered.
            </div>
          </section>
        )}
    </div>
  );
}