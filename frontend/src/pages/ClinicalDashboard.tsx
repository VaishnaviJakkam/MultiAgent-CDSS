import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BedDouble,
  BrainCircuit,
  CircleAlert,
  HeartPulse,
  RefreshCw,
  ShieldCheck,
  Stethoscope,
  TrendingDown,
  TrendingUp,
  Users,
} from "lucide-react";

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  useNavigate,
} from "react-router-dom";


const API_BASE =
  "http://127.0.0.1:8000";


type DiseaseSummary = {
  status?: string;
  probability?: number | null;
  risk_level?: string | null;
  trend?: string | null;
  first_probability?: number | null;
  latest_probability?: number | null;
  probability_change?: number | null;
};


type PrioritySummary = {
  level?: string;
  highest_risk_disease?: string | null;
  reason?: string | null;
  worsening_diseases?: string[];
};


type PatientSummary = {
  patient_id: string;
  admission_id: string;

  observation_count?: number;
  assessment_count?: number;

  sepsis?: DiseaseSummary;
  aki?: DiseaseSummary;

  priority?: PrioritySummary;

  latest_parameters?: Record<
    string,
    number | null
  >;
};


type DashboardResponse = {
  status: string;
  patient_count: number;
  patients: PatientSummary[];
};


// =========================================================
// HELPERS
// =========================================================

function formatProbability(
  probability?: number | null
) {
  if (
    probability === null ||
    probability === undefined
  ) {
    return "—";
  }

  return `${(
    probability * 100
  ).toFixed(1)}%`;
}


function formatValue(
  value?: number | null,
  decimals = 1
) {
  if (
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  return Number(
    value
  ).toFixed(
    decimals
  );
}


function normalizePriority(
  priority?: string
) {
  return String(
    priority || "PENDING"
  ).toUpperCase();
}


function normalizeTrend(
  trend?: string | null
) {
  return String(
    trend || "INSUFFICIENT_DATA"
  ).toUpperCase();
}


function readableTrend(
  trend?: string | null
) {
  const value =
    normalizeTrend(
      trend
    );

  if (
    value ===
    "INSUFFICIENT_DATA"
  ) {
    return "Awaiting Trend";
  }

  if (
    value === "WORSENING"
  ) {
    return "Worsening";
  }

  if (
    value === "IMPROVING"
  ) {
    return "Improving";
  }

  if (
    value === "STABLE"
  ) {
    return "Stable";
  }

  return value;
}


function priorityRank(
  priority?: string
) {
  const value =
    normalizePriority(
      priority
    );

  if (value === "HIGH") {
    return 0;
  }

  if (value === "MEDIUM") {
    return 1;
  }

  if (value === "LOW") {
    return 2;
  }

  return 3;
}


function TrendIcon({
  trend,
}: {
  trend?: string | null;
}) {
  const value =
    normalizeTrend(
      trend
    );

  if (
    value === "WORSENING"
  ) {
    return (
      <TrendingUp
        size={15}
      />
    );
  }

  if (
    value === "IMPROVING"
  ) {
    return (
      <TrendingDown
        size={15}
      />
    );
  }

  return (
    <Activity
      size={15}
    />
  );
}


// =========================================================
// VITAL CHIP
// =========================================================

function VitalChip({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit?: string;
}) {
  return (
    <div className="aegis-vital">
      <span className="aegis-vital-label">
        {label}
      </span>

      <div>
        <strong>
          {value}
        </strong>

        {unit && (
          <small>
            {unit}
          </small>
        )}
      </div>
    </div>
  );
}


// =========================================================
// PATIENT CARD
// =========================================================

function PatientCard({
  patient,
  onOpen,
}: {
  patient: PatientSummary;
  onOpen: () => void;
}) {
  const priority =
    normalizePriority(
      patient.priority?.level
    );

  const trend =
    normalizeTrend(
      patient.sepsis?.trend
    );

  const latest =
    patient.latest_parameters ||
    {};

  const probability =
    patient.sepsis?.probability;

  const hasProbability =
    probability !== null &&
    probability !== undefined;

  const probabilityPercent =
    hasProbability
      ? Math.min(
          100,
          Math.max(
            0,
            probability * 100
          )
        )
      : 0;

  return (
    <button
      type="button"
      className="aegis-patient-card"
      onClick={onOpen}
    >
      <div className="aegis-card-top">
        <div>
          <div className="aegis-patient-id">
            {patient.patient_id}
          </div>

          <div className="aegis-admission">
            <BedDouble
              size={13}
            />

            {
              patient.admission_id
            }
          </div>
        </div>

        <span
          className={`aegis-priority aegis-priority-${priority.toLowerCase()}`}
        >
          {priority}
        </span>
      </div>

      <div className="aegis-card-risk">
        <div className="aegis-risk-heading">
          <span>
            Sepsis risk
          </span>

          <strong>
            {formatProbability(
              probability
            )}
          </strong>
        </div>

        <div className="aegis-risk-track">
          <div
            className="aegis-risk-fill"
            style={{
              width:
                `${probabilityPercent}%`,
            }}
          />
        </div>

        <div className="aegis-risk-meta">
          <span
            className={`aegis-trend aegis-trend-${trend.toLowerCase()}`}
          >
            <TrendIcon
              trend={trend}
            />

            {readableTrend(
              trend
            )}
          </span>

          <span>
            {patient.sepsis
              ?.risk_level ||
              (
                hasProbability
                  ? "Assessed"
                  : "Not yet assessed"
              )}
          </span>
        </div>
      </div>

      <div className="aegis-mini-vitals">
        <div>
          <span>
            HR
          </span>

          <strong>
            {formatValue(
              latest.HR,
              0
            )}
          </strong>
        </div>

        <div>
          <span>
            SpO₂
          </span>

          <strong>
            {formatValue(
              latest.O2Sat,
              0
            )}
          </strong>
        </div>

        <div>
          <span>
            MAP
          </span>

          <strong>
            {formatValue(
              latest.MAP,
              0
            )}
          </strong>
        </div>

        <div>
          <span>
            RR
          </span>

          <strong>
            {formatValue(
              latest.Resp,
              0
            )}
          </strong>
        </div>
      </div>

      <div className="aegis-card-footer">
        <span>
          {
            patient.observation_count ||
            0
          }{" "}
          observations
        </span>

        <span className="aegis-view-link">
          View patient

          <ArrowRight
            size={15}
          />
        </span>
      </div>
    </button>
  );
}


// =========================================================
// DASHBOARD
// =========================================================

export default function ClinicalDashboard() {
  const navigate =
    useNavigate();

  const [
    patients,
    setPatients,
  ] =
    useState<
      PatientSummary[]
    >([]);

  const [
    loading,
    setLoading,
  ] =
    useState(
      true
    );

  const [
    error,
    setError,
  ] =
    useState(
      ""
    );


  async function loadDashboard() {
    try {
      setLoading(
        true
      );

      setError(
        ""
      );

      const response =
        await fetch(
          `${API_BASE}/api/dashboard/patients`
        );

      if (
        !response.ok
      ) {
        throw new Error(
          `Dashboard request failed (${response.status})`
        );
      }

      const data:
        DashboardResponse =
        await response.json();

      console.log(
        "Clinical dashboard data:",
        data
      );

      setPatients(
        data.patients ||
        []
      );
    } catch (err) {
      console.error(
        err
      );

      setError(
        err instanceof Error
          ? err.message
          : "Could not load dashboard."
      );
    } finally {
      setLoading(
        false
      );
    }
  }


  useEffect(
    () => {
      loadDashboard();
    },
    []
  );


  const sortedPatients =
    useMemo(
      () => {
        return [
          ...patients,
        ].sort(
          (
            a,
            b
          ) => {
            const priorityDifference =
              priorityRank(
                a.priority
                  ?.level
              ) -
              priorityRank(
                b.priority
                  ?.level
              );

            if (
              priorityDifference !==
              0
            ) {
              return priorityDifference;
            }

            return (
              (
                b.sepsis
                  ?.probability ||
                0
              ) -
              (
                a.sepsis
                  ?.probability ||
                0
              )
            );
          }
        );
      },
      [
        patients,
      ]
    );


  const statistics =
    useMemo(
      () => {
        const high =
          patients.filter(
            (patient) =>
              normalizePriority(
                patient.priority
                  ?.level
              ) ===
              "HIGH"
          ).length;

        const medium =
          patients.filter(
            (patient) =>
              normalizePriority(
                patient.priority
                  ?.level
              ) ===
              "MEDIUM"
          ).length;

        const worsening =
          patients.filter(
            (patient) =>
              normalizeTrend(
                patient.sepsis
                  ?.trend
              ) ===
              "WORSENING"
          ).length;

        const assessed =
          patients.filter(
            (patient) =>
              patient.sepsis
                ?.probability !==
                null &&
              patient.sepsis
                ?.probability !==
                undefined
          ).length;

        return {
          total:
            patients.length,

          high,

          medium,

          worsening,

          assessed,
        };
      },
      [
        patients,
      ]
    );


  const urgentPatients =
    sortedPatients.filter(
      (patient) =>
        normalizePriority(
          patient.priority
            ?.level
        ) ===
          "HIGH" ||
        normalizeTrend(
          patient.sepsis
            ?.trend
        ) ===
          "WORSENING"
    );


  function openPatient(
    patient:
      PatientSummary
  ) {
    navigate(
      `/doctor?patient_id=${encodeURIComponent(
        patient.patient_id
      )}&admission_id=${encodeURIComponent(
        patient.admission_id
      )}`
    );
  }


  return (
    <>
      <style>
        {`
          .aegis-dashboard {
            min-height: 100%;
            padding: 34px;
            background:
              radial-gradient(
                circle at top right,
                rgba(37, 99, 235, 0.07),
                transparent 30%
              ),
              #f6f8fb;
            color: #132238;
          }

          .aegis-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 24px;
            margin-bottom: 28px;
          }

          .aegis-eyebrow {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.7px;
            color: #2563eb;
            margin-bottom: 8px;
          }

          .aegis-header h1 {
            margin: 0;
            font-size: 30px;
            letter-spacing: -0.7px;
            color: #102039;
          }

          .aegis-header p {
            margin: 8px 0 0;
            color: #68768a;
            font-size: 14px;
          }

          .aegis-live-area {
            display: flex;
            align-items: center;
            gap: 12px;
          }

          .aegis-live {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border: 1px solid #dfe7f1;
            padding: 9px 12px;
            border-radius: 10px;
            background: white;
            font-size: 12px;
            font-weight: 700;
            color: #43536a;
          }

          .aegis-live-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow:
              0 0 0 4px rgba(34, 197, 94, 0.12);
          }

          .aegis-refresh {
            border: 1px solid #dfe7f1;
            background: white;
            color: #34445b;
            padding: 9px 12px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            cursor: pointer;
            font-weight: 700;
          }

          .aegis-refresh:hover {
            background: #f7f9fc;
          }

          .aegis-stat-grid {
            display: grid;
            grid-template-columns:
              repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 26px;
          }

          .aegis-stat {
            background: white;
            border: 1px solid #e3e9f1;
            border-radius: 15px;
            padding: 18px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow:
              0 4px 16px rgba(15, 23, 42, 0.025);
          }

          .aegis-stat-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: #eff6ff;
            color: #2563eb;
            display: flex;
            align-items: center;
            justify-content: center;
          }

          .aegis-stat-value {
            font-size: 23px;
            font-weight: 800;
            color: #132238;
            line-height: 1;
          }

          .aegis-stat-label {
            margin-top: 5px;
            font-size: 12px;
            color: #7a8799;
          }

          .aegis-section {
            margin-top: 28px;
          }

          .aegis-section-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 13px;
          }

          .aegis-section-title {
            display: flex;
            align-items: center;
            gap: 9px;
          }

          .aegis-section-title h2 {
            margin: 0;
            font-size: 17px;
            color: #18283f;
          }

          .aegis-section-caption {
            font-size: 12px;
            color: #8792a3;
          }

          .aegis-attention-card {
            background:
              linear-gradient(
                135deg,
                #ffffff,
                #fff7f5
              );
            border: 1px solid #f3d7d3;
            border-radius: 18px;
            padding: 22px;
            box-shadow:
              0 8px 24px rgba(96, 35, 25, 0.055);
          }

          .aegis-attention-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
          }

          .aegis-attention-id {
            display: flex;
            align-items: center;
            gap: 10px;
          }

          .aegis-attention-id strong {
            font-size: 20px;
          }

          .aegis-attention-admission {
            color: #818b9b;
            font-size: 12px;
            margin-top: 4px;
          }

          .aegis-priority {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.6px;
          }

          .aegis-priority-high {
            background: #fee2e2;
            color: #b42318;
          }

          .aegis-priority-medium {
            background: #fff7d6;
            color: #9a6700;
          }

          .aegis-priority-low {
            background: #dcfce7;
            color: #15803d;
          }

          .aegis-priority-pending {
            background: #edf1f6;
            color: #667085;
          }

          .aegis-attention-risk {
            margin-top: 22px;
            display: grid;
            grid-template-columns:
              180px 1fr;
            gap: 24px;
            align-items: center;
          }

          .aegis-big-risk span {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #7a8797;
            font-weight: 700;
          }

          .aegis-big-risk strong {
            display: block;
            font-size: 39px;
            letter-spacing: -1.5px;
            margin-top: 3px;
            color: #b42318;
          }

          .aegis-reason {
            font-size: 13px;
            line-height: 1.55;
            color: #59677a;
          }

          .aegis-vital-row {
            display: grid;
            grid-template-columns:
              repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin-top: 20px;
          }

          .aegis-vital {
            background: rgba(255,255,255,0.72);
            border: 1px solid #ebedf1;
            border-radius: 11px;
            padding: 11px 12px;
          }

          .aegis-vital-label {
            display: block;
            font-size: 10px;
            font-weight: 700;
            color: #8792a2;
            margin-bottom: 3px;
          }

          .aegis-vital strong {
            font-size: 15px;
          }

          .aegis-vital small {
            font-size: 9px;
            margin-left: 3px;
            color: #929cab;
          }

          .aegis-attention-footer {
            margin-top: 17px;
            display: flex;
            justify-content: flex-end;
          }

          .aegis-open-button {
            border: none;
            background: #162943;
            color: white;
            border-radius: 10px;
            padding: 10px 14px;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            cursor: pointer;
            font-weight: 700;
            font-size: 12px;
          }

          .aegis-patient-grid {
            display: grid;
            grid-template-columns:
              repeat(3, minmax(0, 1fr));
            gap: 14px;
          }

          .aegis-patient-card {
            width: 100%;
            text-align: left;
            border: 1px solid #e1e7ef;
            background: white;
            border-radius: 16px;
            padding: 17px;
            cursor: pointer;
            color: inherit;
            font-family: inherit;
            box-shadow:
              0 4px 15px rgba(15, 23, 42, 0.025);
            transition:
              transform 0.15s ease,
              box-shadow 0.15s ease,
              border-color 0.15s ease;
          }

          .aegis-patient-card:hover {
            transform: translateY(-2px);
            border-color: #cad6e6;
            box-shadow:
              0 10px 24px rgba(15, 23, 42, 0.06);
          }

          .aegis-card-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
          }

          .aegis-patient-id {
            font-size: 16px;
            font-weight: 800;
          }

          .aegis-admission {
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 11px;
            color: #8994a4;
          }

          .aegis-card-risk {
            margin-top: 18px;
          }

          .aegis-risk-heading {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #7c8899;
          }

          .aegis-risk-heading strong {
            color: #1c2d45;
            font-size: 16px;
          }

          .aegis-risk-track {
            height: 5px;
            background: #edf1f5;
            border-radius: 999px;
            margin-top: 8px;
            overflow: hidden;
          }

          .aegis-risk-fill {
            height: 100%;
            border-radius: inherit;
            background:
              linear-gradient(
                90deg,
                #60a5fa,
                #ef4444
              );
          }

          .aegis-risk-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 8px;
            font-size: 10px;
            color: #8b96a5;
          }

          .aegis-trend {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-weight: 700;
          }

          .aegis-trend-worsening {
            color: #c2413a;
          }

          .aegis-trend-improving {
            color: #15803d;
          }

          .aegis-trend-stable {
            color: #2563eb;
          }

          .aegis-trend-insufficient_data {
            color: #7e8997;
          }

          .aegis-mini-vitals {
            display: grid;
            grid-template-columns:
              repeat(4, minmax(0, 1fr));
            gap: 7px;
            margin-top: 15px;
          }

          .aegis-mini-vitals div {
            background: #f7f9fc;
            border-radius: 8px;
            padding: 8px;
          }

          .aegis-mini-vitals span {
            display: block;
            font-size: 9px;
            color: #8d98a7;
          }

          .aegis-mini-vitals strong {
            display: block;
            margin-top: 2px;
            font-size: 12px;
          }

          .aegis-card-footer {
            border-top: 1px solid #edf0f4;
            margin-top: 15px;
            padding-top: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #8490a0;
            font-size: 10px;
          }

          .aegis-view-link {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            color: #2563eb;
            font-weight: 700;
          }

          .aegis-message {
            padding: 40px;
            text-align: center;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            color: #697586;
          }

          .aegis-spin {
            animation: aegisRotate 0.8s linear infinite;
          }

          @keyframes aegisRotate {
            to {
              transform: rotate(360deg);
            }
          }

          @media (max-width: 1150px) {
            .aegis-stat-grid {
              grid-template-columns:
                repeat(2, 1fr);
            }

            .aegis-patient-grid {
              grid-template-columns:
                repeat(2, 1fr);
            }

            .aegis-vital-row {
              grid-template-columns:
                repeat(3, 1fr);
            }
          }

          @media (max-width: 760px) {
            .aegis-dashboard {
              padding: 20px;
            }

            .aegis-header {
              flex-direction: column;
            }

            .aegis-stat-grid,
            .aegis-patient-grid {
              grid-template-columns:
                1fr;
            }

            .aegis-attention-risk {
              grid-template-columns:
                1fr;
            }

            .aegis-vital-row {
              grid-template-columns:
                repeat(2, 1fr);
            }
          }
        `}
      </style>


      <div className="aegis-dashboard">
        {/* HEADER */}

        <div className="aegis-header">
          <div>
            <div className="aegis-eyebrow">
              AEGIS • CLINICAL COMMAND CENTER
            </div>

            <h1>
              Patient Deterioration Overview
            </h1>

            <p>
              Continuous multi-patient
              monitoring, disease-risk
              assessment and clinical
              prioritization.
            </p>
          </div>

          <div className="aegis-live-area">
            <div className="aegis-live">
              <span className="aegis-live-dot" />

              LIVE
            </div>

            <button
              className="aegis-refresh"
              onClick={loadDashboard}
              disabled={loading}
            >
              <RefreshCw
                size={15}
                className={
                  loading
                    ? "aegis-spin"
                    : ""
                }
              />

              Refresh
            </button>
          </div>
        </div>


        {/* SUMMARY */}

        <div className="aegis-stat-grid">
          <div className="aegis-stat">
            <div className="aegis-stat-icon">
              <Users size={20} />
            </div>

            <div>
              <div className="aegis-stat-value">
                {statistics.total}
              </div>

              <div className="aegis-stat-label">
                Active patients
              </div>
            </div>
          </div>


          <div className="aegis-stat">
            <div className="aegis-stat-icon">
              <CircleAlert size={20} />
            </div>

            <div>
              <div className="aegis-stat-value">
                {statistics.high}
              </div>

              <div className="aegis-stat-label">
                High priority
              </div>
            </div>
          </div>


          <div className="aegis-stat">
            <div className="aegis-stat-icon">
              <TrendingUp size={20} />
            </div>

            <div>
              <div className="aegis-stat-value">
                {statistics.worsening}
              </div>

              <div className="aegis-stat-label">
                Worsening disease trend
              </div>
            </div>
          </div>


          <div className="aegis-stat">
            <div className="aegis-stat-icon">
              <BrainCircuit size={20} />
            </div>

            <div>
              <div className="aegis-stat-value">
                {statistics.assessed}
                /
                {statistics.total}
              </div>

              <div className="aegis-stat-label">
                Disease assessments available
              </div>
            </div>
          </div>
        </div>


        {/* LOADING / ERROR */}

        {loading && (
          <div className="aegis-message">
            Loading clinical intelligence...
          </div>
        )}


        {!loading &&
          error && (
            <div className="aegis-message">
              <AlertTriangle
                size={25}
              />

              <h3>
                Dashboard unavailable
              </h3>

              <p>
                {error}
              </p>
            </div>
          )}


        {!loading &&
          !error &&
          patients.length === 0 && (
            <div className="aegis-message">
              No active patients were found.
            </div>
          )}


        {!loading &&
          !error &&
          patients.length > 0 && (
            <>
              {/* REQUIRES ATTENTION */}

              {urgentPatients.length >
                0 && (
                <section className="aegis-section">
                  <div className="aegis-section-heading">
                    <div className="aegis-section-title">
                      <AlertTriangle
                        size={18}
                      />

                      <h2>
                        Requires Attention
                      </h2>
                    </div>

                    <span className="aegis-section-caption">
                      Highest-priority patient
                    </span>
                  </div>


                  {(() => {
                    const patient =
                      urgentPatients[0];

                    const latest =
                      patient.latest_parameters ||
                      {};

                    return (
                      <div className="aegis-attention-card">
                        <div className="aegis-attention-top">
                          <div>
                            <div className="aegis-attention-id">
                              <HeartPulse
                                size={22}
                              />

                              <strong>
                                {
                                  patient.patient_id
                                }
                              </strong>
                            </div>

                            <div className="aegis-attention-admission">
                              {
                                patient.admission_id
                              }
                            </div>
                          </div>

                          <span
                            className={`aegis-priority aegis-priority-${normalizePriority(
                              patient.priority
                                ?.level
                            ).toLowerCase()}`}
                          >
                            {normalizePriority(
                              patient.priority
                                ?.level
                            )}
                          </span>
                        </div>


                        <div className="aegis-attention-risk">
                          <div className="aegis-big-risk">
                            <span>
                              Current Sepsis Risk
                            </span>

                            <strong>
                              {formatProbability(
                                patient.sepsis
                                  ?.probability
                              )}
                            </strong>
                          </div>

                          <div className="aegis-reason">
                            <strong>
                              Why this patient is prioritized
                            </strong>

                            <div
                              style={{
                                marginTop:
                                  "6px",
                              }}
                            >
                              {patient.priority
                                ?.reason ||
                                "Longitudinal clinical risk assessment indicates this patient requires attention."}
                            </div>
                          </div>
                        </div>


                        <div className="aegis-vital-row">
                          <VitalChip
                            label="HEART RATE"
                            value={formatValue(
                              latest.HR,
                              0
                            )}
                            unit="bpm"
                          />

                          <VitalChip
                            label="SpO₂"
                            value={formatValue(
                              latest.O2Sat,
                              0
                            )}
                            unit="%"
                          />

                          <VitalChip
                            label="MAP"
                            value={formatValue(
                              latest.MAP,
                              0
                            )}
                            unit="mmHg"
                          />

                          <VitalChip
                            label="RESP"
                            value={formatValue(
                              latest.Resp,
                              0
                            )}
                            unit="/min"
                          />

                          <VitalChip
                            label="WBC"
                            value={formatValue(
                              latest.WBC
                            )}
                          />

                          <VitalChip
                            label="LACTATE"
                            value={formatValue(
                              latest.Lactate
                            )}
                          />
                        </div>


                        <div className="aegis-attention-footer">
                          <button
                            className="aegis-open-button"
                            onClick={() =>
                              openPatient(
                                patient
                              )
                            }
                          >
                            <Stethoscope
                              size={15}
                            />

                            Open Patient Dashboard

                            <ArrowRight
                              size={15}
                            />
                          </button>
                        </div>
                      </div>
                    );
                  })()}
                </section>
              )}


              {/* ALL PATIENTS */}

              <section className="aegis-section">
                <div className="aegis-section-heading">
                  <div className="aegis-section-title">
                    <ShieldCheck
                      size={18}
                    />

                    <h2>
                      All Active Patients
                    </h2>
                  </div>

                  <span className="aegis-section-caption">
                    Ranked by current clinical priority
                  </span>
                </div>


                <div className="aegis-patient-grid">
                  {sortedPatients.map(
                    (
                      patient
                    ) => (
                      <PatientCard
                        key={`${patient.patient_id}-${patient.admission_id}`}
                        patient={
                          patient
                        }
                        onOpen={() =>
                          openPatient(
                            patient
                          )
                        }
                      />
                    )
                  )}
                </div>
              </section>
            </>
          )}
      </div>
    </>
  );
}