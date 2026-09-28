import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BedDouble,
  HeartPulse,
  RefreshCw,
  Search,
  TrendingDown,
  TrendingUp,
  Minus,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  useNavigate,
  useSearchParams,
} from "react-router-dom";

import {
  getDashboardPatients,
  getPatientHistory,
} from "../api/clinicalApi";

import type {
  PatientSummary,
} from "../api/clinicalApi";

import MetricCard from "../components/MetricCard";
import ParameterCard from "../components/ParameterCard";
import RiskBadge from "../components/RiskBadge";


type ClinicalParameter =
  | "HR"
  | "O2Sat"
  | "Temp"
  | "SBP"
  | "MAP"
  | "Resp"
  | "WBC"
  | "Lactate";


type TrendOption = {
  key: ClinicalParameter;
  label: string;
  title: string;
  unit: string;
};


const TREND_OPTIONS: TrendOption[] = [
  {
    key: "HR",
    label: "HR",
    title: "Heart Rate",
    unit: "bpm",
  },
  {
    key: "O2Sat",
    label: "SpO₂",
    title: "Oxygen Saturation",
    unit: "%",
  },
  {
    key: "Temp",
    label: "Temp",
    title: "Temperature",
    unit: "°C",
  },
  {
    key: "SBP",
    label: "SBP",
    title: "Systolic Blood Pressure",
    unit: "mmHg",
  },
  {
    key: "MAP",
    label: "MAP",
    title: "Mean Arterial Pressure",
    unit: "mmHg",
  },
  {
    key: "Resp",
    label: "Resp",
    title: "Respiratory Rate",
    unit: "/min",
  },
  {
    key: "WBC",
    label: "WBC",
    title: "White Blood Cell Count",
    unit: "",
  },
  {
    key: "Lactate",
    label: "Lactate",
    title: "Lactate",
    unit: "",
  },
];


export default function DoctorDashboard() {
  const navigate = useNavigate();

  const [searchParams] =
    useSearchParams();

  const queryPatientId =
    searchParams.get(
      "patient_id"
    );

  const queryAdmissionId =
    searchParams.get(
      "admission_id"
    );

  const [patients, setPatients] =
    useState<PatientSummary[]>([]);

  const [patientSearch, setPatientSearch] =
    useState("");

  const [patientListLoading, setPatientListLoading] =
    useState(true);

  const [patientListError, setPatientListError] =
    useState("");


  const [
    patientId,
    setPatientId,
  ] = useState(
    queryPatientId ||
    "P1446"
  );


  const [
    admissionId,
    setAdmissionId,
  ] = useState(
    queryAdmissionId ||
    "ADM-1446"
  );


  const [
    history,
    setHistory,
  ] = useState<any>(
    null
  );


  const [
    loading,
    setLoading,
  ] = useState(
    false
  );


  const [
    selectedParameter,
    setSelectedParameter,
  ] =
    useState<ClinicalParameter>(
      "HR"
    );

  async function loadHistory(
    patient: string = patientId,
    admission: string = admissionId
  ) {
    if (
      !patient.trim() ||
      !admission.trim()
    ) {
      return;
    }

    setLoading(
      true
    );

    try {
      const result =
        await getPatientHistory(
          patient,
          admission
        );

      console.log(
        "Patient history response:",
        result
      );

      setHistory(
        result
      );
    } catch (error) {
      console.error(
        "Could not load patient history:",
        error
      );
    } finally {
      setLoading(
        false
      );
    }
  }


  /*
   * If the user clicked a patient from
   * Clinical Dashboard, automatically
   * load that patient.
   */
  useEffect(
    () => {
      async function loadPatients() {
        try {
          setPatientListLoading(true);
          setPatientListError("");
          const result = await getDashboardPatients();
          setPatients(result.patients || []);
        } catch (error) {
          setPatientListError(
            error instanceof Error
              ? error.message
              : "Could not load patients."
          );
        } finally {
          setPatientListLoading(false);
        }
      }

      void loadPatients();
    },
    []
  );


  useEffect(
    () => {
      if (
        queryPatientId &&
        queryAdmissionId
      ) {
        setPatientId(
          queryPatientId
        );

        setAdmissionId(
          queryAdmissionId
        );

        loadHistory(
          queryPatientId,
          queryAdmissionId
        );
      }
    },
    [
      queryPatientId,
      queryAdmissionId,
    ]
  );


  const filteredPatients =
    useMemo(
      () => {
        const search = patientSearch.trim().toLowerCase();
        const priorityOrder: Record<string, number> = {
          HIGH: 0,
          MEDIUM: 1,
          LOW: 2,
          PENDING: 3,
        };

        return [...patients]
          .filter((patient) =>
            patient.patient_id
              .toLowerCase()
              .includes(search)
          )
          .sort((a, b) => {
            const priorityDifference =
              (priorityOrder[String(a.priority?.level || "PENDING").toUpperCase()] ?? 3) -
              (priorityOrder[String(b.priority?.level || "PENDING").toUpperCase()] ?? 3);

            if (priorityDifference !== 0) {
              return priorityDifference;
            }

            return (b.sepsis?.probability || 0) - (a.sepsis?.probability || 0);
          });
      },
      [patients, patientSearch]
    );


  /*
   * -----------------------------
   * OBSERVATIONS
   * -----------------------------
   */

  const observations =
    history?.observations ||
    [];


  const latestObservation =
    observations.length > 0
      ? observations[
          observations.length -
          1
        ]
      : null;


  const latestParams =
    latestObservation
      ?.clinical_parameters ||
    {};


  /*
   * -----------------------------
   * ASSESSMENTS
   * -----------------------------
   */

  const assessments =
    history?.assessments ||
    [];


  const latestAssessment =
    assessments.length > 0
      ? assessments[
          assessments.length -
          1
        ]
      : null;


  const sepsisAssessment =
    latestAssessment?.sepsis ||
    latestAssessment
      ?.sepsis_result ||
    null;


  const akiAssessment =
    latestAssessment?.aki ||
    latestAssessment
      ?.aki_result ||
    null;


  /*
   * -----------------------------
   * SEPSIS DISPLAY
   * -----------------------------
   */

  const sepsisProbability =
    typeof sepsisAssessment
      ?.probability ===
    "number"
      ? sepsisAssessment
          .probability
      : null;


  const sepsisProbabilityDisplay =
    sepsisProbability !== null
      ? `${(
          sepsisProbability *
          100
        ).toFixed(1)}%`
      : sepsisAssessment
          ?.status ===
        "insufficient_data"
      ? "Insufficient Data"
      : "No Data";


  const sepsisRiskLevel =
    sepsisAssessment
      ?.risk_level ||
    sepsisAssessment
      ?.status ||
    "No assessment";


  /*
   * -----------------------------
   * AKI DISPLAY
   * -----------------------------
   */

  const akiProbability =
    typeof akiAssessment
      ?.probability ===
    "number"
      ? akiAssessment
          .probability
      : null;


  const akiProbabilityDisplay =
    akiProbability !== null
      ? `${(
          akiProbability *
          100
        ).toFixed(1)}%`
      : akiAssessment
          ?.status ===
        "insufficient_data"
      ? "Insufficient Data"
      : "No Data";


  /*
   * -----------------------------
   * SEPSIS TREND
   * -----------------------------
   */

  const trends =
    history?.trends ||
    [];


  const sepsisTrends =
    trends.filter(
      (trend: any) =>
        String(
          trend?.disease ||
          ""
        ).toLowerCase() ===
        "sepsis"
    );


  const latestSepsisTrend =
    sepsisTrends.length > 0
      ? sepsisTrends[
          sepsisTrends.length -
          1
        ]
      : null;


  const sepsisTrendLabel =
    latestSepsisTrend
      ?.trend ||
    "INSUFFICIENT_DATA";


  const firstProbability =
    typeof latestSepsisTrend
      ?.first_probability ===
    "number"
      ? latestSepsisTrend
          .first_probability
      : null;


  const latestTrendProbability =
    typeof latestSepsisTrend
      ?.latest_probability ===
    "number"
      ? latestSepsisTrend
          .latest_probability
      : null;


  const probabilityChange =
    typeof latestSepsisTrend
      ?.probability_change ===
    "number"
      ? latestSepsisTrend
          .probability_change
      : null;


  /*
   * -----------------------------
   * PRIORITIZATION
   * -----------------------------
   */

  const priority =
    history
      ?.latest_prioritization;


  /*
   * -----------------------------
   * CLINICAL CHART DATA
   * -----------------------------
   */

  const chartData =
    useMemo(
      () =>
        observations.map(
          (
            observation: any,
            index: number
          ) => ({
            index:
              index + 1,

            HR:
              observation
                .clinical_parameters
                ?.HR ??
              null,

            O2Sat:
              observation
                .clinical_parameters
                ?.O2Sat ??
              null,

            Temp:
              observation
                .clinical_parameters
                ?.Temp ??
              null,

            SBP:
              observation
                .clinical_parameters
                ?.SBP ??
              null,

            MAP:
              observation
                .clinical_parameters
                ?.MAP ??
              null,

            Resp:
              observation
                .clinical_parameters
                ?.Resp ??
              null,

            WBC:
              observation
                .clinical_parameters
                ?.WBC ??
              null,

            Lactate:
              observation
                .clinical_parameters
                ?.Lactate ??
              null,
          })
        ),
      [
        observations,
      ]
    );


  /*
   * Only observations containing the
   * selected measurement are used for
   * the selected chart.
   */
  const selectedChartData =
    useMemo(
      () =>
        chartData.filter(
          (point: any) =>
            point[
              selectedParameter
            ] !== null &&
            point[
              selectedParameter
            ] !== undefined
        ),
      [
        chartData,
        selectedParameter,
      ]
    );


  const selectedOption =
    TREND_OPTIONS.find(
      (option) =>
        option.key ===
        selectedParameter
    ) ||
    TREND_OPTIONS[0];


  /*
   * -----------------------------
   * PARAMETER TREND SUMMARY
   * -----------------------------
   */

  const parameterSummary =
    useMemo(
      () => {
        if (
          selectedChartData
            .length === 0
        ) {
          return {
            first: null,
            latest: null,
            change: null,
          };
        }

        const first =
          Number(
            selectedChartData[0][
              selectedParameter
            ]
          );

        const latest =
          Number(
            selectedChartData[
              selectedChartData
                .length - 1
            ][
              selectedParameter
            ]
          );

        return {
          first,
          latest,
          change:
            latest -
            first,
        };
      },
      [
        selectedChartData,
        selectedParameter,
      ]
    );


  function renderTrendIcon() {
    if (
      sepsisTrendLabel ===
      "WORSENING"
    ) {
      return (
        <TrendingUp
          size={30}
        />
      );
    }

    if (
      sepsisTrendLabel ===
      "IMPROVING"
    ) {
      return (
        <TrendingDown
          size={30}
        />
      );
    }

    return (
      <Minus
        size={30}
      />
    );
  }


  if (!queryPatientId || !queryAdmissionId) {
    return (
      <div className="doctor-patient-list-page">
        <style>
          {`
            .doctor-patient-list-page {
              min-height: 100%;
              padding: 34px;
              background: #f6f8fb;
              color: #132238;
            }

            .doctor-patient-list-header {
              margin-bottom: 22px;
            }

            .doctor-patient-list-header h1 {
              margin: 0;
              color: #102039;
              font-size: 30px;
            }

            .doctor-patient-search {
              position: relative;
              margin-bottom: 18px;
            }

            .doctor-patient-search svg {
              position: absolute;
              left: 14px;
              top: 50%;
              transform: translateY(-50%);
              color: #8792a3;
            }

            .doctor-patient-search input {
              width: 100%;
              box-sizing: border-box;
              border: 1px solid #dfe7f1;
              border-radius: 10px;
              padding: 13px 14px 13px 42px;
              background: white;
              color: #132238;
              font: inherit;
            }

            .doctor-patient-list {
              margin: 0;
              padding-left: 34px;
              display: grid;
              gap: 10px;
            }

            .doctor-patient-list li {
              padding-left: 8px;
            }

            .doctor-patient-row {
              width: 100%;
              box-sizing: border-box;
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 18px;
              border: 1px solid #e1e7ef;
              border-radius: 12px;
              padding: 16px 18px;
              background: white;
              color: inherit;
              text-align: left;
              cursor: pointer;
              font: inherit;
            }

            .doctor-patient-row:hover {
              border-color: #cad6e6;
              box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            }

            .doctor-patient-id {
              font-weight: 800;
              color: #132238;
            }

            .doctor-patient-admission {
              display: flex;
              align-items: center;
              gap: 5px;
              margin-top: 5px;
              color: #8994a4;
              font-size: 12px;
            }

            .doctor-patient-row-right {
              display: flex;
              align-items: center;
              gap: 14px;
            }

            .doctor-patient-priority {
              border-radius: 999px;
              padding: 6px 10px;
              font-size: 10px;
              font-weight: 800;
              letter-spacing: 0.6px;
            }

            .doctor-patient-priority-high { background: #fee2e2; color: #b42318; }
            .doctor-patient-priority-medium { background: #fff7d6; color: #9a6700; }
            .doctor-patient-priority-low { background: #dcfce7; color: #15803d; }
            .doctor-patient-priority-pending { background: #edf1f6; color: #667085; }

            .doctor-patient-risk {
              color: #68768a;
              font-size: 12px;
              white-space: nowrap;
            }

            .doctor-patient-message {
              padding: 40px;
              text-align: center;
              background: white;
              border: 1px solid #e2e8f0;
              border-radius: 16px;
              color: #697586;
            }

            @media (max-width: 700px) {
              .doctor-patient-list-page { padding: 24px 18px; }
              .doctor-patient-row { align-items: flex-start; }
              .doctor-patient-row-right { flex-direction: column; align-items: flex-end; gap: 8px; }
            }
          `}
        </style>

        <div className="doctor-patient-list-header">
          <span className="eyebrow">Doctor Workspace</span>
          <h1>Patients</h1>
        </div>

        <div className="doctor-patient-search">
          <Search size={18} aria-hidden="true" />
          <input
            aria-label="Search patients by Patient ID"
            placeholder="Search by Patient ID"
            value={patientSearch}
            onChange={(event) => setPatientSearch(event.target.value)}
          />
        </div>

        {patientListLoading ? (
          <div className="doctor-patient-message">Loading patients...</div>
        ) : patientListError ? (
          <div className="doctor-patient-message">{patientListError}</div>
        ) : filteredPatients.length === 0 ? (
          <div className="doctor-patient-message">Patient not found</div>
        ) : (
          <ol className="doctor-patient-list">
            {filteredPatients.map((patient) => {
              const priority = String(
                patient.priority?.level || "PENDING"
              ).toUpperCase();

              return (
                <li key={`${patient.patient_id}-${patient.admission_id}`}>
                  <button
                    type="button"
                    className="doctor-patient-row"
                    onClick={() =>
                      navigate(
                        `/doctor?patient_id=${encodeURIComponent(
                          patient.patient_id
                        )}&admission_id=${encodeURIComponent(
                          patient.admission_id
                        )}`
                      )
                    }
                  >
                    <div>
                      <div className="doctor-patient-id">
                        {patient.patient_id}
                      </div>
                      <div className="doctor-patient-admission">
                        <BedDouble size={13} />
                        {patient.admission_id}
                      </div>
                    </div>
                    <div className="doctor-patient-row-right">
                      <span className={`doctor-patient-priority doctor-patient-priority-${priority.toLowerCase()}`}>
                        {priority}
                      </span>
                      <span className="doctor-patient-risk">
                        Sepsis {patient.sepsis?.probability == null
                          ? "—"
                          : `${(patient.sepsis.probability * 100).toFixed(1)}%`}
                      </span>
                      <ArrowRight size={16} aria-hidden="true" />
                    </div>
                  </button>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    );
  }


  return (
    <div className="page">
      {/* PAGE HEADER */}

      <div className="page-header">
        <div>
          <span className="eyebrow">
            Doctor Workspace
          </span>

          <h1>
            Clinical Deterioration
            Dashboard
          </h1>

          <p>
            Longitudinal disease
            assessment, deterioration
            trends and patient
            prioritization.
          </p>
        </div>

        <button
          className="refresh-button"
          onClick={() =>
            loadHistory()
          }
          disabled={loading}
        >
          <RefreshCw
            size={17}
            className={
              loading
                ? "spin"
                : ""
            }
          />

          {loading
            ? "Loading..."
            : "Refresh"}
        </button>
      </div>


      {/* PATIENT SELECTOR */}

      <div className="patient-strip">
        <div>
          <label>
            Patient ID
          </label>

          <input
            value={patientId}
            onChange={(e) =>
              setPatientId(
                e.target.value
              )
            }
          />
        </div>

        <div>
          <label>
            Admission ID
          </label>

          <input
            value={admissionId}
            onChange={(e) =>
              setAdmissionId(
                e.target.value
              )
            }
          />
        </div>

        <button
          className="primary-button compact"
          onClick={() =>
            loadHistory()
          }
          disabled={loading}
        >
          {loading
            ? "Loading..."
            : "Load Patient"}
        </button>
      </div>


      {/* TOP METRICS */}

      <div className="metrics-grid">
        <MetricCard
          title="Observations"
          value={
            observations.length
          }
          subtitle="Longitudinal records"
        />

        <MetricCard
          title="Sepsis Assessment"
          value={
            sepsisProbabilityDisplay
          }
          subtitle={
            sepsisProbability !==
            null
              ? `Risk: ${sepsisRiskLevel}`
              : "Latest Agent 1 result"
          }
        />

        <MetricCard
          title="AKI Assessment"
          value={
            akiProbabilityDisplay
          }
          subtitle={
            akiAssessment
              ?.status ||
            "Latest Agent 1 result"
          }
        />

        <MetricCard
          title="Patient Priority"
          value={
            priority
              ?.priority_level ||
            "Pending"
          }
          subtitle="Agent 2 prioritization"
        />
      </div>


      {/* PRIORITY + OBSERVATION PROGRESS */}

      <div className="doctor-grid">
        <section className="large-card">
          <div className="card-title-row">
            <div>
              <span className="eyebrow">
                Patient Risk
              </span>

              <h2>
                Clinical Priority
              </h2>
            </div>

            <RiskBadge
              level={
                priority
                  ?.priority_level ||
                "UNKNOWN"
              }
            />
          </div>

          {!priority ? (
            <div className="empty-state tall">
              No prioritization
              available yet.
            </div>
          ) : (
            <div className="priority-content">
              <div className="priority-icon">
                <AlertTriangle
                  size={30}
                />
              </div>

              <div>
                <span>
                  Highest risk disease
                </span>

                <h3>
                  {
                    priority
                      .highest_risk_disease
                  }
                </h3>

                <p>
                  {
                    priority.reason
                  }
                </p>
              </div>
            </div>
          )}
        </section>


        <section className="large-card">
          <div className="card-title-row">
            <div>
              <span className="eyebrow">
                Monitoring
              </span>

              <h2>
                Observation Progress
              </h2>
            </div>

            <Activity
              size={22}
            />
          </div>

          <div className="progress-content">
            <div className="progress-number">
              <strong>
                {
                  observations.length
                }
              </strong>

              <span>
                / 12
              </span>
            </div>

            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width:
                    `${Math.min(
                      (
                        observations.length /
                        12
                      ) *
                        100,
                      100
                    )}%`,
                }}
              />
            </div>

            <p>
              Advanced Sepsis LSTM
              requires 12
              longitudinal
              observations.
            </p>
          </div>
        </section>
      </div>


      {/* SEPSIS PROBABILITY TREND */}

      <section className="large-card">
        <div className="card-title-row">
          <div>
            <span className="eyebrow">
              Disease Risk Trend
            </span>

            <h2>
              Sepsis Probability
              Trend
            </h2>
          </div>

          {renderTrendIcon()}
        </div>

        {!latestSepsisTrend ? (
          <div className="empty-state">
            At least two stored
            Sepsis assessments are
            required to calculate a
            probability trend.
          </div>
        ) : (
          <div className="priority-content">
            <div>
              <span>
                Current trend
              </span>

              <h3>
                {
                  sepsisTrendLabel
                }
              </h3>

              {firstProbability !==
                null &&
              latestTrendProbability !==
                null ? (
                <p>
                  First stored
                  probability:{" "}
                  <strong>
                    {(
                      firstProbability *
                      100
                    ).toFixed(1)}
                    %
                  </strong>

                  {" → "}

                  Current
                  probability:{" "}
                  <strong>
                    {(
                      latestTrendProbability *
                      100
                    ).toFixed(1)}
                    %
                  </strong>
                </p>
              ) : (
                <p>
                  More disease
                  assessments are
                  needed before
                  probability change
                  can be calculated.
                </p>
              )}

              {probabilityChange !==
                null && (
                <p>
                  Probability
                  change:{" "}
                  <strong>
                    {(
                      probabilityChange *
                      100
                    ).toFixed(1)}
                    {" "}
                    percentage points
                  </strong>
                </p>
              )}
            </div>
          </div>
        )}
      </section>


      {/* MULTI-PARAMETER LONGITUDINAL TRENDS */}

      <section className="chart-card">
        <div className="card-title-row">
          <div>
            <span className="eyebrow">
              Longitudinal Monitoring
            </span>

            <h2>
              Clinical Parameter
              Trends
            </h2>

            <p>
              Explore changes in
              bedside vitals and
              laboratory measurements
              across stored
              observations.
            </p>
          </div>

          <HeartPulse
            size={23}
          />
        </div>


        {/* PARAMETER SELECTOR */}

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            marginTop: "20px",
            marginBottom: "22px",
          }}
        >
          {TREND_OPTIONS.map(
            (option) => {
              const active =
                selectedParameter ===
                option.key;

              return (
                <button
                  key={
                    option.key
                  }
                  type="button"
                  onClick={() =>
                    setSelectedParameter(
                      option.key
                    )
                  }
                  style={{
                    border:
                      active
                        ? "1px solid #5b7cfa"
                        : "1px solid #dfe5ee",

                    background:
                      active
                        ? "#eef2ff"
                        : "#ffffff",

                    color:
                      active
                        ? "#405bd8"
                        : "#667085",

                    padding:
                      "9px 14px",

                    borderRadius:
                      "9px",

                    fontWeight:
                      700,

                    fontSize:
                      "12px",

                    cursor:
                      "pointer",
                  }}
                >
                  {
                    option.label
                  }
                </button>
              );
            }
          )}
        </div>


        {/* SELECTED PARAMETER SUMMARY */}

        <div
          style={{
            display: "flex",
            gap: "28px",
            flexWrap: "wrap",
            marginBottom: "18px",
            padding: "14px 16px",
            background: "#f8fafc",
            borderRadius: "10px",
          }}
        >
          <div>
            <div
              style={{
                fontSize: "11px",
                color: "#7a8797",
                marginBottom: "3px",
              }}
            >
              PARAMETER
            </div>

            <strong>
              {
                selectedOption.title
              }
            </strong>
          </div>

          <div>
            <div
              style={{
                fontSize: "11px",
                color: "#7a8797",
                marginBottom: "3px",
              }}
            >
              FIRST AVAILABLE
            </div>

            <strong>
              {parameterSummary
                .first !== null
                ? `${parameterSummary.first.toFixed(
                    1
                  )} ${
                    selectedOption.unit
                  }`
                : "—"}
            </strong>
          </div>

          <div>
            <div
              style={{
                fontSize: "11px",
                color: "#7a8797",
                marginBottom: "3px",
              }}
            >
              LATEST
            </div>

            <strong>
              {parameterSummary
                .latest !== null
                ? `${parameterSummary.latest.toFixed(
                    1
                  )} ${
                    selectedOption.unit
                  }`
                : "—"}
            </strong>
          </div>

          <div>
            <div
              style={{
                fontSize: "11px",
                color: "#7a8797",
                marginBottom: "3px",
              }}
            >
              CHANGE
            </div>

            <strong>
              {parameterSummary
                .change !== null
                ? `${
                    parameterSummary
                      .change > 0
                      ? "+"
                      : ""
                  }${parameterSummary.change.toFixed(
                    1
                  )} ${
                    selectedOption.unit
                  }`
                : "—"}
            </strong>
          </div>

          <div>
            <div
              style={{
                fontSize: "11px",
                color: "#7a8797",
                marginBottom: "3px",
              }}
            >
              DATA POINTS
            </div>

            <strong>
              {
                selectedChartData.length
              }
            </strong>
          </div>
        </div>


        {/* DYNAMIC GRAPH */}

        {chartData.length ===
        0 ? (
          <div className="empty-state tall">
            Load a patient to
            visualize longitudinal
            observations.
          </div>
        ) : selectedChartData
            .length === 0 ? (
          <div className="empty-state tall">
            No{" "}
            {
              selectedOption.title
            }{" "}
            measurements are
            available for this
            patient's stored
            observations.
          </div>
        ) : (
          <div className="chart-container">
            <ResponsiveContainer
              width="100%"
              height={320}
            >
              <AreaChart
                data={
                  selectedChartData
                }
              >
                <defs>
                  <linearGradient
                    id="clinicalTrendGradient"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor="#5b7cfa"
                      stopOpacity={
                        0.25
                      }
                    />

                    <stop
                      offset="95%"
                      stopColor="#5b7cfa"
                      stopOpacity={
                        0
                      }
                    />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  strokeDasharray="3 3"
                  vertical={
                    false
                  }
                />

                <XAxis
                  dataKey="index"
                  tickLine={
                    false
                  }
                  label={{
                    value:
                      "Observation",
                    position:
                      "insideBottom",
                    offset:
                      -2,
                  }}
                />

                <YAxis
                  tickLine={
                    false
                  }
                  domain={[
                    "auto",
                    "auto",
                  ]}
                />

                <Tooltip
                  formatter={(
                    value: any
                  ) => [
                    `${Number(
                      value
                    ).toFixed(
                      1
                    )} ${
                      selectedOption.unit
                    }`,
                    selectedOption.title,
                  ]}
                  labelFormatter={(
                    label
                  ) =>
                    `Observation ${label}`
                  }
                />

                <Area
                  type="monotone"
                  dataKey={
                    selectedParameter
                  }
                  stroke="#5b7cfa"
                  fill="url(#clinicalTrendGradient)"
                  strokeWidth={
                    3
                  }
                  connectNulls
                  dot={{
                    r: 3,
                  }}
                  activeDot={{
                    r: 5,
                  }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>


      {/* LATEST OBSERVATION */}

      <section className="latest-card">
        <div className="card-title-row">
          <div>
            <span className="eyebrow">
              Latest Observation
            </span>

            <h2>
              Clinical
              Measurements
            </h2>
          </div>
        </div>

        {Object.keys(
          latestParams
        ).length === 0 ? (
          <div className="empty-state">
            No observations
            available.
          </div>
        ) : (
          <div className="parameter-display-grid">
            {Object.entries(
              latestParams
            ).map(
              (
                [
                  key,
                  value,
                ]
              ) => (
                <ParameterCard
                  key={key}
                  label={key}
                  value={String(
                    value
                  )}
                />
              )
            )}
          </div>
        )}
      </section>
    </div>
  );
}