import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import NurseDashboard from "./NurseDashboard";
import DoctorDashboard from "./DoctorDashboard";
import ClinicalDashboard from "./ClinicalDashboard";

const api = vi.hoisted(() => ({
  uploadLabReport: vi.fn(),
  getNurseTasks: vi.fn(),
  getNurseTask: vi.fn(),
  submitNurseAudio: vi.fn(),
  getDoctorPatient: vi.fn(),
  getDashboardPatients: vi.fn(),
}));

vi.mock("../api/clinicalApi", () => api);

afterEach(() => {
  vi.clearAllMocks();
});

const task = {
  task_id: "task-1",
  task_type: "NURSE_INPUT",
  patient_id: "P1446",
  admission_id: "ADM-1446",
  status: "PENDING",
  required_fields: ["HR"],
  provided_fields: [],
  metadata: {
    report_id: "report-1",
    question_fields: ["HR"],
    question_index: 0,
    current_question: "Please provide HR.",
  },
};

describe("NurseDashboard", () => {
  it("uploads a report and shows the started workflow", async () => {
    api.getNurseTasks.mockResolvedValue([]);
    api.uploadLabReport.mockResolvedValue({
      report_id: "report-1",
      status: "PROCESSED",
      workflow_started: true,
      task_id: null,
    });

    const { container } = render(<NurseDashboard />);
    const reportInput = container.querySelectorAll('input[type="file"]')[0];
    fireEvent.change(reportInput, { target: { files: [new File(["report"], "labs.pdf")] } });
    fireEvent.click(screen.getByRole("button", { name: /upload report/i }));

    expect(await screen.findByText("Report uploaded")).toBeInTheDocument();
    expect(screen.getByText("Workflow started")).toBeInTheDocument();
    expect(screen.getByText("Report ID: report-1")).toBeInTheDocument();
  });

  it("loads tasks and completes the voice questionnaire", async () => {
    api.getNurseTasks.mockResolvedValue([task]);
    api.getNurseTask.mockResolvedValue(task);
    api.submitNurseAudio.mockResolvedValue({
      completed: true,
      observation_id: "observation-1",
      assessment_started: true,
    });

    const { container } = render(<NurseDashboard />);
    const taskButton = await screen.findByRole("button", { name: /task-1/i });
    expect(taskButton).toBeInTheDocument();
    fireEvent.click(taskButton);

    await screen.findByText("Please provide HR.");
    const audioInput = container.querySelectorAll('input[type="file"]')[1];
    fireEvent.change(audioInput, { target: { files: [new File(["audio"], "answer.wav")] } });
    fireEvent.click(screen.getByRole("button", { name: /submit answer/i }));

    expect(await screen.findByText("Assessment started")).toBeInTheDocument();
    expect(api.submitNurseAudio).toHaveBeenCalledWith("task-1", expect.any(File));
  });
});

describe("DoctorDashboard", () => {
  it("renders the event-driven patient history", async () => {
    api.getDoctorPatient.mockResolvedValue({
      patient: { patient_id: "P1446", demographics: { age: 62 } },
      admission: { admission_id: "ADM-1446" },
      observations: [],
      assessments: [{ sepsis: { probability: 0.72, status: "success", risk_level: "ELEVATED" }, aki: { probability: 0.2, status: "success" } }],
      trends: [],
      prioritizations: [{ priority_level: "HIGH", reason: "Sepsis risk elevated" }],
      reports: [],
      tasks: [],
    });

    render(
      <MemoryRouter initialEntries={["/doctor?patient_id=P1446&admission_id=ADM-1446"]}>
        <DoctorDashboard />
      </MemoryRouter>
    );

    await waitFor(() => expect(api.getDoctorPatient).toHaveBeenCalledWith("P1446", "ADM-1446"));
    expect((await screen.findAllByText("HIGH")).length).toBeGreaterThan(0);
  });
});

describe("ClinicalDashboard", () => {
  it("sorts patients by backend priority", async () => {
    api.getDashboardPatients.mockResolvedValue({
      status: "success",
      patient_count: 2,
      patients: [
        { patient_id: "P-LOW", admission_id: "A-LOW", priority: { level: "LOW" }, sepsis: { probability: 0.1 } },
        { patient_id: "P-HIGH", admission_id: "A-HIGH", priority: { level: "HIGH" }, sepsis: { probability: 0.4 } },
      ],
    });

    render(
      <MemoryRouter>
        <ClinicalDashboard />
      </MemoryRouter>
    );

    const patientCards = await screen.findAllByRole("button", { name: /P-(HIGH|LOW)/ });
    const high = patientCards.find((card) => card.textContent?.includes("P-HIGH"));
    const low = patientCards.find((card) => card.textContent?.includes("P-LOW"));
    expect(high).toBeDefined();
    expect(low).toBeDefined();
    expect(high!.compareDocumentPosition(low!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
