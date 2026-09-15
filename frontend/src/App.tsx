import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import Sidebar from "./components/Sidebar";
import NurseDashboard from "./pages/NurseDashboard";
import DoctorDashboard from "./pages/DoctorDashboard";
import ClinicalDashboard from "./pages/ClinicalDashboard";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />

      <main className="main-content">
        <Routes>
          <Route
            path="/"
            element={
              <Navigate
                to="/nurse"
                replace
              />
            }
          />

          <Route
            path="/nurse"
            element={
              <NurseDashboard />
            }
          />

          <Route
            path="/doctor"
            element={
              <DoctorDashboard />
            }
          />

          <Route
            path="/clinical"
            element={
              <ClinicalDashboard />
            }
          />
        </Routes>
      </main>
    </div>
  );
}