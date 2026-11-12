import {
  useState,
} from "react";

import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import Sidebar from "./components/Sidebar";
import {
  canAccessPortal,
  clearAuthentication,
  getAuthenticatedUser,
  getDefaultPortal,
  type PortalPath,
} from "./auth";
import NurseDashboard from "./pages/NurseDashboard";
import LabDashboard from "./pages/LabDashboard";
import DoctorDashboard from "./pages/DoctorDashboard";
import LoginPage from "./pages/LoginPage";

function ProtectedRoute({
  path,
  children,
}: {
  path: PortalPath;
  children: React.ReactNode;
}) {
  const user = getAuthenticatedUser();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return canAccessPortal(user.role, path)
    ? children
    : <Navigate to={getDefaultPortal(user.role)} replace />;
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => getAuthenticatedUser() !== null
  );
  const authenticatedUser = isAuthenticated ? getAuthenticatedUser() : null;

  function handleLogout() {
    clearAuthentication();
    setIsAuthenticated(false);
  }

  return (
    <div className="app-shell">
      {authenticatedUser && (
        <Sidebar onLogout={handleLogout} role={authenticatedUser.role} />
      )}

      <main className={isAuthenticated ? "main-content" : "auth-content"}>
        <Routes>
          <Route
            path="/"
            element={
              <Navigate
                to={authenticatedUser ? getDefaultPortal(authenticatedUser.role) : "/login"}
                replace
              />
            }
          />

          <Route
            path="/login"
            element={
              isAuthenticated
                ? <Navigate to={getDefaultPortal(authenticatedUser?.role || "Doctor")} replace />
                : <LoginPage onLogin={() => setIsAuthenticated(true)} />
            }
          />

          <Route
            path="/nurse"
            element={<ProtectedRoute path="/nurse"><NurseDashboard /></ProtectedRoute>}
          />

          <Route path="/lab" element={<ProtectedRoute path="/lab"><LabDashboard /></ProtectedRoute>} />

          <Route
            path="/doctor"
            element={<ProtectedRoute path="/doctor"><DoctorDashboard /></ProtectedRoute>}
          />
        </Routes>
      </main>
    </div>
  );
}