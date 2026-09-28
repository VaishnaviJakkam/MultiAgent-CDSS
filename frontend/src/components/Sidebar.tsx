import {
  Activity,
  ClipboardPlus,
  FlaskConical,
  LogOut,
  Stethoscope,
} from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import { getAllowedPortals, type PortalRole } from "../auth";

export default function Sidebar({ onLogout, role }: { onLogout: () => void; role: PortalRole }) {
  const navigate = useNavigate();
  const allowedPortals = getAllowedPortals(role);

  function handleLogout() {
    onLogout();
    navigate("/login", { replace: true });
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Activity size={24} />
        </div>

        <div>
          <h2>A4 Clinical</h2>
          <span>
            Deterioration Intelligence
          </span>
        </div>
      </div>

      <nav className="nav-menu">
        {allowedPortals.includes("/lab") && (
          <NavLink
            to="/lab"
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            <FlaskConical size={19} />
            Lab Portal
          </NavLink>
        )}

        {allowedPortals.includes("/nurse") && (
          <NavLink
            to="/nurse"
            className={({ isActive }) =>
              `nav-item ${
                isActive
                  ? "active"
                  : ""
              }`
            }
          >
            <ClipboardPlus size={19} />
            Nurse Portal
          </NavLink>
        )}

        {allowedPortals.includes("/doctor") && (
          <NavLink
            to="/doctor"
            className={({ isActive }) =>
              `nav-item ${
                isActive
                  ? "active"
                  : ""
              }`
            }
          >
            <Stethoscope size={19} />
            Doctor Portal
          </NavLink>
        )}

      </nav>

      <div className="sidebar-footer">
        <div className="status-dot" />

        <div>
          <strong>
            System Online
          </strong>

          <span>
            Agent services connected
          </span>
        </div>
      </div>

      <button className="logout-button" type="button" onClick={handleLogout}>
        <LogOut size={17} />
        Logout
      </button>
    </aside>
  );
}