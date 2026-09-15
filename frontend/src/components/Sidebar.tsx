import {
  Activity,
  ClipboardPlus,
  LayoutDashboard,
  Stethoscope,
} from "lucide-react";
import { NavLink } from "react-router-dom";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Activity size={24} />
        </div>

        <div>
          <h2>Aegis Clinical</h2>
          <span>
            Deterioration Intelligence
          </span>
        </div>
      </div>

      <nav className="nav-menu">
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

        <div className="nav-divider" />

        <NavLink
          to="/clinical"
          className={({ isActive }) =>
            `nav-item ${
              isActive
                ? "active"
                : ""
            }`
          }
        >
          <LayoutDashboard
            size={19}
          />
          Clinical Dashboard
        </NavLink>
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
    </aside>
  );
}