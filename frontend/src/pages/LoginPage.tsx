import { useState } from "react";
import { LockKeyhole, LogIn, UserRound } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { authenticate, type PortalRole } from "../auth";

type RoleOption = {
  role: PortalRole;
  label: string;
  path: string;
};

const ROLE_OPTIONS: RoleOption[] = [
  { role: "Doctor", label: "Doctor", path: "/doctor" },
  { role: "Nurse", label: "Nurse", path: "/nurse" },
  { role: "Lab", label: "Lab", path: "/lab" },
];

export default function LoginPage({ onLogin }: { onLogin: () => void }) {
  const navigate = useNavigate();
  const [role, setRole] = useState<PortalRole>("Doctor");
  const [employeeId, setEmployeeId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const authenticatedUser = authenticate(role, employeeId, password);

    if (!authenticatedUser) {
      setError("Invalid employee ID or password.");
      return;
    }

    onLogin();
    navigate(ROLE_OPTIONS.find((option) => option.role === role)?.path || "/doctor", {
      replace: true,
    });
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="login-title">
        <div className="auth-brand-mark"><LogIn size={22} /></div>
        <span className="eyebrow">A4 Clinical</span>
        <h1 id="login-title">Portal sign in</h1>
        <p className="auth-intro">Sign in to access your clinical workspace.</p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Role
            <select value={role} onChange={(event) => setRole(event.target.value as PortalRole)}>
              {ROLE_OPTIONS.map((option) => (
                <option key={option.role} value={option.role}>{option.label}</option>
              ))}
            </select>
          </label>

          <label>
            Employee ID
            <span className="auth-input-wrap">
              <UserRound size={17} aria-hidden="true" />
              <input
                value={employeeId}
                onChange={(event) => setEmployeeId(event.target.value)}
                autoComplete="username"
                required
              />
            </span>
          </label>

          <label>
            Password
            <span className="auth-input-wrap">
              <LockKeyhole size={17} aria-hidden="true" />
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
                required
              />
            </span>
          </label>

          {error && <p className="auth-error" role="alert">{error}</p>}

          <button className="primary-button auth-submit" type="submit">
            Sign in
          </button>
        </form>
      </section>
    </main>
  );
}
