export type PortalRole = "Doctor" | "Nurse" | "Lab";

export type AuthenticatedUser = {
  role: PortalRole;
  employeeId: string;
};

export type PortalPath = "/doctor" | "/nurse" | "/lab";

const AUTH_STORAGE_KEY = "aegis-authenticated-user";

export const STATIC_CREDENTIALS: Record<PortalRole, { employeeId: string; password: string }> = {
  Doctor: { employeeId: "DOC001", password: "doctor123" },
  Nurse: { employeeId: "NUR001", password: "nurse123" },
  Lab: { employeeId: "LAB001", password: "lab123" },
};

const ROLE_PORTAL_ACCESS: Record<PortalRole, PortalPath[]> = {
  Doctor: ["/doctor"],
  Nurse: ["/nurse", "/doctor"],
  Lab: ["/lab"],
};

export function getAllowedPortals(role: PortalRole): PortalPath[] {
  return ROLE_PORTAL_ACCESS[role];
}

export function getDefaultPortal(role: PortalRole): PortalPath {
  return ROLE_PORTAL_ACCESS[role][0];
}

export function canAccessPortal(role: PortalRole, path: string): path is PortalPath {
  return getAllowedPortals(role).includes(path as PortalPath);
}

export function getAuthenticatedUser(): AuthenticatedUser | null {
  const stored = window.localStorage.getItem(AUTH_STORAGE_KEY);
  if (!stored) return null;

  try {
    const user = JSON.parse(stored) as AuthenticatedUser;
    if (!(user.role in ROLE_PORTAL_ACCESS) || !user.employeeId) {
      throw new Error("Invalid authentication state");
    }
    return user;
  } catch {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

export function authenticate(
  role: PortalRole,
  employeeId: string,
  password: string
): AuthenticatedUser | null {
  const credentials = STATIC_CREDENTIALS[role];
  if (credentials.employeeId !== employeeId.trim() || credentials.password !== password) {
    return null;
  }

  const user = { role, employeeId: credentials.employeeId };
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(user));
  return user;
}

export function clearAuthentication() {
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
}
