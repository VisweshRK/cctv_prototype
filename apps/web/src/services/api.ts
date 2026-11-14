/**
 * Frontend API Service Client for Presence B2C CCTV AI Platform.
 */

const API_BASE = "/api";

function getHeaders(): HeadersInit {
  const token = localStorage.getItem("presence_token");
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export const api = {
  // Auth
  login: async (email: string, password: string) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Login failed");
    const data = await res.json();
    localStorage.setItem("presence_token", data.access_token);
    return data;
  },

  register: async (payload: { email: string; password: string; full_name: string; household_name?: string; role?: string }) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Registration failed");
    const data = await res.json();
    localStorage.setItem("presence_token", data.access_token);
    return data;
  },

  getMe: async () => {
    const res = await fetch(`${API_BASE}/auth/me`, { headers: getHeaders() });
    if (!res.ok) return null;
    return res.json();
  },

  logout: () => {
    localStorage.removeItem("presence_token");
  },

  // Cameras
  getCameras: async () => {
    const res = await fetch(`${API_BASE}/cameras`, { headers: getHeaders() });
    return res.json();
  },

  addCamera: async (data: { name: string; rtsp_url?: string; stream_type?: string; location?: string }) => {
    const res = await fetch(`${API_BASE}/cameras`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    return res.json();
  },

  probeCamera: async (data: { ip: string; rtsp_port?: number; onvif_port?: number; username?: string; password?: string }) => {
    const res = await fetch(`${API_BASE}/cameras/probe`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    return res.json();
  },

  // Rules
  getRules: async () => {
    const res = await fetch(`${API_BASE}/rules`, { headers: getHeaders() });
    return res.json();
  },

  createRule: async (rule: any) => {
    const res = await fetch(`${API_BASE}/rules`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(rule),
    });
    return res.json();
  },

  updateRule: async (id: string, rule: any) => {
    const res = await fetch(`${API_BASE}/rules/${id}`, {
      method: "PUT",
      headers: getHeaders(),
      body: JSON.stringify(rule),
    });
    return res.json();
  },

  deleteRule: async (id: string) => {
    const res = await fetch(`${API_BASE}/rules/${id}`, {
      method: "DELETE",
      headers: getHeaders(),
    });
    return res.json();
  },

  simulateRule: async (rule: any, sample_event: any) => {
    const res = await fetch(`${API_BASE}/rules/simulate`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ rule, sample_event }),
    });
    return res.json();
  },

  // Events & Alerts
  getEvents: async (limit = 50, moduleId?: string) => {
    const url = moduleId ? `${API_BASE}/events?limit=${limit}&module_id=${moduleId}` : `${API_BASE}/events?limit=${limit}`;
    const res = await fetch(url, { headers: getHeaders() });
    return res.json();
  },

  simulateEvent: async (payload: { module_id: string; event_type: string; camera_id?: string; severity?: string; metadata?: any }) => {
    const res = await fetch(`${API_BASE}/events/simulate`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(payload),
    });
    return res.json();
  },

  getAlerts: async () => {
    const res = await fetch(`${API_BASE}/alerts`, { headers: getHeaders() });
    return res.json();
  },

  acknowledgeAlert: async (alertId: string) => {
    const res = await fetch(`${API_BASE}/alerts/${alertId}/ack`, {
      method: "POST",
      headers: getHeaders(),
    });
    return res.json();
  },

  // Zones
  getZones: async (cameraId = "camera_main") => {
    const res = await fetch(`${API_BASE}/zones?camera_id=${cameraId}`, { headers: getHeaders() });
    return res.json();
  },

  createZone: async (zone: any) => {
    const res = await fetch(`${API_BASE}/zones`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(zone),
    });
    return res.json();
  },

  deleteZone: async (zoneId: string) => {
    const res = await fetch(`${API_BASE}/zones/${zoneId}`, {
      method: "DELETE",
      headers: getHeaders(),
    });
    return res.json();
  },

  // Faces
  getFaces: async () => {
    const res = await fetch(`${API_BASE}/faces`, { headers: getHeaders() });
    return res.json();
  },

  enrollFace: async (data: { name: string; relation: string; consent_given: boolean }) => {
    const res = await fetch(`${API_BASE}/faces/enroll`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    return res.json();
  },

  deleteFace: async (id: string) => {
    const res = await fetch(`${API_BASE}/faces/${id}`, {
      method: "DELETE",
      headers: getHeaders(),
    });
    return res.json();
  },

  // Models
  getModels: async () => {
    const res = await fetch(`${API_BASE}/models`, { headers: getHeaders() });
    return res.json();
  },

  // Analytics
  getAnalyticsSummary: async () => {
    const res = await fetch(`${API_BASE}/analytics/summary`, { headers: getHeaders() });
    return res.json();
  },

  getOccupancyTrends: async () => {
    const res = await fetch(`${API_BASE}/analytics/occupancy-trends`, { headers: getHeaders() });
    return res.json();
  },

  getEnergyTrends: async () => {
    const res = await fetch(`${API_BASE}/analytics/energy-trends`, { headers: getHeaders() });
    return res.json();
  },

  getVehicleLogs: async () => {
    const res = await fetch(`${API_BASE}/analytics/vehicle-logs`, { headers: getHeaders() });
    return res.json();
  },

  // Super Admin
  getAdminSystemHealth: async () => {
    const res = await fetch(`${API_BASE}/admin/system-health`, { headers: getHeaders() });
    return res.json();
  },

  getAdminHouseholds: async () => {
    const res = await fetch(`${API_BASE}/admin/households`, { headers: getHeaders() });
    return res.json();
  },

  getAdminAuditLogs: async () => {
    const res = await fetch(`${API_BASE}/admin/audit-logs`, { headers: getHeaders() });
    return res.json();
  },
};
