const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
let statusLogSequence = 0;

export function createCheckingStatus(source = "auto") {
  return {
    key: "checking",
    label: "Checking serverB",
    detail: source === "manual" ? "Manual health check in progress" : "Pinging AI tunnel",
    stage: "checking",
    checkedAt: new Date().toISOString(),
  };
}

function statusFromResponse(response, body, elapsedMs) {
  const checkedAt = new Date().toISOString();
  const message = body?.message || body?.detail || "";
  const payload = body && Object.keys(body).length > 0 ? body : null;

  if (response.ok && body?.ok !== false) {
    return {
      key: "online",
      label: "ServerB connected",
      detail: message || "AI health check passed",
      stage: body?.stage || "ai-health-ok",
      httpStatus: response.status,
      elapsedMs,
      checkedAt,
      payload,
    };
  }

  const stage = body?.stage || `http-${response.status}`;
  const isTunnelFailure = stage.startsWith("ai-tunnel") || stage.startsWith("ai-connect");
  const key = isTunnelFailure || response.status >= 500 ? "offline" : "degraded";

  return {
    key,
    label: key === "offline" ? "ServerB unavailable" : "ServerB degraded",
    detail: message || `AI health check returned HTTP ${response.status}`,
    stage,
    httpStatus: response.status,
    elapsedMs,
    checkedAt,
    payload,
  };
}

export async function checkServerBStatus() {
  const startedAt = performance.now();

  try {
    const response = await fetch(`${API_BASE}/api/ai/health`, {
      credentials: "include",
    });
    const body = await response.json().catch(() => ({}));
    return statusFromResponse(response, body, Math.round(performance.now() - startedAt));
  } catch (error) {
    return {
      key: "offline",
      label: "Backend unreachable",
      detail: error?.message || "The frontend could not reach the backend health endpoint.",
      stage: "frontend-fetch",
      elapsedMs: Math.round(performance.now() - startedAt),
      checkedAt: new Date().toISOString(),
      payload: null,
    };
  }
}

export function createStatusLogEntry(status, source = "auto") {
  statusLogSequence += 1;

  return {
    id: `${Date.now()}-${statusLogSequence}-${source}-${status.stage}`,
    timestamp: status.checkedAt || new Date().toISOString(),
    source,
    statusKey: status.key,
    label: status.label,
    detail: status.detail,
    stage: status.stage,
    httpStatus: status.httpStatus,
    elapsedMs: status.elapsedMs,
    payload: status.payload,
  };
}
