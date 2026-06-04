const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
let statusLogSequence = 0;
const STATUS_LABELS = {
  online: "ServerB connected",
  degraded: "ServerB degraded",
  offline: "ServerB unavailable",
  checking: "Checking serverB",
};

function normalizeStatusKey(value) {
  return Object.prototype.hasOwnProperty.call(STATUS_LABELS, value) ? value : "";
}

function statusKeyFromReachability(reachability) {
  if (!reachability) return "";
  if (reachability.aiService === true) return "online";
  if (reachability.serverB === true && reachability.aiService === false) return "degraded";
  if (reachability.backend === true || reachability.tunnelContainer === true) return "offline";
  return "";
}

export function createCheckingStatus(source = "auto") {
  const isReconnect = source === "reconnect" || source === "auto-reconnect";

  return {
    key: "checking",
    label: isReconnect ? "Reconnecting serverB" : "Checking serverB",
    detail: isReconnect
      ? "Attempting to restore the AI tunnel"
      : source === "manual"
        ? "Manual health check in progress"
        : "Pinging AI tunnel",
    stage: isReconnect ? "ai-reconnect-start" : "checking",
    checkedAt: new Date().toISOString(),
  };
}

function statusFromResponse(response, body, elapsedMs) {
  const checkedAt = new Date().toISOString();
  const message = body?.message || body?.detail || "";
  const payload = body && Object.keys(body).length > 0 ? body : null;
  const explicitKey = normalizeStatusKey(body?.statusKey || body?.severity);
  const reachabilityKey = statusKeyFromReachability(body?.reachability);

  if (response.ok && body?.ok !== false) {
    const key = explicitKey || reachabilityKey || "online";
    return {
      key,
      label: STATUS_LABELS[key],
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
  const key = explicitKey || reachabilityKey || (isTunnelFailure || response.status >= 500 ? "offline" : "degraded");

  return {
    key,
    label: STATUS_LABELS[key],
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

export async function reconnectServerB() {
  const startedAt = performance.now();

  try {
    const response = await fetch(`${API_BASE}/api/ai/reconnect`, {
      method: "POST",
      credentials: "include",
    });
    const body = await response.json().catch(() => ({}));
    return statusFromResponse(response, body, Math.round(performance.now() - startedAt));
  } catch (error) {
    return {
      key: "offline",
      label: "Backend unreachable",
      detail: error?.message || "The frontend could not reach the backend reconnect endpoint.",
      stage: "frontend-reconnect-fetch",
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
