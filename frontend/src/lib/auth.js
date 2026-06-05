const fallbackApiOrigins = [
  "http://localhost:4000",
  "http://127.0.0.1:4000",
];

const defaultRequestError = "We couldn't complete your request. Please try again.";
const networkRequestError = "We couldn't reach the server. Please try again in a moment.";

function parseResponseText(rawText) {
  if (!rawText) {
    return {};
  }

  try {
    return JSON.parse(rawText);
  } catch {
    return { message: rawText };
  }
}

function normalizeBase(base) {
  if (!base || base === "/") {
    return "";
  }

  return base.endsWith("/") ? base.slice(0, -1) : base;
}

function getApiBases() {
  const configuredOrigin = normalizeBase(import.meta.env.VITE_API_URL || "");
  return Array.from(new Set([configuredOrigin, "", ...fallbackApiOrigins].map(normalizeBase)));
}

function isRetryableStatus(status) {
  return status >= 500;
}

function normalizeErrorMessage(message, status) {
  const trimmedMessage = typeof message === "string" ? message.trim() : "";
  if (trimmedMessage) {
    return trimmedMessage;
  }

  if (status) {
    return `The server returned an error (${status}). Please try again.`;
  }

  return defaultRequestError;
}

async function sendRequest(base, path, { method = "POST", payload } = {}) {
  const response = await fetch(`${base}${path}`, {
    method,
    headers: payload ? {
      "Content-Type": "application/json",
    } : undefined,
    credentials: "include",
    body: payload ? JSON.stringify(payload) : undefined,
  });

  const rawText = await response.text();
  const data = parseResponseText(rawText);

  return { response, data };
}

async function request(path, options = {}) {
  const bases = getApiBases();
  let lastMessage = defaultRequestError;
  let lastStatus = 0;

  for (const base of bases) {
    try {
      const { response, data } = await sendRequest(base, path, options);

      if (response.ok) {
        return data;
      }

      const message = normalizeErrorMessage(data.message, response.status);
      lastMessage = message;
      lastStatus = response.status;

      if (!isRetryableStatus(response.status)) {
        const error = new Error(message);
        error.status = response.status;
        throw error;
      }
    } catch (error) {
      lastMessage = error.message || networkRequestError;
      lastStatus = error.status || 0;
    }
  }

  const error = new Error(lastMessage);
  error.status = lastStatus;
  throw error;
}

export function getCurrentUser() {
  return request("/api/me", { method: "GET" });
}

export function loginAccount(payload) {
  return request("/api/login", { payload });
}

export function registerAccount(payload) {
  return request("/api/register", { payload });
}

export function logoutAccount() {
  return request("/api/logout");
}
