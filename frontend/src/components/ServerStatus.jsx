import { useState } from "react";

const STATES = [
  { key: "online",   label: "Server online",  detail: "All systems nominal" },
  { key: "degraded", label: "Server degraded", detail: "Elevated latency" },
  { key: "offline",  label: "Server offline", detail: "No response" },
  { key: "checking", label: "Checking…",      detail: "Pinging backend" },
];

export default function ServerStatus() {
  const [index, setIndex] = useState(0);
  const current = STATES[index];

  function cycle() {
    setIndex((i) => (i + 1) % STATES.length);
  }

  return (
    <button
      type="button"
      className={`sidebar-status sidebar-status--${current.key}`}
      onClick={cycle}
      title={current.detail}
      aria-label={`Server status: ${current.label}. Click to preview next state.`}
    >
      <span className="sidebar-status-dot" aria-hidden="true" />
      <span className="sidebar-status-label">{current.label}</span>
    </button>
  );
}
