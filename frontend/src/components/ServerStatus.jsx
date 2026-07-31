import { NavLink } from "react-router-dom";

const FALLBACK_STATUS = {
  key: "checking",
  label: "Checking mock AI service",
  detail: "Pinging the mock AI service",
};

export default function ServerStatus({ status = FALLBACK_STATUS }) {
  const current = status || FALLBACK_STATUS;

  return (
    <NavLink
      to="/server-b"
      className={({ isActive }) =>
        `sidebar-status sidebar-status--${current.key}${isActive ? " active" : ""}`
      }
      title={current.detail}
      aria-label={`AI service status: ${current.label}. Open the AI service status page.`}
    >
      <span className="sidebar-status-dot" aria-hidden="true" />
      <span className="sidebar-status-label">{current.label}</span>
    </NavLink>
  );
}
