import { NavLink } from "react-router-dom";

const FALLBACK_STATUS = {
  key: "checking",
  label: "Checking serverB",
  detail: "Pinging AI tunnel",
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
      aria-label={`ServerB status: ${current.label}. Open serverB status page.`}
    >
      <span className="sidebar-status-dot" aria-hidden="true" />
      <span className="sidebar-status-label">{current.label}</span>
    </NavLink>
  );
}
