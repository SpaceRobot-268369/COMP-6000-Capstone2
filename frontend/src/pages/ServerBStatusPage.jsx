function formatTime(value) {
  if (!value) return "Never";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function formatPayload(payload) {
  if (!payload) return "";
  return JSON.stringify(payload, null, 2);
}

export default function ServerBStatusPage({
  status,
  logs = [],
  checking = false,
  onRecheck = () => {},
}) {
  const current = status || {
    key: "checking",
    label: "Checking serverB",
    detail: "Pinging AI tunnel",
    stage: "checking",
  };

  return (
    <section className="server-b-page">
      <header className="topbar">
        <div>
          <p className="eyebrow">INFRASTRUCTURE STATUS</p>
          <h1>SERVERB AI LINK</h1>
          <div className="status-line">
            <span className={`status-accent status-accent--${current.key}`} />
            <p>{current.label} / {current.stage || "unknown"}</p>
          </div>
        </div>
        <div className="topbar-tools">
          <button
            type="button"
            className="analyse-btn server-b-recheck"
            onClick={onRecheck}
            disabled={checking}
          >
            {checking ? "Checking" : "Recheck"}
          </button>
        </div>
      </header>

      <section className={`panel server-b-status-card server-b-status-card--${current.key}`}>
        <div className="server-b-status-head">
          <span className="server-b-status-dot" aria-hidden="true" />
          <div>
            <h2>{current.label}</h2>
            <p>{current.detail}</p>
          </div>
        </div>
        <dl className="server-b-facts">
          <div>
            <dt>Last check</dt>
            <dd>{formatTime(current.checkedAt)}</dd>
          </div>
          <div>
            <dt>HTTP</dt>
            <dd>{current.httpStatus ?? "n/a"}</dd>
          </div>
          <div>
            <dt>Latency</dt>
            <dd>{typeof current.elapsedMs === "number" ? `${current.elapsedMs} ms` : "n/a"}</dd>
          </div>
          <div>
            <dt>Stage</dt>
            <dd>{current.stage || "unknown"}</dd>
          </div>
        </dl>
      </section>

      <section className="panel server-b-log-panel">
        <div className="generation-card-head">
          <h2>Live Check Log</h2>
          <p>Automatic checks, reconnect attempts, and manual rechecks appear here in real time</p>
        </div>

        <div className="server-b-log-list" aria-live="polite">
          {logs.length === 0 ? (
            <div className="server-b-log-empty">Waiting for first serverB check.</div>
          ) : (
            logs.map((entry) => (
              <article
                key={entry.id}
                className={`server-b-log-entry server-b-log-entry--${entry.statusKey}`}
              >
                <div className="server-b-log-meta">
                  <span>{formatTime(entry.timestamp)}</span>
                  <span>{entry.source}</span>
                  <span>{entry.stage}</span>
                  {typeof entry.elapsedMs === "number" ? <span>{entry.elapsedMs} ms</span> : null}
                </div>
                <h3>{entry.label}</h3>
                <p>{entry.detail}</p>
                {entry.payload ? (
                  <pre>{formatPayload(entry.payload)}</pre>
                ) : null}
              </article>
            ))
          )}
        </div>
      </section>
    </section>
  );
}
