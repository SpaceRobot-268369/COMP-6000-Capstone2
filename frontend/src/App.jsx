import { useCallback, useEffect, useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import AboutPage from "./pages/AboutPage.jsx";
import ImmersivePage from "./pages/ImmersivePage.jsx";
import GenerationPage from "./pages/GenerationPage.jsx";
import DemoPage from "./pages/DemoPage.jsx";
import LayerATestPage from "./pages/LayerATestPage.jsx";
import DevAnalysisPage from "./pages/DevAnalysisPage.jsx";
import ServerBStatusPage from "./pages/ServerBStatusPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import ServerStatus from "./components/ServerStatus.jsx";
import {
  checkServerBStatus,
  createCheckingStatus,
  createStatusLogEntry,
  reconnectServerB,
} from "./lib/serverBStatus.js";

const accountStorageKey = "sonic-lab-account-name";
const minServerBCheckingMs = 350;

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function sidebarLinkClass({ isActive }) {
  return `nav-item${isActive ? " active" : ""}`;
}

function sidebarActionClass({ isActive }) {
  return `sidebar-action${isActive ? " active" : ""}`;
}

export default function App() {
  const [accountName, setAccountName] = useState("");
  const [serverBStatus, setServerBStatus] = useState(() => createCheckingStatus());
  const [serverBLogs, setServerBLogs] = useState([]);
  const [serverBChecking, setServerBChecking] = useState(false);
  const serverBCheckInFlightRef = useRef(false);
  const serverBPollStartedRef = useRef(false);
  const isLoggedIn = Boolean(accountName);

  useEffect(() => {
    const storedAccountName = window.localStorage.getItem(accountStorageKey);
    if (storedAccountName) {
      setAccountName(storedAccountName);
    }
  }, []);

  const runServerBCheck = useCallback(async (source = "auto") => {
    if (serverBCheckInFlightRef.current) {
      return null;
    }

    serverBCheckInFlightRef.current = true;
    const checkingStatus = createCheckingStatus(source);
    setServerBChecking(true);
    setServerBStatus(checkingStatus);
    setServerBLogs((current) => [
      createStatusLogEntry(checkingStatus, source),
      ...current,
    ].slice(0, 80));

    const startedAt = performance.now();
    try {
      const result = await checkServerBStatus();
      setServerBLogs((current) => [
        createStatusLogEntry(result, source),
        ...current,
      ].slice(0, 80));

      if (result.key !== "offline") {
        const remainingMs = minServerBCheckingMs - (performance.now() - startedAt);
        if (remainingMs > 0) {
          await wait(remainingMs);
        }

        setServerBStatus(result);
        return result;
      }

      const reconnectSource = source === "manual" ? "manual-reconnect" : "auto-reconnect";
      const reconnectingStatus = createCheckingStatus("reconnect");
      setServerBStatus(reconnectingStatus);
      setServerBLogs((current) => [
        createStatusLogEntry(reconnectingStatus, reconnectSource),
        ...current,
      ].slice(0, 80));

      const reconnectResult = await reconnectServerB();
      const remainingMs = minServerBCheckingMs - (performance.now() - startedAt);
      if (remainingMs > 0) {
        await wait(remainingMs);
      }

      setServerBStatus(reconnectResult);
      setServerBLogs((current) => [
        createStatusLogEntry(reconnectResult, reconnectSource),
        ...current,
      ].slice(0, 80));
      return reconnectResult;
    } finally {
      serverBCheckInFlightRef.current = false;
      setServerBChecking(false);
    }
  }, []);

  useEffect(() => {
    if (!serverBPollStartedRef.current) {
      serverBPollStartedRef.current = true;
      runServerBCheck("initial");
    }

    const timer = window.setInterval(() => {
      runServerBCheck("auto");
    }, 30000);
    return () => window.clearInterval(timer);
  }, [runServerBCheck]);

  function handleAuthenticate(value) {
    const nextValue = value.trim();
    setAccountName(nextValue);
    if (nextValue) {
      window.localStorage.setItem(accountStorageKey, nextValue);
      return;
    }

    window.localStorage.removeItem(accountStorageKey);
  }

  function handleLogout() {
    setAccountName("");
    window.localStorage.removeItem(accountStorageKey);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-block">
            <p className="brand-title">SONIC LAB</p>
            <p className="brand-subtitle">PRECISION AUDIO AI</p>
          </div>

          <nav className="sidebar-nav">
            <NavLink to="/about" className={sidebarLinkClass}>
              <span className="nav-icon">✦</span>
              <span>Introduction</span>
            </NavLink>
            <NavLink to="/analysis" className={sidebarLinkClass}>
              <span className="nav-icon">◫</span>
              <span>Analysis</span>
            </NavLink>
            <NavLink to="/generation" className={sidebarLinkClass}>
              <span className="nav-icon">✦</span>
              <span>Generation</span>
            </NavLink>
            <NavLink to="/demo" className={sidebarLinkClass}>
              <span className="nav-icon">❖</span>
              <span>Demo</span>
            </NavLink>
            <NavLink to="/immersive" className={sidebarLinkClass}>
              <span className="nav-icon">❂</span>
              <span>Immersive</span>
            </NavLink>
            <NavLink to="/dev/layers" className={sidebarLinkClass}>
              <span className="nav-icon">⌬</span>
              <span>Dev — Generation</span>
            </NavLink>
            <NavLink to="/dev/analysis" className={sidebarLinkClass}>
              <span className="nav-icon">◉</span>
              <span>Dev — Analysis</span>
            </NavLink>
          </nav>
        </div>

        <ThemeToggle />

        <ServerStatus status={serverBStatus} />

        <section className="sidebar-user panel" aria-label="Account status">
          <div className="sidebar-user-head">
            <p className="sidebar-user-label">
              {isLoggedIn ? "Account logged in" : "Not logged in"}
            </p>
          </div>

          {isLoggedIn ? (
            <div className="sidebar-user-simple">
              <strong className="sidebar-user-name">{accountName}</strong>
              <button type="button" className="sidebar-logout-button" onClick={handleLogout}>
                Log out
              </button>
            </div>
          ) : (
            <>
              <p className="sidebar-user-empty">No account signed in</p>
              <div className="sidebar-user-actions">
                <NavLink to="/login" className={sidebarActionClass}>
                  Login
                </NavLink>
                <NavLink to="/register" className={sidebarActionClass}>
                  Register
                </NavLink>
              </div>
            </>
          )}
        </section>
      </aside>

      <main className="main-panel">
        <Routes>
          <Route path="/" element={<Navigate to="/about" replace />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/analysis" element={<HomePage />} />
          <Route path="/generation" element={<GenerationPage />} />
          <Route path="/demo" element={<DemoPage />} />
          {/* Transformation is out of scope for the demo — redirect any stale links. */}
          <Route path="/transformation" element={<Navigate to="/about" replace />} />
          <Route path="/immersive" element={<ImmersivePage />} />
          <Route
            path="/dev/layers"
            element={
              <LayerATestPage
                mode="generation"
                eyebrow="DEVELOPER TOOLS — GENERATION"
                title="Generation Layers Dev Test"
              />
            }
          />
          <Route path="/dev/analysis" element={<DevAnalysisPage />} />
          <Route
            path="/server-b"
            element={
              <ServerBStatusPage
                status={serverBStatus}
                logs={serverBLogs}
                checking={serverBChecking}
                onRecheck={() => runServerBCheck("manual")}
              />
            }
          />
          <Route
            path="/login"
            element={<LoginPage accountName={accountName} onLogin={handleAuthenticate} />}
          />
          <Route
            path="/register"
            element={<RegisterPage accountName={accountName} onRegister={handleAuthenticate} />}
          />
        </Routes>
      </main>
    </div>
  );
}
