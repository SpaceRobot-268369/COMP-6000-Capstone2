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

function navbarLinkClass({ isActive }) {
  return `navbar-link${isActive ? " active" : ""}`;
}

function navbarDropdownItemClass({ isActive }) {
  return `navbar-dropdown-item${isActive ? " active" : ""}`;
}

function navbarActionClass({ isActive }) {
  return `navbar-action${isActive ? " active" : ""}`;
}

export default function App() {
  const [accountName, setAccountName] = useState("");
  const [serverBStatus, setServerBStatus] = useState(() => createCheckingStatus());
  const [serverBLogs, setServerBLogs] = useState([]);
  const [serverBChecking, setServerBChecking] = useState(false);
  const serverBCheckInFlightRef = useRef(false);
  const serverBPollStartedRef = useRef(false);
  const isLoggedIn = Boolean(accountName);

  const [devDropdownOpen, setDevDropdownOpen] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

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
      <header className="navbar-container">
        <div className="navbar-body">
          {/* Brand Logo and Title */}
          <NavLink to="/about" className="navbar-brand" onClick={() => setMobileMenuOpen(false)}>
            <span className="brand-logo">✦</span>
            <div className="brand-text">
              <span className="brand-title">SONIC LAB</span>
              <span className="brand-subtitle">PRECISION AUDIO AI</span>
            </div>
          </NavLink>

          {/* Mobile Menu Burger Toggle */}
          <button 
            type="button" 
            className={`navbar-mobile-toggle ${mobileMenuOpen ? "active" : ""}`}
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle navigation menu"
          >
            <span className="burger-line"></span>
            <span className="burger-line"></span>
            <span className="burger-line"></span>
          </button>

          {/* Navigation Links and Menus */}
          <nav className={`navbar-menu ${mobileMenuOpen ? "active" : ""}`}>
            <div className="navbar-links">
              <NavLink to="/about" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">✦</span>
                <span>Introduction</span>
              </NavLink>
              <NavLink to="/analysis" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">◫</span>
                <span>Analysis</span>
              </NavLink>
              <NavLink to="/generation" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">✦</span>
                <span>Generation</span>
              </NavLink>
              <NavLink to="/demo" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">❖</span>
                <span>Demo</span>
              </NavLink>
              <NavLink to="/immersive" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">❂</span>
                <span>Immersive</span>
              </NavLink>
            </div>

            <div className="navbar-controls">
              {/* Dev Tools Dropdown */}
              <div 
                className={`navbar-dropdown-wrapper ${devDropdownOpen ? "open" : ""}`}
                onMouseEnter={() => setDevDropdownOpen(true)}
                onMouseLeave={() => setDevDropdownOpen(false)}
              >
                <button 
                  type="button" 
                  className="navbar-dropdown-trigger"
                  onClick={() => setDevDropdownOpen(!devDropdownOpen)}
                >
                  <span className="nav-icon">⌬</span>
                  <span>Developer</span>
                  <span className="dropdown-arrow">▼</span>
                </button>
                <div className="navbar-dropdown-menu">
                  <NavLink to="/dev/layers" className={navbarDropdownItemClass} onClick={() => { setDevDropdownOpen(false); setMobileMenuOpen(false); }}>
                    <span className="nav-icon">⌬</span>
                    <span>Dev — Generation</span>
                  </NavLink>
                  <NavLink to="/dev/analysis" className={navbarDropdownItemClass} onClick={() => { setDevDropdownOpen(false); setMobileMenuOpen(false); }}>
                    <span className="nav-icon">◉</span>
                    <span>Dev — Analysis</span>
                  </NavLink>
                  <div className="dropdown-divider"></div>
                  <div className="navbar-dropdown-status">
                    <ServerStatus status={serverBStatus} />
                  </div>
                </div>
              </div>

              {/* Theme Toggle */}
              <ThemeToggle />

              {/* User Dropdown */}
              <div 
                className={`navbar-dropdown-wrapper ${userDropdownOpen ? "open" : ""}`}
                onMouseEnter={() => setUserDropdownOpen(true)}
                onMouseLeave={() => setUserDropdownOpen(false)}
              >
                <button 
                  type="button" 
                  className="navbar-dropdown-trigger"
                  onClick={() => setUserDropdownOpen(!userDropdownOpen)}
                >
                  <span className="nav-icon">👤</span>
                  <span>{isLoggedIn ? accountName : "Account"}</span>
                  <span className="dropdown-arrow">▼</span>
                </button>
                <div className="navbar-dropdown-menu user-dropdown">
                  <div className="navbar-dropdown-header">
                    {isLoggedIn ? "Account Logged In" : "Not Logged In"}
                  </div>
                  {isLoggedIn ? (
                    <div className="navbar-user-simple">
                      <strong className="navbar-user-name">{accountName}</strong>
                      <button 
                        type="button" 
                        className="navbar-logout-button" 
                        onClick={() => { handleLogout(); setUserDropdownOpen(false); setMobileMenuOpen(false); }}
                      >
                        Log out
                      </button>
                    </div>
                  ) : (
                    <div className="navbar-user-actions">
                      <NavLink 
                        to="/login" 
                        className={navbarActionClass} 
                        onClick={() => { setUserDropdownOpen(false); setMobileMenuOpen(false); }}
                      >
                        Login
                      </NavLink>
                      <NavLink 
                        to="/register" 
                        className={navbarActionClass} 
                        onClick={() => { setUserDropdownOpen(false); setMobileMenuOpen(false); }}
                      >
                        Register
                      </NavLink>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </nav>
        </div>
      </header>

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
