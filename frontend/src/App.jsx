import { useCallback, useEffect, useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import AboutPage from "./pages/AboutPage.jsx";
import ImmersivePage from "./pages/ImmersivePage.jsx";
import GenerationPage from "./pages/GenerationPage.jsx";
import PipelinePage from "./pages/PipelinePage.jsx";
import LayerATestPage from "./pages/LayerATestPage.jsx";
import DevAnalysisPage from "./pages/DevAnalysisPage.jsx";
import ServerBStatusPage from "./pages/ServerBStatusPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import ServerStatus from "./components/ServerStatus.jsx";
import { getCurrentUser, logoutAccount } from "./lib/auth.js";
import {
  checkServerBStatus,
  createCheckingStatus,
  createStatusLogEntry,
  reconnectServerB,
} from "./lib/serverBStatus.js";

const minServerBCheckingMs = 350;
const authExpiredEventName = "sonic-lab-auth-expired";
const sessionRefreshMs = 60 * 1000;

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

function isAdminRoute(pathname) {
  return pathname.startsWith("/dev/") || pathname === "/server-b";
}

function RequireAdmin({ currentUser, authChecking, children }) {
  const routeLocation = useLocation();

  if (authChecking) {
    return (
      <section className="account-page">
        <section className="panel account-card">
          <p className="account-lead">Checking account access...</p>
        </section>
      </section>
    );
  }

  if (!currentUser) {
    return <Navigate to="/login" replace state={{ from: routeLocation }} />;
  }

  if (currentUser.role !== "admin") {
    return (
      <section className="account-page">
        <header className="account-topbar">
          <div>
            <p className="eyebrow">ACCOUNT</p>
            <h1>Admin access required</h1>
            <p className="account-lead">
              Developer tools are only available to admin accounts.
            </p>
          </div>
        </header>
        <section className="panel account-card">
          <p className="account-feedback error">
            Your current account does not have permission to view this page.
          </p>
          <NavLink to="/about" className="auth-primary-btn">
            Back to introduction
          </NavLink>
        </section>
      </section>
    );
  }

  return children;
}

function RequireGuest({ currentUser, authChecking, children }) {
  const routeLocation = useLocation();
  const fromPath = routeLocation.state?.from?.pathname || "/about";

  if (authChecking) {
    return (
      <section className="account-page">
        <section className="panel account-card">
          <p className="account-lead">Checking account access...</p>
        </section>
      </section>
    );
  }

  if (currentUser) {
    return <Navigate to={fromPath} replace />;
  }

  return children;
}

export default function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [serverBStatus, setServerBStatus] = useState(() => createCheckingStatus());
  const [serverBLogs, setServerBLogs] = useState([]);
  const [serverBChecking, setServerBChecking] = useState(false);
  const serverBCheckInFlightRef = useRef(false);
  const serverBPollStartedRef = useRef(false);
  const accountName = currentUser?.username || "";
  const isLoggedIn = Boolean(accountName);
  const isAdmin = currentUser?.role === "admin";

  const [devDropdownOpen, setDevDropdownOpen] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();
  const authReturnLocation = location.pathname === "/login" || location.pathname === "/register"
    ? { pathname: "/about" }
    : location;

  useEffect(() => {
    const path = location.pathname;
    document.body.classList.remove("theme-analysis", "theme-generation");
    if (path === "/generation" || path === "/dev/analysis") {
      document.body.classList.add("theme-analysis");
    } else if (path === "/analysis" || path === "/dev/layers") {
      document.body.classList.add("theme-generation");
    }
    return () => {
      document.body.classList.remove("theme-analysis", "theme-generation");
    };
  }, [location.pathname]);

  const handleSessionExpired = useCallback(() => {
    setCurrentUser(null);
    if (isAdminRoute(location.pathname)) {
      navigate("/login", { replace: true, state: { from: location } });
    }
  }, [location, navigate]);

  const refreshCurrentUser = useCallback(({ showChecking = false } = {}) => {
    let cancelled = false;

    async function restoreSession() {
      if (showChecking) {
        setAuthChecking(true);
      }

      try {
        const data = await getCurrentUser();
        if (!cancelled) {
          setCurrentUser(data.user || null);
        }
      } catch {
        if (!cancelled) {
          handleSessionExpired();
        }
      } finally {
        if (!cancelled && showChecking) {
          setAuthChecking(false);
        }
      }
    }

    restoreSession();
    return () => {
      cancelled = true;
    };
  }, [handleSessionExpired]);

  useEffect(() => {
    return refreshCurrentUser({ showChecking: true });
  }, [refreshCurrentUser]);

  useEffect(() => {
    window.addEventListener(authExpiredEventName, handleSessionExpired);
    return () => window.removeEventListener(authExpiredEventName, handleSessionExpired);
  }, [handleSessionExpired]);

  useEffect(() => {
    if (!currentUser) {
      return undefined;
    }

    function handleFocus() {
      refreshCurrentUser();
    }

    window.addEventListener("focus", handleFocus);
    const timer = window.setInterval(() => {
      refreshCurrentUser();
    }, sessionRefreshMs);

    return () => {
      window.removeEventListener("focus", handleFocus);
      window.clearInterval(timer);
    };
  }, [currentUser, refreshCurrentUser]);

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

  function handleAuthenticate(user) {
    setCurrentUser(user || null);
  }

  async function handleLogout() {
    try {
      await logoutAccount();
    } finally {
      setCurrentUser(null);
      if (isAdminRoute(location.pathname)) {
        navigate("/about", { replace: true });
      }
    }
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
              <NavLink to="/how-it-works" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">⌥</span>
                <span>How it works</span>
              </NavLink>
              <NavLink to="/analysis" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">◫</span>
                <span>Generation</span>
              </NavLink>
              <NavLink to="/generation" className={navbarLinkClass} onClick={() => setMobileMenuOpen(false)}>
                <span className="nav-icon">✦</span>
                <span>Analysis</span>
              </NavLink>
            </div>

            <div className="navbar-controls">
              {/* Dev Tools Dropdown */}
              {isAdmin ? (
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
                    <NavLink to="/immersive" className={navbarDropdownItemClass} onClick={() => { setDevDropdownOpen(false); setMobileMenuOpen(false); }}>
                      <span className="nav-icon">❂</span>
                      <span>Immersive</span>
                    </NavLink>
                    <div className="dropdown-divider"></div>
                    <div className="navbar-dropdown-status">
                      <ServerStatus status={serverBStatus} />
                    </div>
                  </div>
                </div>
              ) : null}

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
                  <span>{authChecking ? "Checking..." : isLoggedIn ? accountName : "Account"}</span>
                  <span className="dropdown-arrow">▼</span>
                </button>
                <div className="navbar-dropdown-menu user-dropdown">
                  <div className="navbar-dropdown-header">
                    {authChecking ? "Checking Account" : isLoggedIn ? "Account Logged In" : "Not Logged In"}
                  </div>
                  {authChecking ? (
                    <div className="navbar-user-simple">
                      <span className="navbar-username">Restoring session...</span>
                    </div>
                  ) : isLoggedIn ? (
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
                        state={{ from: authReturnLocation }}
                        className={navbarActionClass} 
                        onClick={() => { setUserDropdownOpen(false); setMobileMenuOpen(false); }}
                      >
                        Login
                      </NavLink>
                      <NavLink 
                        to="/register" 
                        state={{ from: authReturnLocation }}
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
          <Route path="/how-it-works" element={<PipelinePage />} />
          <Route path="/analysis" element={<GenerationPage />} />
          <Route path="/generation" element={<HomePage />} />
          {/* Transformation is out of scope for the demo — redirect any stale links. */}
          <Route path="/transformation" element={<Navigate to="/about" replace />} />
          <Route path="/immersive" element={<ImmersivePage />} />
          <Route
            path="/dev/layers"
            element={
              <RequireAdmin currentUser={currentUser} authChecking={authChecking}>
                <LayerATestPage
                  mode="generation"
                  eyebrow="DEVELOPER TOOLS — GENERATION"
                  title="Generation Layers Dev Test"
                />
              </RequireAdmin>
            }
          />
          <Route
            path="/dev/analysis"
            element={
              <RequireAdmin currentUser={currentUser} authChecking={authChecking}>
                <DevAnalysisPage />
              </RequireAdmin>
            }
          />
          <Route
            path="/server-b"
            element={
              <RequireAdmin currentUser={currentUser} authChecking={authChecking}>
                <ServerBStatusPage
                  status={serverBStatus}
                  logs={serverBLogs}
                  checking={serverBChecking}
                  onRecheck={() => runServerBCheck("manual")}
                />
              </RequireAdmin>
            }
          />
          <Route
            path="/login"
            element={
              <RequireGuest currentUser={currentUser} authChecking={authChecking}>
                <LoginPage accountName={accountName} onLogin={handleAuthenticate} />
              </RequireGuest>
            }
          />
          <Route
            path="/register"
            element={
              <RequireGuest currentUser={currentUser} authChecking={authChecking}>
                <RegisterPage accountName={accountName} onRegister={handleAuthenticate} />
              </RequireGuest>
            }
          />
        </Routes>
      </main>
    </div>
  );
}
