import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import AboutPage from "./pages/AboutPage.jsx";
import TransformationPage from "./pages/TransformationPage.jsx";
import GenerationPage from "./pages/GenerationPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";

const accountStorageKey = "sonic-lab-account-name";

function sidebarLinkClass({ isActive }) {
  return `nav-item${isActive ? " active" : ""}`;
}

function sidebarActionClass({ isActive }) {
  return `sidebar-action${isActive ? " active" : ""}`;
}

export default function App() {
  const [accountName, setAccountName] = useState("");
  const isLoggedIn = Boolean(accountName);

  useEffect(() => {
    const storedAccountName = window.localStorage.getItem(accountStorageKey);
    if (storedAccountName) {
      setAccountName(storedAccountName);
    }
  }, []);

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
            <NavLink to="/transformation" className={sidebarLinkClass}>
              <span className="nav-icon">≋</span>
              <span>Transformation</span>
            </NavLink>
          </nav>
        </div>

        <ThemeToggle />

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
          <Route path="/transformation" element={<TransformationPage />} />
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
