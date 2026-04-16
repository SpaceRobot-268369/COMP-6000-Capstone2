import { NavLink, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import AboutPage from "./pages/AboutPage.jsx";
import TransformationPage from "./pages/TransformationPage.jsx";
import GenerationPage from "./pages/GenerationPage.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="brand-title">SONIC LAB</p>
          <p className="brand-subtitle">PRECISION AUDIO AI</p>
        </div>

        <nav className="sidebar-nav">
          <NavLink to="/about" className="nav-item">
            <span className="nav-icon">✦</span>
            <span>System</span>
          </NavLink>
          <NavLink to="/" className="nav-item">
            <span className="nav-icon">◫</span>
            <span>Analysis</span>
          </NavLink>
          <NavLink to="/generation" className="nav-item">
            <span className="nav-icon">✦</span>
            <span>Generation</span>
          </NavLink>
          <NavLink to="/transformation" className="nav-item">
            <span className="nav-icon">≋</span>
            <span>Transformation</span>
          </NavLink>
        </nav>
      </aside>

      <main className="main-panel">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/generation" element={<GenerationPage />} />
          <Route path="/transformation" element={<TransformationPage />} />
        </Routes>
      </main>
    </div>
  );
}
