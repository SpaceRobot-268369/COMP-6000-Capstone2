import { useEffect, useState } from "react";

const OPTIONS = [
  { value: "auto",  label: "Auto",  icon: "◑" },
  { value: "light", label: "Light", icon: "☀" },
  { value: "dark",  label: "Dark",  icon: "☾" },
];

/**
 * Adaptive / Light / Dark theme toggle.
 *
 * "auto"  — JS syncs data-theme with prefers-color-scheme in real time
 * "light" — sets data-theme="light" explicitly
 * "dark"  — sets data-theme="dark"  explicitly
 *
 * CSS only needs [data-theme="light"] and [data-theme="dark"] overrides.
 * Preference is saved to localStorage.
 */
export default function ThemeToggle() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("sl-theme") || "auto"
  );

  useEffect(() => {
    const root = document.documentElement;
    localStorage.setItem("sl-theme", theme);

    if (theme !== "auto") {
      root.setAttribute("data-theme", theme);
      return;
    }

    // Auto mode: follow system preference and watch for changes
    const mq = window.matchMedia("(prefers-color-scheme: light)");

    function sync() {
      root.setAttribute("data-theme", mq.matches ? "light" : "dark");
    }

    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, [theme]);

  return (
    <div className="theme-toggle" role="group" aria-label="Colour theme">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          className={`theme-btn${theme === opt.value ? " active" : ""}`}
          onClick={() => setTheme(opt.value)}
          aria-label={`${opt.label} theme`}
          title={opt.label}
        >
          {opt.icon}
        </button>
      ))}
    </div>
  );
}
