import { useEffect, useState } from "react";
import { fetchModelConfig, saveModelConfig } from "../lib/api.js";

const SLOTS = [
  { id: "layer_a", label: "Layer A (Ambient)", type: "generation", description: "Base ambient bed generated via AudioLDM2." },
  { id: "layer_b", label: "Layer B (Weather)", type: "generation", description: "Wind, rain, and storm synthesis or curation." },
  { id: "layer_c", label: "Layer C (Events)", type: "generation", description: "Fauna and specific event insertions via AudioGen." },
  { id: "layer_d", label: "Layer D (Mixer)", type: "generation", description: "Timeline placement, equalization, and final audio mixing." },
  { id: "layer_e_ambient", label: "Layer E: Ambient Detector", type: "analysis", description: "Infers ambient soundscape category, season, and diel distribution." },
  { id: "layer_e_weather", label: "Layer E: Weather Detector", type: "analysis", description: "Detects presence and intensity of wind, rain, and thunder." },
  { id: "layer_e_events", label: "Layer E: Event Detector", type: "analysis", description: "Identifies species calls and places them on the timeline." },
  { id: "layer_e_aggregator", label: "Layer E: Aggregator", type: "analysis", description: "Aggregates head detector findings and resolves conflicts." },
];

export default function SettingsPage() {
  const [layers, setLayers] = useState([]);
  const [slots, setSlots] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});

  useEffect(() => {
    fetchModelConfig()
      .then((data) => {
        setLayers(data.layers || []);

        // Populate initial slots state
        const initialSlots = {};
        for (const slot of SLOTS) {
          initialSlots[slot.id] = data.slots?.[slot.id] || "";
        }
        setSlots(initialSlots);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  function handleSlotChange(slotId, attemptId) {
    setSlots((prev) => ({ ...prev, [slotId]: attemptId }));
    setSuccess(false);
    setError("");
    setValidationErrors((prev) => ({ ...prev, [slotId]: null }));
  }

  function handleSave() {
    setSaving(true);
    setError("");
    setSuccess(false);
    setValidationErrors({});

    saveModelConfig(slots)
      .then(() => {
        setSuccess(true);
        setSaving(false);
      })
      .catch((err) => {
        if (err.errors) setValidationErrors(err.errors);
        setError(err.message);
        setSaving(false);
      });
  }

  function handleReset() {
    const cleared = {};
    for (const slot of SLOTS) {
      cleared[slot.id] = "";
    }
    setSlots(cleared);
    setSuccess(false);
    setError("");
    setValidationErrors({});
  }

  // Get attempts for a slot
  function getAttemptsForSlot(slotId) {
    if (slotId.startsWith("layer_e_")) {
      const head = slotId.slice("layer_e_".length);
      const layerE = layers.find((l) => l.id === "layer_e");
      if (!layerE) return [];
      return (layerE.attempts || []).filter((a) => a.head === head);
    } else {
      const layer = layers.find((l) => l.id === slotId);
      return layer ? layer.attempts || [] : [];
    }
  }

  // Resolve default attempt for a slot
  function getDefaultAttempt(slotId) {
    const attempts = getAttemptsForSlot(slotId);
    let defaultId = "";
    
    if (slotId.startsWith("layer_e_")) {
      const head = slotId.slice("layer_e_".length);
      const layerE = layers.find((l) => l.id === "layer_e");
      if (layerE) {
        const def = layerE.default;
        const defAttempt = (layerE.attempts || []).find((a) => a.id === def);
        if (defAttempt && defAttempt.head === head) {
          defaultId = def;
        } else {
          const firstMatching = (layerE.attempts || []).find((a) => a.head === head);
          defaultId = firstMatching ? firstMatching.id : "";
        }
      }
    } else {
      const layer = layers.find((l) => l.id === slotId);
      if (layer) defaultId = layer.default;
    }

    return attempts.find((a) => a.id === defaultId);
  }

  if (loading) {
    return (
      <div className="generation-page theme-analysis">
        <div style={{ padding: "40px", textAlign: "center", color: "var(--color-text-dim)" }}>
          <div className="upload-icon animate-pulse" style={{ fontSize: "2rem" }}>⌬</div>
          <p style={{ marginTop: "16px" }}>Loading active model configurations...</p>
        </div>
      </div>
    );
  }

  const generationSlots = SLOTS.filter((s) => s.type === "generation");
  const analysisSlots = SLOTS.filter((s) => s.type === "analysis");

  function renderSlot(slot) {
    const attempts = getAttemptsForSlot(slot.id);
    const defaultAttempt = getDefaultAttempt(slot.id);
    const currentVal = slots[slot.id];
    const valError = validationErrors[slot.id];

    return (
      <div key={slot.id} className="settings-slot">
        <div className="settings-slot-head">
          <span className="settings-slot-name">{slot.label}</span>
          {currentVal && (
            <span className="dev-badge" style={{ fontSize: "10px", background: "rgba(99, 102, 241, 0.15)", color: "#818cf8" }}>Overridden</span>
          )}
        </div>
        <p className="settings-slot-desc">{slot.description}</p>
        <select
          className={`layer-a-input ${valError ? "error" : ""}`}
          value={currentVal}
          onChange={(e) => handleSlotChange(slot.id, e.target.value)}
          style={{ border: valError ? "1px solid #ef4444" : "" }}
        >
          <option value="">
            {defaultAttempt
              ? `(default) — ${defaultAttempt.label} (${defaultAttempt.stage})`
              : "Use Registry Default"}
          </option>
          {attempts.map((a) => (
            <option key={a.id} value={a.id} disabled={!a.available}>
              {a.label} ({a.stage} · {a.status}){!a.available ? " — ✗ Weights missing" : ""}
            </option>
          ))}
        </select>
        {valError && <p className="analysis-error">{valError}</p>}
      </div>
    );
  }

  return (
    <section className="generation-page theme-analysis">
      <header className="generation-topbar">
        <div className="generation-brandline">
          <p className="eyebrow">SYSTEM SETTINGS</p>
          <span>Configure the active model attempts used across precision audio generation and analysis pipelines</span>
        </div>
      </header>

      <div className="settings-row">

        {/* The dropdowns are real (the choice is persisted and forwarded), but nothing
            downstream loads weights, and the mock reports every attempt as available —
            so the usual "✗ Weights missing" signal can never appear here. */}
        <p className="analysis-auth-notice">
          Demo build — every attempt lists as available because the mock reports it so. Nothing
          here loads weights.
        </p>

        {/* Status Messages */}
        {success && (
          <div className="panel settings-status" style={{ borderLeft: "4px solid #10b981", background: "rgba(16, 185, 129, 0.05)" }}>
            <strong style={{ color: "#10b981" }}>✓ Settings Saved Successfully</strong>
            <span>The active model configurations were saved to PostgreSQL. In this demo build the selection is recorded and echoed back with each result, but every layer still replays the same fixtures.</span>
          </div>
        )}

        {error && (
          <div className="panel settings-status" style={{ borderLeft: "4px solid #ef4444", background: "rgba(239, 68, 68, 0.05)" }}>
            <strong style={{ color: "#ef4444" }}>✗ Save Failed</strong>
            <span>{error}</span>
          </div>
        )}

        {/* Generation Slots Panel */}
        <main className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Soundscape Generation Slots</h2>
            <p>Define which generative attempt serves each layer of the speculative synthesis pipeline (A to D).</p>
          </div>
          <div className="settings-body">
            {generationSlots.map((slot) => renderSlot(slot))}
          </div>
        </main>

        {/* Analysis Slots Panel */}
        <main className="panel dev-result-card">
          <div className="generation-card-head">
            <h2>Soundscape Analysis Slots</h2>
            <p>Configure which Layer E detection heads and aggregation adapters perform ecological assessment tasks.</p>
          </div>
          <div className="settings-body">
            {analysisSlots.map((slot) => renderSlot(slot))}
          </div>
        </main>

        {/* Action Buttons */}
        <div className="settings-actions">
          <button
            type="button"
            className="navbar-logout-button"
            style={{ 
              background: "transparent", 
              border: "1px solid rgba(127,127,127,0.3)", 
              color: "var(--color-text)",
              cursor: "pointer",
              padding: "10px 20px",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: 500
            }}
            onClick={handleReset}
            disabled={saving}
          >
            Reset to Defaults
          </button>
          <button
            type="button"
            className="analyse-btn"
            style={{ 
              cursor: "pointer",
              padding: "10px 24px",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: "8px"
            }}
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving Configurations..." : "Save Active Configurations"}
          </button>
        </div>

      </div>
    </section>
  );
}
