/**
 * Environmental condition controls.
 * Field names match the training manifest columns exactly so values can be
 * sent to the model inference API without transformation.
 */

const SEASONS   = ["summer", "autumn", "winter", "spring"];
const TIME_BINS = ["dawn", "morning", "afternoon", "night"];

export const DEFAULT_CONDITIONS = {
  temperature_c:    22,
  humidity_pct:     60,
  wind_speed_ms:    3,
  precipitation_mm: 0,
  season:           "summer",
  sample_bin:       "morning",
};

/**
 * Props:
 *   value    — conditions object (see DEFAULT_CONDITIONS)
 *   onChange — (nextValue) => void
 */
export default function EnvControls({ value, onChange }) {
  function set(key, val) {
    onChange({ ...value, [key]: val });
  }

  return (
    <div className="env-controls">
      <Slider
        label="Temperature"
        display={`${value.temperature_c}°C`}
        min={-10} max={45} step={1}
        val={value.temperature_c}
        onChange={(v) => set("temperature_c", v)}
      />
      <Slider
        label="Humidity"
        display={`${value.humidity_pct}%`}
        min={0} max={100} step={1}
        val={value.humidity_pct}
        onChange={(v) => set("humidity_pct", v)}
      />
      <Slider
        label="Wind Speed"
        display={`${value.wind_speed_ms} m/s`}
        min={0} max={20} step={0.5}
        val={value.wind_speed_ms}
        onChange={(v) => set("wind_speed_ms", v)}
      />
      <Slider
        label="Precipitation"
        display={`${value.precipitation_mm} mm`}
        min={0} max={80} step={0.5}
        val={value.precipitation_mm}
        onChange={(v) => set("precipitation_mm", v)}
      />

      <div className="gen-info-block">
        <p>Season</p>
        <div className="season-grid">
          {SEASONS.map((s) => (
            <button
              key={s}
              type="button"
              className={`season-pill${value.season === s ? " active" : ""}`}
              onClick={() => set("season", s)}
            >
              {s[0].toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="gen-info-block">
        <p>Time of Day</p>
        <div className="season-grid">
          {TIME_BINS.map((b) => (
            <button
              key={b}
              type="button"
              className={`season-pill${value.sample_bin === b ? " active" : ""}`}
              onClick={() => set("sample_bin", b)}
            >
              {b[0].toUpperCase() + b.slice(1)}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Slider({ label, display, min, max, step, val, onChange }) {
  const pct = ((val - min) / (max - min)) * 100;

  return (
    <div className="gen-control">
      <div className="gen-control-head">
        <span>{label}</span>
        <strong>{display}</strong>
      </div>
      <div className="env-slider-wrap">
        <div className="env-slider-track">
          <div className="env-slider-fill" style={{ width: `${pct}%` }} />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={val}
          onChange={(e) => onChange(Number(e.target.value))}
          className="env-range"
          aria-label={label}
        />
      </div>
    </div>
  );
}
