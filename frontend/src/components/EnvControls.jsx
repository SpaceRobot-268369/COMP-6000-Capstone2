/**
 * Environmental condition controls.
 * Field names match the training manifest columns exactly so values can be
 * sent to the model inference API without transformation.
 */

const TIME_BINS = ["dawn", "morning", "afternoon", "night"];
export const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

export function monthRangeForMonth(month) {
  const m = Number(month);
  if (m === 12 || m === 1 || m === 2) return "December-February";
  if (m >= 3 && m <= 5) return "March-May";
  if (m >= 6 && m <= 8) return "June-August";
  return "September-November";
}

export function monthLabel(month) {
  const m = Math.min(12, Math.max(1, Math.round(Number(month) || 1)));
  return MONTH_NAMES[m - 1];
}

export const DEFAULT_CONDITIONS = {
  temperature_c:             22,
  humidity_pct:              60,
  wind_speed_ms:             3,
  wind_direction_deg:        180,
  wind_max_ms:               5,
  precipitation_mm:          0,
  precipitation_daily_mm:    0,
  days_since_rain:           7,
  month:                     1,
  month_range:               "December-February",
  sample_bin:                "morning",
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

  function setMonth(month) {
    onChange({ ...value, month, month_range: monthRangeForMonth(month) });
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
        label="Wind Gust"
        display={`${value.wind_max_ms} m/s`}
        min={0} max={30} step={0.5}
        val={value.wind_max_ms}
        onChange={(v) => set("wind_max_ms", v)}
      />
      <Slider
        label="Wind Direction"
        display={`${value.wind_direction_deg}°`}
        min={0} max={359} step={1}
        val={value.wind_direction_deg}
        onChange={(v) => set("wind_direction_deg", v)}
      />
      <Slider
        label="Precipitation"
        display={`${value.precipitation_mm} mm`}
        min={0} max={80} step={0.5}
        val={value.precipitation_mm}
        onChange={(v) => set("precipitation_mm", v)}
      />
      <Slider
        label="Daily Rain"
        display={`${value.precipitation_daily_mm} mm`}
        min={0} max={120} step={0.5}
        val={value.precipitation_daily_mm}
        onChange={(v) => set("precipitation_daily_mm", v)}
      />
      <Slider
        label="Days Since Rain"
        display={`${value.days_since_rain} d`}
        min={0} max={30} step={1}
        val={value.days_since_rain}
        onChange={(v) => set("days_since_rain", v)}
      />

      <div className="gen-info-block">
        <p>Month</p>
        <Slider
          label="Month"
          display={monthLabel(value.month)}
          min={1} max={12} step={1}
          val={value.month ?? 1}
          onChange={setMonth}
        />
      </div>

      <div className="gen-info-block">
        <p>Time of Day</p>
        <div className="time-bin-grid">
          {TIME_BINS.map((b) => (
            <button
              key={b}
              type="button"
              className={`time-bin-pill${value.sample_bin === b ? " active" : ""}`}
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
      <div className="env-slider-wrap" style={{ "--thumb-pct": `${pct}%` }}>
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
