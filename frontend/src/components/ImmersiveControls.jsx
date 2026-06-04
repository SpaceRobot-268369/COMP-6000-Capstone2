import { useState } from "react";
import { SEASONS, TIMES, SEASON_LABEL, TIME_LABEL } from "../immersive/scenes.js";

/* Dev preview panel for the immersive scene — the React port of the artifact's
   `initUI`. Drives the engine through its control `api`. Mounted only when the
   SHOW_DEV_PANEL flag in ImmersivePage is on; flip that off to ship the scene
   without controls. */
export default function ImmersiveControls({ api }) {
  const [season, setSeason] = useState("autumn");
  const [time, setTime] = useState("dawn");
  const [rain, setRain] = useState(false);
  const [rainAmount, setRainAmount] = useState(0.6);
  const [playing, setPlaying] = useState(false);
  const [fileName, setFileName] = useState("no file — controls mock playback");
  const [collapsed, setCollapsed] = useState(false);

  function pickSeason(s) {
    setSeason(s);
    api.setSeason(s);
  }
  function pickTime(t) {
    setTime(t);
    api.setTime(t);
  }
  function toggleRain() {
    const next = !rain;
    setRain(next);
    api.setRain(next);
  }
  function changeRainAmount(v) {
    setRainAmount(v);
    api.setRainAmount(v);
  }
  function togglePlay() {
    setPlaying(api.togglePlay());
  }
  function loadFile(event) {
    const file = event.target.files[0];
    const name = api.loadFile(file);
    if (name) setFileName(name);
  }

  return (
    <aside className={`immersive-panel${collapsed ? " collapsed" : ""}`}>
      <div className="phead">
        <h2>Scene · Preview</h2>
        <button
          type="button"
          className="panel-toggle"
          title="collapse"
          onClick={() => setCollapsed((c) => !c)}
        >
          {collapsed ? "⟨" : "⟩"}
        </button>
      </div>
      <div className="pbody">
        <div className="grp">
          <span className="lbl">Season</span>
          <div className="segrow">
            {SEASONS.map((s) => (
              <button
                key={s}
                type="button"
                className={`seg${season === s ? " active" : ""}`}
                onClick={() => pickSeason(s)}
              >
                {SEASON_LABEL[s]}
              </button>
            ))}
          </div>
        </div>

        <div className="grp">
          <span className="lbl">Time of day</span>
          <div className="segrow">
            {TIMES.map((t) => (
              <button
                key={t}
                type="button"
                className={`seg${time === t ? " active" : ""}`}
                onClick={() => pickTime(t)}
              >
                {TIME_LABEL[t]}
              </button>
            ))}
          </div>
        </div>

        <div className="grp">
          <span className="lbl">Weather overlay</span>
          <div className="row2">
            <button type="button" className={`btn${rain ? " on" : ""}`} onClick={toggleRain}>
              {rain ? "Rain · on" : "Rain · off"}
            </button>
            <button type="button" className="btn" onClick={() => api.thunder()}>
              Thunder ⚡
            </button>
          </div>
          <input
            type="range"
            min="0.15"
            max="1"
            step="0.05"
            value={rainAmount}
            onChange={(e) => changeRainAmount(parseFloat(e.target.value))}
          />
        </div>

        <div className="grp">
          <span className="lbl">Title sequence</span>
          <button type="button" className="btn full" onClick={() => api.replayTitle()}>
            Replay analysis text
          </button>
        </div>

        <div className="grp">
          <span className="lbl">Audio</span>
          <div className="row2">
            <button type="button" className="btn" onClick={togglePlay}>
              {playing ? "Pause" : "Play"}
            </button>
            <label className="btn file-btn">
              Load file
              <input type="file" accept="audio/*" onChange={loadFile} />
            </label>
          </div>
          <span className="file-name">{fileName}</span>
        </div>

        <p className="hint">Dev preview panel. Weather composes over any of the 16 scenes.</p>
      </div>
    </aside>
  );
}
