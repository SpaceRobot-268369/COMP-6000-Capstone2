import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { createImmersive } from "../immersive/engine.js";
import ImmersiveControls from "../components/ImmersiveControls.jsx";
import AudioPlayer from "../components/AudioPlayer.jsx";
import { sceneStateFromAnalysis } from "../lib/analysisScene.js";
import "../immersive/immersive.css";

/* The eco-acoustic immersive experience screen — a procedural Three.js woodland
   re-lit across the 16-cell season×time mood matrix. The Three.js engine is
   imperative, so it's mounted into refs from a useEffect and torn down (GPU +
   WebAudio + rAF) on unmount via the returned dispose().

   Props:
     initial      — optional opening state { season, time, rain, rainAmount,
                    thunder, narration }; defaults to autumn/dawn (the /immersive
                    route). The demo flow passes the resolved prompt here.
     showDevPanel — render the season/time/weather dev control panel
                    (default true; the demo flow turns it off).
     overlay      — optional React node rendered above the scene (the demo flow
                    uses it for a "New scene" affordance). */
export default function ImmersivePage({ initial = null, showDevPanel = true, overlay = null }) {
  const location = useLocation();
  const navigate = useNavigate();

  const analysisInitial = sceneStateFromAnalysis(location.state);
  const isFromDemo = Boolean(location.state?.fromDemo);
  const isFromAnalysis = Boolean(location.state?.fromAnalysis);
  const activeInitial = location.state?.resolved || analysisInitial || initial;
  const activeShowDevPanel = isFromDemo || isFromAnalysis ? false : showDevPanel;

  const sceneRef = useRef(null);
  const boltRef = useRef(null);
  const titleWordsRef = useRef(null);
  const titleScrimRef = useRef(null);
  const audioRef = useRef(null);

  const [api, setApi] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [season, setSeason] = useState(() => activeInitial?.season || "autumn");
  const [time, setTime] = useState(() => activeInitial?.time || "dawn");
  const [rain, setRain] = useState(() => activeInitial?.rain || false);
  const [rainAmount, setRainAmount] = useState(() => activeInitial?.rainAmount || 0.6);
  const [wind, setWind] = useState(() => activeInitial?.wind ?? 0);
  const [audioSrc, setAudioSrc] = useState(() => activeInitial?.audioUrl || "");
  const [audioLabel, setAudioLabel] = useState(() => activeInitial?.resolvedPrompt || activeInitial?.prompt || "Soundscape");
  // Toggle for the supplementary metadata overlays (analytical narration +
  // source caption + models-used). The center caption always shows the
  // immersive register; this just reveals/hides the analytical detail.
  const [showDetails, setShowDetails] = useState(true);

  useEffect(() => {
    const instance = createImmersive({
      sceneEl: sceneRef.current,
      boltEl: boltRef.current,
      titleWordsEl: titleWordsRef.current,
      titleScrimEl: titleScrimRef.current,
      audioEl: audioRef.current,
      initial: activeInitial,
    });
    setApi(instance);

    // Ensure the audio element loads the source (blob URLs need an explicit
    // load() call when set before the element is connected to the DOM).
    const a = audioRef.current;
    if (a && activeInitial?.audioUrl) {
      a.src = activeInitial.audioUrl;
      a.load();
      const playPromise = a.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => setPlaying(true))
          .catch((e) => {
            console.log("Autoplay blocked by browser policy, awaiting interaction:", e);
          });
      }
    }

    return () => {
      setApi(null);
      instance.dispose();
      // Do NOT revoke blob URLs here — React StrictMode double-mounts effects
      // in dev, so the first cleanup would kill the URL before the second mount
      // can use it. The browser GCs blob URLs when the document unloads.
    };
    // activeInitial is read once at mount; remounting for a new scene is the caller's job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep internal play state synchronized with the actual audio element events
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;

    function handlePlay() {
      setPlaying(true);
    }
    function handlePause() {
      setPlaying(false);
    }

    a.addEventListener("play", handlePlay);
    a.addEventListener("pause", handlePause);
    return () => {
      a.removeEventListener("play", handlePlay);
      a.removeEventListener("pause", handlePause);
    };
  }, []);

  // Sync data-theme on document root with actual time of day
  useEffect(() => {
    const isLightMode = time === "morning" || time === "afternoon";
    const computedTheme = isLightMode ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", computedTheme);
  }, [time]);

  // Save the user's previous theme state on mount, restore it on unmount
  useEffect(() => {
    const savedThemeSetting = localStorage.getItem("sl-theme") || "auto";

    return () => {
      const root = document.documentElement;
      if (savedThemeSetting !== "auto") {
        root.setAttribute("data-theme", savedThemeSetting);
      } else {
        const mq = window.matchMedia("(prefers-color-scheme: light)");
        root.setAttribute("data-theme", mq.matches ? "light" : "dark");
      }
    };
  }, []);

  function handleReset() {
    const performReset = () => {
      navigate(location.state?.backPath || "/analysis");
    };

    if (document.startViewTransition) {
      document.startViewTransition(performReset);
    } else {
      performReset();
    }
  }

  function handleDownload() {
    if (audioSrc) {
      const link = document.createElement("a");
      link.href = audioSrc;
      let filename = "soundscape.wav";
      if (audioSrc.startsWith("blob:")) {
        filename = `${season}_${time}_soundscape.wav`;
      } else {
        filename = audioSrc.split("/").pop() || "soundscape.wav";
      }
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  const activeOverlay = overlay || (
    <>
      {(isFromDemo || isFromAnalysis) && (
        <button
          type="button"
          className={`demo-reset ${audioSrc ? "has-audio" : ""}`}
          onClick={handleReset}
        >
          ↺ New scene
        </button>
      )}
      {audioSrc && (
        <button type="button" className="demo-audio-download" onClick={handleDownload}>
          ⬇ Download Audio
        </button>
      )}
    </>
  );

  const analyticalText = activeInitial?.narratives?.analytical || "";
  const hasDetails = Boolean(
    analyticalText || activeInitial?.sourceCaption || activeInitial?.generation?.attempts
  );

  return (
    <div className="immersive-page">
      <div className="immersive-scene" ref={sceneRef} />
      <div className="immersive-scrim" ref={titleScrimRef} />

      {hasDetails && (
        <button
          type="button"
          className={`immersive-details-toggle ${showDetails ? "is-on" : ""}`}
          aria-pressed={showDetails}
          onClick={() => setShowDetails((v) => !v)}
        >
          {showDetails ? "Hide details" : "Show details"}
        </button>
      )}

      {showDetails && analyticalText && (
        <aside className="immersive-analytical" aria-label="Analytical narration">
          <span className="immersive-analytical-tag">Analytical</span>
          <p>{renderNarrative(analyticalText)}</p>
        </aside>
      )}

      {showDetails && (activeInitial?.sourceCaption || activeInitial?.generation?.attempts) && (
        <div className="immersive-meta-stack">
          {activeInitial?.sourceCaption && (
            <p className="immersive-source-caption">
              <span className="immersive-source-tag">
                {activeInitial.generatedAudio ? "Generated audio" : "Source recording"}
              </span>
              {activeInitial.sourceCaption}
            </p>
          )}
          {activeInitial?.generation?.attempts && (
            <GenerationModelLine attempts={activeInitial.generation.attempts} />
          )}
        </div>
      )}
      <div className="immersive-title">
        <div className="immersive-title-words" ref={titleWordsRef} />
      </div>
      <canvas className="immersive-bolt" ref={boltRef} />

      {activeShowDevPanel && api && (
        <ImmersiveControls
          api={api}
          season={season}
          setSeason={setSeason}
          time={time}
          setTime={setTime}
          rain={rain}
          setRain={setRain}
          rainAmount={rainAmount}
          setRainAmount={setRainAmount}
          wind={wind}
          setWind={setWind}
          setAudioSrc={setAudioSrc}
          setAudioLabel={setAudioLabel}
          playing={playing}
        />
      )}
      {activeOverlay}

      <audio ref={audioRef} src={audioSrc} loop crossOrigin={audioSrc && !audioSrc.startsWith("blob:") ? "anonymous" : undefined} />

      {audioSrc && (
        <AudioPlayer
          src={audioSrc}
          label={audioLabel}
          audioRef={audioRef}
        />
      )}
    </div>
  );
}

/* Render the report writer's light markdown (bold + blockquote markers) as React
   nodes for the analytical panel: strip blockquote/heading markers and turn
   **bold** into <strong>. */
function renderNarrative(md) {
  const clean = (md || "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*#{1,6}\s*/gm, "")
    .trim();
  return clean
    .split(/(\*\*[^*]+\*\*)/g)
    .filter(Boolean)
    .map((part, i) => {
      const m = part.match(/^\*\*([^*]+)\*\*$/);
      return m ? <strong key={i}>{m[1]}</strong> : <span key={i}>{part}</span>;
    });
}

function GenerationModelLine({ attempts }) {
  const rows = [
    ["A", "Ambient", attempts.layer_a],
    ["B", "Weather", attempts.layer_b],
    ["C", "Events", attempts.layer_c],
    ["D", "Mixer", attempts.layer_d],
  ].filter(([, , attempt]) => attempt);
  if (!rows.length) return null;
  return (
    <aside className="immersive-model-line" aria-label="Generation models used">
      <strong>Models used</strong>
      <div>
        {rows.map(([letter, label, attempt]) => (
          <span key={letter}>
            <b>Layer {letter}</b>
            <em>{label}</em>
            <code>{attempt}</code>
          </span>
        ))}
      </div>
    </aside>
  );
}
