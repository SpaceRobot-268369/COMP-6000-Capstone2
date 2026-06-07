import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { createImmersive } from "../immersive/engine.js";
import ImmersiveControls from "../components/ImmersiveControls.jsx";
import AudioPlayer from "../components/AudioPlayer.jsx";
import ToneToggle from "../components/ToneToggle.jsx";
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

  const isFromDemo = Boolean(location.state?.fromDemo);
  const activeInitial = location.state?.resolved || initial;
  const activeShowDevPanel = isFromDemo ? false : showDevPanel;

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

    // If there is an audioUrl, attempt autoplay once Three.js and WebAudio are bootstrapped
    if (activeInitial?.audioUrl && audioRef.current) {
      const playPromise = audioRef.current.play();
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
      {isFromDemo && (
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

  return (
    <div className="immersive-page">
      <div className="immersive-scene" ref={sceneRef} />
      <div className="immersive-scrim" ref={titleScrimRef} />
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

      <audio ref={audioRef} src={activeInitial?.audioUrl} loop crossOrigin="anonymous" />

      {audioSrc && (
        <AudioPlayer
          src={audioSrc}
          label={audioLabel}
          audioRef={audioRef}
        />
      )}

      {/* Top-center tone toggle (plan §3.5). Renders only when an analysis
          report is present in page state; switches the narrative register via
          the LLM-OSS report writer without re-running detectors. */}
      <ToneToggle
        report={activeInitial?.report}
        defaultRegister="immersive"
        initialText={activeInitial?.narration}
      />
    </div>
  );
}
