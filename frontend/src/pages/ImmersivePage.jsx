import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { createImmersive } from "../immersive/engine.js";
import ImmersiveControls from "../components/ImmersiveControls.jsx";
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

  function handleReset() {
    const performReset = () => {
      navigate(location.state?.backPath || "/generation");
    };

    if (document.startViewTransition) {
      document.startViewTransition(performReset);
    } else {
      performReset();
    }
  }

  function handleTogglePlay() {
    if (api) {
      setPlaying(api.togglePlay());
    }
  }

  const activeOverlay = overlay || (isFromDemo ? (
    <>
      <button type="button" className="demo-reset" onClick={handleReset}>
        ↺ New scene
      </button>
      {activeInitial?.audioUrl && (
        <button type="button" className="demo-audio-toggle" onClick={handleTogglePlay}>
          {playing ? "⏸ Pause Audio" : "▶ Play Audio"}
        </button>
      )}
    </>
  ) : null);

  return (
    <div className="immersive-page">
      <div className="immersive-scene" ref={sceneRef} />
      <div className="immersive-scrim" ref={titleScrimRef} />
      <div className="immersive-title">
        <div className="immersive-title-words" ref={titleWordsRef} />
      </div>
      <canvas className="immersive-bolt" ref={boltRef} />

      {activeShowDevPanel && api && <ImmersiveControls api={api} />}
      {activeOverlay}

      <audio ref={audioRef} src={activeInitial?.audioUrl} loop crossOrigin="anonymous" />
    </div>
  );
}

