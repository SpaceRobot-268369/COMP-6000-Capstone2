import { useEffect, useRef, useState } from "react";
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
  const sceneRef = useRef(null);
  const boltRef = useRef(null);
  const titleWordsRef = useRef(null);
  const titleScrimRef = useRef(null);
  const audioRef = useRef(null);
  const [api, setApi] = useState(null);

  useEffect(() => {
    const instance = createImmersive({
      sceneEl: sceneRef.current,
      boltEl: boltRef.current,
      titleWordsEl: titleWordsRef.current,
      titleScrimEl: titleScrimRef.current,
      audioEl: audioRef.current,
      initial,
    });
    setApi(instance);
    return () => {
      setApi(null);
      instance.dispose();
    };
    // initial is read once at mount; remounting for a new scene is the caller's job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={`immersive-page${initial?.narration ? " narration" : ""}`}>
      <div className="immersive-scene" ref={sceneRef} />
      <div className="immersive-scrim" ref={titleScrimRef} />
      <div className="immersive-title">
        <div className="immersive-title-words" ref={titleWordsRef} />
      </div>
      <canvas className="immersive-bolt" ref={boltRef} />

      {showDevPanel && api && <ImmersiveControls api={api} />}
      {overlay}

      <audio ref={audioRef} loop crossOrigin="anonymous" />
    </div>
  );
}
