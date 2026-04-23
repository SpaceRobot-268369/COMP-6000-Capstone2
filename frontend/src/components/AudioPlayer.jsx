import { useEffect, useRef, useState } from "react";

function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Shared audio player bar.
 *
 * Props:
 *   src    — blob URL or file URL to play (null = disabled state)
 *   label  — filename / description shown in the player
 *   detail — secondary line (e.g. "Generated soundscape")
 */
export default function AudioPlayer({ src, label = "No file loaded", detail = "" }) {
  const audioRef    = useRef(null);
  const [playing,  setPlaying]  = useState(false);
  const [current,  setCurrent]  = useState(0);
  const [duration, setDuration] = useState(0);

  // Reset when src changes
  useEffect(() => {
    setPlaying(false);
    setCurrent(0);
    setDuration(0);
  }, [src]);

  function toggle() {
    const a = audioRef.current;
    if (!a || !src) return;
    if (playing) { a.pause(); } else { a.play(); }
    setPlaying((p) => !p);
  }

  function onTimeUpdate() {
    setCurrent(audioRef.current?.currentTime ?? 0);
  }

  function onLoadedMetadata() {
    setDuration(audioRef.current?.duration ?? 0);
  }

  function onEnded() {
    setPlaying(false);
    setCurrent(0);
    if (audioRef.current) audioRef.current.currentTime = 0;
  }

  function onScrub(e) {
    const a = audioRef.current;
    if (!a || !duration) return;
    const t = (Number(e.target.value) / 100) * duration;
    a.currentTime = t;
    setCurrent(t);
  }

  const progress = duration > 0 ? (current / duration) * 100 : 0;

  return (
    <footer className="player-bar panel">
      {src && (
        <audio
          ref={audioRef}
          src={src}
          onTimeUpdate={onTimeUpdate}
          onLoadedMetadata={onLoadedMetadata}
          onEnded={onEnded}
        />
      )}

      <button
        type="button"
        className="play-button"
        aria-label={playing ? "Pause" : "Play"}
        onClick={toggle}
        disabled={!src}
        style={{ opacity: src ? 1 : 0.4, cursor: src ? "pointer" : "default" }}
      >
        {playing ? "⏸" : "▶"}
      </button>

      <div className="player-copy">
        <strong>{label}</strong>
        <p>{detail || `${formatTime(current)} / ${formatTime(duration)}`}</p>
      </div>

      <div className="player-scrub">
        <input
          type="range"
          min="0"
          max="100"
          step="0.1"
          value={progress}
          onChange={onScrub}
          disabled={!src}
          aria-label="Seek"
          className="scrub-range"
        />
        <div className="scrub-track">
          <div className="scrub-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="player-levels" aria-hidden="true">
        <span style={{ height: playing ? undefined : "4px" }} />
        <span style={{ height: playing ? undefined : "4px" }} />
        <span style={{ height: playing ? undefined : "4px" }} />
        <span style={{ height: playing ? undefined : "4px" }} />
      </div>
    </footer>
  );
}
