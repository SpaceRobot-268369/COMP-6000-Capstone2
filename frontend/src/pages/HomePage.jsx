import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { resolvePrompt } from "../demo/resolvePrompt.js";
import { composeNarration } from "../demo/composeNarration.js";
import { ambientForCell } from "../demo/sampleCatalog.js";

/* The presets are real Bowra dry-woodland recordings (Layer A `expected/` bank),
   served by the backend straight from the repo checkout. Each card analyses a
   genuine site recording for its season×diel cell — no placeholder audio. The
   `audioUrl` + `sourceCaption` come from the sample catalog so the immersive
   scene plays the real clip and shows where it came from. */
function buildPreset({ title, tags, season, time, events }) {
  const sample = ambientForCell(season, time);
  return {
    title,
    tags,
    audioUrl: sample.audioUrl,
    sourceCaption: sample.sourceCaption,
    resolved: { season, time, rain: false, rainAmount: 0, thunder: false, events },
  };
}

const PRESET_SAMPLES = [
  buildPreset({
    title: "Autumn Dawn",
    tags: "Mild Air • Dry Woodland • Dawn Chorus",
    season: "autumn",
    time: "dawn",
    events: ["birdsong"],
  }),
  buildPreset({
    title: "Summer Night",
    tags: "Warm Night • Dry Air • Insects",
    season: "summer",
    time: "night",
    events: ["insects", "crickets"],
  }),
  buildPreset({
    title: "Spring Afternoon",
    tags: "Warm Light • Light Breeze • Birdsong",
    season: "spring",
    time: "afternoon",
    events: ["birdsong"],
  }),
  buildPreset({
    title: "Winter Morning",
    tags: "Cold Air • Bare Branches • Sparse Calls",
    season: "winter",
    time: "morning",
    events: ["birdsong"],
  }),
];

const STAGES = [
  "Reading audio signal...",
  "Extracting 256-d acoustic embeddings...",
  "Mapping environmental season & diel cues...",
  "Detecting precipitation & event parameters...",
  "Synthesizing immersive 3D woodland scene...",
];
const STAGE_MS = 650;

export default function HomePage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const previewAudioRef = useRef(null);
  const timersRef = useRef([]);

  const [phase, setPhase] = useState("idle"); // idle | analyzing
  const [stageIndex, setStageIndex] = useState(0);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);

  // Audio preview controls
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewPlaying, setPreviewPlaying] = useState(false);

  useEffect(() => {
    return () => {
      timersRef.current.forEach(clearTimeout);
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
      }
    };
  }, []);

  // Update preview element when URL changes
  useEffect(() => {
    if (previewUrl && previewAudioRef.current) {
      previewAudioRef.current.src = previewUrl;
      previewAudioRef.current.load();
      previewAudioRef.current
        .play()
        .then(() => setPreviewPlaying(true))
        .catch((err) => {
          console.log("Audio preview play failed", err);
          setPreviewPlaying(false);
        });
    }
  }, [previewUrl]);

  function handleTogglePreview(url, e) {
    e.stopPropagation(); // Avoid triggering card analysis
    if (previewUrl === url) {
      if (previewPlaying) {
        previewAudioRef.current.pause();
        setPreviewPlaying(false);
      } else {
        previewAudioRef.current
          .play()
          .then(() => setPreviewPlaying(true))
          .catch(() => setPreviewPlaying(false));
      }
    } else {
      setPreviewUrl(url);
    }
  }

  function handleSelectPreset(preset) {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      setPreviewPlaying(false);
    }

    const narration = composeNarration(preset.resolved);
    const resolvedState = {
      ...preset.resolved,
      narration,
      audioUrl: preset.audioUrl,
      sourceCaption: preset.sourceCaption,
    };

    startAnalysis(resolvedState);
  }

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("audio/")) {
      setFile(f);
    }
  }

  function handleAnalyzeUploadedFile() {
    if (!file) return;

    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      setPreviewPlaying(false);
    }

    const audioUrl = URL.createObjectURL(file);
    // Parse the file name for environmental keywords
    const params = resolvePrompt(file.name);
    const narration = composeNarration(params);
    const resolvedState = {
      ...params,
      narration,
      audioUrl,
    };

    startAnalysis(resolvedState);
  }

  function startAnalysis(resolvedState) {
    setStageIndex(0);
    setPhase("analyzing");

    // Clear any active timers
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];

    // Cycle through analysis steps
    STAGES.forEach((_, i) => {
      if (i === 0) return;
      timersRef.current.push(
        setTimeout(() => setStageIndex(i), STAGE_MS * i)
      );
    });

    // Navigate to immersive scene once finished
    timersRef.current.push(
      setTimeout(() => {
        const performNavigation = () => {
          navigate("/immersive", {
            state: { resolved: resolvedState, fromDemo: true, backPath: "/generation" },
          });
        };

        if (document.startViewTransition) {
          document.startViewTransition(performNavigation);
        } else {
          performNavigation();
        }
      }, STAGE_MS * STAGES.length + 300)
    );
  }

  const isAnalyzing = phase === "analyzing";

  return (
    <div className={`demo-chat theme-analysis${isAnalyzing ? " generating" : ""}`}>
      <div className="demo-chat-inner">
        <header className="demo-chat-head">
          <p className="demo-eyebrow">ACOUSTIC MAPPING</p>
          <h1>Upload a soundscape to map its environment</h1>
          <p className="demo-sub">
            We will parse your recording's texture, season, weather, and active species, placing you inside its virtual 3D woodland reconstruction.
          </p>
        </header>

        {isAnalyzing ? (
          <div className="demo-transcript" aria-live="polite">
            <div className="demo-bubble assistant thinking">
              <span className="demo-dots" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
              <span className="demo-status">{STAGES[stageIndex]}</span>
            </div>
          </div>
        ) : (
          <>
            {/* Audio Presets Grid */}
            <div className="analysis-presets-container">
              <p className="analysis-presets-label">Select a Preset Nature Recording</p>
              <div className="analysis-presets-grid">
                {PRESET_SAMPLES.map((preset) => {
                  const isCurrentPlaying =
                    previewUrl === preset.audioUrl && previewPlaying;
                  return (
                    <div
                      key={preset.title}
                      className="analysis-preset-card"
                      onClick={() => handleSelectPreset(preset)}
                    >
                      <div>
                        <h3 className="analysis-card-title">{preset.title}</h3>
                        <p className="analysis-card-tags">{preset.tags}</p>
                      </div>
                      <div className="analysis-card-footer">
                        <button
                          type="button"
                          className="analysis-card-play-btn"
                          onClick={(e) => handleTogglePreview(preset.audioUrl, e)}
                        >
                          {isCurrentPlaying ? "⏸ Pause Preview" : "▶ Play Preview"}
                        </button>
                        <button type="button" className="analysis-card-action-btn">
                          Analyze
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Custom File Upload Card */}
            <div className="analysis-upload-container">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />

              {!file ? (
                <div
                  className={`analysis-upload-zone${dragging ? " drag-over" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <span className="analysis-upload-icon">⇪</span>
                  <h3 className="analysis-upload-title">Drop your recording here</h3>
                  <p className="analysis-upload-sub">WAV, FLAC, MP3 • CLICK OR DRAG</p>
                </div>
              ) : (
                <div className="analysis-upload-zone" style={{ cursor: "default" }}>
                  <span className="analysis-upload-icon">◫</span>
                  <div className="analysis-upload-file-info">
                    <h3 className="analysis-upload-file-name">{file.name}</h3>
                    <p className="analysis-upload-file-size">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <div className="analysis-upload-actions">
                    <button
                      type="button"
                      className="analysis-card-action-btn"
                      onClick={handleAnalyzeUploadedFile}
                    >
                      ✦ Analyze Recording
                    </button>
                    <button
                      type="button"
                      className="analysis-card-play-btn"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Change File
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Hidden audio element for previewing preset tracks */}
      <audio
        ref={previewAudioRef}
        onEnded={() => setPreviewPlaying(false)}
        crossOrigin="anonymous"
      />
    </div>
  );
}
