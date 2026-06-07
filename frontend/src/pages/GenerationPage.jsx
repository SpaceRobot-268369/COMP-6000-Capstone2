import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PromptChat from "../components/PromptChat.jsx";
import {
  fetchLayerRegistry,
  generateAttempt,
  parseGenerationPrompt,
} from "../lib/api.js";
import { resolvePrompt } from "../demo/resolvePrompt.js";
import { composeNarration } from "../demo/composeNarration.js";
import { ambientForCell } from "../demo/sampleCatalog.js";

// Staged "thinking" lines shown while the scene resolves, in order.
const GEN_STAGES = [
  "Reading your scene with the prompt parser…",
  "Choosing the Layer A season and light…",
  "Generating the ambient bed…",
  "Preparing the immersive scene…",
];
const STAGE_MS = 650;
const LAYER_A = "layer_a";
const DEFAULT_SEED = 42;

function generatedWavUrl(audioB64) {
  if (!audioB64) return "";
  const raw = window.atob(audioB64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i);
  }
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

function normalizeLayerA(parseResult, fallback) {
  const layerA = parseResult?.layer_a || {};
  return {
    season: layerA.season || fallback.season,
    time: layerA.diel || fallback.time,
  };
}

function layerASourceCaption({ metadata, parseResult }) {
  const cell = metadata?.cell || [metadata?.season, metadata?.diel].filter(Boolean).join("_");
  const prompt = metadata?.prompt;
  const note = parseResult?.note;
  return [
    "Layer A generated ambient bed",
    cell ? `cell ${cell}` : "",
    prompt ? `locked prompt: ${prompt}` : "",
    note ? `parser note: ${note}` : "",
  ].filter(Boolean).join(" · ");
}

function composeGenerationNarration(resolved) {
  return composeNarration(resolved).replace(
    " This is what the recording remembers.",
    " This is the ambient bed Layer A imagines for the scene.",
  );
}

async function defaultLayerAAttempt() {
  const registry = await fetchLayerRegistry();
  const layerA = registry.layers?.find((layer) => layer.id === LAYER_A);
  if (!layerA?.default) {
    throw new Error("Layer A default attempt is not available.");
  }
  return layerA.default;
}

async function resolveWithLayerA(text, localParams, onStatus, registerTimer) {
  onStatus("Reading your scene with the prompt parser…");
  
  const parseTimer = registerTimer(
    setTimeout(() => {
      onStatus("Reading your scene with the prompt parser (interpreting description)…");
    }, 2000)
  );

  let parseResult;
  try {
    parseResult = await parseGenerationPrompt(text);
  } finally {
    clearTimeout(parseTimer);
  }

  if (parseResult.status === "rejected") {
    throw new Error(parseResult.note || "This prompt is outside the current Bowra dry-woodland generation scope.");
  }

  onStatus("Choosing the Layer A season and light…");
  const layerA = normalizeLayerA(parseResult, localParams);
  const attemptId = await defaultLayerAAttempt();

  onStatus("Generating the ambient bed…");
  const genTimer1 = registerTimer(
    setTimeout(() => {
      onStatus("Generating the ambient bed (running model inference, this takes time)…");
    }, 3500)
  );
  const genTimer2 = registerTimer(
    setTimeout(() => {
      onStatus("Still generating the ambient bed, please wait…");
    }, 9000)
  );
  const genTimer3 = registerTimer(
    setTimeout(() => {
      onStatus("Almost done, finalizing the audio render…");
    }, 18000)
  );

  let generation;
  try {
    generation = await generateAttempt(LAYER_A, attemptId, {
      seed: DEFAULT_SEED,
      season: layerA.season,
      diel: layerA.time,
    });
  } finally {
    clearTimeout(genTimer1);
    clearTimeout(genTimer2);
    clearTimeout(genTimer3);
  }

  onStatus("Preparing the immersive scene…");
  const audioUrl = generatedWavUrl(generation.audio_b64);
  if (!audioUrl) {
    throw new Error("Layer A returned no audio.");
  }

  const resolved = {
    ...localParams,
    season: layerA.season,
    time: layerA.time,
  };
  return {
    ...resolved,
    narration: composeGenerationNarration(resolved),
    audioUrl,
    generatedAudio: true,
    resolvedPrompt: `${layerA.season} ${layerA.time} Layer A ambient bed`,
    sourceCaption: layerASourceCaption({
      metadata: generation.metadata,
      parseResult,
    }),
    generation: {
      mode: "layer_a_ambient_until_layer_d",
      layer: LAYER_A,
      attempt: attemptId,
      parser: parseResult,
      metadata: generation.metadata,
      sampleRate: generation.sample_rate,
      durationS: generation.duration_s,
    },
  };
}

function fallbackResolvedScene(text, localParams, reason) {
  const narration = composeGenerationNarration(localParams);
  const sample = ambientForCell(localParams.season, localParams.time);
  return {
    ...localParams,
    narration,
    ...sample,
    generation: {
      mode: "fallback_expected_layer_a_sample",
      reason,
    },
  };
}

/* The generation page flow: a chatbot-style prompt → a staged generating
   transition → navigating to the immersive woodland placed on the resolved scene,
   with a second-person narration as its centre text. */
export default function GenerationPage() {
  const navigate = useNavigate();
  const [phase, setPhase] = useState("prompt"); // prompt | generating
  const [userMessage, setUserMessage] = useState("");
  const [statusText, setStatusText] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const timersRef = useRef([]);

  function clearTimers() {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }
  useEffect(() => {
    return () => {
      clearTimers();
    };
  }, []);

  async function handleSubmit(text) {
    const params = resolvePrompt(text);
    setUserMessage(text);
    setErrorMsg("");
    setStatusText("Reading your scene with the prompt parser…");
    setPhase("generating");

    clearTimers();
    const registerTimer = (t) => {
      timersRef.current.push(t);
      return t;
    };

    try {
      const resolvedState = await resolveWithLayerA(text, params, setStatusText, registerTimer);
      registerTimer(
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
        }, 450)
      );
    } catch (err) {
      const message = err.message || "Layer A generation failed.";
      if (message.includes("outside") || message.includes("scope")) {
        clearTimers();
        setErrorMsg(message);
        setPhase("prompt");
        return;
      }

      setStatusText("Generation failed. Loading fallback woodland scene…");
      const resolvedState = fallbackResolvedScene(text, params, message);
      registerTimer(
        setTimeout(() => {
          navigate("/immersive", {
            state: { resolved: resolvedState, fromDemo: true, backPath: "/generation" },
          });
        }, 1500)
      );
    }
  }

  return (
    <PromptChat
      phase={phase}
      userMessage={userMessage}
      statusLine={statusText}
      errorMessage={errorMsg}
      onSubmit={handleSubmit}
    />
  );
}
