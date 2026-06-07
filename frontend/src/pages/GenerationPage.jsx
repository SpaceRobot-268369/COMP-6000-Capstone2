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

const LAYER_A = "layer_a";

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

/** Build a concise human-readable summary of the parsed scene from the layer contracts. */
function summarizeParsedScene(parseResult) {
  const parts = [];
  const la = parseResult?.layer_a;
  if (la?.season || la?.diel) {
    parts.push([la.season, la.diel].filter(Boolean).join(" "));
  }
  const lb = parseResult?.layer_b;
  if (lb) {
    const weatherDesc = [lb.intensity, lb.weather_type].filter(Boolean).join(" ");
    if (weatherDesc) parts.push(weatherDesc);
  }
  const lc = parseResult?.layer_c;
  if (lc?.species?.length) {
    parts.push(lc.species.join(", "));
  }
  return parts.length ? parts.join(", ") : null;
}

async function generateFromParsed(parseResult, localParams, onStatus, registerTimer) {
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

  const randomSeed = Math.floor(Math.random() * 2147483648);
  let generation;
  try {
    generation = await generateAttempt(LAYER_A, attemptId, {
      seed: randomSeed,
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

/* The generation page flow:
   1. User types a prompt → submit
   2. Prompt is parsed by the LLM parser
   3a. If "ok": proceed directly to generation
   3b. If "corrected": show the parser's note and let user confirm or cancel
   3c. If "rejected": show the rejection message and return to prompt
   4. After confirmation → generate Layer A audio → navigate to immersive scene */
export default function GenerationPage() {
  const navigate = useNavigate();
  // prompt | parsing | confirm | generating | rejected
  const [phase, setPhase] = useState("prompt");
  const [userMessage, setUserMessage] = useState("");
  const [statusText, setStatusText] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  // For the confirm phase: the parser result + local params to resume generation
  const [pendingParse, setPendingParse] = useState(null);
  const [confirmNote, setConfirmNote] = useState("");
  const [confirmSummary, setConfirmSummary] = useState("");
  // For rejection: the rejection note from the parser
  const [rejectionNote, setRejectionNote] = useState("");
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

  function registerTimer(t) {
    timersRef.current.push(t);
    return t;
  }

  async function handleSubmit(text) {
    const params = resolvePrompt(text);
    setUserMessage(text);
    setErrorMsg("");
    setRejectionNote("");
    setConfirmNote("");
    setConfirmSummary("");
    setPendingParse(null);
    setStatusText("Reading your scene with the prompt parser…");
    setPhase("parsing");

    clearTimers();

    const parseTimer = registerTimer(
      setTimeout(() => {
        setStatusText("Reading your scene with the prompt parser (interpreting description)…");
      }, 2000)
    );

    let parseResult;
    try {
      parseResult = await parseGenerationPrompt(text);
    } catch (err) {
      clearTimeout(parseTimer);
      setErrorMsg(err.message || "Prompt parsing failed.");
      setPhase("prompt");
      return;
    }
    clearTimeout(parseTimer);

    // Handle the three parser statuses
    if (parseResult.status === "rejected") {
      setRejectionNote(
        parseResult.note ||
        "This prompt is outside the current Bowra dry-woodland generation scope."
      );
      setPhase("rejected");
      return;
    }

    if (parseResult.status === "corrected") {
      // Show the correction and let the user confirm
      setConfirmNote(parseResult.note || "The parser adjusted your prompt to fit the site's soundscape.");
      setConfirmSummary(summarizeParsedScene(parseResult) || "");
      setPendingParse({ parseResult, params });
      setPhase("confirm");
      return;
    }

    // status === "ok" — proceed directly to generation
    await runGeneration(parseResult, params);
  }

  async function handleConfirm() {
    if (!pendingParse) return;
    const { parseResult, params } = pendingParse;
    setPendingParse(null);
    setConfirmNote("");
    setConfirmSummary("");
    await runGeneration(parseResult, params);
  }

  function handleCancel() {
    setPendingParse(null);
    setConfirmNote("");
    setConfirmSummary("");
    setPhase("prompt");
  }

  function handleDismissRejection() {
    setRejectionNote("");
    setPhase("prompt");
  }

  async function runGeneration(parseResult, params) {
    setStatusText("Choosing the Layer A season and light…");
    setPhase("generating");

    try {
      const resolvedState = await generateFromParsed(parseResult, params, setStatusText, registerTimer);
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
      setStatusText("Generation encountered an issue. Loading fallback scene…");
      const resolvedState = fallbackResolvedScene(userMessage, params, message);
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
      rejectionNote={rejectionNote}
      confirmNote={confirmNote}
      confirmSummary={confirmSummary}
      onSubmit={handleSubmit}
      onConfirm={handleConfirm}
      onCancel={handleCancel}
      onDismissRejection={handleDismissRejection}
    />
  );
}
