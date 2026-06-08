import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PromptChat from "../components/PromptChat.jsx";
import {
  generateSoundscape,
  parseGenerationPrompt,
} from "../lib/api.js";
import { resolvePrompt } from "../demo/resolvePrompt.js";
import { composeNarration } from "../demo/composeNarration.js";
import { ambientForCell } from "../demo/sampleCatalog.js";

const USER_GENERATION_LAYER_D_ATTEMPT = "songke__mvp_2__multi_clip_mix";

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

function layerDSourceCaption({ metadata, parseResult }) {
  const attempts = metadata?.orchestration?.attempts || {};
  const attemptSummary = [
    attempts.layer_a ? `A ${attempts.layer_a}` : "",
    attempts.layer_b ? `B ${attempts.layer_b}` : "",
    attempts.layer_c ? `C ${attempts.layer_c}` : "",
    attempts.layer_d ? `D ${attempts.layer_d}` : "",
  ].filter(Boolean).join(", ");
  const note = parseResult?.note;
  return [
    "Layer D final mix",
    attemptSummary ? `attempts: ${attemptSummary}` : "",
    note ? `parser note: ${note}` : "",
  ].filter(Boolean).join(" - ");
}

function composeGenerationNarration(resolved) {
  return composeNarration(resolved).replace(
    " This is what the recording remembers.",
    " This is the layered soundscape the system composes for the scene.",
  );
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
  onStatus("Choosing the Layer A/B/C contracts...");
  const layerA = normalizeLayerA(parseResult, localParams);
  const layerB = parseResult?.layer_b || null;
  const layerC = parseResult?.layer_c || null;

  onStatus("Generating A/B/C stems...");
  const genTimer1 = registerTimer(
    setTimeout(() => {
      onStatus("Generating A/B/C stems (running model inference, this takes time)...");
    }, 3500)
  );
  const genTimer2 = registerTimer(
    setTimeout(() => {
      onStatus("Mixing the final soundscape through Layer D...");
    }, 9000)
  );
  const genTimer3 = registerTimer(
    setTimeout(() => {
      onStatus("Almost done, finalizing the audio render...");
    }, 18000)
  );

  const randomSeed = Math.floor(Math.random() * 2147483648);
  let generation;
  try {
    generation = await generateSoundscape({
      seed: randomSeed,
      duration_s: Number(layerB?.duration_s) || 30,
      season: layerA.season,
      diel: layerA.time,
      layer_a: { season: layerA.season, diel: layerA.time },
      layer_b: layerB,
      layer_c: layerC,
      include_weather: Boolean(layerB),
      include_events: Boolean(layerC?.species?.length),
      layer_d_attempt: USER_GENERATION_LAYER_D_ATTEMPT,
    });
  } finally {
    clearTimeout(genTimer1);
    clearTimeout(genTimer2);
    clearTimeout(genTimer3);
  }

  onStatus("Preparing the immersive scene...");
  const audioUrl = generatedWavUrl(generation.audio_b64);
  if (!audioUrl) {
    throw new Error("Layer D generation returned no audio.");
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
    resolvedPrompt: summarizeParsedScene(parseResult) || `${layerA.season} ${layerA.time} Layer D mix`,
    sourceCaption: layerDSourceCaption({
      metadata: generation.metadata,
      parseResult,
    }),
    generation: {
      mode: "abc_layer_d_mix",
      layer: "layer_d",
      parser: parseResult,
      metadata: generation.metadata,
      sampleRate: generation.sample_rate,
      durationS: generation.duration_s,
      attempts: generation.metadata?.orchestration?.attempts,
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
   1. User types a prompt -> submit
   2. Prompt is parsed by the LLM parser
   3a. If "ok": proceed directly to generation
   3b. If "corrected": show the parser's note and let user confirm or cancel
   3c. If "rejected": show the rejection message and return to prompt
   4. After confirmation -> generate A/B/C stems and Layer D mix -> navigate to immersive scene */
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
    setStatusText("Reading your scene with the prompt parser...");
    setPhase("parsing");

    clearTimers();

    const parseTimer = registerTimer(
      setTimeout(() => {
        setStatusText("Reading your scene with the prompt parser (interpreting description)...");
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

    // status === "ok" -> proceed directly to generation
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
    setStatusText("Choosing the Layer A/B/C contracts...");
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
      const message = err.message || "Layer D generation failed.";
      setStatusText("Generation encountered an issue. Loading fallback scene...");
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
