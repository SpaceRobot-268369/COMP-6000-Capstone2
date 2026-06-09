import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PromptChat from "../components/PromptChat.jsx";
import { resolvePrompt } from "../demo/resolvePrompt.js";
import { composeNarration } from "../demo/composeNarration.js";
import { ambientForCell } from "../demo/sampleCatalog.js";
import { speciesForLayerC } from "../demo/layerCSpecies.js";
import { fetchModelConfig, generateSoundscape, parsePrompt } from "../lib/api.js";

// Decode a base64 WAV payload into an object URL the <audio> element can play.
function b64ToObjectUrl(b64, mime = "audio/wav") {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

// Map the resolved scene params onto the backend generation conditions
// (generateSoundscape derives weather type/intensity from wind + precipitation).
function conditionsFromParams(params) {
  return {
    season: params.season,
    sample_bin: params.time,
    duration_s: 30,
    wind_speed_ms: params.wind ? params.wind * 12 : 0,
    precipitation_mm: params.rain ? params.rainAmount * 25 : 0,
  };
}

// Staged "thinking" lines shown while the scene resolves, in order. The first
// few walk through the pipeline once; the trailing lines loop so the user keeps
// seeing live progress for as long as the server takes (no timeout).
const GEN_STAGES = [
  "Reading your scene…",
  "Choosing the season & light…",
  "Placing the weather…",
  "Composing the moment…",
  "Layering the voices…",
  "Mixing the soundscape…",
  "Still generating — almost there…",
];
// Once the lines run out, keep cycling the last LOOP_TAIL of them.
const LOOP_TAIL = 3;
const STAGE_MS = 900;

/* The generation page flow: a chatbot-style prompt → a staged generating
   transition → navigating to the immersive woodland placed on the resolved scene,
   with a second-person narration as its centre text. */
export default function GenerationPage() {
  const navigate = useNavigate();
  const [phase, setPhase] = useState("prompt"); // prompt | generating
  const [userMessage, setUserMessage] = useState("");
  const [stageIndex, setStageIndex] = useState(0);
  const [resolved, setResolved] = useState(null);
  // Parser feedback shown back on the prompt screen: a rejection (invalid scene)
  // or a correction note (something was swapped/dropped). { kind, text } | null.
  const [notice, setNotice] = useState(null);
  // Which Layer C model is active (from /dev/settings) decides which species the
  // user can choose — a model only voices the species it was built for. Default
  // to the broad retrieval library so the rail is populated even if serverB /
  // the model-config lookup is unavailable.
  const [layerCAttempt, setLayerCAttempt] = useState("");
  const timersRef = useRef([]);

  function clearTimers() {
    // Holds both timeout and interval ids; clearInterval clears either kind.
    timersRef.current.forEach(clearInterval);
    timersRef.current = [];
  }
  useEffect(() => clearTimers, []);

  useEffect(() => {
    let active = true;
    fetchModelConfig()
      .then((data) => {
        if (active) setLayerCAttempt(data.slots?.layer_c || "");
      })
      .catch(() => {
        // serverB asleep or not logged in — keep the fallback species list.
      });
    return () => {
      active = false;
    };
  }, []);

  const species = speciesForLayerC(layerCAttempt);

  async function handleSubmit(text) {
    setUserMessage(text);
    setStageIndex(0);
    setResolved(null);
    setNotice(null);
    setPhase("generating");

    // Gate the prompt through the LLM-OSS parser first (shown as the opening
    // "Reading your scene…" stage). An explicit `rejected` stops here with the
    // suggested alternative. A parser that's unavailable (serverB asleep) must
    // NOT block the demo, so parse errors are swallowed and we generate anyway.
    try {
      const parsed = await parsePrompt(text);
      if (parsed?.status === "rejected") {
        clearTimers();
        setNotice({
          kind: "rejected",
          text: parsed.note
            || "That scene isn't something this remote dry-woodland site can voice. Try a quieter, in-domain scene.",
        });
        setPhase("prompt");
        return;
      }
      if (parsed?.status === "corrected" && parsed.note) {
        setNotice({ kind: "corrected", text: parsed.note });
      }
    } catch {
      // Parser unavailable — proceed with generation unchanged.
    }

    const params = resolvePrompt(text);
    const narration = composeNarration(params);

    // Cycle the "thinking" lines while the real generation runs server-side.
    // Walk through the stages once, then loop the last few so the user always
    // sees live progress — generation takes as long as it takes (no timeout,
    // no frozen line). The await below is what actually drives navigation.
    clearTimers();
    let step = 0;
    timersRef.current.push(
      setInterval(() => {
        step += 1;
        const idx =
          step < GEN_STAGES.length
            ? step
            : GEN_STAGES.length - LOOP_TAIL + (step % LOOP_TAIL);
        setStageIndex(idx);
      }, STAGE_MS)
    );

    // Drive the real A+B+C→D orchestrator. If serverB is asleep or the call
    // fails, fall back to the real Layer A recording for the resolved cell so
    // the immersive scene still has audio + a sample-grounded source caption.
    let scene;
    try {
      const result = await generateSoundscape(conditionsFromParams(params));
      scene = { audioUrl: b64ToObjectUrl(result.audio_b64), metadata: result.metadata };
    } catch {
      scene = ambientForCell(params.season, params.time);
    }

    clearTimers();
    const resolvedState = { ...params, narration, ...scene };
    setResolved(resolvedState);
    const performNavigation = () => {
      navigate("/immersive", {
        state: { resolved: resolvedState, fromDemo: true, backPath: "/analysis" },
      });
    };
    if (document.startViewTransition) {
      document.startViewTransition(performNavigation);
    } else {
      performNavigation();
    }
  }

  return (
    <PromptChat
      phase={phase}
      userMessage={userMessage}
      statusLine={GEN_STAGES[stageIndex]}
      species={species}
      notice={notice}
      onSubmit={handleSubmit}
    />
  );
}
