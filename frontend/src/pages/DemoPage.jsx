import { useEffect, useRef, useState } from "react";
import PromptChat from "../components/PromptChat.jsx";
import ImmersivePage from "./ImmersivePage.jsx";
import { resolvePrompt } from "../demo/resolvePrompt.js";
import { composeNarration } from "../demo/composeNarration.js";
import "../demo/demo.css";

// Staged "thinking" lines shown while the scene resolves, in order.
const GEN_STAGES = [
  "Reading your scene…",
  "Choosing the season & light…",
  "Placing the weather…",
  "Composing the moment…",
];
const STAGE_MS = 650;

/* The generation demo flow: a chatbot-style prompt → a staged generating
   transition → the immersive woodland placed on the resolved scene, with a
   second-person narration as its centre text.

   One page owns a `phase` state machine so the GPU-heavy immersive engine is
   mounted only in the immersive phase and disposed cleanly on "New scene". */
export default function DemoPage() {
  const [phase, setPhase] = useState("prompt"); // prompt | generating | immersive
  const [userMessage, setUserMessage] = useState("");
  const [stageIndex, setStageIndex] = useState(0);
  const [resolved, setResolved] = useState(null);
  const timersRef = useRef([]);

  function clearTimers() {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }
  useEffect(() => clearTimers, []);

  function handleSubmit(text) {
    const params = resolvePrompt(text);
    const narration = composeNarration(params);
    setUserMessage(text);
    setResolved({ ...params, narration });
    setStageIndex(0);
    setPhase("generating");

    // advance the thinking lines, then cross into the immersive scene
    clearTimers();
    GEN_STAGES.forEach((_, i) => {
      if (i === 0) return;
      timersRef.current.push(setTimeout(() => setStageIndex(i), STAGE_MS * i));
    });
    timersRef.current.push(
      setTimeout(() => setPhase("immersive"), STAGE_MS * GEN_STAGES.length + 300),
    );
  }

  function handleReset() {
    clearTimers();
    setPhase("prompt");
    setResolved(null);
    setUserMessage("");
    setStageIndex(0);
  }

  if (phase === "immersive" && resolved) {
    return (
      <ImmersivePage
        initial={resolved}
        showDevPanel={false}
        overlay={
          <button type="button" className="demo-reset" onClick={handleReset}>
            ↺ New scene
          </button>
        }
      />
    );
  }

  return (
    <PromptChat
      phase={phase}
      userMessage={userMessage}
      statusLine={GEN_STAGES[stageIndex]}
      onSubmit={handleSubmit}
    />
  );
}
