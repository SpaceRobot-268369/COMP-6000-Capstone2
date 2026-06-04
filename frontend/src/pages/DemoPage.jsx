import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PromptChat from "../components/PromptChat.jsx";
import { resolvePrompt } from "../demo/resolvePrompt.js";
import { composeNarration } from "../demo/composeNarration.js";

// Staged "thinking" lines shown while the scene resolves, in order.
const GEN_STAGES = [
  "Reading your scene…",
  "Choosing the season & light…",
  "Placing the weather…",
  "Composing the moment…",
];
const STAGE_MS = 650;

/* The generation demo flow: a chatbot-style prompt → a staged generating
   transition → navigating to the immersive woodland placed on the resolved scene,
   with a second-person narration as its centre text. */
export default function DemoPage() {
  const navigate = useNavigate();
  const [phase, setPhase] = useState("prompt"); // prompt | generating
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
      setTimeout(() => {
        const resolvedState = { ...params, narration };
        const performNavigation = () => {
          navigate("/immersive", {
            state: { resolved: resolvedState, fromDemo: true },
          });
        };

        if (document.startViewTransition) {
          document.startViewTransition(performNavigation);
        } else {
          performNavigation();
        }
      }, STAGE_MS * GEN_STAGES.length + 300)
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

