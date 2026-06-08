const VALID_SEASONS = new Set(["spring", "summer", "autumn", "winter"]);
const VALID_TIMES = new Set(["dawn", "morning", "afternoon", "night"]);

export function clamp01(value, fallback = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(1, Math.max(0, num));
}

function firstValidChoice(values, allowed, fallback) {
  for (const value of values) {
    const choice = String(value || "").toLowerCase();
    if (choice && choice !== "undetermined" && allowed.has(choice)) {
      return choice;
    }
  }
  return fallback;
}

function componentIsPresent(component) {
  if (!component || typeof component !== "object") return false;
  const label = String(component.label || "").toLowerCase();
  return Boolean(label && label !== "none") || Number(component.intensity) > 0.05;
}

function callEventTags(calls = []) {
  const tags = new Set();
  calls.forEach((call) => {
    const name = `${call?.common_name || ""} ${call?.label || ""}`.toLowerCase();
    if (!name.trim()) return;
    if (name.includes("cicada")) tags.add("insects");
    if (name.includes("cricket")) tags.add("crickets");
    if (!name.includes("cicada") && !name.includes("cricket")) tags.add("birdsong");
  });
  return [...tags];
}

export function sceneStateFromAnalysis(routeState) {
  const report = routeState?.report;
  if (!report) return null;

  const decision = report.decision || {};
  const inferred = report.inferred_context || {};
  const weather = decision.weather || {};
  const rainComponent = weather.rain || {};
  const windComponent = weather.wind || {};
  const thunderComponent = weather.thunder || {};
  const calls = Array.isArray(decision.detected_calls) ? decision.detected_calls : [];
  const rain = componentIsPresent(rainComponent);
  const thunder = componentIsPresent(thunderComponent) ||
    (Array.isArray(thunderComponent.events) && thunderComponent.events.length > 0);
  const rainAmount = rain ? clamp01(rainComponent.intensity, 0.6) : 0;
  const narrativeText = routeState.narrative?.text || report.narration?.summary || "";

  return {
    season: firstValidChoice(
      [decision.season?.value, inferred.season?.estimate],
      VALID_SEASONS,
      "autumn",
    ),
    time: firstValidChoice(
      [decision.time_of_day?.value, inferred.diel?.estimate],
      VALID_TIMES,
      "dawn",
    ),
    rain,
    rainAmount,
    wind: clamp01(windComponent.intensity, 0),
    thunder,
    narration: narrativeText,
    events: callEventTags(calls),
    audioUrl: routeState.audioUrl || "",
    sourceCaption: routeState.sourceCaption || "",
    report,
    register: routeState.register || routeState.narrative?.register || "immersive",
  };
}
