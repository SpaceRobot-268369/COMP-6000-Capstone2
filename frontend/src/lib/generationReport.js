/* generationReport.js — synthesize a fused-analysis-shaped report from a
   generation contract.

   The immersive tone toggle (Immersive ⇄ Analytical) narrates an analysis
   `report` through the LLM-OSS report writer. Generated scenes never run the
   detectors, so they carry no report — but we already know the *ground truth*
   of what we composed (season, time, weather, species), so we build the report
   directly from the parsed layer contracts + resolved local params.

   The shape mirrors the analysis aggregator's `decision` slice
   (schema `analysis_aggregator.v1`) that `write_report` narrates, so the same
   /analysis/narrative endpoint renders both registers with no backend change. */

function clamp01(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

function titleCase(value) {
  return String(value || "")
    .split(/[\s_]+/)
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Build a fused-report-shaped object from the generation contract so the tone
 * toggle can narrate a generated scene faithfully (the species we list are the
 * species the faithfulness guard will allow the analytical prose to name).
 *
 * @param {object} localParams  resolved scene params { season, time, rain, rainAmount, wind, thunder }
 * @param {object} parseResult  parser output { layer_a:{season,diel}, layer_c:{species[]} }
 * @returns {object} report with schema_version + decision slice
 */
export function reportFromGeneration(localParams = {}, parseResult = {}) {
  const layerA = parseResult.layer_a || {};
  const layerC = parseResult.layer_c || {};
  const season = layerA.season || localParams.season || "autumn";
  const time = layerA.diel || localParams.time || "dawn";

  const rain = Boolean(localParams.rain);
  const thunder = Boolean(localParams.thunder);
  const rainAmount = rain ? clamp01(localParams.rainAmount, 0.6) : 0;
  const windAmount = clamp01(localParams.wind, 0);

  const species = Array.isArray(layerC.species) ? layerC.species : [];
  const detected_calls = species.map((name) => ({
    common_name: titleCase(name),
    label: titleCase(name),
  }));

  return {
    schema_version: "analysis_aggregator.v1",
    source: "generation_contract",
    decision: {
      season: { value: season },
      time_of_day: { value: time },
      weather: {
        rain: { label: rain ? "rain" : "none", intensity: rainAmount },
        wind: { label: windAmount > 0.05 ? "wind" : "none", intensity: windAmount },
        thunder: { label: thunder ? "thunder" : "none", events: thunder ? [{ t: 0 }] : [] },
      },
      detected_calls,
    },
  };
}
