/* sampleCatalog.js — real project recordings used as the demo's audio + scene
   description source while Layer D (mixing) and the LLM-OSS scene writer are not
   yet built.

   Both demo flows (the Analysis page presets and the Generation page prompt →
   immersive scene) need a real soundscape to play and a sample-grounded caption
   to show. Until the generative path is wired end-to-end we stand in with the
   Layer A `expected/` bank: one real Bowra dry-woodland recording per season×diel
   cell (16 cells), served by the Express backend straight from the repo checkout:

     /api/layers/layer_a/attempts/<attempt>/samples/expected/<cell>/<case>/audio.wav

   Paths are stable (checked-in DVC-pulled artefacts), so the cell→clip map is
   hardcoded — no runtime listing fetch, no auth/loading states. Captions are the
   recordings' real metadata captions, lightly trimmed for display. */

const A_LAYER = "layer_a";
const A_ATTEMPT = "lucas__prod_1__per_cell_loras";
const SAMPLES_BASE = `/api/layers/${A_LAYER}/attempts/${A_ATTEMPT}/samples/expected`;

// season_diel → { dir: <case folder>, caption: <real metadata caption> }
const CELLS = {
  autumn_afternoon: { dir: "real_0015_1313480_clip001_s000", caption: "afternoon autumn ambient soundscape, Bowra dry woodland, Australia, mild (23C), dry air, still" },
  autumn_dawn:      { dir: "real_0182_216086_clip002_s000",  caption: "dawn autumn ambient soundscape, Bowra dry woodland, Australia, mild (24C), dry air, light breeze" },
  autumn_morning:   { dir: "real_0241_216088_clip002_s000",  caption: "morning autumn ambient soundscape, Bowra dry woodland, Australia, warm (31C), dry air, moderate wind" },
  autumn_night:     { dir: "real_0279_216473_clip006_s000",  caption: "night autumn ambient soundscape, Bowra dry woodland, Australia, cold (14C), humid air, light breeze" },
  spring_afternoon: { dir: "real_0500_1401567_clip001_s000", caption: "afternoon spring ambient soundscape, Bowra dry woodland, Australia, warm (31C), dry air, light breeze" },
  spring_dawn:      { dir: "real_0575_4881_clip001_s000",    caption: "dawn spring ambient soundscape, Bowra dry woodland, Australia, mild (18C), dry air, moderate wind" },
  spring_morning:   { dir: "real_0682_4887_clip001_s000",    caption: "morning spring ambient soundscape, Bowra dry woodland, Australia, warm (25C), dry air, moderate wind" },
  spring_night:     { dir: "real_0771_5392_clip001_s000",    caption: "night spring ambient soundscape, Bowra dry woodland, Australia, mild (15C), dry air, light breeze" },
  summer_afternoon: { dir: "real_0978_215469_clip005_s000",  caption: "afternoon summer ambient soundscape, Bowra dry woodland, Australia, very hot (48C), dry air, moderate wind" },
  summer_dawn:      { dir: "real_1032_215467_clip001_s000",  caption: "dawn summer ambient soundscape, Bowra dry woodland, Australia, hot (33C), dry air, moderate wind" },
  summer_morning:   { dir: "real_1129_215185_clip002_s000",  caption: "morning summer ambient soundscape, Bowra dry woodland, Australia, warm (31C), dry air, moderate wind" },
  summer_night:     { dir: "real_1164_215190_clip001_s000",  caption: "night summer ambient soundscape, Bowra dry woodland, Australia, warm (30C), dry air, moderate wind" },
  winter_afternoon: { dir: "real_1331_1313831_clip011_s000", caption: "afternoon winter ambient soundscape, Bowra dry woodland, Australia, mild (19C), dry air, light breeze" },
  winter_dawn:      { dir: "real_1397_1313802_clip001_s000", caption: "dawn winter ambient soundscape, Bowra dry woodland, Australia, cold (9C), moderate humidity, light breeze" },
  winter_morning:   { dir: "real_1467_5489_clip001_s000",    caption: "morning winter ambient soundscape, Bowra dry woodland, Australia, cold (14C), dry air, moderate wind" },
  winter_night:     { dir: "real_1494_5486_clip001_s000",    caption: "night winter ambient soundscape, Bowra dry woodland, Australia, cold (15C), moderate humidity, moderate wind" },
};

const FALLBACK_CELL = "autumn_dawn";

function cellKey(season, time) {
  const key = `${season}_${time}`;
  return CELLS[key] ? key : FALLBACK_CELL;
}

/**
 * Resolve a (season, diel) cell to its real recording URL + display caption.
 * `time` is the immersive "time of day" token, which matches the diel bins
 * (dawn / morning / afternoon / night).
 *
 * @returns {{ cell:string, audioUrl:string, sourceCaption:string }}
 */
export function ambientForCell(season, time) {
  const key = cellKey(season, time);
  const { dir, caption } = CELLS[key];
  return {
    cell: key,
    audioUrl: `${SAMPLES_BASE}/${key}/${dir}/audio.wav`,
    sourceCaption: caption,
  };
}

export { CELLS };
