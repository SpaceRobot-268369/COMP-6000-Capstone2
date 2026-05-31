# Implementation Plan — E-A Ambient Analysis · **Approach 1: CLAP cell-match**

| | |
|---|---|
| **Attempt ID** | `lucas__smoke_2__clap_cell_match` |
| **Layer / role** | `layer_e` — Analysis · E-A ambient context |
| **Stage** | `smoke_2` (lucas/layer_e; `smoke_1` = detectors) |
| **Backbone** | **Frozen LAION-CLAP** (`laion/clap-htsat-unfused` via `transformers.ClapModel`) |
| **Scope** | Python module + **offline eval only** — no server / Express / frontend wiring |
| **Author / date** | lucas · 2026-06-01 |
| **Branch** | `feat/lucas/layer-e-ambient-analysis` |

> One of a **three-attempt bake-off** for ambient analysis. Siblings:
> [`smoke_3__clap_knn_env`](../lucas__smoke_3__clap_knn_env/PLAN.md) (k-NN to real
> segments → env metadata) and
> [`smoke_4__clap_cell_plus_knn`](../lucas__smoke_4__clap_cell_plus_knn/PLAN.md)
> (both heads + agreement). All three share the frozen-CLAP foundation below;
> they differ only in the **reference target** and the **report**.

---

## 1. Purpose / hypothesis

**The "reverse prompt" question, closed-set form.** Generation takes
`(season, diel)` → locked caption → audio. This attempt inverts it: **audio →
nearest of the 16 `(season, diel)` cells → that cell's caption**, with a
confidence. It is the cleanest inverse of the per-cell bank
(`lucas__prod_1__per_cell_loras`) because the output vocabulary is exactly the
16 captions the generator was conditioned on.

**Hypothesis:** a frozen CLAP embedding separates the 16 Bowra season×diel cells
well enough that nearest-anchor classification recovers the correct season and
diel materially above chance — with *zero training* — because CLAP was
pre-trained on environmental audio and the cells are acoustically distinct
(dawn chorus vs. cicada-saturated summer afternoon vs. still winter night).

**Why this is in-distribution:** the 16 cells are *our* cells. No off-the-shelf
tagger knows "Bowra dry woodland autumn dawn"; here the label set is closed and
matches the generation contract 1:1.

---

## 2. Shared foundation (common to all three attempts)

### 2.1 Embedding backbone
- **Model:** `laion/clap-htsat-unfused` loaded via `transformers.ClapModel` +
  `ClapProcessor`. This is the **same CLAP that lives inside `cvssp/audioldm2`**,
  so the analysis embedding space matches the generation conditioning space, and
  **no new dependency is added** — `transformers >= 4.40` is already in
  `acoustic_ai/requirements.txt`.
- **Interpreter:** `acoustic_ai/.venv` only (MPS). Never system Python.
- **Audio contract:** CLAP expects **48 kHz mono**. Source clips are 300 s
  `.webm` (Opus); ambient segments are ~10 s WAV. Load with `librosa`/`torchaudio`
  (both present), resample to 48 kHz, mono-mix, window into 10 s chunks, embed
  each, **mean-pool** chunk embeddings → one L2-normalised vector per item.
- **Frozen:** `model.eval()`, `torch.no_grad()`. No CLAP weights are updated in
  Phase 1.

### 2.2 Data inputs (all verified present)
| Use | Path | Notes |
|---|---|---|
| Ground-truth labels | `resources/site_257_bowra-dry-a/site_257_training_manifest.csv` | 6,148 clips; cols incl. `clip_path, sample_bin (=diel), season, hour_local, month, day_of_year` |
| Cleaned ambient pool | `…/lucas__smoke_4__vae_baseline/data/ambient/ambient_index.csv` (+ `ambient_segments/*.wav`) | 1,982 segments on disk (DVC-materialised); cols `segment_id, source_clip, diel_bin, season, hour_sin/cos, month_sin/cos` |
| Cell captions | `acoustic_ai/registry.yaml` → `layer_a.lucas__prod_1__per_cell_loras.params.cells` | 16 locked prompts |

**Cell definition:** the 16 cells = `season ∈ {spring,summer,autumn,winter}` ×
`diel ∈ {dawn,morning,afternoon,night}`. In the training manifest, diel is the
`sample_bin` column.

### 2.3 Determinism
Canonical seed **42** for every split, sample, and shuffle. Embedding caches are
`.npy` → **DVC-tracked**; csv/json/md → git.

---

## 3. Approach mechanics (this attempt)

Build **16 cell anchors**, then classify a query clip by nearest anchor. Two
anchor constructions are evaluated head-to-head:

**(a) Text anchors (pure zero-shot).**
`anchor_c = CLAP_text(cells[c].prompt)` for each of the 16 captions. Query =
`CLAP_audio(clip)`. Prediction = `argmax_c cos(query, anchor_c)`. This is the
literal audio↔text reverse and needs **no training data at all**.

**(b) Audio-prototype anchors.**
Group `site_257_training_manifest` by `(season, sample_bin)`; for each cell embed
its clips (10 s windows) and average → `anchor_c = mean(CLAP_audio(clip_i))`.
Query classified by nearest prototype. This is audio↔audio (no text), usually
tighter but uses labelled audio.

Report (either variant):
```json
{
  "predicted_cell": "summer_afternoon",
  "season": "summer", "diel": "afternoon",
  "caption": "<the cell's locked generation prompt>",
  "confidence": 0.0,                       // softmax(sim/τ)[argmax]
  "topk": [{"cell": "...", "score": 0.0}]  // top-3
}
```
`caption` is the literal "reverse prompt" — the text the generator would use to
*re-synthesise* a bed like the input.

---

## 4. Implementation workflow

```
code/
  clap_backbone.py      # load ClapModel+processor; embed_audio(paths)->E; embed_text(strs)->T  (SHARED design)
  build_anchors.py      # (a) text anchors from registry cells; (b) audio prototypes from manifest -> anchors.npy
  ambient_cell_match.py # load anchors; classify(clip)->report JSON   (the E-A entry point)
  eval.py               # held-out classification metrics + confusion matrix -> metrics.json + report.md
  train_probe.py        # Phase 2 only (see §6)
data/
  anchors_text.npy / anchors_audio.npy   # DVC
  embeddings_cache.npy                    # DVC (per-clip CLAP vectors, keyed by clip_path)
  splits/{train,val}.csv                  # git (seed-42 stratified)
README.md  metrics.json
```

Steps:
1. **Scaffold** the attempt: `code/`, `data/`, `expected/.gitkeep`,
   `showcase/.gitkeep`, `dev-artifacts-self-testing/.gitkeep`, `README.md`
   (experiment-log template, conventions §6).
2. **`clap_backbone.py`** — load CLAP once, expose `embed_audio` / `embed_text`
   with the 48 kHz / 10 s-window / mean-pool / L2-norm contract from §2.1.
3. **Split** `site_257_training_manifest` stratified by the 16 cells, seed 42
   (e.g. 80/20). Write `splits/`. (Anchors come from train only; eval on val.)
4. **`build_anchors.py`** — produce both `anchors_text.npy` (16×D) and
   `anchors_audio.npy` (16×D). Cache per-clip embeddings to avoid recompute.
5. **`ambient_cell_match.py`** — `classify(path) -> report`; the future E-A
   handler entry point.
6. **`eval.py`** — run on val split; emit metrics (§5) for **both** anchor
   variants; render the 16×16 confusion matrix PNG.
7. **Audit** misclassifications: are confusions *adjacent* cells (e.g.
   spring_dawn↔autumn_dawn)? Record in README "Results".
8. **Decide** Phase 2 (§6) only if the smoke bar (§5) is missed.

**Not in this plan (deferred):** registry entry, `handler.py` `analyze()`,
FastAPI upload endpoint, Express route, frontend. The module is built and proven
offline first; wiring is a separate follow-up once a winner is chosen.

---

## 5. Offline evaluation design

**Split:** seed-42 stratified train/val over the 6,148 manifest clips (anchors
from train, scored on val). Audio-prototype anchors must never see val clips.

| Metric | Definition | Chance baseline |
|---|---|---|
| Cell top-1 / top-3 acc | 16-way over `(season,diel)` | 6.25% / 18.75% |
| Season acc | 4-way (collapse diel) | 25% (or majority-class) |
| Diel acc | 4-way (collapse season) | 25% |
| Confusion matrix | 16×16, inspect adjacency structure | — |
| Confidence calibration | acc vs. softmax confidence bins | — |

**Baselines reported alongside:** random, majority-cell, and **text-anchor vs.
audio-prototype** head-to-head (the key internal comparison).

**Smoke success bar (recognizable target, not production):**
- Season acc **≥ 70%** and diel acc **≥ 55%** for at least one anchor variant, **and**
- Cell top-3 **≥ 50%**, **and**
- the confusion matrix shows *structured* errors (adjacent season/diel), not noise.

Numbers are starting targets — adjust once the first run lands; the gate is
"clearly better than chance with interpretable confusions."

---

## 6. Phase 2 — optional fine-tuning (only if the bar is missed)

Backbone stays **frozen**; train a tiny head on cached CLAP embeddings (cheap,
local MPS, seed 42):
1. **Linear probe** — softmax(16) on the 512-d CLAP vector, cross-entropy on the
   train split. Usually recovers 10–20 pts over zero-shot text anchors.
2. **MLP probe** — 1 hidden layer if linear underfits.
3. **CLAP audio-encoder LoRA** — *last resort*, larger scope: LoRA-tune CLAP's
   audio tower on the cell-classification objective. Flag explicitly before
   doing this; it changes the embedding space and breaks the "shared with
   generation" property.

Trigger: take Phase 2 **only** if season/diel zero-shot is below the §5 bar.
The user's "new attempt with fine-tuning" choice is satisfied by the probe; the
LoRA path is reserved and gated.

---

## 7. Risks / open questions
- **CLAP domain gap.** CLAP's training audio skews general/AudioSet; fine Bowra
  diel distinctions (dawn vs. morning) may be subtle → diel acc the likely weak
  spot. Mitigation: audio prototypes + Phase-2 probe.
- **Caption realism for text anchors.** The locked captions contain
  recording-date and temperature tokens ("recorded 2019-10-04", "(26C)") that
  CLAP can't hear. Consider an **ablation**: strip non-acoustic tokens from the
  anchor text and re-measure.
- **Class imbalance.** Some cells are source-thin (`winter_morning`,
  `summer_morning`, `autumn_morning` per the prod card). Stratified split +
  per-class metrics, not just global accuracy.
- **Window aggregation.** Mean-pool vs. max-pool vs. per-window vote over the
  10 s chunks — cheap ablation, note in README.

---

## 8. Dependencies
**None new for Phase 1** — `transformers`, `librosa`, `torchaudio`, `numpy`,
`scipy` already in `acoustic_ai/requirements.txt`. Phase-2 probe uses `torch`
(present). If anything is added, update `requirements.txt` in the same change.

---

## 9. Definition of done (this attempt)
- `metrics.json` + `report.md` committed with both anchor variants scored on the
  held-out split and a confusion matrix.
- README "Results" states whether the §5 bar was met and which anchor variant
  won.
- A one-paragraph **bake-off verdict** vs. siblings (cell-match alone enough? or
  does k-NN/env add value?) — feeds the decision on which attempt graduates to
  `mvp` and gets wired to the server.
