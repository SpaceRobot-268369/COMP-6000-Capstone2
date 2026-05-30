# DEVLOG — lucas__mvp_1_2__shared_lora_maxed

## Hypothesis

With 8× more LoRA capacity (r=64 vs MVP-1's r=8), alpha scaled in proportion
(128 vs 32), and a longer schedule (8 vs 3 epochs), a single shared LoRA can
cover all 16 (season, diel) cells of the MVP-1 manifest without the per-cell
blur observed in MVP-1.

If true: per-cell LoRA bank (MVP-2) becomes unnecessary.
If false: capacity wasn't the limiting factor at r=64; per-cell partitioning
is justified, and Phase 2 proceeds.

Secondary hypothesis: the `recorded YYYY-MM-DD` fragment was noise, not
signal. Caption v3 drops it; if val loss / showcase quality holds, v3
becomes the MVP-2 default.

## Setup

- Branch / commit at training: `model/lucas/layer-a-mvp-1-all-conditioned`
  (commit recorded at tag `attempt/lucas__mvp_1_2__shared_lora_maxed`).
- Parent dataset: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/clips.dvc`
  (same DVC blob as MVP-1 and MVP-1.1; no rebuild).
- Training manifest: `manifest_v3.csv` — derived from `manifest.csv` by
  stripping `, recorded \d{4}-\d{2}-\d{2}` from the `caption` column.
  All other columns preserved (so `recording_date` is still logged for
  retrieval / future analysis).
- Build command (run once on server before training):
  ```
  M=resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset
  python3 - <<'PY'
  import csv, re
  src = f"{M}/manifest.csv"; dst = f"{M}/manifest_v3.csv"
  pat = re.compile(r", recorded \d{4}-\d{2}-\d{2}")
  with open(src) as fi, open(dst, "w", newline="") as fo:
      r = csv.DictReader(fi); w = csv.DictWriter(fo, fieldnames=r.fieldnames); w.writeheader()
      for row in r:
          row["caption"] = pat.sub("", row["caption"])
          w.writerow(row)
  PY
  ```
- Environment (planned): AWS Tesla T4, torch 2.12.0+cu130, fp16 single-GPU,
  same venv as MVP-1 / MVP-1.1.

## Run log

### TBD — training run #1

To execute (from `~/lucano/COMP-6000-Capstone2/acoustic_ai/`, inside tmux
session `mvp12`):

```
./.venv/bin/accelerate launch --mixed_precision fp16 \
  layers/layer_a/attempts/lucas__mvp_1_2__shared_lora_maxed/code/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest_v3.csv \
  --output_dir ../model/candidates/lucas/mvp_1_2__shared_lora_maxed \
  --batch_size 4 --num_epochs 8 --learning_rate 1e-5 \
  --lora_r 64 --lora_alpha 128
```

Expected: ~242 steps/epoch × 8 = ~1,936 steps × ~1.3 s = ~42 min total
wall-clock. GPU peak ≈ 10–11 GB. Note: the train script may or may not
accept `--lora_r` / `--lora_alpha` — to be confirmed during dry-run; if
not, will be patched in (small, surgical change) before launch.

(Epoch losses + anomalies appended here after the run.)

### TBD — showcase generation

3 seeds across 3 cells (same 3 prompts MVP-1 will showcase) for direct A/B:

| seed | cell | caption (from `params.yaml inference.showcase_prompts`) |
|---|---|---|
| 42 | spring_night | "night spring ambient soundscape, …, mild (18C), moderate humidity, still, …" |
| 42 | summer_afternoon | "afternoon summer ambient soundscape, …, hot (33C), dry air, moderate wind, …" |
| 42 | autumn_night | "night autumn ambient soundscape, …, cool (12C), moderate humidity, still, …" |

Artifacts → `dev-artifacts-self-testing/`. Keepers promoted to `showcase/`
after listening.

### TBD — listening verdict

(Filled in after generation.)

## Decision gate

| outcome | next step |
|---|---|
| All 3 cells sound smoke-quality | shared LoRA wins → skip per-cell, head to Phase 3 inference router |
| Some cells good, some blurred | hybrid: shared base LoRA + targeted per-cell LoRA on weak cells (Phase 2.1) |
| All cells still blurred | capacity isn't enough → proceed to Phase 2 per-cell LoRA bank as planned |

## Comparison vs prior attempts

(Filled in after listening — table will include MVP-1, MVP-1.1, this attempt.)

## Retrospective

(Written when attempt is closed.)
