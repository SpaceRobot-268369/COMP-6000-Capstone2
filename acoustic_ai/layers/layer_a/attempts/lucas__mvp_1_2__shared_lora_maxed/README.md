# mvp_1_2__shared_lora_maxed

## Summary

- Owner: Lucas
- Layer / role: Layer A — ambient bed (capacity diagnostic)
- Status: candidate (Phase 1B)
- Base model: cvssp/audioldm2 (frozen)
- Trained at: _pending first training run_
- Parent attempts: `lucas__mvp_1__audioldm2_all_conditioned`,
  `lucas__mvp_1_1__spring_night_replica`

## Purpose / hypothesis

Capacity diagnostic for the Layer A quality-push roadmap. Phase 1A
(`mvp_1_1`) confirmed the MVP-1 data pipeline is sound on a single cell.
This attempt asks the next question:

> With 8× more LoRA capacity (r=64 vs r=8) and a longer schedule (8 vs 3
> epochs), can a single shared LoRA cover all 16 (season, diel) cells
> without per-cell quality blur?

If yes → MVP-2's per-cell LoRA bank is unnecessary; we ship a shared
adapter. If no → per-cell partitioning is justified and MVP-2 proceeds.

## Dataset / inputs

- Source manifest: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv`
  (the MVP-1 manifest; same DVC `clips.dvc` blob — no rebuild).
- Training manifest: `manifest_v3.csv` derived from `manifest.csv` by
  rewriting the `caption` column to drop the `, recorded YYYY-MM-DD`
  fragment. All other columns (including `recording_date`) are preserved.
- Build command (run once on server before training):
  ```
  M=resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset
  python3 - <<'PY'
  import csv, re
  src = "$M/manifest.csv"; dst = "$M/manifest_v3.csv"
  pat = re.compile(r", recorded \d{4}-\d{2}-\d{2}")
  with open(src) as fi, open(dst, "w", newline="") as fo:
      r = csv.DictReader(fi); w = csv.DictWriter(fo, fieldnames=r.fieldnames); w.writeheader()
      for row in r:
          row["caption"] = pat.sub("", row["caption"])
          w.writerow(row)
  PY
  ```
- Filtering: identical to MVP-1 (rules 1–7 + §6.1 content filters).
- Caption schema: **v3** — date dropped, env fields retained. See
  `.claude/context/ai/logs/caption_schema_log.md`.

## Training context

- Command (from `acoustic_ai/`):
  ```bash
  ./.venv/bin/accelerate launch --mixed_precision fp16 \
    layers/layer_a/attempts/lucas__mvp_1_2__shared_lora_maxed/code/train_audioldm2.py \
    --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest_v3.csv \
    --output_dir ../model/candidates/lucas/mvp_1_2__shared_lora_maxed \
    --batch_size 4 --num_epochs 8 --learning_rate 1e-5 \
    --lora_r 64 --lora_alpha 128
  ```
- Hardware: AWS EC2 Tesla T4 (host `shinypokemon`).
- Expected runtime: ~45 min wall-clock (~242 steps/epoch × 8 × 1.3 s/step,
  ~2.5× MVP-1 — slight overhead for the larger LoRA forward/backward).
- GPU peak: ≈ 10–11 GB (r=64 is ~8× MVP-1's r=8 trainable params, but the
  adapter still dwarfs nothing — base model dominates VRAM).
- Code branch: `model/lucas/layer-a-mvp-1-all-conditioned` (TBD sha).
- Important settings: see [params.yaml](./params.yaml).

## Artifacts

- Checkpoint: `adapter_model.safetensors` (DVC after training).
- Pointer: `adapter_model.safetensors.dvc`.
- Metrics: `metrics.json` (TBD — epoch-by-epoch losses, runtime, GPU peak).
- Sample outputs:
  - Self-test: `dev-artifacts-self-testing/seed_<N>_<cell>/`.
  - Showcase: `showcase/seed_<N>_<cell>/` after curation.

## Results / metrics

Pending — see DEVLOG.

## Results analysis / audit

Pending — written in the DEVLOG retrospective.

## Known limitations

- Caption v3 deliberately strips date — if val loss / quality degrades vs
  MVP-1, can't disambiguate "lost date signal" from "still capacity-bound"
  without a v3+date sibling attempt. Mitigation: the comparison against
  MVP-1.1 (caption v2, single cell) anchors what "good per-cell quality"
  sounds like; if MVP-1.2 hits that bar across cells, capacity wins
  regardless of caption schema.
- Per-cell sample count is unchanged from MVP-1; thin cells (summer
  afternoon, 29 raw clips) are still thin. Phase 2.5 expands them.

## Follow-up actions

- Build `manifest_v3.csv` on server → train → write `metrics.json`.
- Generate showcase across 3 cells (spring_night, summer_afternoon,
  autumn_night) — same prompts as MVP-1's eventual showcase — for direct
  A/B vs MVP-1's blurred outputs.
- Decision gate documented in [DEVLOG.md](./DEVLOG.md):
  | outcome | next step |
  |---|---|
  | All 3 cells sound smoke-quality | shared LoRA wins → skip per-cell, head to Phase 3 inference router |
  | Some cells good, some blurred | hybrid: shared base + targeted per-cell LoRA on weak cells |
  | All cells still blurred | capacity isn't enough → proceed to Phase 2 per-cell LoRA bank |
