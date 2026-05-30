# lucas__mvp_1__audioldm2_all_conditioned

Layer A MVP-1: first MVP-stage attempt for the ambient-bed role.

## Purpose / hypothesis

The smoke tests proved AudioLDM2 + LoRA can reproduce **one** narrow scene
each (smoke_1 spring night, smoke_2 summer-afternoon insects). MVP-1 keeps
the **same method** and tests whether a single LoRA can cover the **whole
clean ambient pool from site 257** when captions encode environmental
conditions.

Expected listener experience: changing the caption's (season, diel,
temperature, humidity, wind) fields should audibly change the generated
ambient bed — e.g. "night spring" should sound darker/quieter than "afternoon
summer", "moderate wind" should sound windier than "still".

## Method

Identical to [smoke_1 spring night](../lucas__smoke_1__audioldm2_spring_night/)
and [smoke_2 insects](../lucas__smoke_2__audioldm2_insects/):

- Frozen base: `cvssp/audioldm2`
- PEFT LoRA injected into UNet `to_q / to_k / to_v / to_out.0`
- Code in `code/` is copied verbatim from smoke_1 (`train_audioldm2.py`,
  `sample_audioldm2.py`, `audioldm2_dataset.py`, `handler.py`,
  `layer_a_visualization.py`). No method changes.

What's different vs smokes:

- **Dataset scale**: ~1,082 clips vs 35 (smoke_2) / 50 (smoke_1)
- **Caption template**: uniform conditioned schema (no scene-specific text)

## Dataset

Built by [script/dataset/build_mvp1_all_conditioned_dataset.py](../../../../../script/dataset/build_mvp1_all_conditioned_dataset.py):

- Source: site 257 ambient segment pool (1,982 clips, DVC).
- Hygiene filters: `wind_speed < 4.5 m/s`, `precipitation < 0.1 mm`,
  `duration ≥ 10 s`, no annotated-event overlap.
- Balance: `--per-cell-cap 100` per (season, diel_bin).
- Resulting size: ~1,082 clips.
- Output: `resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv`

Caption template (same at train + inference):

```
{diel} {season} ambient soundscape, Bowra dry woodland, Australia,
{temp_bucket} ({temp}C), {humidity_bucket}, {wind_bucket},
recorded {YYYY-MM-DD}, no music, no machinery
```

Buckets:
- temp: cold <15, mild 15–25, warm 25–32, hot 32–40, very hot ≥40
- humidity: dry <40, moderate 40–70, humid ≥70
- wind: still <0.5, light breeze 0.5–2, moderate 2–4.5

## Train

From `acoustic_ai/`, on a CUDA host (MPS may not handle batch_size 4):

```bash
./.venv/bin/accelerate launch \
  layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/mvp1_all_conditioned_dataset/manifest.csv \
  --output_dir ../model/candidates/lucas/mvp_1__audioldm2_all_conditioned \
  --batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-5
```

Hyperparams above are the first-run proposal — re-tune after seeing the loss
curve.

## Sample (manual baseline check)

```bash
for seed in 42 43 44; do
  ./.venv/bin/python \
    layers/layer_a/attempts/lucas__mvp_1__audioldm2_all_conditioned/code/sample_audioldm2.py \
    --prompt "night spring ambient soundscape, Bowra dry woodland, Australia, mild (12C), moderate humidity, light breeze, no music, no machinery" \
    --lora_dir ../model/candidates/lucas/mvp_1__audioldm2_all_conditioned \
    --run_name mvp1_spring_night_seed${seed} \
    --seed ${seed} \
    --num_inference_steps 100 \
    --guidance_scale 2.0 \
    --output_target_rms 0.0015 \
    --highpass_hz 80
done
```

## Dev endpoint contract — TENTATIVE

Smoke_1/smoke_2 lock the prompt server-side and expose only `seed`. MVP-1
needs a richer contract because the model is **driven by the caption**:

```
{ seed, season, diel_bin, temperature_c, humidity_pct, wind_speed_ms }
```

The server would assemble the caption via the same template the builder uses
and forward it to the pipeline. This contract change is **deferred** until
after the first training run validates that the captions actually influence
output. Until then, `code/handler.py` is the smoke_1 verbatim copy (seed-only,
fixed prompt).

## Results

_Pending first training run._

## Open questions

- Does ~1k clips across 16 cells give the LoRA enough per-cell data to
  differentiate condition fields? If not, fall back to fewer cells or
  per-cell LoRAs.
- Is `num_epochs=3` × ~270 batches/epoch enough total updates relative to
  smoke_2's 5 × ~35 batches? May need to bump epochs.
