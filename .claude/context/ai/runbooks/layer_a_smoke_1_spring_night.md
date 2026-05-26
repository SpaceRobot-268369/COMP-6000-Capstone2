# Layer A Smoke Test 1 — Spring Night Ambient (AudioLDM2 LoRA)

User-validated on 2026-05-06. Active in `inference.py` as the "Dev > Smoke 1" path.

## What this checkpoint is for

A narrow Layer A ambient-bed smoke model trained on **50 quiet spring-night clips**
from Bowra dry woodland (site 257). It produces low-volume, environmental-like
ambient beds with only minor artifacts. It is **not** a general seasonal or
weather-conditioned soundscape model — it should only be used with the
spring-night smoke prompt that excludes foreground events, music, and machinery.

| Asset | Path |
|---|---|
| Base model | `cvssp/audioldm2` (frozen) |
| LoRA checkpoint | `model/candidates/lucas/layer-a-audioldm2-raw-smoke/` |
| Training dataset | `resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv` |
| Per-run params | `model/candidates/lucas/layer-a-audioldm2-raw-smoke/params.yaml` |

## Train

From `acoustic_ai/`:

```bash
./.venv/bin/accelerate launch layers/layer_a/attempts/lucas__smoke_1__audioldm2_spring_night/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/smoking_test_dataset/manifest.csv \
  --output_dir ../model/candidates/lucas/layer-a-audioldm2-raw-smoke \
  --batch_size 1 \
  --num_epochs 5 \
  --learning_rate 1e-5
```

Keep raw field-recording levels — **do not** normalize to `target_rms 0.05`;
that over-amplifies background recorder noise and produces pulsing /
machine-like artifacts. See the deprecated `layer-a-audioldm2-rms005-smoke`
candidate as the documented negative result.

## Sample

```bash
for seed in 42 43 44; do
  ./.venv/bin/python layers/layer_a/attempts/lucas__smoke_1__audioldm2_spring_night/sample_audioldm2.py \
    --prompt "quiet spring night ambient soundscape, Bowra dry woodland, Australia, distant environmental bed, no foreground events, no music, no machinery" \
    --lora_dir ../model/candidates/lucas/layer-a-audioldm2-raw-smoke \
    --run_name spring_night_raw_smoke_seed${seed} \
    --seed ${seed} \
    --num_inference_steps 100 \
    --guidance_scale 2.0 \
    --output_target_rms 0.0015 \
    --highpass_hz 80
done
```

Outputs (WAV + PNG spectrogram + JSON metadata) land under
`debug/layer_a/audioldm2/samples/audioldm2-lora-raw-smoke/spring_night_raw_smoke_seed{42,43,44}/`.

## Notes

- `sample_audioldm2.py` and the dev frontend/backend response both render Layer
  A spectrogram PNGs through `modules.ambient.diffusion.layer_a_visualization`
  with the same log-mel parameters. If the two diverge, check that both
  services are restarted and the compared WAVs share seed/model/prompt/settings.
- Different LoRA checkpoint outputs must remain in separate
  checkpoint-named folders under `debug/layer_a/audioldm2/samples/`. Don't
  reuse one `--output_dir` across checkpoints unless it still keeps
  checkpoint-named subfolders.
- **Seed semantics:** the seed initializes the diffusion model's random
  starting noise. Same model + prompt + params + seed = effectively the same
  audio on the same code path. Different seed = different variation. Seed is
  **not** temperature; temperature is not exposed on this path. Use
  non-negative integers; portable range `0`–`2147483647`.

## Dev endpoint contract

Because this smoke model was trained on a very small dataset, the dev
generation path is locked down:

- Frontend exposes **only** a non-negative integer `seed`.
- Express backend forwards **only** `{ seed }`.
- FastAPI AI server owns the fixed prompt, checkpoint, guidance, step count,
  audio length, RMS, and high-pass settings. It returns all of them in
  response metadata for debugging.
