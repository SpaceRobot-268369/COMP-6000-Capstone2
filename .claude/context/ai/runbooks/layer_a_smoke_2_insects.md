# Layer A Smoke Test 2 — Summer Insect/Cicada Ambient (AudioLDM2 LoRA)

Active in `inference.py` as the "Dev > Smoke 2" path.

## What this checkpoint is for

A Layer A ambient-bed smoke model trained on a **manually-audited
insect/cicada dataset** from Bowra dry woodland. The dataset is intentionally
small (≤50 clips after audit) and excludes segments overlapping annotated
events and strong-wind rows.

Use prompts that stay inside the observed scene distribution: summer afternoon
insect/cicada texture, distant environmental bed, no birds, no foreground
events, no music, no machinery, no strong wind.

| Asset | Path |
|---|---|
| Base model | `cvssp/audioldm2` (frozen) |
| LoRA checkpoint | `model/candidates/lucas/layer-a-audioldm2-insects-smoke/` |
| Training dataset | `resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv` |
| Per-run params | `model/candidates/lucas/layer-a-audioldm2-insects-smoke/params.yaml` |

## Train

From `acoustic_ai/`:

```bash
./.venv/bin/accelerate launch modules/ambient/diffusion/train_audioldm2.py \
  --manifest_path ../resources/site_257_bowra-dry-a/smoking_test2_insects_dataset/manifest.csv \
  --output_dir ../model/candidates/lucas/layer-a-audioldm2-insects-smoke \
  --batch_size 1 \
  --num_epochs 5 \
  --learning_rate 1e-5
```

## Sample

```bash
for seed in 42 43 44; do
  ./.venv/bin/python modules/ambient/diffusion/sample_audioldm2.py \
    --prompt "summer afternoon insect-rich ambient soundscape, cicada and insect texture, Bowra dry woodland, Australia, dry hot air, distant environmental bed, no birds, no foreground events, no music, no machinery, no strong wind" \
    --lora_dir ../model/candidates/lucas/layer-a-audioldm2-insects-smoke \
    --run_name insects_smoke_seed${seed} \
    --seed ${seed} \
    --num_inference_steps 100 \
    --guidance_scale 2.0 \
    --output_target_rms 0.0015 \
    --highpass_hz 80
done
```

Outputs land under
`debug/layer_a/audioldm2/samples/audioldm2-lora-insects-smoke/insects_smoke_seed{42,43,44}/`.

## Notes

See [layer_a_smoke_1_spring_night.md](layer_a_smoke_1_spring_night.md) for shared
notes on spectrogram rendering, output-folder discipline, seed semantics, and
the dev endpoint contract — they apply identically here. Keep insect-smoke
outputs in their own checkpoint-named folder, separate from smoke-test-1.
