# Layer C Smoke Test 1 — Bird Events (AudioGen LoRA per species)

Per-species event-generation LoRA. Trial stage — not yet wired into the dev
endpoint or the mixer.

## Scope

Train or validate **bird vocal events only**. The current annotation dataset
contains only bird species detections, so treat event types as species-level
classes. Exclude insects/cicadas, frogs, mammals, weather events, wind gusts,
human/vehicle/anthropogenic sounds, and generic `Unknown` from this source —
those need separate Layer B assets, another detector, or a later annotation
pass.

Default smoke set: two event types × 50 segments each:
- `Southern Boobook` / *Ninox boobook* — nocturnal owl calls
- `Splendid Fairywren` / *Malurus splendens* — common dawn/diurnal passerine

Selection rules used by the manifest builder:
- BirdNET score `>= 0.9`
- Raw event duration `1.0–10.0 s`
- Event-type-specific diel preference (nocturnal / diurnal)
- Distinct recordings where possible
- Standard `±3.0 s` event buffer

| Asset | Path |
|---|---|
| Base model | `facebook/audiogen-medium` (frozen, 16 kHz native) |
| Annotation source | `resources/site_257_bowra-dry-a/all_items_annotation/.../BirdNET.results.csv` (use as index — do NOT download the full per-event segment archive) |
| Dataset root | `resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/` |
| First trained LoRA | `model/candidates/lucas/layer-c-audiogen-boobook-smoke/` |
| Per-run params | `model/candidates/lucas/layer-c-audiogen-boobook-smoke/params.yaml` |

## Build event manifest and download segments

```bash
python3 script/dataset/build_layer_c_smoke_manifest.py \
  --output resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/manifest.csv \
  --event-type boobook \
  --event-type splendid_fairywren \
  --segments-per-type 50

python3 script/download/download_site_257_event_segments.py \
  --event-manifest resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/manifest.csv \
  --output-dir resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/segments \
  --min-score 0.9 \
  --min-duration 1.0 \
  --max-duration 10.0 \
  --workers 2
```

## Train (per-species)

Train one LoRA per species. Boobook example:

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/train_audiogen_lora.py \
  --manifest resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/prepared_manifest_boobook.csv \
  --output_dir model/candidates/lucas/layer-c-audiogen-boobook-smoke \
  --num_epochs 5 \
  --batch_size 1 \
  --learning_rate 1e-5
```

Output: 32 MB LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`
+ `training_metadata.json`). The binary is tracked by DVC.

## Sample

```bash
./acoustic_ai/.venv/bin/python acoustic_ai/layers/layer_c/attempts/lucas__smoke_1__audiogen_boobook/sample_audiogen_lora.py \
  --lora_dir model/candidates/lucas/layer-c-audiogen-boobook-smoke \
  --output_dir debug/layer_c/audiogen/samples/audiogen-lora-boobook-smoke/
```

## Notes

- AudioGen runs natively at 16 kHz. Resample to 22.05 kHz at the **mixer
  boundary** (Module D), not earlier — keeping native rate through Layer C
  inference avoids cascaded resampling artifacts.
- Each per-species LoRA gets its own candidate folder: `layer-c-audiogen-<species>-<context>/`.
- The currently-trained `boobook` checkpoint hit a JSON-serialization crash at
  the end of training (int64 not serializable). Adapter weights + config saved
  successfully; `training_metadata.json` was reconstructed afterward.
