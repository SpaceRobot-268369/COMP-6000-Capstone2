# layer-c-audiogen-boobook-smoke

## Summary

- Owner: Lucas
- Layer / role: Module C — event layer (per-species AudioGen LoRA)
- Status: candidate
- Base model: `facebook/audiogen-medium` (frozen, 16 kHz native)
- Trained at: 2026-05-08

## Purpose / hypothesis

First Layer C smoke test. Validate that a per-species LoRA over AudioGen can
produce recognizable Southern Boobook (*Ninox boobook*) nocturnal calls when
trained on a small curated event-segment set. The goal is qualitative
plausibility, not high-fidelity species reproduction.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/prepared_manifest_boobook.csv`
- Source clips / manifests: 50 segments selected from `BirdNET.results.csv` annotations on site 257 recordings
- Filtering or preprocessing: BirdNET score `>= 0.9`, raw event duration `1.0-10.0 s`, nocturnal preference, distinct recordings where possible, ±3.0 s event buffer
- Known data caveats: small dataset (50 segments); event quality varies with annotation noise

## Training or promotion context

- Training command: see [layer_c_smoke_1_birds runbook](../../../../.claude/context/ai/runbooks/layer_c_smoke_1_birds.md)
- Code branch / commit: `model/lucas/layer-c-event-attemp-1`
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 5 epochs, batch_size 1, lr 1e-5, sample_rate 16000, frame_rate 50

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors`
- DVC pointer files: `adapter_model.safetensors.dvc`
- Params: `params.yaml`
- Metrics: n/a
- Sample outputs: TBD — generate under `debug/layer_c/audiogen/samples/audiogen-lora-boobook-smoke/`
- Related runbook or log: [.claude/context/ai/runbooks/layer_c_smoke_1_birds.md](../../../../.claude/context/ai/runbooks/layer_c_smoke_1_birds.md)
- Training metadata: `training_metadata.json` (reconstructed after a JSON-serialization crash at end of training — adapter weights + config saved successfully)

## Results / metrics

Not evaluated yet. Adapter weights saved; needs an inference run + listening
test to assess whether boobook calls are recognizable.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Trained on only 50 segments; expect mode collapse / repetition.
- No inference defaults yet documented — populate `params.yaml`'s `inference:` block after the first validated generation.
- Layer C wiring into the mixer (Module D) does not exist yet; this candidate is for standalone evaluation.

## Follow-up actions

- Run a first inference pass and write sample outputs.
- If quality is plausible, train the second species (`splendid_fairywren`) per the runbook.
- Resolve the int64 JSON serialization bug at end of training so `training_metadata.json` is written automatically next run.
