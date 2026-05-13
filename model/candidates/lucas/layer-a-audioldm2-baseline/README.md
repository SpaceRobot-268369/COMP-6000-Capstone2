# layer-a-audioldm2-baseline

## Summary

- Owner: Lucas
- Layer / role: Module A — undocumented orphan AudioLDM2 LoRA run (predates the smoke-test discipline)
- Status: deprecated (context lost)
- Base model: `cvssp/audioldm2` (frozen UNet; PEFT LoRA on attention projections)
- Trained at: 2026-05 (best.pt timestamp; exact date unknown)

## Purpose / hypothesis

Originally landed in the repo as `acoustic_ai/checkpoints/audioldm2-lora/` —
an early AudioLDM2 LoRA training run from before the smoke-test workflow
was formalized. Purpose, dataset, and hypothesis were not documented at
training time. Renamed to `layer-a-audioldm2-baseline` during the checkpoint
reorganization and retained for archival reference, but the run is not
reproducible from current docs alone.

## Dataset / inputs

Unknown. The training manifest used for this run was not recorded in the
candidate folder or in any git-tracked log. To reconstruct, inspect
`acoustic_ai/modules/ambient/diffusion/train_audioldm2.py` history around
the `best.pt` mtime and look for matching commits.

## Training or promotion context

Not documented. The adapter_config.json reveals the LoRA structure
(`r=8, alpha=32, dropout=0.1, target_modules=[to_q, to_k, to_v, to_out.0]`,
PEFT 0.19.1) but training hyperparameters and dataset path are lost.

## Artifacts

- Checkpoint binaries: `adapter_model.safetensors` (LoRA adapter only)
- DVC pointer files: `adapter_model.safetensors.dvc`
- Config: `adapter_config.json`
- Params: not recorded (params.yaml intentionally absent — no honest reconstruction)
- Metrics: n/a
- Sample outputs: n/a
- Related runbook or log: n/a

## Results / metrics

Unknown. Outputs of this run, if any were generated, are not retained.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Training context lost. Do not treat this checkpoint as a meaningful baseline
  or reference point.
- If a true AudioLDM2 baseline (no LoRA) is needed for comparison, generate
  one fresh from `cvssp/audioldm2` directly rather than relying on this run.

## Follow-up actions

- Delete in a future cleanup pass once it's been confirmed that no other
  branch or experiment references this checkpoint.
