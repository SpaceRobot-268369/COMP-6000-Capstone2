# layer-a-ambient-diffusion

## Summary

- Owner: Lucas
- Layer / role: Module A — early ambient-bed generation attempt (superseded)
- Status: deprecated
- Base model: Custom latent-diffusion architecture (not a pretrained base; trained from scratch on Bowra latents)
- Trained at: 2026-05-04 (best.pt timestamp)

## Purpose / hypothesis

Early ambient-bed generation attempt using a latent diffusion model conditioned
on environmental vectors rather than text. Superseded by AudioLDM2 LoRA after
the env-conditioned model produced low-quality output and the text-prompt
approach proved more controllable.

## Dataset / inputs

- Dataset: site 257 latent-encoded clips (via the VAE)
- Filtering or preprocessing: not fully documented
- Known data caveats: training context lost during the AudioLDM2 transition — see git history if reconstruction is needed

## Training or promotion context

Training command and exact hyperparameters not recovered. The checkpoint
predates the per-candidate `params.yaml` discipline; if you need to re-run,
inspect `acoustic_ai/modules/ambient/diffusion/` git history from around
2026-05-04.

## Artifacts

- Checkpoint binaries: `best.pt`, `last.pt`
- DVC pointer files: `best.pt.dvc`, `last.pt.dvc`
- Params: not recorded (params.yaml intentionally absent)
- Metrics: n/a
- Sample outputs: n/a
- Related runbook or log: [.claude/context/ai/logs/audioldm2_transition_log.md](../../../../.claude/context/ai/logs/audioldm2_transition_log.md)

## Results / metrics

Not retained. Output quality was qualitatively worse than the later AudioLDM2
LoRA candidates; this was the motivation to switch.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

Deprecated. Do not use for current Layer A generation. Retained for
historical reproducibility and to document the env-conditioned-diffusion
dead end.

## Follow-up actions

None. Kept only for archival reference.
