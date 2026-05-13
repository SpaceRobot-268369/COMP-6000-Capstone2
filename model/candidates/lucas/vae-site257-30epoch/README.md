# vae-site257-30epoch

## Summary

- Owner: Lucas
- Layer / role: Module A (legacy retrieval baseline) + Module E (analysis similarity)
- Status: candidate
- Base model: Custom convolutional VAE (`SoundscapeModel`) — not a pretrained base
- Trained at: 2026-04 (best.pt epoch)

## Purpose / hypothesis

A 256-dim latent VAE trained on Bowra site-257 mel spectrograms with 29-dim
environmental conditioning. Originally produced as the Layer A retrieval
baseline (nearest-neighbour over `latent_clips.npy`); now retained mainly
as the embedding for Module E analysis similarity. Generation has migrated
to AudioLDM2 LoRA candidates; this checkpoint is no longer on the active
Layer A generation path.

## Dataset / inputs

- Dataset: `resources/site_257_bowra-dry-a/site_257_training_manifest.csv` (5,318 rows / 30 s crops)
- Source clips / manifests: site 257 Bowra dry woodland, ~510 hrs of FLAC re-rendered to mel at 22.05 kHz mono
- Filtering or preprocessing: stratified train/val split by `sample_bin` (val_frac = 0.15); mel normalised `(dB + 80) / 80 → [0, 1]`; random 30 s crop per step
- Known data caveats: trained on uncleaned full clips — latent manifold mixes ambient with foreground events, so it is **not** an "ambient-only" embedding

## Training or promotion context

- Training command: `python3 acoustic_ai/modules/ambient/train.py` (driven by `dvc.yaml`'s `train_vae` stage)
- Code branch / commit: `main` family at training time
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 30 epochs, batch_size 16, lr 1e-4, beta_kl 0.01, AdamW + CosineAnnealingLR

## Artifacts

- Checkpoint binaries: `best.pt`
- DVC pointer files: tracked via `dvc.lock` under stage `train_vae` (no separate `.dvc` pointer)
- Params: `params.yaml`
- Metrics: not currently written by `train.py` (TODO)
- Sample outputs: n/a (this candidate is for retrieval / similarity, not generation)
- Related runbook or log: [.claude/context/ai/architecture.md](../../../../.claude/context/ai/architecture.md), [.claude/context/ai/logs/mvp_decision_log.md](../../../../.claude/context/ai/logs/mvp_decision_log.md)

## Results / metrics

- Val loss ≈ 0.003580 at best epoch
- KL / element ≈ 0.05

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- Latent manifold mixes ambient with foreground events; do not use for sampling/decoding ambient beds.
- Currently used by inference as the retrieval baseline; Layer A generation has moved to AudioLDM2 LoRA.

## Follow-up actions

- Write `metrics.json` from `train.py` so it can be tracked by `dvc.yaml`.
- Evaluate whether the VAE remains useful for Module E similarity after the AudioLDM2 transition, or replace with a CLAP-derived embedding.
