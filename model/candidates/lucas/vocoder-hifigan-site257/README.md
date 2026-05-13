# vocoder-hifigan-site257

## Summary

- Owner: Lucas
- Layer / role: Cross-cutting infrastructure — converts mel spectrograms back to waveform for any layer that outputs in mel space
- Status: candidate
- Base model: HiFi-GAN (V1-style generator, no pretrained weights — trained from scratch on ecoacoustic audio)
- Trained at: 2026-04

## Purpose / hypothesis

A HiFi-GAN vocoder fine-tuned on site 257 audio at 22.05 kHz, 128 mel bins. The
goal: produce ecoacoustic-faithful waveform reconstruction from mel
spectrograms, replacing generic speech-domain vocoders that introduce
artefacts on bird calls and ambient textures.

## Dataset / inputs

- Dataset: site 257 clips listed in `resources/site_257_bowra-dry-a/site_257_training_manifest.csv`
- Source clips / manifests: up to 500 clips sampled (`max_clips: 500`)
- Filtering or preprocessing: 22.05 kHz mono mel, 128 bins, hop 512
- Known data caveats: trained on the same source as the VAE — anthropogenic noise (vehicles, etc.) is present in the clips

## Training or promotion context

- Training command: `python3 acoustic_ai/modules/ambient/train_vocoder.py` (driven by `dvc.yaml`'s `train_vocoder` stage)
- Code branch / commit: `main` family at training time
- Hardware: single Apple Silicon machine, MPS
- Important settings: see `params.yaml` — 100 epochs, batch_size 8, lr 2e-4, base_channels 128, upsample rates `[8, 8, 4, 2]`

## Artifacts

- Checkpoint binaries: `best.pt`
- DVC pointer files: tracked via `dvc.lock` under stage `train_vocoder` (no separate `.dvc` pointer)
- Params: `params.yaml`
- Metrics: not currently written by `train_vocoder.py` (TODO)
- Sample outputs: n/a (vocoder; outputs are produced indirectly via Layer A/C generation pipelines)
- Related runbook or log: [.claude/context/ai/logs/audioldm2_transition_log.md](../../../../.claude/context/ai/logs/audioldm2_transition_log.md)

## Results / metrics

Not formally evaluated. Used by `acoustic_ai/server/inference.py` as the
canonical mel-to-wav step for Layer A spectrogram output. Smoke-tested
qualitatively during the AudioLDM2 transition.

## Results analysis / audit

_Empty until developer evaluation notes are provided._

## Known limitations

- No formal MOS or objective vocoder metrics recorded.
- Trained on a relatively small subset (500 clips); broader-distribution coverage might need a larger training set.

## Follow-up actions

- Add metrics output to `train_vocoder.py` (loss curves, qualitative samples).
- Compare against an off-the-shelf HiFi-GAN baseline to justify the custom train.
