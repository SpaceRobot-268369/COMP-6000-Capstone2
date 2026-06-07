# BWE prototype workspace

Offline prototype workspace for bandwidth expansion experiments on the rain
showcase samples.

Scope for phase 0:

- Read-only input: `../../showcase/seed_*_generated/audio.wav`
- Output only inside this `bwe_prototype/` directory
- Do not modify `handler.py`, `sample_audioldm2.py`, registry wiring, or the
  generation pipeline

Planned outputs when later phases run:

- `outputs/seed_*_generated/audio_bwe.wav`
- `outputs/seed_*_generated/metadata.json`
- comparison figures under `figures/`

Current script:

- `scripts/bwe_prototype.py`
