"""Latent diffusion module for Layer A (Method X — CLAP-conditioned).

Pipeline:
  Audio → Mel-spec → VAE Encoder → Latent
  Text  → CLAP Encoder          → Text embedding
  (noisy latent, timestep, text embedding) → LatentDenoiser → v-pred
  Sampled latent → VAE Decoder → Mel-spec → Vocoder → Waveform

See `.claude/context/branches/layer-a-ambient-method-X/` for design rationale
(promote to a permanent doc under `.claude/context/ai/` on merge).
"""
