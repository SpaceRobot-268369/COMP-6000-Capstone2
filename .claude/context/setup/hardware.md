# Hardware Requirements

## Minimum

- OS: macOS 12+
- CPU: 4-core (Intel Core i5 10th gen / AMD Ryzen 5)
- GPU: NVIDIA GTX 1660 / RTX 2060, 6 GB VRAM
- RAM: 32 GB
- Storage: 1 TB SSD

## Recommended

- OS: macOS 13+
- CPU: 8-core (Intel Core i7 / Apple Silicon M2 Pro)
- GPU: NVIDIA RTX 3080 / RTX 4070 Ti, 12–16 GB VRAM
- RAM: 128 GB
- Storage: 2 TB NVMe SSD

## Notes

- The AI server runs natively for GPU access. On Apple Silicon it uses MPS via
  PyTorch; on Linux/Windows it uses CUDA. Docker cannot reach MPS on macOS.
- AudioLDM2 + AudioGen LoRA training fits in the **minimum** spec at
  `batch_size=1`. The smoke-test runs in this repo were all trained on a
  single Apple Silicon machine.
- Inference is the limiting factor for VRAM: AudioLDM2 needs ~8 GB to
  generate at the smoke defaults (100 steps, ~10 s audio).
