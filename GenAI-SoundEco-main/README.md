# GenAI-SoundEco

GenAI-SoundEco is a soundscape-analysis project focused on linking location-and-time-specific acoustic recordings with nearby environmental observations.

## Current Scope

The current repository work is centered on the backend processing pipeline:

- audio preprocessing
- acoustic feature extraction
- Mel spectrogram export for inspection
- PyTorch VGGish embedding generation
- batch processing for collections of audio clips

The short-term goal is not full audio generation. The current goal is to build a reusable backend that can help support downstream models for predicting likely soundscape patterns or sound categories under given environmental conditions.

## Quick Start

Recommended Python version:

- Python `3.11`

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python backend\scripts\run_batch_pipeline.py --audio_dir path\to\audio_dir
```

Optional:

- add `--save_spectrograms` to export Mel spectrogram PNGs
- add `--env_file path\to\weather.csv` to include environment data

Note:

- the first VGGish run will download model files from GitHub and cache them locally

## Repository Layout

- `backend/`: main backend project
- `backend/app/`: source code
- `backend/data/`: runtime input/output directories
- `backend/scripts/`: command-line entrypoints

## Main Documentation

Detailed backend setup and usage instructions are in:

- [backend/README.md](backend/README.md)

## Current Backend Capabilities

- load and standardize audio
- extract RMS, ZCR, spectral, and MFCC features
- generate Mel spectrogram PNGs
- generate real VGGish embeddings with PyTorch
- export feature tables and per-file embedding outputs

## Suggested Next Steps

1. add automated tests
2. improve environment-data alignment
3. expand acoustic and ecoacoustic feature sets
4. cleanly expose the backend through APIs or services
