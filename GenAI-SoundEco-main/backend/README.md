# GenAI-SoundEco Backend

Backend for acoustic feature extraction, Mel spectrogram export, PyTorch VGGish embedding generation, and future environment-data alignment.

## Project Goal

This backend is designed for a soundscape analysis workflow:

1. ingest raw audio collected at specific locations and times
2. standardize the audio into a consistent processing format
3. extract structured acoustic features
4. generate VGGish embeddings
5. optionally align audio records with nearby environmental observations
6. export a flat feature table for downstream modelling

The current project focus is not full audio generation. The short-term goal is to build a reusable backend that can help predict likely soundscape patterns or sound categories under given environmental conditions.

## Current Status

Implemented:

- audio loading with `librosa`
- resampling and mono conversion
- audio validation and normalization
- acoustic feature extraction
- Mel spectrogram PNG export
- PyTorch VGGish integration
- file-level and segment-level embedding export
- batch processing for a directory of audio files

Not fully implemented yet:

- robust environment-data alignment rules
- dedicated automated tests
- alternate embedding export formats such as `.npy` or Parquet
- API layer beyond the current script-oriented workflow

## Features Extracted

The backend currently extracts these acoustic features and summarizes each over the full clip with `mean`, `std`, `min`, and `max`:

- RMS energy
- zero crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- MFCC 1 through MFCC 13

Additional metadata fields are also exported, such as:

- `audio_id`
- `file_path`
- `recorded_at`
- `duration`
- `sample_rate`
- `status`

## VGGish Output

The project uses a PyTorch VGGish integration via `torch.hub` with `harritaylor/torchvggish`.

For each audio file, the backend produces:

- segment-level VGGish embeddings
- one aggregated file-level embedding

The file-level embedding is stored in the feature table.  
The segment-level embeddings are saved as separate JSON files under `data/embeddings/`.

On the first run that needs embeddings, PyTorch downloads:

- the `torchvggish` repository
- the VGGish model weights
- the VGGish PCA parameters

These are cached locally under:

- `backend/models/torch_hub/`

## Project Structure

- `app/core`: settings and shared configuration
- `app/schemas`: data contracts
- `app/services`: audio processing, feature extraction, VGGish, storage, and visualization
- `app/pipelines`: single-file and batch orchestration
- `data/raw`: raw audio and raw weather inputs
- `data/features`: feature-table outputs
- `data/embeddings`: segment-level embedding files
- `data/metadata`: processing logs
- `data/visualizations/spectrograms`: Mel spectrogram PNGs
- `scripts`: command-line entrypoints
- `models/torch_hub`: cached VGGish model code and weights

## Environment Setup

Recommended Python version:

- Python `3.11`

Create a virtual environment from the repository root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r backend\requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

## How To Run

Place your input audio files in any accessible directory, for example:

- `backend/data/raw/audio/`
- or another local folder such as `path/to/audio_dir`

Run batch processing on a directory of audio files:

```powershell
python backend\scripts\run_batch_pipeline.py --audio_dir path\to\audio_dir
```

Run with an environment file:

```powershell
python backend\scripts\run_batch_pipeline.py --audio_dir path\to\audio_dir --env_file path\to\weather.csv
```

Run and export Mel spectrogram images:

```powershell
python backend\scripts\run_batch_pipeline.py --audio_dir path\to\audio_dir --save_spectrograms
```

Run without saving segment-level embedding files:

```powershell
python backend\scripts\run_batch_pipeline.py --audio_dir path\to\audio_dir --disable_segment_embedding_export
```

Example with the repository's default raw-audio directory:

```powershell
python backend\scripts\run_batch_pipeline.py --audio_dir backend\data\raw\audio
```

Important:

- on the first run that uses VGGish, the project will download model files from GitHub
- those files are cached under `backend/models/torch_hub/`
- if you delete that cache, the next VGGish run will download them again

## API

The backend now includes a minimal FastAPI service.

Start the API from the repository root:

```powershell
python backend\scripts\run_api.py
```

Available endpoints:

- `GET /health`
- `POST /process/file`
- `POST /process/batch`
- `GET /results/{audio_id}`

Example:

- upload one audio file through `/process/file`
- optionally set `save_spectrogram=true`
- optionally set `save_segment_embeddings=false`

The file-processing endpoint returns the extracted result row and any generated output paths.

For batch processing, send a JSON body such as:

```json
{
  "audio_dir": "D:\\path\\to\\audio_dir",
  "save_spectrogram": true,
  "save_segment_embeddings": true
}
```

The result-lookup endpoint reads the latest matching record from `data/metadata/processing_log.jsonl`.

## Output Files

Main outputs:

- feature table: `backend/data/features/audio_feature_table.csv`
- processing log: `backend/data/metadata/processing_log.jsonl`
- segment embeddings: `backend/data/embeddings/*.json`
- Mel spectrograms: `backend/data/visualizations/spectrograms/*.png`

The feature table includes:

- structured acoustic features
- aggregated VGGish embedding
- processing status
- optional environment-match fields
- optional Mel spectrogram path
- optional segment embedding path

## Verified Dataset

The current pipeline has already been validated on:

- `489` `.wav` files
- approximately `10` seconds per file
- mono recordings

Observed results during validation:

- audio preprocessing succeeded
- acoustic feature extraction succeeded
- Mel spectrogram export succeeded
- real VGGish embeddings were generated successfully

## Notes

- the current VGGish integration is practical and working, but it depends on first-run network access to download the model files
- a NumPy-related warning may appear during PyTorch deserialization; it does not currently block embedding generation
- the current environment-alignment logic is intentionally minimal and should be expanded later

## Next Build Targets

1. add automated tests for preprocessing, feature extraction, VGGish, and visualization
2. improve environment-data alignment and matching rules
3. support additional export formats for embeddings
4. add more acoustic and ecoacoustic features when needed
5. expose the pipeline through a cleaner API layer
