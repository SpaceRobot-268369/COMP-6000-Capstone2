"""FastAPI inference server for the SoundscapeModel.

Runs on port 8000 (internal only — not exposed to the browser directly).
Express backend proxies requests here.

Endpoints:
  POST /analysis    — encode an uploaded .wav clip → latent vector JSON
  POST /generation  — env conditions JSON → generated spectrogram as base64 PNG
  GET  /health      — liveness check

Usage (from project root):
  pip install -r acoustic_ai/requirements.txt
  python3 acoustic_ai/server.py
  # or with auto-reload during development:
  uvicorn acoustic_ai.server:app --reload --port 8000
"""

import base64
import io
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure acoustic_ai modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import encode_clip, generate_spectrogram, DEFAULT_CKPT

app = FastAPI(title="Soundscape Inference API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EnvFeatures(BaseModel):
    temperature_c:          float = 20.0
    humidity_pct:           float = 60.0
    wind_speed_ms:          float = 2.0
    precipitation_mm:       float = 0.0
    solar_radiation_wm2:    float = 300.0
    cloud_clearness_index:  float = 0.5
    surface_pressure_kpa:   float = 101.3
    temp_max_c:             float = 25.0
    temp_min_c:             float = 15.0
    precipitation_daily_mm: float = 0.0
    wind_max_ms:            float = 5.0
    days_since_rain:        float = 3.0
    daylight_hours:         float = 11.0
    hour_utc:               float = 6.0
    hour_local:             float = 16.0
    wind_direction_deg:     float = 180.0
    month:                  float = 9.0
    day_of_year:            float = 260.0
    season:                 str   = "spring"
    sample_bin:             str   = "afternoon"
    noise_std:              float = 0.5   # generation only
    seed:                   Optional[int] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "checkpoint": str(DEFAULT_CKPT), "exists": DEFAULT_CKPT.exists()}


@app.post("/analysis")
async def analysis(
    file: UploadFile = File(...),
    temperature_c:          float = 20.0,
    humidity_pct:           float = 60.0,
    wind_speed_ms:          float = 2.0,
    precipitation_mm:       float = 0.0,
    solar_radiation_wm2:    float = 300.0,
    cloud_clearness_index:  float = 0.5,
    surface_pressure_kpa:   float = 101.3,
    temp_max_c:             float = 25.0,
    temp_min_c:             float = 15.0,
    precipitation_daily_mm: float = 0.0,
    wind_max_ms:            float = 5.0,
    days_since_rain:        float = 3.0,
    daylight_hours:         float = 11.0,
    hour_utc:               float = 6.0,
    hour_local:             float = 16.0,
    wind_direction_deg:     float = 180.0,
    month:                  float = 9.0,
    day_of_year:            float = 260.0,
    season:                 str   = "spring",
    sample_bin:             str   = "afternoon",
):
    """Encode an uploaded audio file into a latent vector.

    Accepts multipart/form-data with a 'file' field (.wav or .webm.wav).
    Returns the 256-dim latent vector as a JSON array.
    """
    if not DEFAULT_CKPT.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found.")

    env_dict = {
        "temperature_c": temperature_c, "humidity_pct": humidity_pct,
        "wind_speed_ms": wind_speed_ms, "precipitation_mm": precipitation_mm,
        "solar_radiation_wm2": solar_radiation_wm2,
        "cloud_clearness_index": cloud_clearness_index,
        "surface_pressure_kpa": surface_pressure_kpa,
        "temp_max_c": temp_max_c, "temp_min_c": temp_min_c,
        "precipitation_daily_mm": precipitation_daily_mm,
        "wind_max_ms": wind_max_ms, "days_since_rain": days_since_rain,
        "daylight_hours": daylight_hours, "hour_utc": hour_utc,
        "hour_local": hour_local, "wind_direction_deg": wind_direction_deg,
        "month": month, "day_of_year": day_of_year,
        "season": season, "sample_bin": sample_bin,
    }

    # Save upload to a temp file so librosa can read it
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        latent = encode_clip(tmp_path, env_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Encoding failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "ok":          True,
        "latent_dim":  len(latent),
        "latent":      latent.tolist(),
    }


@app.post("/generation")
def generation(body: EnvFeatures):
    """Generate a spectrogram from environmental conditions.

    Returns a base64-encoded PNG of the mel-spectrogram and the raw
    dB matrix as a nested JSON array.
    """
    if not DEFAULT_CKPT.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found.")

    env_dict = body.model_dump(exclude={"noise_std", "seed"})

    try:
        mel_db = generate_spectrogram(env_dict, noise_std=body.noise_std, seed=body.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    # Encode as base64 PNG for easy display in the frontend
    png_b64 = _mel_to_png_b64(mel_db)

    return {
        "ok":        True,
        "shape":     list(mel_db.shape),
        "mel_db":    mel_db.tolist(),
        "image_b64": png_b64,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mel_to_png_b64(mel_db: np.ndarray) -> str:
    """Convert a (128, T) dB spectrogram to a base64 PNG string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
                  vmin=-80, vmax=0)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.set_title("Generated Mel-Spectrogram")
        plt.colorbar(ax.images[0], ax=ax, label="dB")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        # matplotlib not installed — return empty string, mel_db still returned
        return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
