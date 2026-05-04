"""FastAPI inference server for the SoundscapeModel.

Runs on port 8000 (internal only — not exposed to the browser directly).
Express backend proxies requests here.

Endpoints:
  POST /analysis    — encode an uploaded .wav clip → latent vector JSON
  POST /generation  — env conditions JSON → generated spectrogram as base64 PNG
  GET  /health      — liveness check

Usage (from acoustic_ai/):
  pip install -r requirements.txt
  uvicorn server.server:app --reload --port 8000
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

# Ensure acoustic_ai root is importable (for modules.* and server.*)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.inference import (
    encode_clip, generate_spectrogram, estimate_env_conditions,
    mel_db_to_wav, mel_db_to_wav_hifigan, mel_db_to_wav_ecoacoustic,
    DEFAULT_CKPT, VOCODER_CKPT, CLIPS_PATH,
)

from modules.ambient.retrieval import AmbientRetriever
from modules.ambient.preprocess import SPEC_CFG, waveform_to_melspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LAYER_A_INDEX = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_index.csv"
LAYER_A_SEGMENTS = PROJECT_ROOT / "acoustic_ai" / "data" / "ambient" / "ambient_segments"
_layer_a_retriever: Optional[AmbientRetriever] = None


def _get_layer_a_retriever() -> AmbientRetriever:
    global _layer_a_retriever
    if _layer_a_retriever is None:
        _layer_a_retriever = AmbientRetriever(LAYER_A_INDEX, LAYER_A_SEGMENTS)
    return _layer_a_retriever

app = FastAPI(title="Soundscape Inference API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:5173", "http://localhost:5174"],
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

    # Nearest-neighbour env estimation — requires latent_clips.npy
    estimated_conditions: dict = {}
    if CLIPS_PATH.exists():
        try:
            import numpy as _np
            clips = _np.load(str(CLIPS_PATH), allow_pickle=True).item()
            estimated_conditions = estimate_env_conditions(latent, clips, top_k=5)
        except Exception as exc:
            print(f"[WARN] Env estimation failed: {exc}")

    return {
        "ok":                    True,
        "latent_dim":            len(latent),
        "latent":                latent.tolist(),
        "estimated_conditions":  estimated_conditions,
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

    png_b64 = _mel_to_png_b64(mel_db)

    # Vocoder priority:
    #   1. Ecoacoustic HiFi-GAN (fine-tuned on site 257, 128-bin 22kHz) — best quality
    #   2. Speech HiFi-GAN (SpeechT5, 80-bin 16kHz + interp)             — Stage 2 fallback
    #   3. Griffin-Lim (no neural model required)                          — last resort
    if VOCODER_CKPT.exists():
        try:
            wav_bytes = mel_db_to_wav_ecoacoustic(mel_db)
            print("[INFO] Ecoacoustic vocoder succeeded.")
        except Exception as exc:
            print(f"[WARN] Ecoacoustic vocoder failed ({exc}), trying speech HiFi-GAN.")
            wav_bytes = None
    else:
        wav_bytes = None

    if wav_bytes is None:
        try:
            wav_bytes = mel_db_to_wav_hifigan(mel_db)
            print("[INFO] Speech HiFi-GAN vocoding succeeded.")
        except Exception as exc:
            print(f"[WARN] Speech HiFi-GAN failed ({exc}), falling back to Griffin-Lim.")
            try:
                wav_bytes = mel_db_to_wav(mel_db)
            except Exception as exc2:
                wav_bytes = b""
                print(f"[WARN] Griffin-Lim also failed: {exc2}")
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8") if wav_bytes else ""

    return {
        "ok":        True,
        "shape":     list(mel_db.shape),
        "image_b64": png_b64,
        "audio_b64": audio_b64,
    }


# ---------------------------------------------------------------------------
# Layer A — Ambient bed (dev test endpoint)
# ---------------------------------------------------------------------------

class LayerARequest(BaseModel):
    diel_bin: str           # dawn | morning | afternoon | night
    season:   str           # spring | summer | autumn | winter
    hour:     int           # 0..23
    month:    int           # 1..12
    duration: float = 60.0
    k:        int   = 5


@app.post("/layer_a/generate")
def layer_a_generate(body: LayerARequest):
    """Retrieve+blend an ambient bed and return WAV+PNG+metadata."""
    if not (0 <= body.hour <= 23):
        raise HTTPException(status_code=400, detail="hour must be 0..23")
    if not (1 <= body.month <= 12):
        raise HTTPException(status_code=400, detail="month must be 1..12")
    if body.diel_bin not in {"dawn", "morning", "afternoon", "night"}:
        raise HTTPException(status_code=400, detail="invalid diel_bin")
    if body.season not in {"spring", "summer", "autumn", "winter"}:
        raise HTTPException(status_code=400, detail="invalid season")

    try:
        retriever = _get_layer_a_retriever()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ambient pool not available: {exc}")

    try:
        result = retriever.retrieve(
            diel_bin=body.diel_bin,
            season=body.season,
            hour=body.hour,
            month=body.month,
            k=body.k,
            target_duration_s=body.duration,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Retrieval failed: {exc}")

    # Encode WAV (PCM16) to base64.
    import soundfile as sf
    wav_buf = io.BytesIO()
    sf.write(wav_buf, result.audio, result.sample_rate, subtype="PCM_16", format="WAV")
    audio_b64 = base64.b64encode(wav_buf.getvalue()).decode("utf-8")

    # Render mel-spectrogram PNG.
    mel_db = waveform_to_melspec(result.audio)
    png_b64 = _layer_a_mel_to_png_b64(mel_db, result.audio.size / result.sample_rate)

    return {
        "ok":         True,
        "audio_b64":  audio_b64,
        "image_b64":  png_b64,
        "metadata":   result.metadata,
        "gain_db":    result.gain_db,
        "sample_rate": result.sample_rate,
        "duration_s": float(result.audio.size / result.sample_rate),
    }


def _layer_a_mel_to_png_b64(mel_db: np.ndarray, duration_s: float) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    img = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        extent=[0, duration_s, 0, SPEC_CFG["n_mels"]],
        cmap="magma",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title("Layer A — Ambient Bed Spectrogram")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


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
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
