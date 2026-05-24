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
    encode_clip, generate_ambient_audio, estimate_env_conditions,
    generate_layer_a_ambient_audio, generate_layer_a_smoke_test_audio,
    DEFAULT_CKPT, CLIPS_PATH,
)

from modules.ambient.diffusion.layer_a_visualization import render_layer_a_mel_png_bytes
from modules.weather.segment_selector import select_weather_segments

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
        mel_db, wav_bytes = generate_ambient_audio(env_dict, noise_std=body.noise_std, seed=body.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    png_b64 = _mel_to_png_b64(mel_db)

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
    seed: Optional[int] = 42


class LayerBSegmentRequest(BaseModel):
    query: Optional[str] = None
    weather_types: Optional[list[str]] = None
    wind_speed_ms: Optional[float] = None
    precipitation_mm: Optional[float] = None
    include_thunder: bool = False
    target_duration: float = 30.0
    top_assets: int = 3
    segments_per_type: int = 2
    window_seconds: float = 10.0
    overlap_seconds: float = 2.0


@app.post("/layer_a/generate")
def layer_a_generate(body: LayerARequest):
    """Generate Layer A with the trained AudioLDM2 LoRA smoke-test model.

    The prompt and checkpoint are fixed at this stage because the model has only
    been validated on the tiny smoke dataset.
    """
    try:
        mel_db, wav_bytes, metadata = generate_layer_a_ambient_audio(seed=body.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer A generation failed: {exc}")

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    png_b64 = _layer_a_mel_to_png_b64(mel_db, metadata["audio"]["duration_s"])

    return {
        "ok":         True,
        "audio_b64":  audio_b64,
        "image_b64":  png_b64,
        "metadata":   metadata,
        "gain_db":    0.0,
        "sample_rate": metadata["audio"]["sample_rate"],
        "duration_s": metadata["audio"]["duration_s"],
    }


@app.post("/layer_a/smoke_test_1/generate")
def layer_a_smoke_test_1_generate(body: LayerARequest):
    """Generate Layer A smoke test 1 with the spring-night LoRA."""
    return _layer_a_smoke_test_response("smoke_test_1", body.seed)


@app.post("/layer_a/smoke_test_2/generate")
def layer_a_smoke_test_2_generate(body: LayerARequest):
    """Generate Layer A smoke test 2 with the insect/cicada LoRA."""
    return _layer_a_smoke_test_response("smoke_test_2", body.seed)


@app.post("/layer_b/select_segments")
def layer_b_select_segments(body: LayerBSegmentRequest):
    """Select Layer B weather segments for Layer D.

    This endpoint returns metadata only. It does not render the final weather
    layer and does not decide timeline placement; Layer D owns those steps.
    """
    allowed_types = {"wind", "rain", "thunder"}
    weather_types = body.weather_types
    if weather_types:
        invalid = sorted(set(weather_types).difference(allowed_types))
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid weather_types: {invalid}. Allowed: wind, rain, thunder.",
            )

    try:
        return select_weather_segments(
            query=body.query,
            weather_types=weather_types,  # type: ignore[arg-type]
            wind_speed_ms=body.wind_speed_ms,
            precipitation_mm=body.precipitation_mm,
            include_thunder=body.include_thunder,
            target_duration=body.target_duration,
            top_assets=body.top_assets,
            segments_per_type=body.segments_per_type,
            window_seconds=body.window_seconds,
            overlap_seconds=body.overlap_seconds,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer B segment selection failed: {exc}")


def _layer_a_smoke_test_response(smoke_test_id: str, seed: Optional[int]):
    try:
        mel_db, wav_bytes, metadata = generate_layer_a_smoke_test_audio(
            smoke_test_id,
            seed=seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Layer A generation failed: {exc}")

    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    png_b64 = _layer_a_mel_to_png_b64(mel_db, metadata["audio"]["duration_s"])

    return {
        "ok":         True,
        "audio_b64":  audio_b64,
        "image_b64":  png_b64,
        "metadata":   metadata,
        "gain_db":    0.0,
        "sample_rate": metadata["audio"]["sample_rate"],
        "duration_s": metadata["audio"]["duration_s"],
    }


def _layer_a_mel_to_png_b64(mel_db: np.ndarray, duration_s: float) -> str:
    png_bytes = render_layer_a_mel_png_bytes(mel_db, duration_s)
    return base64.b64encode(png_bytes).decode("utf-8")


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
