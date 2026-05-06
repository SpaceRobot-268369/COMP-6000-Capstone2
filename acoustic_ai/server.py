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

from inference import (
    encode_clip, generate_spectrogram, estimate_env_conditions,
    mel_db_to_wav, mel_db_to_wav_hifigan, mel_db_to_wav_ecoacoustic,
    DEFAULT_CKPT, VOCODER_CKPT, CLIPS_PATH,
)
from layer_a import generate_layer_a_response
from layer_b import prepare_weather_layers
from layer_c import prepare_event_layers
from layer_d import mix_generation_layers


def month_range_for_month(month: float) -> str:
    m = int(round(float(month)))
    if m == 12 or m in (1, 2):
        return "December-February"
    if 3 <= m <= 5:
        return "March-May"
    if 6 <= m <= 8:
        return "June-August"
    return "September-November"


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _level(value: float, light: float, strong: float, labels: tuple[str, str, str]) -> str:
    if value >= strong:
        return labels[2]
    if value >= light:
        return labels[1]
    return labels[0]


def _analysis_spectrogram_png_b64(mel_db: np.ndarray) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.set_title("Analysis Mel-Spectrogram")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return ""


def basic_audio_fallback_analysis(audio_path: str) -> dict:
    """Return explainable analysis that does not require a trained checkpoint."""
    import librosa

    from preprocess import SPEC_CFG, waveform_to_melspec

    sample_rate = SPEC_CFG["sample_rate"]
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=300.0)
    if waveform.size == 0:
        raise ValueError("Uploaded audio contains no readable samples.")

    duration_sec = float(waveform.size / sr)
    rms = librosa.feature.rms(y=waveform, frame_length=SPEC_CFG["n_fft"], hop_length=SPEC_CFG["hop_length"])[0]
    rms_mean = float(np.mean(rms))
    rms_db = float(20.0 * np.log10(rms_mean + 1e-8))
    peak = float(np.max(np.abs(waveform)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(waveform)[0]))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)[0]))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)[0]))
    onset_env = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=SPEC_CFG["hop_length"])
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=SPEC_CFG["hop_length"],
        units="frames",
        backtrack=False,
    )
    transient_rate = float(len(onset_frames) / max(duration_sec, 1e-6))
    onset_density = _clip01(transient_rate / 3.0)

    stft = np.abs(librosa.stft(waveform, n_fft=SPEC_CFG["n_fft"], hop_length=SPEC_CFG["hop_length"])) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=SPEC_CFG["n_fft"])
    total_energy = float(np.sum(stft) + 1e-8)
    low_ratio = float(np.sum(stft[freqs < 500.0]) / total_energy)
    mid_ratio = float(np.sum(stft[(freqs >= 500.0) & (freqs < 4000.0)]) / total_energy)
    high_ratio = float(np.sum(stft[freqs >= 4000.0]) / total_energy)
    low_high_ratio = float(low_ratio / (high_ratio + 1e-8))

    loudness_norm = _clip01((rms_db + 55.0) / 45.0)
    brightness = _clip01((centroid - 600.0) / 4200.0)
    brightness_label = _level(brightness, 0.35, 0.65, ("dark", "balanced", "bright"))
    sound_density = _clip01(loudness_norm * 0.45 + flatness * 1.65 + onset_density * 0.35)
    activity_index = _clip01(onset_density * 0.55 + high_ratio * 0.65 + mid_ratio * 0.25 + loudness_norm * 0.15)
    wind_index = _clip01((low_ratio * 1.9 + mid_ratio * 0.25 + (1.0 - onset_density) * 0.25) * (0.35 + loudness_norm))
    rain_index = _clip01(high_ratio * 1.35 + flatness * 2.6 + zcr * 0.25 + sound_density * 0.45)
    bio_index = _clip01(onset_density * 0.70 + high_ratio * 0.60 + mid_ratio * 0.20 - flatness * 0.35)
    time_hint_score = _clip01(bio_index * 0.65 + brightness * 0.25 + onset_density * 0.25)
    time_hint = "dawn/morning" if time_hint_score >= 0.55 else "night" if brightness < 0.22 and activity_index < 0.35 else "day/afternoon"
    mel_db = waveform_to_melspec(waveform)

    acoustic_features = {
        "duration_sec": round(duration_sec, 2),
        "sample_rate": int(sr),
        "rms_db": round(rms_db, 2),
        "peak_amplitude": round(peak, 4),
        "zero_crossing_rate": round(zcr, 4),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_bandwidth_hz": round(bandwidth, 2),
        "spectral_rolloff_hz": round(rolloff, 2),
        "spectral_flatness": round(flatness, 4),
        "transient_rate_per_sec": round(transient_rate, 4),
        "onset_density": round(onset_density, 4),
        "low_energy_ratio": round(low_ratio, 4),
        "mid_energy_ratio": round(mid_ratio, 4),
        "high_energy_ratio": round(high_ratio, 4),
        "low_high_energy_ratio": round(low_high_ratio, 4),
        "sound_density": round(sound_density, 3),
        "brightness": round(brightness, 3),
        "brightness_label": brightness_label,
        "activity_level": _level(activity_index, 0.35, 0.65, ("low", "moderate", "high")),
        "activity_score": round(activity_index, 3),
        "wind_texture_proxy": round(wind_index, 3),
        "rain_texture_proxy": round(rain_index, 3),
    }

    wind_level = _level(wind_index, 0.35, 0.62, ("none", "light", "strong"))
    rain_level = _level(rain_index, 0.38, 0.68, ("none", "light", "dense"))
    activity_level = _level(activity_index, 0.35, 0.65, ("low", "moderate", "high"))
    heuristic_environment = {
        "wind": {
            "level": wind_level,
            "confidence": round(wind_index, 3),
            "explanation": (
                f"{wind_level.capitalize()} wind likelihood from low-frequency sustained energy "
                f"({low_ratio:.2f}) and low transient density ({onset_density:.2f})."
            ),
        },
        "rain": {
            "level": rain_level,
            "confidence": round(rain_index, 3),
            "explanation": (
                f"{rain_level.capitalize()} rain likelihood from high-frequency energy "
                f"({high_ratio:.2f}), spectral flatness ({flatness:.2f}), and dense texture."
            ),
        },
        "activity": {
            "level": activity_level,
            "confidence": round(activity_index, 3),
            "biological_activity_score": round(bio_index, 3),
            "explanation": (
                f"{activity_level.capitalize()} activity from onset density "
                f"({onset_density:.2f}) plus mid/high-frequency burst energy."
            ),
        },
        "time_of_day_hint": {
            "label": time_hint,
            "confidence": round(time_hint_score, 3),
            "explanation": (
                "Estimated from brightness and biological activity patterns only; "
                "not derived from recording metadata."
            ),
        },
    }

    estimated_conditions = {
        "wind_speed_ms": round(0.5 + wind_index * 5.5, 2),
        "wind_max_ms": round(1.0 + wind_index * 8.0, 2),
        "precipitation_mm": round(max(0.0, rain_index - 0.28) * 7.0, 2),
        "precipitation_daily_mm": round(max(0.0, rain_index - 0.22) * 10.0, 2),
        "humidity_pct": round(42.0 + rain_index * 38.0, 2),
        "days_since_rain": round(max(0.0, 7.0 * (1.0 - rain_index)), 2),
        "confidence": 0.25,
        "inference_method": "basic_audio_feature_fallback",
        "wind": wind_level,
        "rain": rain_level,
        "activity": activity_level,
        "time_of_day_hint": time_hint,
    }

    return {
        "acoustic_features": acoustic_features,
        "heuristic_environment": heuristic_environment,
        "estimated_conditions": estimated_conditions,
        "spectrogram": {
            "image_b64": _analysis_spectrogram_png_b64(mel_db),
            "shape": list(mel_db.shape),
            "mime": "image/png",
        },
    }


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
    month_range:            str   = "September-November"
    sample_bin:             str   = "afternoon"
    noise_std:              float = 0.5   # generation only
    seed:                   Optional[int] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    checkpoint_exists = DEFAULT_CKPT.exists()
    return {
        "ok": True,
        "checkpoint": str(DEFAULT_CKPT),
        "exists": checkpoint_exists,
        "analysis_modes": {
            "vae_latent": checkpoint_exists,
            "basic_audio_fallback": True,
        },
    }


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
    month_range:            str   = "September-November",
    sample_bin:             str   = "afternoon",
):
    """Encode an uploaded audio file into a latent vector.

    Accepts multipart/form-data with a 'file' field (.wav or .webm.wav).
    Returns the 256-dim latent vector as a JSON array.
    """
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
        "month_range": month_range or month_range_for_month(month),
        "sample_bin": sample_bin,
    }

    # Save upload to a temp file so librosa can read it
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    base_analysis: dict = {}
    try:
        base_analysis = basic_audio_fallback_analysis(tmp_path)
    except Exception as exc:
        print(f"[WARN] Basic audio analysis failed: {exc}")

    if not DEFAULT_CKPT.exists():
        if not base_analysis:
            Path(tmp_path).unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="Basic audio analysis failed.")

        try:
            fallback = base_analysis
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Basic audio analysis failed: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return {
            "ok": True,
            "analysis_mode": "basic_fallback",
            "checkpoint_available": False,
            "checkpoint": str(DEFAULT_CKPT),
            "latent_dim": 0,
            "latent": [],
            "estimated_conditions": fallback["estimated_conditions"],
            "acoustic_features": fallback["acoustic_features"],
            "heuristic_environment": fallback["heuristic_environment"],
            "spectrogram": fallback["spectrogram"],
            "limitations": [
                "VAE checkpoint is unavailable, so no learned latent embedding was computed.",
                "Environmental values are low-confidence audio-feature proxies, not NASA-aligned nearest-neighbour estimates.",
                "Species-specific labels are not produced in Analysis mode.",
            ],
            "summary": "Basic Analysis Mode completed without VAE latent analysis.",
        }

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
        "analysis_mode":         "vae_latent",
        "checkpoint_available":  True,
        "latent_dim":            len(latent),
        "latent":                latent.tolist(),
        "estimated_conditions":  estimated_conditions,
        "acoustic_features":     base_analysis.get("acoustic_features", {}),
        "heuristic_environment": base_analysis.get("heuristic_environment", {}),
        "spectrogram":           base_analysis.get("spectrogram", {}),
        "limitations":           [],
    }


@app.post("/generation")
def generation(body: EnvFeatures):
    """Generate a soundscape from environmental conditions.

    Revised architecture MVP: prefer Layer A retrieval of a real ambient bed.
    Falls back to the older VAE/vocoder path if no downloaded clips are present.
    """
    env_dict = body.model_dump(exclude={"noise_std", "seed"})
    env_dict["month_range"] = env_dict.get("month_range") or month_range_for_month(body.month)
    layer_a_env = body.model_dump(exclude={"noise_std", "seed"}, exclude_unset=True) or env_dict
    layer_a_env["month_range"] = layer_a_env.get("month_range") or month_range_for_month(body.month)

    try:
        response = generate_layer_a_response(layer_a_env, seed=body.seed)
        weather = prepare_weather_layers(env_dict, seed=body.seed)
        events = prepare_event_layers(env_dict, seed=body.seed)
        mixed = mix_generation_layers(response, weather, events)
        ambient_audio = {
            "audio_b64": response.get("audio_b64", ""),
            "audio_mime": response.get("audio_mime", ""),
            "audio_ext": response.get("audio_ext", ""),
        }
        response["weather"] = weather
        response["events"] = events
        response["ambient_audio"] = ambient_audio
        response["final_audio_b64"] = mixed["final_audio_b64"]
        response["final_audio_mime"] = mixed["final_audio_mime"]
        response["final_audio_ext"] = mixed["final_audio_ext"]
        response["final_image_b64"] = mixed.get("final_image_b64", "")
        response["mixer"] = mixed["mixer"]
        response["audio_b64"] = mixed["final_audio_b64"] or response.get("audio_b64", "")
        response["audio_mime"] = mixed["final_audio_mime"] if mixed["final_audio_b64"] else response.get("audio_mime")
        response["audio_ext"] = mixed["final_audio_ext"] if mixed["final_audio_b64"] else response.get("audio_ext")
        if mixed.get("final_image_b64"):
            response["image_b64"] = mixed["final_image_b64"]
        response.setdefault("layer_status", {})["weather_layer"] = weather["status"]
        response.setdefault("layer_status", {})["species_event_layer"] = events["status"]
        response.setdefault("layer_status", {})["mixer"] = mixed["status"]
        if weather["status"] == "prepared":
            response["explanation"] = f"{response.get('explanation', '')} {weather['explanation']}".strip()
        if events["status"] == "prepared":
            response["explanation"] = f"{response.get('explanation', '')} {events['explanation']}".strip()
        if mixed["status"] == "mixed":
            response["explanation"] = f"{response.get('explanation', '')} {mixed['explanation']}".strip()
        return response
    except FileNotFoundError as exc:
        print(f"[WARN] Layer A unavailable ({exc}); falling back to VAE generation.")

    if not DEFAULT_CKPT.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not found and Layer A clips unavailable.")

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
        "mode":      "vae_decoder",
        "shape":     list(mel_db.shape),
        "image_b64": png_b64,
        "audio_b64": audio_b64,
        "audio_mime": "audio/wav",
        "audio_ext":  "wav",
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
