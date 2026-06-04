"""FastAPI server for the Soundscape layer registry.

Generic dropdown-driven endpoints — no per-layer hard-coding. All available
attempts are declared in ``acoustic_ai/registry.yaml`` and dispatched to a
per-attempt ``handler.py``.

Endpoints:
  GET  /health
  GET  /layers
  GET  /layers/{layer_id}/attempts
  POST /layers/{layer_id}/attempts/{attempt_id}/generate
  POST /layers/{layer_id}/attempts/{attempt_id}/analyze   (multipart upload)
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make `layers.*` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import registry  # noqa: E402


app = FastAPI(title="Soundscape Inference API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    seed: Optional[int] = None
    retrieval_seed: Optional[int] = None
    # Cell selector for bank attempts (e.g. Layer A mvp_2 per-cell LoRAs).
    # Single-adapter attempts ignore these (their handler swallows extras).
    season: Optional[str] = None
    diel: Optional[str] = None
    # Layer B weather-stem controls. Other attempts ignore these.
    weather_type: Optional[str] = None
    intensity: Optional[str] = None
    wind_intensity: Optional[str] = None
    duration_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    layers = registry.list_layers()
    return {
        "ok": True,
        "registry_layers": [l["id"] for l in layers],
        "total_attempts": sum(len(l["attempts"]) for l in layers),
    }


@app.get("/layers")
def list_layers() -> dict:
    """Frontend dropdown payload: every layer + its registered attempts."""
    return {"layers": registry.list_layers()}


@app.get("/layers/{layer_id}/attempts")
def list_attempts(layer_id: str) -> dict:
    for layer in registry.list_layers():
        if layer["id"] == layer_id:
            return layer
    raise HTTPException(status_code=404, detail=f"unknown layer: {layer_id}")


@app.post("/layers/{layer_id}/attempts/{attempt_id}/generate")
def generate(layer_id: str, attempt_id: str, body: GenerateRequest) -> dict:
    """Dispatch a generation call to the attempt's handler.

    Returns JSON containing base64-encoded WAV + PNG and the handler's
    metadata block.
    """
    try:
        run_seed = body.retrieval_seed if body.retrieval_seed is not None else body.seed
        result = registry.generate(
            layer_id, attempt_id,
            seed=run_seed, season=body.season, diel=body.diel,
            weather_type=body.weather_type,
            intensity=body.intensity,
            wind_intensity=body.wind_intensity,
            duration_s=body.duration_s,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    wav_bytes = result.get("wav_bytes", b"")
    mel_db = result.get("mel_db")
    metadata = result.get("metadata", {})
    duration_s = float(metadata.get("audio", {}).get("duration_s", 0.0))
    sample_rate = int(metadata.get("audio", {}).get("sample_rate", 0))

    png_b64 = _mel_to_png_b64(layer_id, attempt_id, mel_db, duration_s)

    return {
        "ok":          True,
        "audio_b64":   base64.b64encode(wav_bytes).decode("utf-8"),
        "image_b64":   png_b64,
        "metadata":    metadata,
        "sample_rate": sample_rate,
        "duration_s":  duration_s,
    }


@app.post("/layers/{layer_id}/attempts/{attempt_id}/analyze")
async def analyze(layer_id: str, attempt_id: str, file: UploadFile = File(...)) -> dict:
    """Dispatch an upload-based analysis call to the attempt's handler.

    Layer E analysis is upload-based (not seed-based): the client posts an
    audio clip as multipart ``file``; the handler embeds it and returns a
    per-head report. Returns ``{ok, report, attempt}``.
    """
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    try:
        data = await file.read()
    finally:
        await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    analysis_path = tmp_path
    converted_path = None
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(data)
        if suffix.lower() != ".wav":
            converted_fd, converted_path = tempfile.mkstemp(suffix=".wav")
            os.close(converted_fd)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    tmp_path,
                    "-ar",
                    "22050",
                    "-ac",
                    "1",
                    converted_path,
                ],
                check=True,
            )
            analysis_path = converted_path
        result = registry.analyze(layer_id, attempt_id, analysis_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail=f"audio conversion failed: {exc}")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        if converted_path:
            try:
                os.unlink(converted_path)
            except OSError:
                pass

    return {"ok": True, **result}


@app.get("/layers/{layer_id}/attempts/{attempt_id}/samples")
def list_samples(layer_id: str, attempt_id: str) -> dict:
    """Cached reference + showcase samples for an attempt (see
    .claude/context/dev/artifact_policy.md). The frontend uses this to show
    a preview without running generation."""
    try:
        return registry.list_samples(layer_id, attempt_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/layers/{layer_id}/attempts/{attempt_id}/samples/{tier}/{rel_path:path}")
def get_sample_wav(layer_id: str, attempt_id: str, tier: str, rel_path: str):
    """Serve a sample WAV inline (so the browser <audio> tag can play it).

    `rel_path` is whatever the sample's `wav_url` puts after the tier — see
    registry.list_samples() for the three supported layouts. Examples:
        expected/<stem>.wav                       (legacy flat)
        expected/<case>/audio.wav                 (canonical case-dir)
        expected/<cell>/<case>/audio.wav          (cell-grouped bank)
    """
    if not rel_path.endswith(".wav"):
        raise HTTPException(status_code=404, detail="only .wav samples are served")
    try:
        path = registry.sample_wav_path(layer_id, attempt_id, tier, rel_path)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"WAV not materialised locally ({path.name}). Run `dvc pull` then retry.",
        )
    return FileResponse(path, media_type="audio/wav", filename=path.name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mel_to_png_b64(layer_id: str, attempt_id: str,
                    mel_db, duration_s: float) -> str:
    """Render a mel-spectrogram PNG. For Layer A we use the attempt-local
    visualization helper to keep visual style consistent across attempts.
    """
    if mel_db is None:
        return ""

    # Try attempt-local renderer first (Layer A + Layer C ship one).
    try:
        import importlib
        viz_mod = importlib.import_module(
            f"layers.{layer_id}.attempts.{attempt_id}.layer_a_visualization"
        )
        return base64.b64encode(
            viz_mod.render_layer_a_mel_png_bytes(mel_db, duration_s)
        ).decode("utf-8")
    except ModuleNotFoundError:
        pass

    # Generic fallback.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
                  vmin=-80, vmax=0)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.set_title(f"{layer_id} / {attempt_id}")
        plt.colorbar(ax.images[0], ax=ax, label="dB")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
