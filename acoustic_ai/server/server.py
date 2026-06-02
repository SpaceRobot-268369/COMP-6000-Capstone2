"""FastAPI server for the Soundscape layer registry.

Generic dropdown-driven endpoints — no per-layer hard-coding. All available
attempts are declared in ``acoustic_ai/registry.yaml`` and dispatched to a
per-attempt ``handler.py``.

Endpoints:
  GET  /health
  GET  /layers
  GET  /layers/{layer_id}/attempts
  POST /layers/{layer_id}/attempts/{attempt_id}/generate
"""

from __future__ import annotations

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
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make `layers.*` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import registry  # noqa: E402
from layers.layer_e.attempts.liting__smoke_1__e_b_weather_analysis.code.weather_detector import (  # noqa: E402
    analyse_weather,
    discover_legacy_weather_assets,
    load_site_promoted_weather_assets,
    load_weather_assets_from_index,
)


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
    # Cell selector for bank attempts (e.g. Layer A mvp_2 per-cell LoRAs).
    # Single-adapter attempts ignore these (their handler swallows extras).
    season: Optional[str] = None
    diel: Optional[str] = None


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


@app.post("/analysis")
async def analyse_upload(file: UploadFile = File(...)) -> dict:
    """Layer E smoke endpoint.

    MVP-1 currently wires the E-B weather detector only. E-A and E-C return
    explicit placeholders so the existing dev analysis UI can render a complete
    report shape while those heads are still being implemented.
    """
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        calibration_assets = _weather_calibration_assets()
        weather = analyse_weather(tmp_path, calibration_assets=calibration_assets)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"E-B analysis failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "ok": True,
        "ambient": {
            "estimated_conditions": None,
            "similar_clips": [],
            "confidence": 0.0,
            "status": "not_implemented",
        },
        "weather": weather,
        "events": {
            "detections": [],
            "confidence": 0.0,
            "status": "not_implemented",
        },
        "limitations": [
            "E-B is a smoke-test spectral baseline calibrated on Layer B/site-weather labels.",
            "Murphy's site257 promoted labels come from a Server A CLAP-first candidate policy.",
            "E-A ambient context and E-C event detection are placeholders in this endpoint.",
        ],
        "metadata": {
            "filename": file.filename,
            "content_type": file.content_type,
            "analysis_heads": {"E-A": "placeholder", "E-B": "smoke", "E-C": "placeholder"},
        },
    }


@app.post("/layers/{layer_id}/attempts/{attempt_id}/generate")
def generate(layer_id: str, attempt_id: str, body: GenerateRequest) -> dict:
    """Dispatch a generation call to the attempt's handler.

    Returns JSON containing base64-encoded WAV + PNG and the handler's
    metadata block.
    """
    try:
        result = registry.generate(
            layer_id, attempt_id,
            seed=body.seed, season=body.season, diel=body.diel,
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


def _weather_calibration_assets() -> list:
    """Load Layer B labels for E-B calibration when materialised locally."""
    analysis_dir = (
        Path(__file__).resolve().parent.parent
        / "layers"
        / "layer_e"
        / "attempts"
        / "liting__smoke_1__e_b_weather_analysis"
        / "data"
        / "analysis"
    )
    weather_dir = (
        Path(__file__).resolve().parent.parent
        / "layers"
        / "layer_b"
        / "attempts"
        / "lucas__smoke_1__curated_assets"
        / "data"
        / "weather"
    )
    site_manifest = analysis_dir / "site257_clap_promoted" / "layer_d_ready_manifest.csv"
    index = weather_dir / "asset_index.csv"
    assets = []
    if site_manifest.exists():
        try:
            assets = [
                asset
                for asset in load_site_promoted_weather_assets(site_manifest)
                if asset.audio_path.exists()
            ]
        except Exception:
            assets = []
    if assets:
        return assets

    if index.exists():
        try:
            assets = [asset for asset in load_weather_assets_from_index(index) if asset.audio_path.exists()]
        except Exception:
            assets = []
    return assets or discover_legacy_weather_assets()


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
