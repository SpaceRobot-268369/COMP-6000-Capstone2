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
import logging
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Make `layers.*` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import registry  # noqa: E402

log = logging.getLogger("soundscape.server")


# ---------------------------------------------------------------------------
# Model pre-warming
# ---------------------------------------------------------------------------
#
# Handler state loads lazily on first use (registry._get_state). For heavy
# generative layers (Layer A = AudioLDM2 + 16 LoRA adapters, Layer C =
# AudioGen) that cold load runs *inside* the first request and can take
# minutes — long enough to blow past the backend's AI_REQUEST_TIMEOUT_MS, so
# the user sees "AI request timed out" instead of audio. Pre-warming on
# startup moves the load off the request path: the first generate then pays
# only inference. serverB is a disposable worker that reboots fresh, so this
# runs on every boot.
#
# Controlled by the AI_PREWARM env var:
#   unset / "1" / "all"   -> warm every layer's default attempt
#   "0" / "false" / "none"-> disabled (pure lazy loading)
#   "layer_a,layer_c"     -> warm only these layers' defaults
#
# Warming runs in a daemon thread so uvicorn binds the port immediately (the
# SSH-tunnel health check stays green while models load). Concurrent requests
# share the same cached state via registry's state lock.


def _prewarm_selection() -> Optional[set[str]]:
    """Parse AI_PREWARM into a layer-id filter. None => warm all defaults;
    empty set => disabled."""
    raw = os.environ.get("AI_PREWARM", "all").strip().lower()
    if raw in {"0", "false", "none", "off"}:
        return set()
    if raw in {"1", "all", "true", "on", ""}:
        return None
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def _run_prewarm(selection: Optional[set[str]]) -> None:
    log.info("[prewarm] starting (selection=%s)", "all" if selection is None else selection)
    for row in registry.prewarm_defaults(layers=selection):
        if row["ok"]:
            log.info("[prewarm] ready: %s/%s", row["layer"], row["attempt"])
        else:
            log.warning("[prewarm] skipped %s/%s: %s",
                        row["layer"], row["attempt"], row["error"])
    log.info("[prewarm] done")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    selection = _prewarm_selection()
    if selection == set():
        log.info("[prewarm] disabled via AI_PREWARM")
    else:
        threading.Thread(
            target=_run_prewarm, args=(selection,), name="prewarm", daemon=True,
        ).start()
    yield


app = FastAPI(title="Soundscape Inference API", version="0.2.0", lifespan=lifespan)


def _clean_form_value(value: str | None) -> str | None:
    value = value.strip() if isinstance(value, str) else ""
    return value or None
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


@app.post("/analysis/run")
async def orchestrated_analysis(
    file: UploadFile = File(...),
    ambient_attempt: str | None = Form(default=None),
    weather_attempt: str | None = Form(default=None),
    events_attempt: str | None = Form(default=None),
    aggregator_attempt: str | None = Form(default=None),
) -> dict:
    """Run the full Layer E analysis stack over one uploaded audio clip."""
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
        result = registry.orchestrate_analysis(
            analysis_path,
            ambient_attempt=_clean_form_value(ambient_attempt),
            weather_attempt=_clean_form_value(weather_attempt),
            events_attempt=_clean_form_value(events_attempt),
            aggregator_attempt=_clean_form_value(aggregator_attempt),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail=f"audio conversion failed: {exc}")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"orchestrated analysis failed: {exc}")
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
