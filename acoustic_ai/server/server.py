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
from pydantic import BaseModel, Field

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


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _llm_narrative_enabled() -> bool:
    """Whether /analyze attaches an inline narrative. Off by default."""
    return _env_truthy("AI_LLM_NARRATIVE", False)


def _fallback_narrative(report: dict, register: str = "immersive") -> dict:
    narration = (report or {}).get("narration") if isinstance(report, dict) else {}
    text = ""
    if isinstance(narration, dict):
        text = str(narration.get("summary") or "")
    if not text:
        text = "The analysis completed, but no narrative summary was available."
    return {
        "register": register,
        "text": text,
        "source": "deterministic_fallback",
        "faithful": True,
        "violations": [],
    }


def _run_llm_prewarm() -> None:
    try:
        from llm import warm
        warm()
        log.info("[prewarm] ready: llm")
    except Exception as exc:  # noqa: BLE001 — boot must survive LLM load failure
        log.warning("[prewarm] skipped llm: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    selection = _prewarm_selection()
    if selection == set():
        log.info("[prewarm] disabled via AI_PREWARM")
    else:
        threading.Thread(
            target=_run_prewarm, args=(selection,), name="prewarm", daemon=True,
        ).start()
    # LLM-OSS pre-warm is opt-in (AI_LLM_PREWARM, default off) so existing boots
    # are unchanged until the model is validated on serverB (plan Phase 5).
    if _env_truthy("AI_LLM_PREWARM", False):
        threading.Thread(
            target=_run_llm_prewarm, name="prewarm-llm", daemon=True,
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
    # Layer C live generative/retrieval attempts can expose multiple species
    # behind one attempt id; the frontend passes this selector through.
    species_common_name: Optional[str] = None


class OrchestratedGenerationRequest(BaseModel):
    seed: Optional[int] = None
    duration_s: float = 30.0
    season: Optional[str] = None
    diel: Optional[str] = None
    weather_type: str = "wind"
    intensity: str = "light"
    include_weather: bool = True
    include_events: bool = True
    layer_a_attempt: Optional[str] = None
    layer_b_attempt: Optional[str] = None
    layer_c_attempt: Optional[str] = None
    layer_d_attempt: Optional[str] = None
    include_stems: bool = False


class ParseRequest(BaseModel):
    """Raw NL prompt for the generation Prompt Parser (LLM-OSS)."""
    prompt: str


class NarrativeRequest(BaseModel):
    """Re-render prose from an already-computed fused report in a chosen
    register (backs the scene-page tone toggle). No detectors re-run."""
    report: dict
    narrative_register: str = Field(default="analytical", alias="register")


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
            species_common_name=body.species_common_name,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    return _generation_response(layer_id, attempt_id, result)


@app.post("/generation/render")
def orchestrated_generation(body: OrchestratedGenerationRequest) -> dict:
    """Generate A/B/C stems and render the final soundscape through Layer D."""
    try:
        result = registry.orchestrate_generation(**body.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"orchestrated generation failed: {exc}")
    attempt_id = body.layer_d_attempt or registry.default_attempt_id("layer_d")
    response = _generation_response("layer_d", attempt_id, result)
    if body.include_stems:
        stems = result.get("_stems") or {}
        response["stems"] = {
            layer_id: _generation_response(
                layer_id,
                body_layer_attempt(layer_id, body),
                stem_result,
            )
            for layer_id, stem_result in stems.items()
            if stem_result is not None
        }
    return response


@app.post("/generation/parse")
def generation_parse(body: ParseRequest) -> dict:
    """Run the LLM-OSS Prompt Parser on a raw NL prompt and return the
    parse-result contract (prompt_parser_policy.md §5). Generates no audio."""
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        from llm import parse_prompt
        result = parse_prompt(prompt)
    except HTTPException:
        raise
    except Exception as exc:  # model/dep unavailable, etc.
        raise HTTPException(status_code=503, detail=f"prompt parser unavailable: {exc}")
    return {"ok": True, **result}


@app.post("/analysis/narrative")
def analysis_narrative(body: NarrativeRequest) -> dict:
    """Re-render prose from a prior fused report in a chosen register, without
    re-running detectors. Backs the scene-page tone toggle (plan §3.4/§3.5)."""
    if not isinstance(body.report, dict) or not body.report:
        raise HTTPException(status_code=400, detail="report JSON is required")
    try:
        from llm import write_report
        narrative = write_report(body.report, body.narrative_register)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"report writer unavailable: {exc}")
    return {"ok": True, "narrative": narrative}


@app.post("/analysis/run")
async def orchestrated_analysis(
    file: UploadFile = File(...),
    narrative_register: str = Form(default="immersive", alias="register"),
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

    register = (narrative_register or "immersive").strip().lower() or "immersive"
    report = result.get("report") or {}
    response = {"ok": True, **result}
    try:
        from llm import write_report
        narrative = write_report(report, register)
        if not narrative.get("faithful", True):
            response["narrative_violations"] = narrative.get("violations", [])
            narrative = _fallback_narrative(report, narrative.get("register") or register)
        response["narrative"] = narrative
    except Exception as exc:  # product mode should still return the fused report
        response["narrative"] = _fallback_narrative(report, register)
        response["narrative_error"] = str(exc)
    return response


@app.post("/layers/{layer_id}/attempts/{attempt_id}/analyze")
async def analyze(layer_id: str, attempt_id: str, file: UploadFile = File(...),
                  register: str = "analytical") -> dict:
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

    response = {"ok": True, **result}
    # Opt-in inline narrative (AI_LLM_NARRATIVE). Off by default so analysis
    # never pays a cold LLM load on the request path; the scene-page toggle uses
    # the dedicated /analysis/narrative endpoint instead.
    if _llm_narrative_enabled():
        try:
            from llm import write_report
            response["narrative"] = write_report(result.get("report") or {}, register)
        except Exception as exc:  # never break analysis if the LLM is down
            response["narrative"] = None
            response["narrative_error"] = str(exc)
    return response


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


def _generation_response(layer_id: str, attempt_id: str, result: dict) -> dict:
    wav_bytes = result.get("wav_bytes", b"")
    mel_db = result.get("mel_db")
    metadata = result.get("metadata", {})
    duration_s = float(metadata.get("audio", {}).get("duration_s", 0.0))
    sample_rate = int(metadata.get("audio", {}).get("sample_rate", 0))
    return {
        "ok": True,
        "audio_b64": base64.b64encode(wav_bytes).decode("utf-8"),
        "image_b64": _mel_to_png_b64(layer_id, attempt_id, mel_db, metadata),
        "metadata": metadata,
        "sample_rate": sample_rate,
        "duration_s": duration_s,
    }


def body_layer_attempt(layer_id: str, body: OrchestratedGenerationRequest) -> str:
    explicit = {
        "layer_a": body.layer_a_attempt,
        "layer_b": body.layer_b_attempt,
        "layer_c": body.layer_c_attempt,
        "layer_d": body.layer_d_attempt,
    }.get(layer_id)
    return explicit or registry.default_attempt_id(layer_id)


def _mel_to_png_b64(layer_id: str, attempt_id: str,
                    mel_db, metadata: dict) -> str:
    """Render a mel-spectrogram PNG. For Layer A we use the attempt-local
    visualization helper to keep visual style consistent across attempts.
    """
    if mel_db is None:
        return ""

    audio_meta = metadata.get("audio", {}) if isinstance(metadata, dict) else {}
    duration_s = float(audio_meta.get("duration_s", 0.0))
    sample_rate = int(audio_meta.get("sample_rate", 0))

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
        img = None
        if sample_rate > 0 and duration_s > 0:
            import librosa
            import librosa.display

            hop_length = max(1, int(round(duration_s * sample_rate / mel_db.shape[1])))
            if layer_id == "layer_c":
                mel_arr = np.asarray(mel_db)
                time_edges = np.linspace(0, duration_s, mel_arr.shape[1] + 1)
                low_hz = 0.0
                high_hz = min(sample_rate / 2.0, 11025)
                mel_centers = librosa.mel_frequencies(
                    n_mels=mel_arr.shape[0],
                    fmin=low_hz,
                    fmax=high_hz,
                )
                freq_edges = np.concatenate(
                    ([low_hz], (mel_centers[:-1] + mel_centers[1:]) / 2.0, [high_hz])
                )
                img = ax.pcolormesh(
                    time_edges,
                    freq_edges,
                    mel_arr,
                    cmap="magma",
                    vmin=-80,
                    vmax=0,
                    shading="auto",
                )
            else:
                img = librosa.display.specshow(
                    np.asarray(mel_db),
                    sr=sample_rate,
                    hop_length=hop_length,
                    x_axis="time",
                    y_axis="mel",
                    cmap="magma",
                    vmin=-80,
                    vmax=0,
                    ax=ax,
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Hz")
            _format_layer_c_axes(ax, layer_id, duration_s, sample_rate, metadata)
        else:
            img = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
                            vmin=-80, vmax=0)
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
        ax.set_title(_spectrogram_title(layer_id, attempt_id, metadata))
        plt.colorbar(img, ax=ax, label="dB")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        return ""


def _format_layer_c_axes(
    ax,
    layer_id: str,
    duration_s: float,
    sample_rate: int,
    metadata: dict | None = None,
) -> None:
    """Use audit-friendly seconds/Hz axes for Layer C plots."""

    if layer_id != "layer_c" or duration_s <= 0 or sample_rate <= 0:
        return

    method = (metadata or {}).get("method") if isinstance(metadata, dict) else None
    is_generative_event = isinstance(method, str) and method.startswith("sa3_lora_generative")
    target_duration = duration_s if is_generative_event else max(60.0, duration_s)
    ax.set_xlim(0, target_duration)
    if is_generative_event:
        tick_step = 0.5 if target_duration <= 6.0 else 1.0
        x_ticks = np.arange(0, target_duration + 1e-6, tick_step)
        if len(x_ticks) < 2:
            x_ticks = np.array([0.0, target_duration])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{tick:g}s" for tick in x_ticks])
    else:
        x_ticks = np.arange(0, target_duration + 0.1, 10.0)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(tick)}s" for tick in x_ticks])

    max_hz = min(10000, sample_rate / 2.0)
    y_ticks = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    y_ticks = [tick for tick in y_ticks if 0 <= tick <= max_hz]
    ax.set_ylim(0, max_hz)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(tick) for tick in y_ticks])


def _spectrogram_title(layer_id: str, attempt_id: str, metadata: dict) -> str:
    if layer_id == "layer_c" and isinstance(metadata, dict):
        species = metadata.get("species")
        method = metadata.get("method")
        if species and method:
            method_label = {
                "sa3_lora_generative_live": "SA3 LoRA live generation",
                "layer_c_retrieval_v2": "retrieval library",
            }.get(str(method), str(method))
            return f"{species} · {method_label}"
        if species:
            return str(species)
    return f"{layer_id} / {attempt_id}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
