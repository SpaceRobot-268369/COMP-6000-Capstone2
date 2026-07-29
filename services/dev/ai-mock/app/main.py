"""Fake AI service for the `demo` branch.

Speaks the exact HTTP contract the Express backend expects from the serverB
FastAPI worker (acoustic_ai/server/server.py), so neither backend nor frontend
code changes for the demo. Every response is a pre-baked fixture or a template
— no model is ever loaded and no audio is processed at request time.

Endpoints (the eight the backend actually calls):
    GET  /health
    GET  /layers
    GET  /layers/{layer_id}/attempts
    POST /layers/{layer_id}/attempts/{attempt_id}/generate
    POST /generation/render
    POST /generation/parse
    POST /analysis/run
    POST /analysis/narrative
    POST /layers/{layer_id}/attempts/{attempt_id}/analyze
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import analysis, heads, parser, registry, replay
from .settings import LATENCY_MS

log = logging.getLogger("ai-mock")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Soundscape Inference API (demo mock)", version="0.2.0-mock")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _think(kind: str) -> None:
    """Fake think-time so the frontend's staged progress text has room to run."""
    await asyncio.sleep(LATENCY_MS.get(kind, 0) / 1000.0)


# ------------------------------------------------------------------ models
class GenerateRequest(BaseModel):
    seed: Optional[int] = None
    retrieval_seed: Optional[int] = None
    season: Optional[str] = None
    diel: Optional[str] = None
    weather_type: Optional[str] = None
    intensity: Optional[str] = None
    wind_intensity: Optional[str] = None
    duration_s: Optional[float] = None
    species_common_name: Optional[str] = None


class OrchestratedGenerationRequest(BaseModel):
    seed: Optional[int] = None
    duration_s: float = 30.0
    season: Optional[str] = None
    diel: Optional[str] = None
    weather_type: Optional[str] = "wind"
    intensity: Optional[str] = "light"
    include_weather: bool = True
    include_events: bool = True
    species_common_name: Optional[str] = None
    layer_a_attempt: Optional[str] = None
    layer_b_attempt: Optional[str] = None
    layer_c_attempt: Optional[str] = None
    layer_d_attempt: Optional[str] = None
    include_stems: bool = False


class ParseRequest(BaseModel):
    prompt: str


class NarrativeRequest(BaseModel):
    report: dict
    narrative_register: str = Field(default="analytical", alias="register")

    model_config = {"populate_by_name": True}


# ------------------------------------------------------------------ routes
@app.get("/health")
def health() -> dict:
    layers = registry.list_layers()
    return {
        "ok": True,
        "mock": True,
        "statusKey": "online",
        "stage": "ai-health-ok",
        "message": "Mock AI service healthy (demo build — no real inference)",
        "reachability": {
            "backend": True,
            "tunnelContainer": True,
            "ssh": True,
            "serverB": True,
            "aiService": True,
        },
        "registry_layers": [layer["id"] for layer in layers],
        "total_attempts": sum(len(layer["attempts"]) for layer in layers),
    }


@app.get("/layers")
def list_layers() -> dict:
    return {"layers": registry.list_layers()}


@app.get("/layers/{layer_id}/attempts")
def layer_attempts(layer_id: str) -> dict:
    layer = registry.get_layer(layer_id)
    if not layer:
        raise HTTPException(status_code=404, detail=f"unknown layer: {layer_id}")
    return layer


@app.post("/layers/{layer_id}/attempts/{attempt_id}/generate")
async def generate(layer_id: str, attempt_id: str, body: GenerateRequest) -> dict:
    if not registry.get_attempt(layer_id, attempt_id):
        raise HTTPException(status_code=404, detail=f"unknown attempt: {layer_id}/{attempt_id}")
    await _think("generate")
    try:
        return replay.generate(layer_id, attempt_id, body.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"demo fixture missing: {exc}") from exc


@app.post("/generation/render")
async def render(body: OrchestratedGenerationRequest) -> dict:
    if body.duration_s is not None and (body.duration_s <= 0 or body.duration_s > 30):
        raise HTTPException(status_code=400, detail="duration_s must be greater than 0 and at most 30 seconds")
    await _think("generate")
    try:
        return replay.render(body.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generation/parse")
async def parse_prompt(body: ParseRequest) -> dict:
    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    await _think("parse")
    return {"ok": True, **parser.parse(prompt)}


@app.post("/analysis/narrative")
async def narrative(body: NarrativeRequest) -> dict:
    if not isinstance(body.report, dict) or not body.report:
        raise HTTPException(status_code=400, detail="report JSON is required")
    await _think("narrative")
    return {"ok": True, "narrative": analysis.narrate(body.report, body.narrative_register)}


@app.post("/analysis/run")
async def analysis_run(
    file: UploadFile = File(...),
    narrative_register: str = Form(default="immersive", alias="register"),
    ambient_attempt: Optional[str] = Form(default=None),
    weather_attempt: Optional[str] = Form(default=None),
    events_attempt: Optional[str] = Form(default=None),
    aggregator_attempt: Optional[str] = Form(default=None),
    ambient_override: Optional[str] = None,
    weather_override: Optional[str] = None,
    events_override: Optional[str] = None,
    aggregator_override: Optional[str] = None,
) -> dict:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    await _think("analysis")

    bundle = analysis.bundle_for_upload(payload)
    report = bundle["report"]

    # Slot overrides (query params, set by the backend from /dev/settings) win
    # over the per-head pickers in the form — same precedence as the real server.
    chosen = {
        "ambient": ambient_override or ambient_attempt,
        "weather": weather_override or weather_attempt,
        "events": events_override or events_attempt,
        "aggregator": aggregator_override or aggregator_attempt,
    }
    attempts = dict(bundle["attempts"])
    for head, attempt_id in chosen.items():
        if attempt_id:
            attempts[head] = registry.attempt_snapshot("layer_e", attempt_id)
    report["model_lineage"] = attempts

    log.info(
        "analysis/run: %s bytes -> cell %s (%s)",
        len(payload),
        bundle.get("cell"),
        report.get("mock_source"),
    )
    return {
        "ok": True,
        "report": report,
        "attempts": attempts,
        "head_reports": bundle["head_reports"],
        "narrative": analysis.narrate(report, narrative_register),
    }


@app.post("/layers/{layer_id}/attempts/{attempt_id}/analyze")
async def analyze_head(
    layer_id: str,
    attempt_id: str,
    file: UploadFile = File(...),
    register: str = "analytical",
) -> dict:
    attempt = registry.get_attempt(layer_id, attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail=f"unknown attempt: {layer_id}/{attempt_id}")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    await _think("analysis")

    bundle = analysis.bundle_for_upload(payload)
    head = attempt.get("head")
    return {
        "ok": True,
        "report": heads.report_for(head, bundle),
        "attempt": registry.attempt_snapshot(layer_id, attempt_id),
    }
