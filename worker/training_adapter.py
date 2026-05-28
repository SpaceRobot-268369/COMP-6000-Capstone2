"""Training job adapter.

Keeps the fake training path for infrastructure smoke tests, and plugs in the
Layer C Stable Audio 3 smoke-training handoff when requested by payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from config import WorkerConfig
from generation_adapter import HeartbeatCallback, sleep_with_heartbeats


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYER_C_SA3_SCRIPT = REPO_ROOT / "script" / "events" / "train_sa3_lora_core6_smoke.sh"
LAYER_C_SA3_OUTPUT_DIR = (
    REPO_ROOT
    / "model"
    / "candidates"
    / "burger"
    / "layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke"
)
LAYER_C_SA3_CHECKPOINT_DIR = LAYER_C_SA3_OUTPUT_DIR / "lora_checkpoints"


@dataclass(frozen=True)
class TrainingResult:
    result: dict[str, Any]
    checkpoint_uri: str
    log_uri: str


def _payload_str(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    return str(value).strip()


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _is_layer_c_sa3_job(payload: dict[str, Any]) -> bool:
    layer = _payload_str(payload, "layer").upper()
    model_family = _payload_str(payload, "model_family").lower()
    base_model = _payload_str(payload, "base_model").lower()
    task = _payload_str(payload, "task").lower()
    return layer == "C" and (
        model_family in {"stable_audio_3", "sa3"}
        or "small-sfx-base" in base_model
        or task in {"sa3_lora_smoke", "train_sa3_lora"}
    )


def _latest_checkpoint() -> Path | None:
    checkpoints = sorted(
        LAYER_C_SA3_CHECKPOINT_DIR.glob("epoch=*-step=*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1] if checkpoints else None


def _run_subprocess_with_heartbeats(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    heartbeat: HeartbeatCallback,
) -> int | None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while process.poll() is None:
            if not heartbeat():
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                log.write("\n[cancelled] worker received cancel_requested\n")
                return None
            time.sleep(5)

        return process.returncode


def run_layer_c_sa3_training_job(
    job: dict[str, Any],
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> TrainingResult | None:
    job_id = str(job["id"])
    payload = job.get("payload") or {}
    run_id = _payload_str(
        payload,
        "run_id",
        "layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke",
    )
    owner = _payload_str(payload, "owner", "burger")
    steps = _payload_int(payload, "steps", _payload_int(payload, "max_train_steps", 300))
    checkpoint_every = _payload_int(payload, "checkpoint_every", 100)
    demo_every = _payload_int(payload, "demo_every", 999999)

    if not LAYER_C_SA3_SCRIPT.exists():
        raise FileNotFoundError(f"missing training script: {LAYER_C_SA3_SCRIPT}")

    sa3_repo = _payload_str(payload, "sa3_repo", os.getenv("SA3_REPO", "/home/ubuntu/stable-audio-3"))
    if not (Path(sa3_repo) / "scripts" / "train_lora.py").exists():
        raise FileNotFoundError(
            f"missing Stable Audio 3 upstream repo at {sa3_repo}; "
            "clone https://github.com/Stability-AI/stable-audio-3.git first"
        )

    log_dir = REPO_ROOT / "logs" / "layer-c" / run_id
    log_path = log_dir / "train.log"
    payload_snapshot = log_dir / "job_payload.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload_snapshot.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "SA3_REPO": sa3_repo,
            "LAYER_C_DATA_DIR": _payload_str(
                payload,
                "data_dir",
                str(REPO_ROOT / "acoustic_ai" / "data" / "events" / "layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke" / "sa3_lora_core6_data"),
            ),
            "MPLCONFIGDIR": _payload_str(payload, "mplconfigdir", os.getenv("MPLCONFIGDIR", "/tmp/mpl")),
            "SA3_STEPS": str(steps),
            "SA3_CHECKPOINT_EVERY": str(checkpoint_every),
            "SA3_DEMO_EVERY": str(demo_every),
            "SA3_NUM_WORKERS": str(_payload_int(payload, "num_workers", 0)),
        }
    )
    if "PYTHON" not in env:
        env["PYTHON"] = str(REPO_ROOT / "acoustic_ai" / ".venv-audiogen" / "bin" / "python")

    started = time.monotonic()
    code = _run_subprocess_with_heartbeats(
        ["bash", str(LAYER_C_SA3_SCRIPT)],
        env=env,
        log_path=log_path,
        heartbeat=heartbeat,
    )
    if code is None:
        return None
    if code != 0:
        raise RuntimeError(f"Layer C SA3 training script failed with exit code {code}; see {log_path}")

    checkpoint = _latest_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(f"training finished but no checkpoint found in {LAYER_C_SA3_CHECKPOINT_DIR}")

    metrics_path = LAYER_C_SA3_OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        metrics_path.write_text(
            json.dumps(
                {
                    "status": "training_completed_no_eval",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "job_id": job_id,
                    "run_id": run_id,
                    "checkpoint_path": str(checkpoint.relative_to(REPO_ROOT)),
                    "manual_eval_required": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    duration_s = round(time.monotonic() - started, 2)
    checkpoint_uri = str(checkpoint.relative_to(REPO_ROOT))
    log_uri = str(log_path.relative_to(REPO_ROOT))
    result = {
        "mock": False,
        "worker_id": config.worker_id,
        "layer": "C",
        "run_id": run_id,
        "owner": owner,
        "model_family": "stable_audio_3",
        "base_model": "small-sfx-base",
        "duration_s": duration_s,
        "local_output_dir": str(LAYER_C_SA3_OUTPUT_DIR.relative_to(REPO_ROOT)),
        "checkpoint_path": checkpoint_uri,
        "log_uri": log_uri,
        "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
        "payload_snapshot": str(payload_snapshot.relative_to(REPO_ROOT)),
        "artifact_uploaded": False,
        "upload_note": "DVC/S3 upload is not wired in this adapter yet.",
    }
    return TrainingResult(result=result, checkpoint_uri=checkpoint_uri, log_uri=log_uri)


def run_training_job(
    job: dict[str, Any],
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> TrainingResult | None:
    """Run a training job and return final metadata.

    Returning None means the job was cancelled and the caller should not mark it
    completed.
    """

    job_id = str(job["id"])
    payload = job.get("payload") or {}
    layer = str(payload.get("layer", "unknown"))
    run_id = str(payload.get("run_id", f"job-{job_id}"))

    if _is_layer_c_sa3_job(payload):
        return run_layer_c_sa3_training_job(job, config, heartbeat)

    if not sleep_with_heartbeats(config.fake_training_seconds, heartbeat):
        return None

    checkpoint_uri = f"{config.checkpoint_base_uri}/{run_id}/checkpoint.safetensors"
    log_uri = f"{config.log_base_uri}/{run_id}/train.log"
    metrics_uri = f"{config.metrics_base_uri}/{run_id}/metrics.json"
    result = {
        "mock": True,
        "worker_id": config.worker_id,
        "layer": layer,
        "run_id": run_id,
        "duration_s": config.fake_training_seconds,
        "checkpoint_uri": checkpoint_uri,
        "metrics_uri": metrics_uri,
        "artifact_uploaded": False,
    }
    return TrainingResult(result=result, checkpoint_uri=checkpoint_uri, log_uri=log_uri)


def upload_training_outputs(
    job: dict[str, Any],
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> bool:
    """Placeholder upload phase for checkpoints, logs, and metrics."""

    return sleep_with_heartbeats(config.fake_upload_seconds, heartbeat)
