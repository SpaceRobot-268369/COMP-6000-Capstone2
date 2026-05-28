"""Training job adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import os
import subprocess
import threading
import time
from typing import Any

from config import WorkerConfig
from generation_adapter import HeartbeatCallback, sleep_with_heartbeats


@dataclass(frozen=True)
class TrainingResult:
    result: dict[str, Any]
    checkpoint_uri: str
    log_uri: str


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYER_C_SA3_SCRIPT = Path("script/events/train_sa3_lora_core6_smoke.sh")
LAYER_C_SA3_OUTPUT_DIR = Path(
    "model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints"
)
LAYER_C_SA3_LOG_DIR = Path("logs")


def _payload_int(payload: dict[str, Any], name: str, default: int) -> int:
    raw = payload.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [path for path in output_dir.glob("*.ckpt") if path.is_file()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def _run_with_heartbeats(
    command: list[str],
    *,
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
    log_path: Path,
    env: dict[str, str],
) -> int | None:
    """Run a foreground command while keeping Server A heartbeats alive."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        cancelled = threading.Event()
        finished = threading.Event()

        def heartbeat_loop() -> None:
            while not finished.wait(config.heartbeat_interval_seconds):
                if not heartbeat():
                    cancelled.set()
                    process.terminate()
                    return

        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        finished.set()
        heartbeat_thread.join(timeout=5)
        if cancelled.is_set():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return None
        return return_code


def _run_dvc(
    args: list[str],
    *,
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> None:
    command = [config.dvc_python, "-m", "dvc", *args]
    process = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if process.stdout:
        print(process.stdout, end="", flush=True)
    if process.stderr:
        print(process.stderr, end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed with exit code {process.returncode}")
    heartbeat()


def _run_layer_c_sa3_training(
    job: dict[str, Any],
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> TrainingResult | None:
    job_id = str(job["id"])
    payload = job.get("payload") or {}
    run_id = str(payload.get("run_id", f"job-{job_id}"))
    owner = str(payload.get("owner", "burger"))
    steps = _payload_int(payload, "steps", _payload_int(payload, "max_train_steps", 10))
    checkpoint_every = _payload_int(payload, "checkpoint_every", steps)
    demo_every = _payload_int(payload, "demo_every", 999999)
    num_workers = _payload_int(payload, "num_workers", 0)

    script_path = REPO_ROOT / LAYER_C_SA3_SCRIPT
    output_dir = REPO_ROOT / LAYER_C_SA3_OUTPUT_DIR
    if not script_path.is_file():
        raise RuntimeError(f"Layer C SA3 script not found: {script_path}")
    if not (Path(config.sa3_repo) / "scripts" / "train_lora.py").is_file():
        raise RuntimeError(f"Stable Audio 3 train_lora.py not found under {config.sa3_repo}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = REPO_ROOT / LAYER_C_SA3_LOG_DIR / f"sa3_lora_{run_id}_{timestamp}.log"
    env = os.environ.copy()
    env.update(
        {
            "SA3_REPO": config.sa3_repo,
            "PYTHON": str((REPO_ROOT / config.sa3_python).resolve())
            if not Path(config.sa3_python).is_absolute()
            else config.sa3_python,
            "MPLCONFIGDIR": config.sa3_mplconfigdir,
            "SA3_STEPS": str(steps),
            "SA3_CHECKPOINT_EVERY": str(checkpoint_every),
            "SA3_DEMO_EVERY": str(demo_every),
            "SA3_NUM_WORKERS": str(num_workers),
        }
    )

    started_at = time.monotonic()
    return_code = _run_with_heartbeats(
        ["bash", str(LAYER_C_SA3_SCRIPT)],
        config=config,
        heartbeat=heartbeat,
        log_path=log_path,
        env=env,
    )
    if return_code is None:
        return None
    if return_code != 0:
        raise RuntimeError(f"Layer C SA3 training failed with exit code {return_code}")

    checkpoint = _latest_checkpoint(output_dir)
    if checkpoint is None:
        raise RuntimeError(f"Layer C SA3 training completed but no checkpoint found in {output_dir}")

    checkpoint_relative = checkpoint.relative_to(REPO_ROOT)
    if config.dvc_push_enabled:
        _run_dvc(["add", str(checkpoint_relative)], config=config, heartbeat=heartbeat)
        _run_dvc(["push", str(checkpoint_relative) + ".dvc"], config=config, heartbeat=heartbeat)

    checkpoint_dvc_path = str(checkpoint_relative) + ".dvc"
    duration_s = int(time.monotonic() - started_at)
    result = {
        "mock": False,
        "worker_id": config.worker_id,
        "layer": "C",
        "run_id": run_id,
        "owner": owner,
        "training_backend": "sa3_lora",
        "duration_s": duration_s,
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "gpu": "cuda",
        "local_output_dir": str(LAYER_C_SA3_OUTPUT_DIR),
        "checkpoint_path": str(checkpoint_relative),
        "checkpoint_dvc_path": checkpoint_dvc_path,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "artifact_uploaded": config.dvc_push_enabled,
    }
    return TrainingResult(
        result=result,
        checkpoint_uri=checkpoint_dvc_path,
        log_uri=str(log_path.relative_to(REPO_ROOT)),
    )


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
    training_backend = str(payload.get("training_backend", payload.get("backend", ""))).lower()

    if config.real_training_enabled and layer.upper() == "C" and training_backend == "sa3_lora":
        return _run_layer_c_sa3_training(job, config, heartbeat)

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
