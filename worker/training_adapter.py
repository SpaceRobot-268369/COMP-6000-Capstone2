"""Training job adapter.

The current implementation is a fake training adapter. It verifies the automatic
Server A -> Server B training job flow before a real Layer A/C training script
is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import WorkerConfig
from generation_adapter import HeartbeatCallback, sleep_with_heartbeats


@dataclass(frozen=True)
class TrainingResult:
    result: dict[str, Any]
    checkpoint_uri: str
    log_uri: str


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
