"""Generation job adapter.

The MVP adapter is intentionally fake. It preserves the job contract while
leaving one small replacement point for real acoustic_ai generation and artifact
upload.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

from config import WorkerConfig


class HeartbeatCallback(Protocol):
    def __call__(self) -> bool:
        """Return False when the current job should stop."""


@dataclass(frozen=True)
class GenerationResult:
    result: dict[str, Any]
    artifact_uri: str
    log_uri: str


def sleep_with_heartbeats(total_seconds: int, heartbeat: HeartbeatCallback) -> bool:
    deadline = time.monotonic() + total_seconds

    while time.monotonic() < deadline:
        if not heartbeat():
            return False
        time.sleep(min(1, max(0, deadline - time.monotonic())))

    return True


def run_generation_job(
    job: dict[str, Any],
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> GenerationResult | None:
    """Run a generation job and return final metadata.

    Returning None means the job was cancelled and the caller should not mark it
    completed.
    """

    job_id = str(job["id"])

    if not sleep_with_heartbeats(config.fake_run_seconds, heartbeat):
        return None

    artifact_uri = f"{config.artifact_base_uri}/job-{job_id}.wav"
    log_uri = f"{config.log_base_uri}/job-{job_id}.log"
    result = {
        "mock": True,
        "worker_id": config.worker_id,
        "duration_s": config.fake_run_seconds,
        "artifact_uploaded": False,
    }
    return GenerationResult(result=result, artifact_uri=artifact_uri, log_uri=log_uri)


def upload_generation_outputs(
    job: dict[str, Any],
    config: WorkerConfig,
    heartbeat: HeartbeatCallback,
) -> bool:
    """Placeholder upload phase for generated files and logs."""

    return sleep_with_heartbeats(config.fake_upload_seconds, heartbeat)

