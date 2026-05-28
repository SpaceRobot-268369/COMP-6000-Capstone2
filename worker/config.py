"""Environment configuration for the Server B worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import socket


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WorkerConfig:
    server_a_url: str
    worker_api_token: str
    worker_id: str
    worker_job_types: list[str]
    poll_interval_seconds: int
    heartbeat_interval_seconds: int
    fake_run_seconds: int
    fake_upload_seconds: int
    fake_training_seconds: int
    real_training_enabled: bool
    sa3_repo: str
    sa3_python: str
    sa3_mplconfigdir: str
    dvc_python: str
    dvc_push_enabled: bool
    artifact_base_uri: str
    log_base_uri: str
    checkpoint_base_uri: str
    metrics_base_uri: str
    idle_shutdown_enabled: bool
    idle_shutdown_dry_run: bool
    idle_shutdown_seconds: int
    shutdown_command: list[str]


def load_config() -> WorkerConfig:
    worker_job_types = [
        item.strip()
        for item in os.getenv("WORKER_JOB_TYPES", "generation,training").split(",")
        if item.strip()
    ]

    return WorkerConfig(
        server_a_url=os.getenv("SERVER_A_URL", "http://localhost").rstrip("/"),
        worker_api_token=os.getenv("WORKER_API_TOKEN", ""),
        worker_id=os.getenv("WORKER_ID") or f"{socket.gethostname()}-worker",
        worker_job_types=worker_job_types,
        poll_interval_seconds=env_int("POLL_INTERVAL_SECONDS", 10),
        heartbeat_interval_seconds=env_int("HEARTBEAT_INTERVAL_SECONDS", 30),
        fake_run_seconds=env_int("FAKE_RUN_SECONDS", 5),
        fake_upload_seconds=env_int("FAKE_UPLOAD_SECONDS", 2),
        fake_training_seconds=env_int("FAKE_TRAINING_SECONDS", 10),
        real_training_enabled=env_bool("REAL_TRAINING_ENABLED", False),
        sa3_repo=os.getenv("SA3_REPO", "/home/ubuntu/stable-audio-3"),
        sa3_python=os.getenv(
            "SA3_PYTHON",
            "acoustic_ai/.venv-audiogen/bin/python",
        ),
        sa3_mplconfigdir=os.getenv("SA3_MPLCONFIGDIR", "/tmp/mpl"),
        dvc_python=os.getenv("DVC_PYTHON", "python3"),
        dvc_push_enabled=env_bool("DVC_PUSH_ENABLED", True),
        artifact_base_uri=os.getenv("ARTIFACT_BASE_URI", "s3://placeholder/generated").rstrip("/"),
        log_base_uri=os.getenv("LOG_BASE_URI", "s3://placeholder/logs").rstrip("/"),
        checkpoint_base_uri=os.getenv("CHECKPOINT_BASE_URI", "s3://placeholder/checkpoints").rstrip("/"),
        metrics_base_uri=os.getenv("METRICS_BASE_URI", "s3://placeholder/metrics").rstrip("/"),
        idle_shutdown_enabled=env_bool("IDLE_SHUTDOWN_ENABLED", False),
        idle_shutdown_dry_run=env_bool("IDLE_SHUTDOWN_DRY_RUN", True),
        idle_shutdown_seconds=env_int("IDLE_SHUTDOWN_SECONDS", 600),
        shutdown_command=shlex.split(os.getenv("SHUTDOWN_COMMAND", "sudo shutdown -h now")),
    )
