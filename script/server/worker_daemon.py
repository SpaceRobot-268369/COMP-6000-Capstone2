#!/usr/bin/env python3
"""Small local worker for testing the Server A/B job bridge.

This script intentionally uses only the Python standard library so it can run
inside the project Docker environment or on a plain Server B shell.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_A_URL = os.environ.get("SERVER_A_URL", "http://localhost:4000").rstrip("/")
WORKER_API_TOKEN = os.environ.get("WORKER_API_TOKEN", "local-dev-worker-token")
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname() or "local-worker")
WORKER_POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "2"))
WORKER_ONCE = os.environ.get("WORKER_ONCE", "").lower() in {"1", "true", "yes"}
WORKER_LOG_DIR = Path(os.environ.get("WORKER_LOG_DIR", "debug/server_worker/jobs"))


def api_post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{SERVER_A_URL}{path}",
        data=data,
        method="POST",
        headers={
            "authorization": f"Bearer {WORKER_API_TOKEN}",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} failed with HTTP {err.code}: {detail}") from err


def worker_payload(status: str, current_job_id: int | None = None) -> dict:
    return {
        "worker_id": WORKER_ID,
        "status": status,
        "current_job_id": current_job_id,
        "metadata_json": {
            "host": socket.gethostname(),
            "server_a_url": SERVER_A_URL,
            "mode": "local-test",
        },
    }


def post_status(job_id: int, status: str, message: str = "", error_message: str = "", metadata: dict | None = None) -> None:
    api_post(
        f"/api/worker/jobs/{job_id}/status",
        {
            "worker_id": WORKER_ID,
            "status": status,
            "message": message,
            "error_message": error_message,
            "metadata_json": metadata or {},
        },
    )


def write_log(job_id: int, command: str, result: subprocess.CompletedProcess[str]) -> Path:
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = WORKER_LOG_DIR / f"job_{job_id}_{timestamp}.log"
    log_path.write_text(
        "\n".join(
            [
                f"job_id={job_id}",
                f"worker_id={WORKER_ID}",
                f"started_at_utc={timestamp}",
                f"returncode={result.returncode}",
                f"command={command}",
                "",
                "STDOUT:",
                result.stdout,
                "",
                "STDERR:",
                result.stderr,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def run_job(job: dict) -> bool:
    job_id = int(job["id"])
    command = job.get("command") or ""
    if not command:
        post_status(job_id, "failed", "Job has no command.", "Missing command.")
        return False

    post_status(job_id, "running", "Worker started command.")
    result = subprocess.run(
        command,
        shell=True,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("WORKER_JOB_TIMEOUT_SECONDS", "3600")),
    )
    log_path = write_log(job_id, command, result)
    api_post(
        f"/api/worker/jobs/{job_id}/artifacts",
        {
            "worker_id": WORKER_ID,
            "artifacts": [
                {
                    "kind": "log",
                    "path": str(log_path),
                    "metadata_json": {
                        "returncode": result.returncode,
                        "local_test": True,
                    },
                }
            ],
        },
    )

    if result.returncode == 0:
        post_status(job_id, "completed", "Command completed.", metadata={"log_path": str(log_path)})
        return True

    post_status(
        job_id,
        "failed",
        "Command failed.",
        error_message=f"Command exited with code {result.returncode}.",
        metadata={"log_path": str(log_path)},
    )
    return False


def main() -> int:
    print(f"Worker {WORKER_ID} polling {SERVER_A_URL}")
    while True:
        api_post("/api/worker/heartbeat", worker_payload("idle"))
        response = api_post("/api/worker/claim", {"worker_id": WORKER_ID, "lease_seconds": 3600})
        job = response.get("job")
        if not job:
            if WORKER_ONCE:
                print("No queued job found.")
                return 0
            time.sleep(WORKER_POLL_SECONDS)
            continue

        print(f"Claimed job {job['id']}: {job.get('type')} {job.get('layer')}")
        try:
            ok = run_job(job)
        except Exception as err:  # noqa: BLE001 - local worker should report every failure.
            post_status(int(job["id"]), "failed", "Worker exception.", error_message=str(err))
            raise
        if WORKER_ONCE:
            return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Worker stopped.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Worker failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
