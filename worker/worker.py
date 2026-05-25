#!/usr/bin/env python3
"""Minimal Server B worker loop for the Milestone 3 API contract.

This worker intentionally performs fake generation. It verifies that a Python
daemon can claim jobs from Server A, send heartbeats, and drive the job state
machine through running, uploading, and completed.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


SERVER_A_URL = os.getenv("SERVER_A_URL", "http://localhost").rstrip("/")
WORKER_API_TOKEN = os.getenv("WORKER_API_TOKEN", "")
WORKER_ID = os.getenv("WORKER_ID") or f"{socket.gethostname()}-worker"
WORKER_JOB_TYPES = [
    item.strip()
    for item in os.getenv("WORKER_JOB_TYPES", "generation").split(",")
    if item.strip()
]
POLL_INTERVAL_SECONDS = env_int("POLL_INTERVAL_SECONDS", 10)
HEARTBEAT_INTERVAL_SECONDS = env_int("HEARTBEAT_INTERVAL_SECONDS", 30)
FAKE_RUN_SECONDS = env_int("FAKE_RUN_SECONDS", 5)
FAKE_UPLOAD_SECONDS = env_int("FAKE_UPLOAD_SECONDS", 2)
ARTIFACT_BASE_URI = os.getenv("ARTIFACT_BASE_URI", "s3://placeholder/generated").rstrip("/")
LOG_BASE_URI = os.getenv("LOG_BASE_URI", "s3://placeholder/logs").rstrip("/")


class ApiError(RuntimeError):
    pass


def api_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    if not WORKER_API_TOKEN:
        raise ApiError("WORKER_API_TOKEN is required")

    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{SERVER_A_URL}{path}",
        data=payload,
        headers={
            "Authorization": f"Bearer {WORKER_API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ApiError(f"POST {path} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"POST {path} failed: {exc}") from exc

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ApiError(f"POST {path} returned invalid JSON: {data}") from exc

    if not parsed.get("ok"):
        raise ApiError(f"POST {path} returned ok=false: {parsed}")
    return parsed


def claim_job() -> dict[str, Any] | None:
    response = api_post(
        "/api/worker/jobs/claim",
        {
            "worker_id": WORKER_ID,
            "types": WORKER_JOB_TYPES,
            "capabilities": {
                "mode": "fake",
            },
        },
    )
    return response.get("job")


def heartbeat(job_id: str) -> dict[str, Any]:
    response = api_post(
        f"/api/worker/jobs/{job_id}/heartbeat",
        {
            "worker_id": WORKER_ID,
        },
    )
    return response["job"]


def update_status(
    job_id: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    artifact_uri: str | None = None,
    log_uri: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "worker_id": WORKER_ID,
        "status": status,
    }
    if result is not None:
        body["result"] = result
    if artifact_uri is not None:
        body["artifact_uri"] = artifact_uri
    if log_uri is not None:
        body["log_uri"] = log_uri
    if error_message is not None:
        body["error_message"] = error_message

    response = api_post(f"/api/worker/jobs/{job_id}/status", body)
    return response["job"]


def sleep_with_heartbeats(job_id: str, total_seconds: int) -> bool:
    deadline = time.monotonic() + total_seconds
    next_heartbeat = time.monotonic()

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            job = heartbeat(job_id)
            if job["status"] == "cancel_requested":
                update_status(job_id, "cancelled")
                print(f"job {job_id}: cancelled after cancel_requested", flush=True)
                return False
            next_heartbeat = now + HEARTBEAT_INTERVAL_SECONDS
        time.sleep(min(1, max(0, deadline - time.monotonic())))

    return True


def process_job(job: dict[str, Any]) -> None:
    job_id = str(job["id"])
    print(f"job {job_id}: claimed type={job['type']}", flush=True)

    try:
        update_status(job_id, "running")
        if not sleep_with_heartbeats(job_id, FAKE_RUN_SECONDS):
            return

        update_status(job_id, "uploading")
        if not sleep_with_heartbeats(job_id, FAKE_UPLOAD_SECONDS):
            return

        artifact_uri = f"{ARTIFACT_BASE_URI}/job-{job_id}.wav"
        log_uri = f"{LOG_BASE_URI}/job-{job_id}.log"
        result = {
            "mock": True,
            "worker_id": WORKER_ID,
            "duration_s": FAKE_RUN_SECONDS,
            "artifact_uploaded": False,
        }
        update_status(
            job_id,
            "completed",
            result=result,
            artifact_uri=artifact_uri,
            log_uri=log_uri,
        )
        print(f"job {job_id}: completed artifact_uri={artifact_uri}", flush=True)
    except Exception as exc:
        try:
            update_status(job_id, "failed", error_message=str(exc))
        except Exception as status_exc:
            print(f"job {job_id}: failed to report failure: {status_exc}", file=sys.stderr, flush=True)
        raise


def main() -> int:
    print(
        f"worker starting id={WORKER_ID} server={SERVER_A_URL} types={','.join(WORKER_JOB_TYPES)}",
        flush=True,
    )

    while True:
        try:
            job = claim_job()
            if not job:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            process_job(job)
        except KeyboardInterrupt:
            print("worker stopping", flush=True)
            return 0
        except Exception as exc:
            print(f"worker error: {exc}", file=sys.stderr, flush=True)
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
