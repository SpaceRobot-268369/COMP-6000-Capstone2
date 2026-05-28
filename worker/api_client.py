"""Small Server A API client for worker job operations."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from config import WorkerConfig


class ApiError(RuntimeError):
    pass


class ServerAClient:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if not self.config.worker_api_token:
            raise ApiError("WORKER_API_TOKEN is required")

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.server_a_url}{path}",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.worker_api_token}",
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

    def claim_job(self) -> dict[str, Any] | None:
        response = self.post(
            "/api/worker/jobs/claim",
            {
                "worker_id": self.config.worker_id,
                "types": self.config.worker_job_types,
                "capabilities": {
                    "mode": "fake",
                },
            },
        )
        return response.get("job")

    def heartbeat(self, job_id: str) -> dict[str, Any]:
        response = self.post(
            f"/api/worker/jobs/{job_id}/heartbeat",
            {
                "worker_id": self.config.worker_id,
            },
        )
        return response["job"]

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        artifact_uri: str | None = None,
        log_uri: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "worker_id": self.config.worker_id,
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

        response = self.post(f"/api/worker/jobs/{job_id}/status", body)
        return response["job"]

    def idle_check(self) -> dict[str, Any]:
        return self.post(
            "/api/worker/jobs/idle-check",
            {
                "worker_id": self.config.worker_id,
                "types": self.config.worker_job_types,
            },
        )
