#!/usr/bin/env python3
"""Server B worker loop for the Milestone 3 API contract.

The worker currently uses a fake generation adapter. It verifies that a Python
daemon can claim jobs from Server A, send heartbeats, and drive the job state
machine through running, uploading, and completed. Real generation should be
plugged into generation_adapter.py without changing the Server A API flow.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from api_client import ServerAClient
from config import WorkerConfig, load_config
from generation_adapter import run_generation_job, upload_generation_outputs


class Worker:
    def __init__(self, config: WorkerConfig, client: ServerAClient) -> None:
        self.config = config
        self.client = client
        self._next_heartbeat_by_job: dict[str, float] = {}

    def heartbeat_or_cancel(self, job_id: str) -> bool:
        now = time.monotonic()
        next_heartbeat = self._next_heartbeat_by_job.get(job_id, 0)
        if now < next_heartbeat:
            return True

        job = self.client.heartbeat(job_id)
        self._next_heartbeat_by_job[job_id] = now + self.config.heartbeat_interval_seconds

        if job["status"] == "cancel_requested":
            self.client.update_status(job_id, "cancelled")
            print(f"job {job_id}: cancelled after cancel_requested", flush=True)
            return False
        return True

    def process_generation_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        self.client.update_status(job_id, "running")

        generation_result = run_generation_job(
            job,
            self.config,
            lambda: self.heartbeat_or_cancel(job_id),
        )
        if generation_result is None:
            return

        self.client.update_status(job_id, "uploading")
        if not upload_generation_outputs(
            job,
            self.config,
            lambda: self.heartbeat_or_cancel(job_id),
        ):
            return

        self.client.update_status(
            job_id,
            "completed",
            result=generation_result.result,
            artifact_uri=generation_result.artifact_uri,
            log_uri=generation_result.log_uri,
        )
        print(f"job {job_id}: completed artifact_uri={generation_result.artifact_uri}", flush=True)

    def process_job(self, job: dict[str, Any]) -> None:
        job_id = str(job["id"])
        print(f"job {job_id}: claimed type={job['type']}", flush=True)
        self._next_heartbeat_by_job[job_id] = 0

        try:
            if job["type"] == "generation":
                self.process_generation_job(job)
            else:
                raise RuntimeError(f"unsupported job type: {job['type']}")
        except Exception as exc:
            try:
                self.client.update_status(job_id, "failed", error_message=str(exc))
            except Exception as status_exc:
                print(
                    f"job {job_id}: failed to report failure: {status_exc}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        finally:
            self._next_heartbeat_by_job.pop(job_id, None)

    def run_forever(self) -> int:
        print(
            "worker starting "
            f"id={self.config.worker_id} "
            f"server={self.config.server_a_url} "
            f"types={','.join(self.config.worker_job_types)}",
            flush=True,
        )

        while True:
            try:
                job = self.client.claim_job()
                if not job:
                    time.sleep(self.config.poll_interval_seconds)
                    continue
                self.process_job(job)
            except KeyboardInterrupt:
                print("worker stopping", flush=True)
                return 0
            except Exception as exc:
                print(f"worker error: {exc}", file=sys.stderr, flush=True)
                time.sleep(self.config.poll_interval_seconds)


def main() -> int:
    config = load_config()
    client = ServerAClient(config)
    return Worker(config, client).run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
