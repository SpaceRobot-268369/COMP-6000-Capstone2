# Server B GPU Worker Runbook

## Purpose

Server B (`shinypokemon`) runs GPU work only. It claims jobs from Server A,
sends heartbeats, runs generation or training, uploads artifacts, and records
final job state through Server A APIs.

The first Milestone 3 implementation is intentionally a fake worker skeleton. It
does not run real generation, DVC, S3, or model code yet. Its job is to verify
the worker loop contract:

```text
queued -> claimed -> running -> uploading -> completed
```

## Current Worker Skeleton

Files:

```text
worker/worker.py
worker/.env.example
worker/requirements.txt
```

The worker uses only Python standard library modules for the MVP.

## Environment

Required:

```env
SERVER_A_URL=http://localhost
WORKER_API_TOKEN=change-me
WORKER_ID=shinypokemon-manual-worker
```

Optional:

```env
WORKER_JOB_TYPES=generation
POLL_INTERVAL_SECONDS=10
HEARTBEAT_INTERVAL_SECONDS=30
FAKE_RUN_SECONDS=5
FAKE_UPLOAD_SECONDS=2
ARTIFACT_BASE_URI=s3://placeholder/generated
LOG_BASE_URI=s3://placeholder/logs
```

For Server A testing through SSH port forwarding, `SERVER_A_URL` can point at
the forwarded localhost URL.

## Run

From the repository root:

```bash
python worker/worker.py
```

Or with an env file loaded by the shell:

```bash
set -a
. worker/.env
set +a
python worker/worker.py
```

## MVP Behavior

Loop:

1. Claim one job from Server A.
2. Sleep if no job is available.
3. Mark claimed job `running`.
4. Send heartbeats while fake work runs.
5. If heartbeat returns `cancel_requested`, mark job `cancelled`.
6. Mark job `uploading`.
7. Send placeholder artifact/log/result metadata.
8. Mark job `completed`.
9. On unhandled error, try to mark job `failed`.

Placeholder completion metadata:

```text
artifact_uri = <ARTIFACT_BASE_URI>/job-<id>.wav
log_uri = <LOG_BASE_URI>/job-<id>.log
result = {"mock": true, ...}
```

## Not Implemented Yet

- real Python GPU environment setup;
- DVC/S3 pull/push;
- model checkpoint validation;
- real generation;
- artifact upload;
- idle shutdown;
- worker registry;
- local disk cleanup;
- OOM detection.

These are later Milestone 3/4/P1 tasks.
