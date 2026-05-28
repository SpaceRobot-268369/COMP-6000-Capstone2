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
worker/api_client.py
worker/config.py
worker/generation_adapter.py
worker/training_adapter.py
worker/worker.py
worker/.env.example
worker/requirements.txt
```

The worker uses only Python standard library modules for the MVP. The main loop
and Server A API contract live in `worker/worker.py` and `worker/api_client.py`.
The current fake generation implementation lives in
`worker/generation_adapter.py`; real `acoustic_ai` generation and artifact
upload should be added there first. The current fake training implementation
lives in `worker/training_adapter.py`; real Layer A/C training commands and
checkpoint upload should be added there first.

## Environment

Required:

```env
SERVER_A_URL=http://localhost
WORKER_API_TOKEN=change-me
WORKER_ID=shinypokemon-manual-worker
WORKER_JOB_TYPES=generation,training
```

Optional:

```env
POLL_INTERVAL_SECONDS=10
HEARTBEAT_INTERVAL_SECONDS=30
FAKE_RUN_SECONDS=5
FAKE_TRAINING_SECONDS=10
FAKE_UPLOAD_SECONDS=2
ARTIFACT_BASE_URI=s3://placeholder/generated
LOG_BASE_URI=s3://placeholder/logs
CHECKPOINT_BASE_URI=s3://placeholder/checkpoints
METRICS_BASE_URI=s3://placeholder/metrics
IDLE_SHUTDOWN_ENABLED=false
IDLE_SHUTDOWN_DRY_RUN=true
IDLE_SHUTDOWN_SECONDS=600
SHUTDOWN_COMMAND=sudo shutdown -h now
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

Training jobs use the same state flow and currently return placeholder training
metadata:

```text
artifact_uri = <CHECKPOINT_BASE_URI>/<run-id>/checkpoint.safetensors
log_uri = <LOG_BASE_URI>/<run-id>/train.log
result = {"mock": true, "checkpoint_uri": ..., "metrics_uri": ...}
```

Idle shutdown is disabled by default. When enabled, the worker asks Server A for
queue state after failed claims. If the queue is idle for
`IDLE_SHUTDOWN_SECONDS`, the worker runs `SHUTDOWN_COMMAND`. Keep
`IDLE_SHUTDOWN_DRY_RUN=true` while testing so the worker exits after printing
the command instead of powering off Server B.

## Smoke Test

Verified on 2026-05-25 through SSH port forwarding from local Windows to Server
A:

```powershell
ssh -L 8080:localhost:80 spacerobot-268369
```

Worker environment:

```powershell
$env:SERVER_A_URL="http://localhost:8080"
$env:WORKER_API_TOKEN="server-a-worker-test-token"
$env:WORKER_ID="local-fake-worker"
$env:WORKER_JOB_TYPES="generation"
$env:POLL_INTERVAL_SECONDS="3"
$env:HEARTBEAT_INTERVAL_SECONDS="2"
$env:FAKE_RUN_SECONDS="5"
$env:FAKE_UPLOAD_SECONDS="2"
python worker\worker.py
```

Observed worker output:

```text
worker starting id=local-fake-worker server=http://localhost:8080 types=generation
job 7: claimed type=generation
job 7: completed artifact_uri=s3://placeholder/generated/job-7.wav
```

Server A API verification:

```text
GET /api/jobs/7 -> 200 OK
status = completed
result.mock = true
result.worker_id = local-fake-worker
artifact_uri = s3://placeholder/generated/job-7.wav
```

## Server B Smoke Test

Verified on 2026-05-28 using the real Server B machine `shinypokemon`.

Machine details observed during setup:

```text
Hostname: shinypokemon.adelaideuni.cloud
Internal IP: 10.0.9.27
Instance type: g4dn.2xlarge
GPU: Tesla T4
VRAM: 15360 MiB
RAM: 32 GiB
vCPU: 8
OS: Ubuntu 22.04.5 LTS
Python: 3.12.10
Git: 2.34.1
NVIDIA driver: 580.126.09
```

Server B can reach Server A through the private network:

```bash
curl -i http://10.0.9.8/api/health
```

Response:

```text
HTTP/1.1 200 OK
{"ok":true,"db":"connected",...}
```

Code deployed on Server B:

```bash
git clone https://github.com/SpaceRobot-268369/COMP-6000-Capstone2.git COMP-6000-Capstone2-worker
cd ~/COMP-6000-Capstone2-worker
git checkout infra/songke/server-a-deployment
git log -1 --oneline
```

Verified commit:

```text
7699345 refactor: split worker generation adapter
```

Worker environment:

```bash
export SERVER_A_URL="http://10.0.9.8"
export WORKER_API_TOKEN="server-a-worker-test-token"
export WORKER_ID="shinypokemon-fake-worker"
export WORKER_JOB_TYPES="generation"
export POLL_INTERVAL_SECONDS="3"
export HEARTBEAT_INTERVAL_SECONDS="2"
export FAKE_RUN_SECONDS="5"
export FAKE_UPLOAD_SECONDS="2"
python worker/worker.py
```

Server A test job:

```text
POST /api/jobs
payload = {"seed":404,"source":"server-b-smoke"}
job id = 8
```

Server A API verification:

```text
GET /api/jobs/8 -> 200 OK
status = completed
result.mock = true
result.worker_id = shinypokemon-fake-worker
```

After the smoke test, the worker was stopped and Server B was stopped in RONIN
to avoid GPU budget usage.

## Server B Training Smoke Test

Verified on 2026-05-28 using the real Server B machine `shinypokemon`, after
adding the fake training adapter.

Verified commit on Server B:

```text
19ae0fe feat: add fake training worker adapter
```

Worker environment:

```bash
export SERVER_A_URL="http://10.0.9.8"
export WORKER_API_TOKEN="server-a-worker-test-token"
export WORKER_ID="shinypokemon-fake-worker"
export WORKER_JOB_TYPES="generation,training"
export POLL_INTERVAL_SECONDS="3"
export HEARTBEAT_INTERVAL_SECONDS="2"
export FAKE_RUN_SECONDS="5"
export FAKE_TRAINING_SECONDS="10"
export FAKE_UPLOAD_SECONDS="2"
python worker/worker.py
```

Server A test job:

```text
POST /api/jobs
type = training
payload = {
  "layer": "C",
  "run_id": "layer-c-fake-smoke-1",
  "species": "southern_boobook",
  "source": "server-b-training-smoke"
}
job id = 9
```

Server A API verification:

```text
GET /api/jobs/9 -> 200 OK
type = training
status = completed
payload.layer = C
payload.run_id = layer-c-fake-smoke-1
result.mock = true
result.worker_id = shinypokemon-fake-worker
```

This verifies the automatic fake training flow:

```text
Server A creates training job
-> Server B claims it
-> Server B marks running
-> Server B marks uploading
-> Server B records placeholder checkpoint/log/metrics metadata
-> Server A stores completed result
```

After the smoke test, the worker was stopped and Server B was stopped in RONIN
to avoid GPU budget usage.

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
