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
worker/run_worker.sh
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

Required in `worker/.env` on Server B:

```env
SERVER_A_URL=http://10.0.9.8
WORKER_API_TOKEN=change-me
WORKER_ID=shinypokemon-worker
WORKER_JOB_TYPES=generation,training
```

Optional:

```env
POLL_INTERVAL_SECONDS=10
HEARTBEAT_INTERVAL_SECONDS=30
FAKE_RUN_SECONDS=5
FAKE_TRAINING_SECONDS=10
FAKE_UPLOAD_SECONDS=2
REAL_TRAINING_ENABLED=false
SA3_REPO=/home/ubuntu/stable-audio-3
SA3_PYTHON=acoustic_ai/.venv-audiogen/bin/python
SA3_MPLCONFIGDIR=/tmp/mpl
DVC_PYTHON=python3
DVC_PUSH_ENABLED=true
ARTIFACT_BASE_URI=s3://placeholder/generated
LOG_BASE_URI=s3://placeholder/logs
CHECKPOINT_BASE_URI=s3://placeholder/checkpoints
METRICS_BASE_URI=s3://placeholder/metrics
IDLE_SHUTDOWN_ENABLED=true
IDLE_SHUTDOWN_DRY_RUN=false
IDLE_SHUTDOWN_SECONDS=600
SHUTDOWN_COMMAND=sudo shutdown -h now
```

For Server A testing through SSH port forwarding, `SERVER_A_URL` can point at
the forwarded localhost URL.

## Run

On Server B, create the local env file once:

```bash
cd ~/COMP-6000-Capstone2-worker
cp worker/.env.example worker/.env
nano worker/.env
```

Set the real `WORKER_API_TOKEN` in `worker/.env`. The file is ignored by git.

Then start the worker from the repository root:

```bash
bash worker/run_worker.sh
```

The script loads `worker/.env`, validates required variables, and runs
`python worker/worker.py`.

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

When `REAL_TRAINING_ENABLED=true`, Layer C jobs with
`payload.training_backend = "sa3_lora"` run the real Stable Audio 3 LoRA script
instead of the fake training adapter. Other training jobs remain fake until
their layer-specific adapters are implemented.

Minimal real Layer C payload:

```json
{
  "type": "training",
  "payload": {
    "layer": "C",
    "training_backend": "sa3_lora",
    "run_id": "layer-c-sa3-smoke-10",
    "owner": "burger",
    "steps": 10,
    "checkpoint_every": 10,
    "demo_every": 999999,
    "num_workers": 0
  }
}
```

The real adapter:

1. runs `script/events/train_sa3_lora_core6_smoke.sh`;
2. keeps Server A heartbeats alive while the subprocess runs;
3. writes a local log under `logs/`;
4. finds the newest `.ckpt` in the Layer C SA3 checkpoint directory;
5. runs `python3 -m dvc add <checkpoint>`;
6. runs `python3 -m dvc push <checkpoint>.dvc`;
7. marks the job completed with `artifact_uri = <checkpoint>.dvc`.

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

## Server B Idle Shutdown Smoke Test

Verified on 2026-05-28 using the real Server B machine `shinypokemon`.

Server A idle-check API verification:

```bash
printf '%s' '{"worker_id":"manual-idle-check","types":["generation","training"]}' > /tmp/idle-check.json
curl -i -H "Authorization: Bearer server-a-worker-test-token" \
  -H "Content-Type: application/json" \
  -d @/tmp/idle-check.json \
  http://localhost/api/worker/jobs/idle-check
```

Response:

```text
HTTP/1.1 200 OK
{"ok":true,"idle":true,"queued_count":0,"active_count":0,"uploading_count":0,"checked_types":["generation","training"]}
```

Dry-run shutdown test on Server B:

```bash
export SERVER_A_URL="http://10.0.9.8"
export WORKER_API_TOKEN="server-a-worker-test-token"
export WORKER_ID="shinypokemon-idle-dry-run"
export WORKER_JOB_TYPES="generation,training"
export POLL_INTERVAL_SECONDS="3"
export HEARTBEAT_INTERVAL_SECONDS="2"
export FAKE_RUN_SECONDS="5"
export FAKE_TRAINING_SECONDS="10"
export FAKE_UPLOAD_SECONDS="2"
export IDLE_SHUTDOWN_ENABLED="true"
export IDLE_SHUTDOWN_DRY_RUN="true"
export IDLE_SHUTDOWN_SECONDS="30"
python worker/worker.py
```

Observed output:

```text
worker starting id=shinypokemon-idle-dry-run server=http://10.0.9.8 types=generation,training
idle-check: queue idle; starting shutdown timer
idle shutdown dry-run: would run `sudo shutdown -h now`
```

Real shutdown pre-check:

```bash
sudo -n true
echo $?
```

Observed output:

```text
0
```

Real shutdown test:

```bash
export SERVER_A_URL="http://10.0.9.8"
export WORKER_API_TOKEN="server-a-worker-test-token"
export WORKER_ID="shinypokemon-idle-real-shutdown"
export WORKER_JOB_TYPES="generation,training"
export POLL_INTERVAL_SECONDS="3"
export HEARTBEAT_INTERVAL_SECONDS="2"
export IDLE_SHUTDOWN_ENABLED="true"
export IDLE_SHUTDOWN_DRY_RUN="false"
export IDLE_SHUTDOWN_SECONDS="30"
python worker/worker.py
```

Observed behavior:

```text
idle-check: queue idle; starting shutdown timer
idle shutdown: running `sudo shutdown -h now`
```

Server B then shut down successfully and RONIN reported that it could stop.
For normal use, set `IDLE_SHUTDOWN_SECONDS=600` for a 10-minute idle timeout.

## Server B Real Layer C SA3 Smoke Test

Verified on 2026-05-28 using the real Server B machine `shinypokemon`.

This test was run manually on Server B to prove the real GPU training stack,
DVC/S3 data flow, Hugging Face model access, and checkpoint artifact storage.
It was not run through the automatic worker training adapter yet.

Verified Git commit on Server B:

```text
8d81066 data: track layer c sa3 10-step smoke checkpoint
```

Runtime environment:

```text
GPU: Tesla T4, 15360 MiB VRAM
CUDA available through torch: true
torch: 2.7.1+cu126
Stable Audio 3 upstream repo: /home/ubuntu/stable-audio-3
Stable Audio 3 commit: fa5ee841dd49bae0fa361fac26904adc27fd400e
Layer C training env: acoustic_ai/.venv-audiogen
Hugging Face auth: verified with hf auth whoami
```

DVC/S3 verification:

```text
Layer C smoke wavs were pushed from local Windows with DVC.
Server B ran dvc pull and materialized all six Layer C wav files.
The 10-step checkpoint was pushed from Server B with DVC.
```

Training command:

```bash
cd ~/COMP-6000-Capstone2-worker
mkdir -p logs

SA3_REPO=/home/ubuntu/stable-audio-3 \
PYTHON=/home/ubuntu/COMP-6000-Capstone2-worker/acoustic_ai/.venv-audiogen/bin/python \
MPLCONFIGDIR=/tmp/mpl \
SA3_STEPS=10 \
SA3_CHECKPOINT_EVERY=10 \
SA3_DEMO_EVERY=999999 \
SA3_NUM_WORKERS=0 \
bash script/events/train_sa3_lora_core6_smoke.sh \
  2>&1 | tee logs/sa3_lora_core6_train_10.log
```

Observed result:

```text
GPU available: True (cuda), used: True
Found 6 files
LoRA config: rank=8, alpha=8.0, adapter_type=dora-rows
lora layers: 192
5.6 M trainable params
Trainer.fit stopped: max_steps=10 reached
```

Generated checkpoint:

```text
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/epoch=1-step=10.ckpt
size: 22 MB
```

Git/DVC artifact pointer:

```text
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/epoch=1-step=10.ckpt.dvc
```

This proves a minimal real training path:

```text
DVC/S3 data -> Server B CUDA -> Stable Audio 3 base model -> Layer C LoRA training -> checkpoint -> DVC/S3 artifact
```

The 300-step Layer C run was intentionally not started during this handoff.
Treat 300-step training as Layer C model-quality work, not as required evidence
for the Server A/B infrastructure MVP.

## Not Implemented Yet

- Server A -> Server B end-to-end verification of the real Layer C SA3 training
  adapter;
- durable S3 upload or retention policy for training logs;
- 300-step Layer C model-quality smoke run;
- real generation inference path;
- worker registry;
- local disk cleanup;
- OOM detection and retry classification.

These are later Milestone 3/4/P1 tasks.
