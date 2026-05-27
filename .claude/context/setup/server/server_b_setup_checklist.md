# Server B Setup Checklist

## Purpose

This checklist is for the first real Server B (`shinypokemon`) setup after the
MVP GPU tier has been approved. Server B should not be created before the sweet
spot tier is decided for the MVP workload: audio generation and model training.

Server B is an on-demand GPU worker. It should run only when generation or
training work is required, and it must be stopped after verification or idle
work completion according to the shutdown policy in
`on_demand_ai_worker.md`.

## Before Creation

Confirm these details before creating Server B in RONIN:

- approved GPU tier;
- GPU model and VRAM;
- hourly price;
- expected storage size;
- Ubuntu version;
- SSH key name;
- who is responsible for stopping the server after testing;
- planned test window start and end time.

Do not create Server B if the GPU tier is still undecided.

## Creation Details To Record

After the machine is created, record:

```text
Machine name:
Public hostname:
Internal IP:
SSH username:
SSH key:
GPU tier:
GPU model:
VRAM:
Hourly price:
Storage size:
Created by:
Created at:
Current state: running/stopped
```

Recommended machine name:

```text
shinypokemon
```

## First SSH Login

From Windows PowerShell:

```powershell
ssh -i C:\path\to\key.pem ubuntu@SERVER_B_HOSTNAME
```

Optional local SSH config entry:

```sshconfig
Host shinypokemon
    HostName SERVER_B_HOSTNAME
    User ubuntu
    IdentityFile ~/.ssh/YOUR_KEY.pem
```

Then connect with:

```powershell
ssh shinypokemon
```

## System Checks

Run these on Server B:

```bash
uname -a
cat /etc/os-release
python3 --version
git --version
nvidia-smi
```

Expected result:

- Ubuntu is available;
- Python 3 is available;
- git is available or can be installed;
- `nvidia-smi` shows the approved GPU.

If `nvidia-smi` fails, stop before installing project dependencies and confirm
whether the RONIN image includes NVIDIA drivers.

## Base Packages

Install only basic tools first:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

Docker is optional for the first fake worker test. If Docker is required later:

```bash
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu
exit
```

Reconnect after adding the `docker` group.

## Clone Project

```bash
cd ~
git clone https://github.com/SpaceRobot-268369/COMP-6000-Capstone2.git COMP-6000-Capstone2-worker
cd ~/COMP-6000-Capstone2-worker
git checkout infra/songke/server-a-deployment
git log -1 --oneline
```

Confirm the worker exists:

```bash
ls worker/worker.py
```

## Python Environment

Project rule: AI Python work uses `acoustic_ai/.venv`.

```bash
python3 -m venv acoustic_ai/.venv
source acoustic_ai/.venv/bin/activate
python --version
```

The fake worker currently uses only the Python standard library, so no package
install is required for the first Server B communication test.

## Server A Connectivity

Try the direct internal address first:

```bash
curl -i http://10.0.9.8/api/health
```

If that fails, try the RONIN hostname:

```bash
curl -i http://SPACEROBOT-268369.ADELAIDEUNI.CLOUD/api/health
```

If both fail, use SSH port forwarding as a temporary development path:

```bash
ssh -L 8080:localhost:80 spacerobot-268369
```

The worker `SERVER_A_URL` should use the first working option:

```text
http://10.0.9.8
http://SPACEROBOT-268369.ADELAIDEUNI.CLOUD
http://localhost:8080
```

## Fake Worker Environment

Set environment variables on Server B:

```bash
export SERVER_A_URL="http://10.0.9.8"
export WORKER_API_TOKEN="server-a-worker-test-token"
export WORKER_ID="shinypokemon-fake-worker"
export WORKER_JOB_TYPES="generation"
export POLL_INTERVAL_SECONDS="3"
export HEARTBEAT_INTERVAL_SECONDS="2"
export FAKE_RUN_SECONDS="5"
export FAKE_UPLOAD_SECONDS="2"
```

Replace `SERVER_A_URL` with whichever connectivity test passed.

The test token is temporary. Replace it with a strong production token before
real Server B use.

## Run Fake Worker

From the repository root on Server B:

```bash
python worker/worker.py
```

Expected startup:

```text
worker starting id=shinypokemon-fake-worker server=<SERVER_A_URL> types=generation
```

Keep this session open while creating a test job on Server A.

## Create Test Job

Create a `generation` job through Server A. The exact command can be run from a
local machine through SSH tunnel, or directly on Server A.

Expected worker output:

```text
job <id>: claimed type=generation
job <id>: completed artifact_uri=s3://placeholder/generated/job-<id>.wav
```

Expected Server A job state:

```text
status = completed
result.mock = true
result.worker_id = shinypokemon-fake-worker
artifact_uri = s3://placeholder/generated/job-<id>.wav
```

## Stop Rules After Test

Before stopping Server B, confirm:

- the worker is not processing a job;
- no required job is in `claimed`, `running`, or `uploading`;
- Server A has recorded final job status;
- generated artifacts, logs, and checkpoints are uploaded when real generation
  or training is enabled.

For the fake worker test, stop the Python worker with `Ctrl+C`, then stop the
machine from RONIN. Closing SSH does not stop billing.

## Handoff To Real GPU Work

Only after the fake Server B test passes:

- install real PyTorch/CUDA dependencies;
- configure DVC/S3 access;
- validate model checkpoints;
- replace fake sleep with real generation;
- upload real artifacts and logs;
- keep GPU concurrency at `1` for MVP unless proven safe.

