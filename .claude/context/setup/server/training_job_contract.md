# Training Job Contract

## Purpose

This document defines the MVP contract for Server A training jobs handled by
Server B (`shinypokemon`). It is the interface between the job system, the
worker, and layer owners. It does not define the internals of Layer A/B/C
training; it defines what the worker receives and what it must return.

Server A stores the job row and final result. Server B claims the job, runs the
layer-specific training command, uploads durable outputs, and updates Server A.

## Job Type

Use the existing job type:

```json
{
  "type": "training",
  "payload": {}
}
```

Layer-specific behavior is selected by `payload.layer`.

## Required Payload Fields

All real training jobs should include:

```json
{
  "layer": "C",
  "run_id": "layer-c-boobook-v1",
  "owner": "team-member-name",
  "dataset_uri": "s3://bucket/path/or/dvc/path",
  "base_model": "facebook/audiogen-medium",
  "seed": 42
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `layer` | `A`, `B`, `C`, or later `mixer`; worker uses this to select the adapter |
| `run_id` | Stable training run id; used in output paths and logs |
| `owner` | Team member responsible for the run |
| `dataset_uri` | Input dataset location; S3/DVC/local path agreed by the layer owner |
| `base_model` | Base model/checkpoint identifier or URI |
| `seed` | Reproducibility seed |

The worker derives output locations from the existing DVC/S3 conventions:

```text
local_output_dir = model/candidates/<owner>/<run_id>/
log_s3_prefix = s3://eco-acoustic-data.store.adelaideuni.cloud/logs/<layer>/<run_id>/<YYYY-MM-DD>/
```

Do not put model candidates directly under a human-readable S3
`model/candidates/` prefix. Candidate checkpoint binaries live in the repo
working tree under `model/candidates/...`, are tracked with DVC, and are pushed
to the DVC remote:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/
```

## Common Optional Payload Fields

```json
{
  "epochs": 5,
  "batch_size": 1,
  "learning_rate": 0.0001,
  "precision": "fp16",
  "max_train_steps": 1000,
  "resume_from_checkpoint": null,
  "notes": "short human-readable run purpose"
}
```

The worker should pass supported fields to the layer-specific script. Unknown
fields should be preserved in logs/configs where possible, but should not be
silently interpreted.

## Layer A Payload

Layer A is the ambient bed model path.

Expected MVP fields:

```json
{
  "type": "training",
  "payload": {
    "layer": "A",
    "run_id": "layer-a-site257-v1",
    "owner": "team-member-name",
    "dataset_uri": "s3://.../layer-a-dataset",
    "base_model": "cvssp/audioldm2",
    "site_id": "257",
    "prompt_set": "ambient_site257_mvp",
    "epochs": 3,
    "batch_size": 1,
    "precision": "fp16",
    "seed": 42
  }
}
```

Layer A owner must confirm:

- exact training script path;
- minimum VRAM for T4 15 GB;
- whether gradient checkpointing or small batch is required;
- expected checkpoint filename.

## Layer B Payload

Layer B covers weather/assets/parameterized mixing. It may not require GPU.

Expected MVP fields if executed through the worker:

```json
{
  "type": "training",
  "payload": {
    "layer": "B",
    "run_id": "layer-b-weather-assets-v1",
    "owner": "team-member-name",
    "dataset_uri": "s3://.../weather-assets",
    "task": "prepare_assets",
    "weather_types": ["rain", "wind"],
    "seed": 42
  }
}
```

Layer B owner must confirm whether this should run on Server B or stay as a CPU
asset-processing workflow.

## Layer C Payload

Layer C is the species/event model path.

Expected MVP fields:

```json
{
  "type": "training",
  "payload": {
    "layer": "C",
    "run_id": "layer-c-boobook-v1",
    "owner": "team-member-name",
    "dataset_uri": "s3://.../layer-c-boobook-dataset",
    "base_model": "facebook/audiogen-medium",
    "species": "southern_boobook",
    "epochs": 5,
    "batch_size": 1,
    "precision": "fp16",
    "seed": 42
  }
}
```

Layer C owner must confirm:

- exact training script path;
- species identifier format;
- dataset manifest format;
- expected checkpoint filename;
- expected metrics filename.

### Layer C SA3 LoRA Worker Payload

The first real training adapter supports the existing Layer C Stable Audio 3
LoRA smoke script. It is opt-in: Server B must set
`REAL_TRAINING_ENABLED=true`, and the job payload must include
`training_backend: "sa3_lora"`.

Example:

```json
{
  "type": "training",
  "payload": {
    "layer": "C",
    "training_backend": "sa3_lora",
    "run_id": "layer-c-sa3-smoke-10",
    "owner": "burger",
    "base_model": "stable-audio-3 small-sfx-base",
    "species": "horsfields_bronze_cuckoo",
    "steps": 10,
    "checkpoint_every": 10,
    "demo_every": 999999,
    "num_workers": 0,
    "seed": 42
  }
}
```

For the current MVP, the script path and output directory are fixed:

```text
script/events/train_sa3_lora_core6_smoke.sh
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/
```

The worker keeps heartbeats alive while the subprocess runs, DVC-tracks the
newest generated `.ckpt`, pushes it to the DVC remote, and returns the `.dvc`
pointer path as `artifact_uri`.

## Worker Result

On success, Server B should update the job to `completed` with:

```json
{
  "mock": false,
  "worker_id": "shinypokemon-worker",
  "layer": "C",
  "run_id": "layer-c-boobook-v1",
  "owner": "team-member-name",
  "duration_s": 3600,
  "gpu": "Tesla T4",
  "peak_vram_mb": 14000,
  "local_output_dir": "model/candidates/team-member-name/layer-c-boobook-v1",
  "checkpoint_path": "model/candidates/team-member-name/layer-c-boobook-v1/adapter_model.safetensors",
  "checkpoint_dvc_path": "model/candidates/team-member-name/layer-c-boobook-v1/adapter_model.safetensors.dvc",
  "log_uri": "s3://eco-acoustic-data.store.adelaideuni.cloud/logs/layer-c/layer-c-boobook-v1/2026-05-28/train.log",
  "config_path": "model/candidates/team-member-name/layer-c-boobook-v1/params.yaml",
  "metrics_path": "model/candidates/team-member-name/layer-c-boobook-v1/metrics.json",
  "sample_uri": "s3://eco-acoustic-data.store.adelaideuni.cloud/logs/layer-c/layer-c-boobook-v1/2026-05-28/samples/sample.wav",
  "artifact_uploaded": true
}
```

The job row should also set:

```text
artifact_uri = checkpoint_path or checkpoint_dvc_path
log_uri = durable training log URI under s3://eco-acoustic-data.store.adelaideuni.cloud/logs/...
```

For fake training jobs, `mock` remains `true` and placeholder URIs are allowed.

## Required Durable Outputs

A real training job is not complete until these are uploaded or otherwise made
durable:

- checkpoint or LoRA adapter;
- training log;
- training config/payload snapshot;
- metrics JSON;
- sample output if the layer produces one;
- any README or notes needed to interpret the run.

Do not leave the only copy of outputs on Server B local disk. Server B is
disposable and may shut down after idle.

Follow the existing project rules:

- local candidate output folder:
  `model/candidates/<owner>/<run_id>/`;
- checkpoint binaries (`.pt`, `.safetensors`, `.bin`, `.ckpt`) are DVC-tracked;
- metadata (`README.md`, `params.yaml`, `metrics.json`,
  `training_metadata.json`, `.dvc` pointers) is git-tracked;
- DVC pushes binary bytes to
  `s3://eco-acoustic-data.store.adelaideuni.cloud/dvc-cache/`;
- human-readable logs and audit samples can be mirrored to
  `s3://eco-acoustic-data.store.adelaideuni.cloud/logs/<layer>/<run_id>/<date>/`;
- do not write directly to `dvc-cache/` with `aws s3 cp`;
- do not write into `model/production/<role>/` until explicit validation and
  promotion sign-off.

## Failure Result

On failure, Server B should update the job to `failed` with:

```json
{
  "mock": false,
  "worker_id": "shinypokemon-worker",
  "layer": "C",
  "run_id": "layer-c-boobook-v1",
  "error_type": "oom",
  "error_message": "CUDA out of memory",
  "log_uri": "s3://eco-acoustic-data.store.adelaideuni.cloud/logs/layer-c/layer-c-boobook-v1/2026-05-28/train.log",
  "partial_artifact_uri": null
}
```

The job row should set:

```text
error_message = short human-readable reason
log_uri = durable log URI when available
```

Common failure types:

- `oom`;
- `missing_dataset`;
- `missing_base_model`;
- `invalid_payload`;
- `script_failed`;
- `upload_failed`;
- `cancelled`.

## Cancellation

Cancellation is cooperative:

```text
Server A marks cancel_requested
-> Worker sees cancel_requested on heartbeat
-> Worker stops between safe steps
-> Worker writes cancelled
```

Layer scripts should expose safe interruption points where possible. Long
single-call training commands may not cancel immediately unless the adapter can
terminate the subprocess safely.

## Adapter Responsibilities

`worker/training_adapter.py` should:

1. Validate payload fields for the selected layer.
2. Build the layer-specific training command.
3. Run the command and stream/capture logs.
4. Send heartbeats while training runs.
5. Detect cancellation requests.
6. Verify expected output files exist.
7. DVC-track checkpoint binaries and push them to the configured DVC remote.
8. Mirror logs and audit samples to the human-readable S3 `logs/` prefix when
   required.
9. Return result metadata to `worker/worker.py`.

The worker should not connect directly to PostgreSQL. All job state updates go
through Server A APIs.

## Current Implementation Status

Current implementation is fake:

```text
worker/training_adapter.py
```

It verifies:

```text
queued -> claimed -> running -> uploading -> completed
```

and returns placeholder checkpoint/log/metrics URIs. Real Layer A/C commands are
not connected yet.
