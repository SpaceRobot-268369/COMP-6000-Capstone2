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
  "output_prefix": "s3://bucket/model/candidates/team-member-name/layer-c-boobook-v1",
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
| `output_prefix` | Durable output prefix for checkpoints, logs, configs, and metrics |
| `base_model` | Base model/checkpoint identifier or URI |
| `seed` | Reproducibility seed |

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
    "output_prefix": "s3://.../model/candidates/team-member-name/layer-a-site257-v1",
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
    "output_prefix": "s3://.../model/candidates/team-member-name/layer-b-weather-assets-v1",
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
    "output_prefix": "s3://.../model/candidates/team-member-name/layer-c-boobook-v1",
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
  "checkpoint_uri": "s3://.../checkpoint.safetensors",
  "config_uri": "s3://.../train_config.json",
  "metrics_uri": "s3://.../metrics.json",
  "sample_uri": "s3://.../samples/sample.wav",
  "artifact_uploaded": true
}
```

The job row should also set:

```text
artifact_uri = checkpoint_uri
log_uri = durable training log URI
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
  "log_uri": "s3://.../train.log",
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
7. Upload checkpoint/log/config/metrics/sample outputs.
8. Return result metadata to `worker/worker.py`.

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

