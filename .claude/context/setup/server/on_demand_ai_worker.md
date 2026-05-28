# On-Demand AI Worker Topology

## Purpose

Use a low-cost always-on application server for normal product traffic, and
start the GPU AI server only when generation or training work is required.

Server A is `spacerobot-268369`. It owns the frontend, backend, PostgreSQL,
job state, and worker orchestration. Server B is `shinypokemon`. It owns AI
generation and training work only, and should be stopped after it is idle and
all artifacts are durable.

## Server Roles

| Server | Role | Responsibilities |
|---|---|---|
| A - `spacerobot-268369` | Control plane + app server | Frontend, backend API, PostgreSQL, job table, auth, worker startup, job status/results |
| B - `shinypokemon` | On-demand AI worker | Layer A/C generation, model training, log capture, artifact upload, job status updates |

Server A is the source of truth. Server B must be disposable: if it stops or
crashes, Server A's PostgreSQL job records still describe what happened and
what needs retry, cancellation, or human review.

## Job Management Model

Jobs are stored in PostgreSQL on Server A. Server B checks Server A for work
when it starts, claims one job with a lock or lease, runs it, uploads outputs,
updates the final state, and then looks for more work.

Recommended MVP job statuses:

| Status | Meaning |
|---|---|
| `queued` | Job has been accepted by Server A but no worker owns it yet |
| `claimed` | Server B has leased the job but has not started the heavy task |
| `running` | Generation or training is actively executing |
| `uploading` | Work finished locally and outputs/logs/checkpoints are being synced |
| `completed` | Durable outputs are available and Server A has recorded the result |
| `failed` | Job ended with an error; record an error message and log URI |
| `cancel_requested` | User/dev requested cancellation; worker should stop cooperatively |
| `cancelled` | Worker stopped cleanly after a cancellation request |

Use a lease or heartbeat field so a crashed worker does not permanently hold a
job. A later worker may recover stale `claimed` or `running` jobs according to
the retry policy chosen by the backend.

## Multiple Job Rules

- Keep separate logical queues for `generation` and `training`.
- Give generation priority because users wait on it interactively.
- Use GPU concurrency `1` for the MVP unless a specific model path is proven to
  fit concurrently in VRAM.
- Do not run training and user-facing generation at the same time on one GPU
  unless the scheduler explicitly supports preemption or separate workers.
- Server B must claim jobs atomically, so two workers can never execute the same
  job.

## Server B Startup Flow

1. User or developer requests generation/training through Server A.
2. Server A inserts a PostgreSQL job row.
3. Server A starts Server B if no healthy worker is available.
4. Server B starts the AI worker process and validates readiness.
5. Server B claims the highest-priority eligible job.
6. Server B runs generation or training and sends heartbeat/status updates.

Readiness should mean more than process liveness. Check that required code,
Python environments, DVC/S3 artifacts, model checkpoints, and GPU access are
available before marking the worker ready.

## Server B Shutdown Flow

Server B may shut down only after:

- there are no `queued`, `claimed`, `running`, or `uploading` jobs that require
  the worker;
- logs, generated clips, metadata, and checkpoint artifacts are uploaded;
- Server A has recorded the final job status;
- an idle timeout has elapsed.

Use an idle timeout such as 5-15 minutes to avoid stopping and restarting B for
back-to-back requests.

The worker checks Server A before shutdown using the worker idle-check API.
Shutdown should remain disabled or dry-run until it has been tested on Server B:

```env
IDLE_SHUTDOWN_ENABLED=true
IDLE_SHUTDOWN_DRY_RUN=true
IDLE_SHUTDOWN_SECONDS=600
SHUTDOWN_COMMAND=sudo shutdown -h now
```

## Artifacts and Logs

Generated clips, spectrogram previews, explanation JSON, training logs, and
checkpoints must be persisted before Server B stops.

Follow the existing model discipline:

- training outputs go under `model/candidates/<member>/<run-id>/`;
- do not write directly into `model/production/<role>/` without explicit
  validation and promotion;
- binary artifacts are DVC-tracked and pushed to the configured S3 remote;
- readable release/log mirrors are synced separately when required.

## Failure and Cancellation

Cancellation is cooperative. Server A marks the job `cancel_requested`; Server B
checks that flag between long-running steps and then records `cancelled` once
cleanup is complete.

On failure, Server B records:

- final status `failed`;
- a short error message;
- a log URI or path;
- any partial artifact locations worth preserving for debugging.

Server A should treat missing heartbeats or expired leases as stale work and
either retry the job, mark it failed, or require manual intervention depending
on job type.

## Recommended MVP Implementation

- PostgreSQL job table on Server A.
- Express job APIs for create/status/cancel.
- Python worker daemon on Server B.
- Worker claims jobs from Server A using an atomic lock or lease.
- Generation and training write outputs to durable storage before final status.
- Idle-shutdown script stops Server B only after the queue is clear and syncs
  are complete.
