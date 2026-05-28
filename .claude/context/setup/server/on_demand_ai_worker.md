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

## Network Topology

Server A (`spacerobot-268369`) is the only publicly reachable host. It exposes
exactly three inbound ports:

| Port | Purpose |
|---|---|
| 22 | SSH (admin + outbound tunnel origin to Server B) |
| 80 | HTTP (redirect to 443) |
| 443 | HTTPS (frontend + backend API) |

Server B (`shinypokemon`) has **no public ingress**. It is not exposed to the
internet — no open HTTP/HTTPS/API ports, no public DNS record needed for the
worker API. All communication from Server A to Server B (health checks, job
dispatch, status polling, log retrieval) goes through an **SSH tunnel**
initiated from Server A to Server B. Server A health-checks the worker by
hitting the tunnelled local port; Server B's worker API binds to localhost
only.

This keeps the attack surface on Server B minimal: the only way in is SSH from
Server A's key, and the worker HTTP endpoint is never directly reachable from
the public internet.

## Hosting and Control Plane (RONIN / AWS)

Both servers are RONIN-managed instances on AWS. RONIN owns authentication and
is the only sanctioned control surface for instance lifecycle.

- **Start:** must be done from the **RONIN dashboard**. `aws-cli` instance
  management is not available to this project, so Server A cannot auto-start
  Server B. For the MVP, starting `shinypokemon` is a manual action.
- **Stop:** can be done over SSH on the instance itself (`sudo shutdown -h now`
  or equivalent). The idle-shutdown automation described below uses this path.
- **Implication for the startup flow:** the "Server A starts Server B" step is
  aspirational. Until RONIN exposes a programmatic start API to us, the human
  triggering generation/training is responsible for booting `shinypokemon` via
  the dashboard. Once it boots, the rest of the flow (poll, claim, run, upload,
  shutdown) is automated.

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
| `paused` | Training job suspended at a checkpoint so a higher-priority generation job can run; resumable |

Use a lease or heartbeat field so a crashed worker does not permanently hold a
job. A later worker may recover stale `claimed` or `running` jobs according to
the retry policy chosen by the backend.

## Multiple Job Rules

- Keep separate logical queues for `generation` and `training`.
- Give generation absolute priority — users wait on it interactively.
- Use GPU concurrency `1` for the MVP unless a specific model path is proven to
  fit concurrently in VRAM.
- Generation and training never run at the same time on the GPU; preemption
  (see below) is how priority is enforced.
- Server B must claim jobs atomically, so two workers can never execute the same
  job.

## Generation Preempts Training

When `shinypokemon` boots, it drains all pending generation jobs on Server A
before claiming any training job. Generation is always served first.

While a training job is `running`, the worker must remain responsive to new
generation work:

- The training loop checkpoints every **N steps** (target cadence: every
  ~30–60 seconds of wall-clock, or every few hundred steps — whichever comes
  first). Tune N per model so the checkpoint cost stays under ~5% of step time.
- Between training steps, the worker polls Server A for `queued` generation
  jobs.
- On a hit: save a checkpoint *now*, transition the training job to `paused`
  (recording the checkpoint path, step count, and optimizer state location),
  release the GPU, drain all pending generation jobs to `completed`, then
  resume the paused training job from its checkpoint.
- Resuming a `paused` training job means reloading model + optimizer state and
  continuing from the recorded step — not restarting the epoch.

Rationale: an epoch-boundary preemption strategy would respect priority on
paper but force generation requests to wait minutes to tens of minutes behind a
running epoch, which defeats the interactive UX goal. Step-level checkpointing
bounds generation latency to roughly one N-step window.

## Runtime Monitoring (cron + Discord)

A monitor on Server A logs `shinypokemon` runtime, health-check status, current
job state, and per-job durations. If the worker has been up beyond a configured
threshold, is still running at midnight, or stops responding, a Discord bot posts
a warning to a designated channel so the GPU does not burn hours unnoticed.

Discord routing:

- `shinypokemon-bot-general`: lifecycle status only, limited to starting and
  stopping messages.
- `shinypokemon-bot-warning`: all cost-control, health-check, runtime, idle,
  timeout, and midnight warning messages.

Recommended MVP warning policy:

| Trigger | Threshold | Warning / action |
|---|---:|---|
| Worker booted but health check is unavailable | 5 min | Warn that `shinypokemon` may have booted without the worker API |
| Worker idle with no `queued`, `claimed`, `running`, or `uploading` jobs | 5 min | Warn that the worker is idle and should shut down soon |
| Worker still idle | 10-15 min | Auto-stop if logs/artifacts are durable and Server A has no eligible jobs |
| Generation job running too long | 15-20 min | Warn that the interactive job may be stuck |
| Generation job still running | 30 min | Mark failed or require manual approval to continue |
| Training job reaches 75% of declared `max_runtime_minutes` | Per job | Warn owner that the budget is nearly consumed |
| Training job exceeds `max_runtime_minutes` | Per job | Checkpoint, upload artifacts, transition to `paused`, and stop if no other work is pending |
| Total Server B uptime | 1 hour | Warn that the GPU server is still running |
| Total Server B uptime | 2 hours | Escalate warning |
| Total Server B uptime | 3-4 hours | Auto-stop if safe, otherwise send urgent warning |
| Server B is still running at `00:00` | Daily | Send a midnight warning regardless of current job state |

Training jobs must declare `max_runtime_minutes`. If no value is provided, use a
conservative default such as 60-120 minutes and record that default in the job
metadata. Generation jobs should have a much shorter hard cap because they are
interactive and are likely stuck if they run for tens of minutes.

Use a single configured project timezone for scheduled warnings; default to
Australia/Adelaide unless deployment chooses a different timezone explicitly.

### Discord message format

All Discord messages follow this pattern so humans can scan severity at a
glance and so the source (sample vs. real) is never ambiguous:

```
<emoji> <one-line title — what happened>
<key>: <value>
<key>: <value>
...
action / next step (when applicable)
```

Rules:

- First line is `<emoji> <title>`. Use one severity emoji from the table below.
- Body lines are `key: value` pairs separated by ` · ` when short, or one per
  line when long. Keep it copy-pasteable plain text inside a fenced code block.
- Timestamps use Australia/Adelaide (ACST/ACDT) and ISO-ish `YYYY-MM-DD HH:MM:SS`.
- Always include `worker_id` when a worker is running, and `current_job_id` /
  `current_job_type` when a job is in flight.
- End with an `action:` or `auto-action:` line whenever the message implies
  the bot or a human should do something.
- **Sample / test messages must be prefixed with `[SAMPLE]`** on the title
  line so they cannot be mistaken for a real production alert.

Severity emoji legend:

| Emoji | Meaning | Channel |
|---|---|---|
| 🟢 | Booting | general |
| ✅ | Ready / healthy | general |
| 🌙 | Shutdown initiated | general |
| ⚫ | Stopped | general |
| 💤 | Idle (soft) | warning |
| ⏱️ / ⏳ | Runtime / budget notice | warning |
| ⚠️ | Warning — needs attention | warning |
| 🛑 | Auto-stop / hard limit hit | warning |
| 🚨 | Escalation / urgent | warning |
| 🌒 | Midnight check | warning |

Emojis are allowed (and encouraged) in Discord messages from these bots even
though the rest of the project avoids emojis. This exception applies only to
bot output posted to Discord.

Open TODOs:

- Decide Discord webhook owner for `shinypokemon-bot-general` and
  `shinypokemon-bot-warning`.

## Server B Health Check API

`shinypokemon` should expose a lightweight health-check API that Server A can
poll after the instance is manually started from RONIN. The endpoint is for
liveness, readiness, and cost-control monitoring; it should not perform heavy
model loading on every request.

Recommended endpoint:

```http
GET /health
```

Recommended response fields:

| Field | Meaning |
|---|---|
| `status` | `starting`, `ready`, `busy`, `idle`, `draining`, or `error` |
| `worker_id` | Stable ID for the current worker process |
| `uptime_seconds` | Server or worker uptime used for warning thresholds |
| `started_at` | ISO timestamp for the current boot or worker process |
| `last_heartbeat_at` | ISO timestamp of the latest successful worker loop heartbeat |
| `current_job_id` | Active job ID, or `null` when idle |
| `current_job_type` | `generation`, `training`, or `null` |
| `current_job_started_at` | ISO timestamp for the active job, or `null` |
| `queue_poll_ok` | Whether Server B can reach Server A's job API |
| `storage_ok` | Whether required DVC/S3 artifact access is available |
| `gpu_ok` | Whether required GPU/MPS/CUDA access is available |
| `shutdown_eligible` | Whether Server B believes it is safe to stop after the idle timeout |

Server A should treat health-check failures as meaningful:

- no response after boot grace period: warn that the worker API is unavailable;
- repeated failures while jobs are `claimed` or `running`: mark the lease stale
  or require manual review;
- healthy but idle beyond the idle timeout: allow or request shutdown;
- healthy and still running at `00:00`: send the daily midnight warning even if
  the active job appears valid.

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
