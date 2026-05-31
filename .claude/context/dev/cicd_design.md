# CI/CD Design

**Status:** Draft
**Branch:** `infra/jay/cicd-pipeline`
**Last updated:** 2026-05-31

## Purpose

Add a CI/CD pipeline that keeps `main` deployable, validates pull requests
before merge, and supports the current Server A / Server B deployment model.

The deployment model is:

```text
GitHub
  -> CI/CD

Server A: spacerobot-268369
  - frontend
  - backend
  - PostgreSQL
  - ai-tunnel sidecar
  - public HTTP/HTTPS entrypoint

Server B: shinypokemon
  - acoustic_ai FastAPI service
  - native Python runtime in ~/shiny-pikachu
  - binds 127.0.0.1:8000 only
  - no public HTTP ingress

Backend -> ai-tunnel:8000 -> SSH tunnel -> Server B 127.0.0.1:8000
```

References:

- Local services: `../setup/local/services.md`
- Server A/B topology: `../setup/server/on_demand_ai_worker.md`
- Git workflow: `git_workflow.md`
- DVC workflow: `dvc_workflow.md`

## Goals

- Run fast, reliable checks for every pull request.
- Build deployable frontend, backend, and ai-tunnel artifacts from `main`.
- Deploy Server A through Docker images and Docker Compose.
- Automatically deploy Server A after a pull request is merged to `main` and
  GHCR images are published.
- Keep Server B private. CI/CD must not expose its FastAPI port publicly.
- Keep secrets, SSH keys, `.pem` files, model binaries, and generated audio out
  of git and out of Docker images.

## Non-goals

- Do not run full model training in CI.
- Do not run real generation in normal pull-request CI.
- Do not require Server B to be online for ordinary PR checks.
- Do not containerize the Server B AI service in the first implementation
  phase. The current design runs it natively on `shinypokemon`.
- Do not auto-start Server B from CI/CD. RONIN dashboard startup is currently a
  manual action.
- Do not deploy the Server B AI worker, job polling loop, Discord bot, or
  idle-shutdown automation in the current CI/CD scope. Those are runtime
  orchestration components.

## Pipeline Overview

```text
Pull request
  -> repo hygiene checks
  -> frontend checks
  -> backend checks
  -> acoustic_ai light checks
  -> Docker / Compose validation

Merge to main
  -> repeat CI checks
  -> build Docker images
  -> push images to registry
  -> deploy Server A
```

## CI: Pull Request Checks

Suggested workflow file:

```text
.github/workflows/ci.yml
```

Triggers:

```text
pull_request
push to non-main branches
workflow_dispatch
```

### Repo Hygiene

Checks:

- Reject committed private keys and certificates:
  - `*.pem`
  - `*.key`
  - `id_rsa`
  - `id_ed25519`
- Reject local environment files:
  - `.env.local`
  - `*.local`
  - `.env.*.local`
- Reject generated or heavy artifacts:
  - `node_modules/`
  - `dist/`
  - `build/`
  - `coverage/`
  - `*.wav`
  - `*.pt`
  - `*.ckpt`
  - `*.bin`
  - `*.safetensors` unless tracked through `.dvc`
- Verify that expected DVC pointer files are small text files.
- Run `git status --short` at the end of generated checks to ensure CI did not
  create tracked changes.

### Frontend

Current state:

- Package: `frontend/package.json`
- Current script: `npm run dev`
- Dockerfile currently starts the dev server.

Recommended first CI checks:

```bash
cd frontend
npm install
npx vite build
```

Recommended follow-up package scripts:

```json
{
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 5173",
    "build": "vite build",
    "check": "vite build"
  }
}
```

### Backend

Current state:

- Package: `backend/package.json`
- Current script: `npm run dev`
- Dockerfile currently starts the dev server.

Recommended first CI checks:

```bash
cd backend
npm install
node --check src/index.js
```

Recommended follow-up package scripts:

```json
{
  "scripts": {
    "dev": "node --watch src/index.js",
    "check": "node --check src/index.js"
  }
}
```

### Acoustic AI

Normal PR CI must stay lightweight. It should verify that the API module and
registry are structurally valid without downloading model checkpoints or
running generation.

Recommended checks:

```bash
python -m compileall acoustic_ai/server acoustic_ai/scripts
python -m compileall acoustic_ai/layers
python -c "from acoustic_ai.server.server import app; print(app.title)"
```

If dependency installation is too slow for every PR, split this into:

- always-on syntax checks using the system Python;
- scheduled or manual dependency import checks using `acoustic_ai/requirements.txt`.

Do not run:

- full training;
- real diffusion generation;
- broad `dvc pull`;
- GPU assertions.

### Docker and Compose Validation

Suggested checks:

```bash
docker build -t eco-frontend-ci frontend
docker build -t eco-backend-ci backend
docker build -t eco-ai-tunnel-ci services/dev/ai-tunnel
docker compose -f services/dev/docker-compose.yml config
```

CI should not start the full Compose stack by default because:

- `ai-tunnel` requires the Server B pem;
- Server B may be stopped;
- ordinary PR validation should not depend on private infrastructure.

Use synthetic environment values for `docker compose config`, but never provide
real `.pem` material to PR workflows.

## Image Registry

Recommended registry:

```text
GitHub Container Registry: ghcr.io
```

Images:

```text
ghcr.io/spacerobot-268369/eco-frontend
ghcr.io/spacerobot-268369/eco-backend
ghcr.io/spacerobot-268369/eco-ai-tunnel
```

Tags:

```text
<git-sha>
main
latest
```

The first production image phase should cover Server A services only:

- frontend;
- backend;
- ai-tunnel.

The Server B AI service should not be pushed as a Docker image until the project
decides to run GPU inference/training in containers on `shinypokemon`.

## CD: Server A

Suggested workflow file:

```text
.github/workflows/deploy-server-a.yml
```

Triggers:

```text
workflow_run after Images succeeds on main
workflow_dispatch for manual redeploy / rollback
```

Target:

```text
spacerobot-268369
```

Confirmed SSH target:

```text
HostName: SPACEROBOT-268369.ADELAIDEUNI.CLOUD
User: ubuntu
```

Confirmed production deployment checkout:

```text
/home/ubuntu/eco-acoustic/COMP-6000-Capstone2
```

Responsibilities:

- pull approved Docker images;
- run frontend, backend, PostgreSQL, and ai-tunnel;
- keep backend configured with `AI_SERVER_URL=http://ai-tunnel:8000`;
- expose only the intended public HTTP/HTTPS entrypoints;
- keep Server B access behind the SSH tunnel.

Deployment flow:

```text
PR merged to main
  -> CI runs on main
  -> Images workflow builds/pushes GHCR images tagged `main`
  -> Deploy Server A workflow starts after Images succeeds
  -> SSH to spacerobot-268369
  -> cd /home/ubuntu/eco-acoustic/COMP-6000-Capstone2
  -> git pull --ff-only origin main
  -> cd services/server-a
  -> docker compose pull
  -> docker compose up -d
  -> docker compose ps
```

Post-deploy checks:

- frontend responds over HTTP/HTTPS;
- backend health or root status endpoint responds;
- backend ServerB status endpoint returns a meaningful state;
- `ai-tunnel` container is either healthy or reports a clear ServerB-not-running
  state;
- no private key is present in image layers.

Server B may be stopped by design. Production Server A must still start the
frontend and backend when `ai-tunnel` is unhealthy; the backend status endpoint
should report the AI link state rather than blocking the whole app.

Server B runtime behavior is intentionally outside this deployment workflow:
operators start Server B manually in RONIN, then Server B reads Server A's job
list, runs work, reports status, posts Discord notifications, and shuts itself
down when idle.

### Server A Preflight

Suggested workflow file:

```text
.github/workflows/server-a-preflight.yml
```

Trigger:

```text
workflow_dispatch only
```

Purpose:

- verify the five Server A GitHub secrets are present;
- verify required keys exist inside `SERVER_A_PROD_ENV`;
- verify GitHub Actions can SSH to Server A as `ubuntu`;
- verify Server A has `git`, `curl`, Docker, and Docker Compose available;
- verify the Server B pem exists at `SERVERB_PEM_PATH`, is readable, and can be
  parsed by `ssh-keygen`.

This workflow must not deploy, restart containers, create the deployment
checkout, or modify the running Server A application. It is a safe pre-deploy
connectivity and configuration test.

Rollback strategy:

- keep the previous image SHA in the deploy metadata;
- redeploy the previous SHA if post-deploy checks fail;
- preserve logs from the failed deployment for review.

## Future: Server B AI Service Deployment

This is not part of the current CI/CD scope. The current deployment target is:

```text
PR merge to main -> GHCR images -> automatic Server A deploy
```

The notes below are retained for a later Server B worker/service deployment
phase.

Suggested workflow file:

```text
.github/workflows/deploy-server-b-ai.yml
```

Recommended trigger for MVP:

```text
workflow_dispatch only
```

Target:

```text
shinypokemon
```

Server B working-tree rule:

```text
~/shiny-pikachu tracks origin/main and runs the deployed AI service.
Never checkout feature branches in this tree.
```

Per-member experiment clones should remain separate, for example:

```text
~/lucano/COMP-6000-Capstone2
```

Deployment flow:

```bash
ssh shinypokemon '
  cd ~/shiny-pikachu &&
  git pull --ff-only origin main &&
  ./acoustic_ai/.venv/bin/python -m pip install -r acoustic_ai/requirements.txt &&
  ./acoustic_ai/.venv/bin/python -m compileall acoustic_ai/server acoustic_ai/layers &&
  if [ -f /tmp/shiny-pikachu-ai.pid ]; then
    kill "$(cat /tmp/shiny-pikachu-ai.pid)" || true
  fi &&
  nohup ./acoustic_ai/.venv/bin/python -m uvicorn acoustic_ai.server.server:app \
    --host 127.0.0.1 --port 8000 \
    > /tmp/shiny-pikachu-ai.log 2>&1 &
  echo $! > /tmp/shiny-pikachu-ai.pid &&
  curl -fsS http://127.0.0.1:8000/health
'
```

DVC policy:

- Pull only the production model pointers needed by `acoustic_ai/registry.yaml`.
- Do not run broad `dvc pull` from CI unless explicitly requested.
- Ensure binary model files remain DVC/S3 managed and are not committed to git.

Server B post-deploy checks:

- `curl http://127.0.0.1:8000/health` works on Server B.
- Server A can reach Server B through `ai-tunnel`.
- health response includes registry metadata and does not require model loading.
- logs are available at `/tmp/shiny-pikachu-ai.log`.

## Secrets

Current MVP Server A GitHub Actions secrets:

```text
SERVER_A_HOST=SPACEROBOT-268369.ADELAIDEUNI.CLOUD
SERVER_A_USER=ubuntu
SERVER_A_SSH_KEY
SERVER_A_DEPLOY_DIR=/home/ubuntu/eco-acoustic/COMP-6000-Capstone2
SERVER_A_PROD_ENV
```

`GITHUB_TOKEN` is provided automatically by GitHub Actions and is used for GHCR
login in the image publish and deploy workflows.

Future Server B deployment secrets may include:

```text
SERVER_B_HOST
SERVER_B_USER
SERVER_B_SSH_KEY
DVC or S3 credentials, if deployment runs dvc pull
```

Important rules:

- Do not commit `.pem` files.
- Do not bake SSH keys into Docker images.
- Mount the Server B pem into `ai-tunnel` as a Docker secret or host-side
  read-only file.
- Do not expose Server B FastAPI to the public internet.
- Do not print secrets in workflow logs.

## Server B Pem Handling

The GHCR image must not contain the Server B pem. The pem lives on Server A at
a documented host path. Deployment pulls the latest `ai-tunnel` image, then
Docker Compose mounts the host-side pem into the container.

Current development Compose convention:

```yaml
secrets:
  shinypokemon_pem:
    file: ${HOME}/.ssh/itds-eap/shinypokemon.pem
```

Container convention:

```text
/run/secrets/shinypokemon.pem
```

Production should prefer an explicit host path variable so the path does not
depend on which user runs `docker compose`:

```text
SERVERB_PEM_PATH=/home/ubuntu/.ssh/itds-eap/shinypokemon.pem
```

Then Compose can use:

```yaml
secrets:
  shinypokemon_pem:
    file: ${SERVERB_PEM_PATH}
```

The deployment preflight should verify:

- the pem exists on Server A;
- permissions are `600`;
- owner is the deploy user or otherwise readable by the deploy process;
- `ai-tunnel` receives it only through the secret mount.

## Environment Strategy

Recommended environments:

```text
ci
staging
production
```

MVP can start with:

```text
ci
production
```

GitHub Environment protections:

- require manual approval for production deploys;
- restrict production secrets to production workflows;
- keep Server B deploy manual until the worker lifecycle is more mature.

Current local development uses `services/dev/.env` as the shared Compose env
file for PostgreSQL, backend, frontend, and ai-tunnel. The frontend also has
`frontend/.env` for direct Vite runs.

Production should separate public frontend config from backend secrets:

```text
Server A Compose env
  - ports
  - non-secret AI tunnel labels / hosts
  - image tags

Backend secrets / env
  - DATABASE_URL
  - SESSION_SECRET
  - APP_SECRET
  - database password

Frontend public env
  - VITE_API_URL
```

Anything exposed through `VITE_*` is browser-visible after build and must not
contain secrets.

## Current Gaps Before Implementation

The repository currently has dev-focused Dockerfiles and package scripts.
Before a production-grade deploy, decide whether to:

- keep dev-style images for the MVP;
- add production Dockerfile targets;
- or create separate Dockerfiles for production.

Recommended production direction:

- frontend image builds static Vite assets and serves them through nginx or the
  Server A reverse proxy;
- backend image installs production dependencies and runs `node src/index.js`;
- ai-tunnel image stays small and dedicated to SSH tunnel setup and health
  checks.

The backend should also gain a simple health endpoint dedicated to deploy
verification if the current API surface is not sufficient.

## Implementation Phases

### Phase 1: CI Documentation and Checks

- Add this design document.
- Add `frontend` and `backend` check scripts.
- Add `.github/workflows/ci.yml`.
- Validate Docker builds and Compose config.

### Phase 2: Image Publishing

- Add GHCR image publishing from `main`.
- Build and tag frontend, backend, and ai-tunnel images.
- Store image SHA metadata for rollback.

### Phase 3: Automatic Server A Deploy

- Add automatic deploy to Server A after the `Images` workflow succeeds on
  `main`.
- Keep `workflow_dispatch` for manual redeploy / rollback.
- Use GHCR images.
- Run post-deploy checks.
- For direct auto-deploy, the GitHub `production` environment must not require
  manual reviewers. If reviewers are configured, deployment will pause for
  approval.

### Phase 4: Future Server B AI Deploy

- Out of current scope.
- Later, add `workflow_dispatch` deploy to Server B if the team wants CI/CD to
  update the worker/service code.
- Update `~/shiny-pikachu` only from `main`.
- Restart uvicorn and verify `/health`.
- Add optional targeted DVC pull for production model artifacts.

### Phase 5: Hardening

- Add rollback workflow.
- Add deploy notifications.
- Add Discord health and runtime notifications for Server B.
- Add scheduled drift checks for Server A and Server B.
- Add production Docker targets if MVP deploy used dev images.

## Open Decisions

- Should the GitHub `production` environment require manual reviewers, or stay
  unprotected to allow direct automatic deploy?
- Which DVC/S3 credentials are safe for Server B deployment automation?
- Should `frontend` be served as a static production build or continue through
  Vite dev server for the MVP?
- Should backend gain an explicit `/health` endpoint for deploy checks?
