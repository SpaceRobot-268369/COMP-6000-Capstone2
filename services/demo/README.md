# Demo deployment — image-only stack

Deploys the `demo` branch to any Docker host with **no repo checkout, no DVC
pull, no serverB, and no GPU**. Everything the stack needs is baked into five
images; the host only needs `docker-compose.yml` and a filled-in `.env`.

**There is no AI in this stack.** `ai-mock` replays pre-baked fixtures over the
same HTTP contract as the real FastAPI worker, so backend and frontend code are
unchanged. Nothing it returns is model output, and only the preset prompts and
preset recordings are faithful to their input — read
[services/dev/ai-mock/README.md](../dev/ai-mock/README.md) before demoing.

This is separate from [services/server-a/](../server-a/), which is the real
deployment: GHCR images built from `main`, an `ai-tunnel` sidecar, and an SSH
key to serverB. Nothing here touches that stack or its image tags.

---

## The images

| Image | Contents beyond the base |
|---|---|
| `eco-demo-frontend` | `frontend/`, npm deps prebuilt |
| `eco-demo-backend` | `backend/`, npm deps prebuilt, **+ the `layers` sample fixtures at `/mock/layers`** (`AI_LAYERS_ROOT`) |
| `eco-demo-ai-mock` | mock app **+ its 65 MB fixture tree + `acoustic_ai/registry.yaml`** |
| `eco-demo-postgres` | `services/dev/db_init.sql` in the init dir (seeds `test@test.com / test1234`) |
| `eco-demo-nginx` | [`nginx/default.conf`](nginx/default.conf) — HTTP :80, `/api/` → backend, `/` → frontend |

Two deliberate differences from the `main` app images:

- **No entrypoint.** `backend/entrypoint.sh` and `frontend/entrypoint.sh` wipe
  `node_modules` and re-run `npm install` at every start — correct for the
  bind-mounted dev stack, wrong here, where it would make the demo host need
  npm registry access to boot. The baked `node_modules` is used as-is.
- **`NODE_ENV=development`** in the backend image. Under `production`,
  `index.js` marks the session cookie `secure`, which silently breaks login on
  a plain-HTTP demo host.

Both images still run the Vite / `node --watch` dev servers, exactly as Server A
does today.

---

## Deploy

On the demo host:

```bash
mkdir -p ~/eco-demo && cd ~/eco-demo
# copy services/demo/docker-compose.yml and services/demo/.env.example here
cp .env.example .env   # then edit it — see below
docker login ghcr.io   # GHCR packages are private by default; needs a PAT with read:packages
docker compose pull
docker compose up -d
```

Only `nginx` publishes a port (`PUBLIC_PORT`, default 80). Postgres, backend,
frontend and the mock stay on the internal network.

### The two settings that actually matter

**`FRONTEND_URL`** is the backend's CORS allowlist and is matched **exactly**
against the browser `Origin` header — scheme + host + port, lowercase, no
trailing slash. Get it wrong and login, register and generate all fail. Set it
to the origin you will type in the browser:

| Reached as | `FRONTEND_URL` |
|---|---|
| `http://203.0.113.10` | `http://203.0.113.10` |
| `http://demo.example.com` | `http://demo.example.com` |
| `http://demo.example.com:8080` (`PUBLIC_PORT=8080`) | `http://demo.example.com:8080` |

**`IMAGE_TAG`** — `demo` follows the branch tip. Pin the short commit sha
instead if you want the exact build you rehearsed with.

Leave `VITE_API_URL` alone (empty, pinned in the compose file). The app calls
`/api` same-origin through nginx; an absolute base breaks the moment the host
is renamed or fronted by HTTPS.

### Verify

```bash
curl -fsS http://localhost/api/health
```

Then open the site and run one preset prompt end to end. `GET /api/ai/health`
should report `"mock": true`.

---

## Before you point this at the public internet

- **Auth is disabled in the application itself.** `requireAuth` in
  `backend/src/index.js:93` is commented out (pre-existing on `main`, not
  something this stack introduces), so every `/api` route is reachable without
  a session. On an exposed host, anyone who finds it can call every endpoint.
  Put it behind a firewall rule, an IP allowlist, or HTTP basic auth at a proxy
  in front of nginx if that matters for your demo.
- **HTTP only.** Baking a cert path into the nginx image would break every host
  that lacks it. For HTTPS, terminate TLS in front of this stack (host nginx,
  Caddy, or a load balancer) with `nginx` as the upstream, and set
  `FRONTEND_URL` to the `https://` origin.
- Change `POSTGRES_PASSWORD` and `SESSION_SECRET` from the example values.

---

## Building the images

CI does this on every push to `demo`
([.github/workflows/images-demo.yml](../../.github/workflows/images-demo.yml)),
tagging `demo` and the short sha. It never writes the `main` / `latest` tags,
and no deploy workflow consumes it — demo deployment is manual on purpose.

To build locally, from the **repo root**. Each image uses a tiny main context
(`services/demo/images`) plus named build contexts for the directories it
actually needs — the repo root is a DVC checkout with `resources/` and `model/`
in it and must never be a build context.

```bash
docker buildx build -f services/demo/images/frontend.Dockerfile --build-context app=frontend -t eco-demo-frontend:local --load services/demo/images
```

```bash
docker buildx build -f services/demo/images/backend.Dockerfile --build-context app=backend --build-context fixtures=services/dev/ai-mock/fixtures/layers -t eco-demo-backend:local --load services/demo/images
```

```bash
docker buildx build -f services/demo/images/ai-mock.Dockerfile --build-context app=services/dev/ai-mock --build-context registry=acoustic_ai -t eco-demo-ai-mock:local --load services/demo/images
```

```bash
docker buildx build -f services/demo/images/postgres.Dockerfile --build-context sql=services/dev -t eco-demo-postgres:local --load services/demo/images
```

```bash
docker buildx build -f services/demo/images/nginx.Dockerfile --build-context conf=services/demo/nginx -t eco-demo-nginx:local --load services/demo/images
```

Named build contexts need Buildx ≥ 0.8 (`docker buildx build`, not plain
`docker build` on older engines).

To run the locally built set, add the image overrides to `.env`:

```
AI_MOCK_IMAGE=eco-demo-ai-mock:local
POSTGRES_IMAGE=eco-demo-postgres:local
BACKEND_IMAGE=eco-demo-backend:local
FRONTEND_IMAGE=eco-demo-frontend:local
NGINX_IMAGE=eco-demo-nginx:local
```

Changing what the demo *says* (preset prompts, canned reports) is a fixture
rebuild, not an image change — see the ai-mock README. Rebuild fixtures first,
then rebuild `eco-demo-ai-mock` and `eco-demo-backend` (it carries a copy of
the `layers` tier).

---

## Local development

Unchanged: `services/dev/docker-compose.yml` still builds from source with
bind-mounted code and the same mock. Use that for iterating; this directory is
only for putting the demo on a machine that has nothing else on it.
