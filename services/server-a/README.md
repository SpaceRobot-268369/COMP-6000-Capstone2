# Server A Deployment

Server A is `spacerobot-268369`. The MVP deployment runs the public app stack
from the `ubuntu` user's deployment checkout:

```text
/home/ubuntu/eco-acoustic/COMP-6000-Capstone2
```

The deployment checkout stores Compose files and environment configuration.
Docker images are pulled from GHCR into Docker's own image store; they are not
stored inside this repository directory.

## Boot auto-start

Every service in `docker-compose.yml` (`ai-tunnel`, `postgres`, `backend`,
`frontend`) plus `nginx` in `docker-compose.override.yml` declares
`restart: unless-stopped`. The Docker daemon is enabled on boot
(`systemctl is-enabled docker`), so after a reboot or a power-cycle the whole
stack comes back automatically — no manual `docker compose up`. `unless-stopped`
(rather than `always`) means an explicit `docker compose down` / `docker stop`
stays down until you bring it back, so taking the host offline deliberately is
not fought by the restart policy.

To take the stack down and keep it down across reboots, `docker compose down`;
to bring it back, `docker compose up -d` from the deploy checkout.

The production database init script creates schema only. It does not seed the
development `test@test.com / test1234` account.

Required host-side Server B key:

```text
/home/ubuntu/.ssh/itds-eap/shinypokemon.pem
```

The key is mounted into `ai-tunnel` as:

```text
/run/secrets/shinypokemon.pem
```

Expected Server A production environment values:

```text
IMAGE_TAG=main
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DB=...
POSTGRES_PORT=5432
BACKEND_PORT=4000
FRONTEND_PORT=5173
FRONTEND_URL=https://spacerobot-268369.adelaideuni.cloud,http://16.176.232.101
VITE_API_URL=
SESSION_SECRET=...
APP_SECRET=...
AI_SERVER_LABEL=shinypokemon
AI_SSH_USER=ubuntu
AI_SSH_HOST=shinypokemon.adelaideuni.cloud
AI_REQUEST_TIMEOUT_MS=300000
AI_SERVICE_START_COMMAND=
AI_TUNNEL_CONTAINER_NAME=eco-ai-tunnel-server-a
AI_RECONNECT_MODE=docker-container
SERVERB_PEM_PATH=/home/ubuntu/.ssh/itds-eap/shinypokemon.pem
```

`FRONTEND_URL` is the backend CORS allowlist and is matched **exactly** against
the browser `Origin` header (scheme + host + port, case-sensitive). In
production it must be the public HTTPS origin
(`https://spacerobot-268369.adelaideuni.cloud`), not the `:5173`/`http` dev
value — a mismatch makes the backend reject generate requests with a 500.

`AI_REQUEST_TIMEOUT_MS` must exceed the synchronous generation time (30-90s warm
on serverB, far longer for a cold model load) or the backend returns 504 before
the audio comes back. It must also stay **under** the nginx `/api/`
`proxy_read_timeout` (600s) so the backend's descriptive error surfaces ahead of
a bare nginx 504 — the deployed default is `540000` (9 min). serverB pre-warms
its default models on boot (`AI_PREWARM`, see the on-demand worker doc), so the
warm path is the norm; the headroom mainly covers cold loads of non-default
attempts.

Automatic serverB reconnect is split into two layers. The backend restarts the
configured `ai-tunnel` container through Docker when health checks go offline.
Inside that container, SSH reachability to serverB is checked before the tunnel
opens. If serverB is reachable but the remote AI service `/health` is down,
`AI_SERVICE_START_COMMAND` is run when configured; leave it empty to report a
degraded status without attempting to start anything remotely.

For local validation, copy `.env.example` to `.env` and replace all placeholder
secret values. Do not commit `.env`.

`FRONTEND_URL` must list the **exact** public origin(s) the browser sends as
its `Origin` header — lowercase scheme + host, no trailing slash, and no port
when nginx serves on the default `:443`/`:80`. The backend treats it as a CORS
allow-list; a case/scheme/port mismatch makes every credentialed `POST`
(login, register, generate) fail CORS. Leave `VITE_API_URL` empty so the
frontend calls `/api` same-origin through nginx.

## TLS / HTTPS

The reverse proxy terminates HTTPS on `:443` for the public hostname
`spacerobot-268369.adelaideuni.cloud` using a Let's Encrypt certificate. This
matters because browsers default to HTTPS-First when a user types the bare
hostname — without `:443` they hit a dead port and the site appears
unreachable (the raw IP still works because IPs aren't auto-upgraded).

- Cert material lives on the host at `/etc/letsencrypt`, mounted read-only into
  the `nginx` container by `docker-compose.override.yml`. The `:443` server
  block in `nginx/default.conf` references the `live/<domain>/` symlinks.
- The bare IP stays on `:80` with **no forced http→https redirect** (it has no
  cert), so `http://16.176.232.101/...` keeps working.
- Initial issuance used the certbot container in `--standalone` mode (HTTP-01
  over the already-open `:80`, nginx briefly stopped).
- Renewal: `/usr/local/bin/renew-letsencrypt.sh` (root cron, weekly) renews only
  within 30 days of expiry — it stops nginx, runs `certbot renew --standalone`,
  and brings nginx back up. Logs to `/var/log/letsencrypt-renew.log`.

Before enabling automatic deployment, run the manual `Server A Preflight`
GitHub Actions workflow. It verifies the five Server A repository secrets, SSH
access to `spacerobot-268369`, Docker/Compose availability, and the host-side
Server B pem path without deploying or restarting the app.
