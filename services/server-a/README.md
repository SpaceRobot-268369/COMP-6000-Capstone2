# Server A Deployment

Server A is `spacerobot-268369`. The MVP deployment runs the public app stack
from the `ubuntu` user's deployment checkout:

```text
/home/ubuntu/eco-acoustic/COMP-6000-Capstone2
```

The deployment checkout stores Compose files and environment configuration.
Docker images are pulled from GHCR into Docker's own image store; they are not
stored inside this repository directory.

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
FRONTEND_URL=https://...
VITE_API_URL=https://...
SESSION_SECRET=...
APP_SECRET=...
AI_SERVER_LABEL=shinypokemon
AI_SSH_USER=ubuntu
AI_SSH_HOST=shinypokemon.adelaideuni.cloud
SERVERB_PEM_PATH=/home/ubuntu/.ssh/itds-eap/shinypokemon.pem
```

For local validation, copy `.env.example` to `.env` and replace all placeholder
secret values. Do not commit `.env`.

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
