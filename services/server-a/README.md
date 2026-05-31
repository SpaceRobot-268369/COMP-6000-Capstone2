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
