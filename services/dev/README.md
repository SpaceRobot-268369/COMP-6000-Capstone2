# Dev Services

## SSH Key Convention

Do not store `.pem` files in this repository or bake them into Docker images.

Use these host-side paths:

```text
serverB AI host: $HOME/.ssh/itds-eap/shinypokemon.pem
serverA host:    $HOME/.ssh/itds-eap/spacerobot-268369.pem
```

The Compose-managed tunnel mounts the serverB key into the tunnel container
only, read-only, at:

```text
/run/secrets/shinypokemon.pem
```

Recommended permissions:

```bash
chmod 700 "$HOME/.ssh" "$HOME/.ssh/itds-eap"
chmod 600 "$HOME/.ssh/itds-eap/shinypokemon.pem"
chmod 600 "$HOME/.ssh/itds-eap/spacerobot-268369.pem"
```

## ServerB AI Tunnel

The default dev Compose setup starts an `ai-tunnel` sidecar:

```text
Docker backend -> ai-tunnel:8000 -> SSH tunnel -> shinypokemon:127.0.0.1:8000
```

Start the stack from this directory:

```bash
docker compose up
```

The tunnel container checks for missing pem files, invalid pem format, SSH
authentication failures, DNS failures, basic network failures, and serverB
AI `/health` failures before opening the tunnel.

If SSH to serverB works but the AI service `/health` is unavailable, the tunnel
can run a fixed remote startup command before it fails:

```bash
AI_SERVICE_START_COMMAND='sudo systemctl restart eco-acoustic-ai.service'
```

Leave `AI_SERVICE_START_COMMAND` empty to diagnose this case without starting
anything remotely. The frontend will show it as degraded/yellow after the
backend reconnect attempt confirms serverB is reachable but AI service health
did not recover.

For manual diagnosis without Compose, run:

```bash
./start-ai-tunnel.sh
```

On serverB, the FastAPI process needs to be running on port 8000. For an SSH tunnel, binding to localhost is enough:

```bash
cd /path/to/COMP-6000-Capstone2/acoustic_ai
source .venv/bin/activate
python -m uvicorn server.server:app --host 127.0.0.1 --reload --port 8000
```

### Optional Layer B Generate Wind Test Port

Keep the main AI registry server on port 8000. If Layer B wind generation is
being tested on a separate serverB process, run that process on a separate
800x port and point the backend at it with environment variables instead of
hard-coding the port in frontend code.

Example shape, using the URL as seen from the backend process:

```bash
LAYER_B_GENERATE_SERVER_URL=http://<backend-reachable-host>:<local-800x-port>
LAYER_B_GENERATE_ATTEMPTS=murphy__mvp_1__wind_intensity_bank,murphy__mvp_1__rain_intensity_seed_pool
```

The backend only uses `LAYER_B_GENERATE_SERVER_URL` for the listed Layer B wind
generator attempts. The existing Layer B weather stem selector continues to use
the normal `AI_SERVER_URL` path.
