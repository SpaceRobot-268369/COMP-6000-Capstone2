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
