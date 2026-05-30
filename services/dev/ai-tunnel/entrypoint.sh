#!/bin/sh
set -eu

AI_SERVER_LABEL="${AI_SERVER_LABEL:-shinypokemon}"
AI_SSH_USER="${AI_SSH_USER:-ubuntu}"
AI_SSH_HOST="${AI_SSH_HOST:-shinypokemon.adelaideuni.cloud}"
AI_SSH_KEY_PATH="${AI_SSH_KEY_PATH:-/run/secrets/shinypokemon.pem}"
AI_TUNNEL_BIND_HOST="${AI_TUNNEL_BIND_HOST:-0.0.0.0}"
AI_TUNNEL_LOCAL_PORT="${AI_TUNNEL_LOCAL_PORT:-8000}"
AI_TUNNEL_REMOTE_HOST="${AI_TUNNEL_REMOTE_HOST:-127.0.0.1}"
AI_TUNNEL_REMOTE_PORT="${AI_TUNNEL_REMOTE_PORT:-8000}"
runtime_key="/tmp/shinypokemon.pem"

fail() {
  code="$1"
  message="$2"
  detail="${3:-}"
  printf 'ERROR [%s] %s\n' "$code" "$message" >&2
  if [ -n "$detail" ]; then
    printf '%s\n' "$detail" >&2
  fi
  exit 1
}

[ -f "$AI_SSH_KEY_PATH" ] || fail "pem-missing" "pem file does not exist: $AI_SSH_KEY_PATH"

mkdir -p /root/.ssh
chmod 700 /root/.ssh
cp "$AI_SSH_KEY_PATH" "$runtime_key" || fail "pem-copy" "Failed to copy pem into the container runtime path: $runtime_key"
chmod 600 "$runtime_key" || fail "pem-permission" "Failed to set pem permissions to 600: $runtime_key"
ssh-keygen -y -f "$runtime_key" >/dev/null 2>&1 || fail "pem-invalid" "pem file is invalid, or the container cannot read it: $AI_SSH_KEY_PATH"

ssh_options="
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o StrictHostKeyChecking=accept-new
  -i $runtime_key
"

printf 'Checking SSH login: %s@%s\n' "$AI_SSH_USER" "$AI_SSH_HOST"
ssh_output=$(ssh $ssh_options "$AI_SSH_USER@$AI_SSH_HOST" "true" 2>&1) || {
  case "$ssh_output" in
    *"Permission denied"*|*"publickey"*)
      fail "ssh-auth" "pem is incorrect, or the SSH username is not $AI_SSH_USER." "$ssh_output"
      ;;
    *"Could not resolve hostname"*|*"Name or service not known"*)
      fail "ssh-dns" "Could not resolve serverB hostname: $AI_SSH_HOST" "$ssh_output"
      ;;
    *"Connection timed out"*|*"Operation timed out"*|*"No route to host"*)
      fail "ssh-network" "Could not connect to serverB: $AI_SERVER_LABEL may be stopped, or the network/security group may block SSH." "$ssh_output"
      ;;
    *"Connection refused"*)
      fail "ssh-refused" "serverB refused the SSH connection: sshd may be stopped, or port 22 may be unreachable." "$ssh_output"
      ;;
    *)
      fail "ssh-login" "SSH login check failed." "$ssh_output"
      ;;
  esac
}

printf 'Checking remote AI service: %s:%s\n' "$AI_TUNNEL_REMOTE_HOST" "$AI_TUNNEL_REMOTE_PORT"
remote_health_cmd="if command -v curl >/dev/null 2>&1; then curl -fsS http://$AI_TUNNEL_REMOTE_HOST:$AI_TUNNEL_REMOTE_PORT/health >/dev/null; else wget -qO- http://$AI_TUNNEL_REMOTE_HOST:$AI_TUNNEL_REMOTE_PORT/health >/dev/null; fi"
health_output=$(ssh $ssh_options "$AI_SSH_USER@$AI_SSH_HOST" "$remote_health_cmd" 2>&1) || {
  case "$health_output" in
    *"Connection refused"*|*"Failed to connect"*)
      fail "ai-service-not-running" "$AI_SERVER_LABEL is reachable over SSH, but the AI service is not listening on $AI_TUNNEL_REMOTE_HOST:$AI_TUNNEL_REMOTE_PORT." "$health_output"
      ;;
    *"Connection timed out"*|*"Operation timed out"*)
      fail "ai-service-timeout" "$AI_SERVER_LABEL is reachable over SSH, but the AI service health check timed out." "$health_output"
      ;;
    *)
      fail "ai-service-health" "$AI_SERVER_LABEL is reachable over SSH, but the AI service /health check failed." "$health_output"
      ;;
  esac
}

printf 'Starting tunnel: %s:%s -> %s:%s via %s@%s\n' \
  "$AI_TUNNEL_BIND_HOST" "$AI_TUNNEL_LOCAL_PORT" "$AI_TUNNEL_REMOTE_HOST" "$AI_TUNNEL_REMOTE_PORT" "$AI_SSH_USER" "$AI_SSH_HOST"

exec ssh $ssh_options \
  -N \
  -L "$AI_TUNNEL_BIND_HOST:$AI_TUNNEL_LOCAL_PORT:$AI_TUNNEL_REMOTE_HOST:$AI_TUNNEL_REMOTE_PORT" \
  "$AI_SSH_USER@$AI_SSH_HOST"
