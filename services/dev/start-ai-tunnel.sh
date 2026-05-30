#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
env_file="${ENV_FILE:-$script_dir/.env}"

if [ -f "$env_file" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
fi

AI_SSH_USER="${AI_SSH_USER:-ubuntu}"
AI_SSH_HOST="${AI_SSH_HOST:-shinypokemon.adelaideuni.cloud}"
AI_SSH_KEY_PATH="${AI_SSH_KEY_PATH:-$HOME/.ssh/itds-eap/shinypokemon.pem}"
AI_TUNNEL_BIND_HOST="${AI_TUNNEL_BIND_HOST:-127.0.0.1}"
AI_TUNNEL_LOCAL_PORT="${AI_TUNNEL_LOCAL_PORT:-8000}"
AI_TUNNEL_REMOTE_HOST="${AI_TUNNEL_REMOTE_HOST:-127.0.0.1}"
AI_TUNNEL_REMOTE_PORT="${AI_TUNNEL_REMOTE_PORT:-8000}"

case "$AI_SSH_KEY_PATH" in
  "~/"*) AI_SSH_KEY_PATH="$HOME/${AI_SSH_KEY_PATH#~/}" ;;
esac

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

command -v ssh >/dev/null 2>&1 || fail "ssh-missing" "ssh command not found on this host."
command -v ssh-keygen >/dev/null 2>&1 || fail "ssh-keygen-missing" "ssh-keygen command not found on this host."

[ -f "$AI_SSH_KEY_PATH" ] || fail "pem-missing" "pem file does not exist: $AI_SSH_KEY_PATH"
chmod 600 "$AI_SSH_KEY_PATH" 2>/dev/null || fail "pem-permission" "Failed to set pem permissions to 600: $AI_SSH_KEY_PATH"
ssh-keygen -y -f "$AI_SSH_KEY_PATH" >/dev/null 2>&1 || fail "pem-invalid" "pem file is invalid, or the current user cannot read it: $AI_SSH_KEY_PATH"

ssh_options="
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
  -o StrictHostKeyChecking=accept-new
  -i $AI_SSH_KEY_PATH
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
      fail "ssh-network" "Could not connect to serverB: shinypokemon may be stopped, or the network/security group may block SSH." "$ssh_output"
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

printf 'Starting tunnel: localhost:%s -> %s:%s via %s@%s\n' \
  "$AI_TUNNEL_LOCAL_PORT" "$AI_TUNNEL_REMOTE_HOST" "$AI_TUNNEL_REMOTE_PORT" "$AI_SSH_USER" "$AI_SSH_HOST"
printf 'Leave this process running while Docker Compose is using the AI service.\n'

exec ssh $ssh_options \
  -N \
  -L "$AI_TUNNEL_BIND_HOST:$AI_TUNNEL_LOCAL_PORT:$AI_TUNNEL_REMOTE_HOST:$AI_TUNNEL_REMOTE_PORT" \
  "$AI_SSH_USER@$AI_SSH_HOST"
