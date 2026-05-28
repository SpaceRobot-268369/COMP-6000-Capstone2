#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${WORKER_ENV_FILE:-"$ROOT_DIR/worker/.env"}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Worker env file not found: $ENV_FILE" >&2
  echo "Create it from worker/.env.example and set WORKER_API_TOKEN." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

required_vars=(
  SERVER_A_URL
  WORKER_API_TOKEN
  WORKER_ID
  WORKER_JOB_TYPES
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" || "${!name}" == "change-me" ]]; then
    echo "Required worker environment variable is missing or unchanged: $name" >&2
    exit 1
  fi
done

cd "$ROOT_DIR"
exec python worker/worker.py
