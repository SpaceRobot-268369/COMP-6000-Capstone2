#!/usr/bin/env bash
set -euo pipefail

# Sync the Server B main checkout and materialise production model artifacts.
# This is intentionally sync-only: it never starts, stops, or restarts uvicorn.

deploy_dir="${SERVER_B_DEPLOY_DIR:-$HOME/shiny-pikachu}"
remote_name="${SERVER_B_REMOTE:-origin}"
main_branch="${SERVER_B_MAIN_BRANCH:-main}"
service_port="${SERVER_B_SERVICE_PORT:-8000}"

log() {
  printf '[server-b-sync] %s\n' "$*"
}

warn() {
  printf '[server-b-sync] WARNING: %s\n' "$*" >&2
}

fail() {
  printf '[server-b-sync] ERROR: %s\n' "$*" >&2
  exit 1
}

find_dvc() {
  if [ -n "${SERVER_B_DVC_BIN:-}" ]; then
    [ -x "$SERVER_B_DVC_BIN" ] || fail "SERVER_B_DVC_BIN is not executable: $SERVER_B_DVC_BIN"
    printf '%s\n' "$SERVER_B_DVC_BIN"
    return 0
  fi

  if command -v dvc >/dev/null 2>&1; then
    command -v dvc
    return 0
  fi

  if [ -x "$HOME/.local/bin/dvc" ]; then
    printf '%s\n' "$HOME/.local/bin/dvc"
    return 0
  fi

  fail "DVC executable not found; expected dvc on PATH or at $HOME/.local/bin/dvc"
}

ensure_git_checkout() {
  [ -d "$deploy_dir" ] || fail "Deploy checkout does not exist: $deploy_dir"
  [ -d "$deploy_dir/.git" ] || fail "Deploy checkout is not a git repository: $deploy_dir"

  cd "$deploy_dir"

  current_branch="$(git branch --show-current)"
  [ "$current_branch" = "$main_branch" ] || fail "$deploy_dir is on '$current_branch', expected '$main_branch'"

  if [ -n "$(git status --porcelain)" ]; then
    git status --short
    fail "$deploy_dir has local changes; refusing to sync live main checkout"
  fi
}

sync_git() {
  before_sha="$(git rev-parse --short HEAD)"
  log "Checkout before sync: $before_sha"

  git fetch "$remote_name" "$main_branch"
  git checkout "$main_branch"
  git pull --ff-only "$remote_name" "$main_branch"

  after_sha="$(git rev-parse --short HEAD)"
  log "Checkout after sync: $after_sha"
}

pull_production_models() {
  dvc_bin="$(find_dvc)"
  log "Using DVC: $("$dvc_bin" --version)"

  if [ ! -d model/production ]; then
    log "No model/production directory found; nothing to materialise"
    return 0
  fi

  mapfile -d '' dvc_pointers < <(find model/production -name '*.dvc' -type f -print0 | sort -z)
  if [ "${#dvc_pointers[@]}" -eq 0 ]; then
    log "No production DVC pointers found; nothing to materialise"
    return 0
  fi

  log "Pulling ${#dvc_pointers[@]} production DVC pointer(s)"
  "$dvc_bin" pull "${dvc_pointers[@]}"

  missing=()
  for pointer in "${dvc_pointers[@]}"; do
    materialised="${pointer%.dvc}"
    if [ ! -e "$materialised" ]; then
      missing+=("$materialised")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    printf '%s\n' "${missing[@]}" >&2
    fail "Some production model artifacts are still missing after dvc pull"
  fi

  log "Production model artifacts are materialised"
}

report_service_state() {
  log "Checking local AI service state on 127.0.0.1:${service_port}"

  pid=""
  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnp 2>/dev/null \
      | awk -v port=":${service_port}" '$0 ~ port { if (match($0, /pid=[0-9]+/)) { print substr($0, RSTART + 4, RLENGTH - 4); exit } }' \
      || true)"
  fi

  if [ -z "$pid" ] && command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -tiTCP:"$service_port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  fi

  if [ -z "$pid" ]; then
    log "No process is listening on 127.0.0.1:${service_port}; sync-only workflow will not start it"
    return 0
  fi

  cwd=""
  if [ -e "/proc/$pid/cwd" ]; then
    cwd="$(readlink "/proc/$pid/cwd" || true)"
  fi

  cmdline=""
  if [ -r "/proc/$pid/cmdline" ]; then
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" || true)"
  fi

  log "Port ${service_port} listener pid: $pid"
  [ -n "$cwd" ] && log "Listener cwd: $cwd"
  [ -n "$cmdline" ] && log "Listener cmd: $cmdline"

  case "$cwd $cmdline" in
    *"$deploy_dir"*)
      log "AI service appears to use $deploy_dir; no restart performed in sync-only mode"
      ;;
    *)
      warn "AI service does not appear to run from $deploy_dir; leaving it untouched"
      ;;
  esac

  if command -v curl >/dev/null 2>&1; then
    log "Health probe, best effort:"
    curl -fsS --max-time 5 "http://127.0.0.1:${service_port}/health" || warn "Health probe failed"
    printf '\n'
  fi
}

main() {
  log "Starting Server B model sync"
  log "Deploy checkout: $deploy_dir"
  ensure_git_checkout
  sync_git
  pull_production_models
  report_service_state
  log "Server B model sync complete"
}

main "$@"
