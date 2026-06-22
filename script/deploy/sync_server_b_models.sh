#!/usr/bin/env bash
set -euo pipefail

# Full Server B deploy: sync the main checkout, install Python deps, materialise
# the served model/sample DVC artifacts (production AND served candidates), then
# restart uvicorn so the new code and models take effect with no manual SSH.
#
# NOTE: the restart unconditionally stops and relaunches the AI service. Any
# generation/training job in flight at deploy time is interrupted. This is the
# accepted trade-off for hands-off deploys (deploys fire on merge to main, and
# serverB is an on-demand worker that is frequently off). Set SERVER_B_RESTART=0
# to fall back to sync-only behaviour (no dep install gate on restart, no
# relaunch).

deploy_dir="${SERVER_B_DEPLOY_DIR:-$HOME/shiny-pikachu}"
remote_name="${SERVER_B_REMOTE:-origin}"
main_branch="${SERVER_B_MAIN_BRANCH:-main}"
service_port="${SERVER_B_SERVICE_PORT:-8000}"
service_host="${SERVER_B_SERVICE_HOST:-127.0.0.1}"
service_app="${SERVER_B_SERVICE_APP:-acoustic_ai.server.server:app}"
service_log="${SERVER_B_SERVICE_LOG:-/tmp/shiny-pikachu-ai.log}"
service_pidfile="${SERVER_B_SERVICE_PIDFILE:-/tmp/shiny-pikachu-ai.pid}"
venv_python="${SERVER_B_VENV_PYTHON:-$deploy_dir/acoustic_ai/.venv/bin/python}"
restart_enabled="${SERVER_B_RESTART:-1}"
service_unit="${SERVER_B_SERVICE_UNIT:-shiny-pikachu-ai.service}"

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

move_known_stale_untracked_paths() {
  # Repo renames can leave DVC-materialised, now-untracked attempt directories
  # in the long-lived deploy checkout. Move only explicitly-known stale paths
  # out of the checkout; any other dirty state still fails the deploy.
  local stale_paths=(
    "acoustic_ai/layers/layer_b/attempts/lucas__smoke_1__curated_assets"
  )
  local backup_root=""
  local rel_path=""
  local status_for_path=""
  local moved=0

  for rel_path in "${stale_paths[@]}"; do
    [ -e "$rel_path" ] || continue

    status_for_path="$(git status --porcelain -- "$rel_path" || true)"
    [ -n "$status_for_path" ] || continue

    if printf '%s\n' "$status_for_path" | awk '$1 != "??" { exit 1 }'; then
      if [ -z "$backup_root" ]; then
        backup_root="/tmp/server-b-sync-stale-$(date +%Y%m%dT%H%M%S)-$$"
      fi
      mkdir -p "$backup_root/$(dirname "$rel_path")"
      mv "$rel_path" "$backup_root/$rel_path"
      moved=1
      warn "Moved stale untracked deploy path to $backup_root/$rel_path"
    fi
  done

  if [ "$moved" -eq 1 ]; then
    log "Known stale untracked deploy paths moved out of checkout"
  fi
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

  move_known_stale_untracked_paths

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

install_python_deps() {
  req="acoustic_ai/requirements.txt"
  if [ ! -f "$req" ]; then
    log "No $req found; skipping Python dependency install"
    return 0
  fi
  [ -x "$venv_python" ] || fail "venv python not found: $venv_python (expected the deployed acoustic_ai/.venv)"
  log "Installing Python deps from $req into the deployed venv"
  "$venv_python" -m pip install -r "$req"
  log "Python dependencies are in sync with $req"
}

pull_dvc_artifacts() {
  dvc_bin="$(find_dvc)"
  log "Using DVC: $("$dvc_bin" --version)"

  # Materialise everything the running server actually loads: promoted models
  # (model/production), the served candidate checkpoints (model/candidates —
  # every Layer E head and the Layer A/C candidate banks live here, so omitting
  # them is how analysis breaks with "best_probe.pt: No such file"), and the
  # expected/showcase sample tiers the dev UI serves, and registry-declared
  # served media-bank artifacts. Datasets under resources/ are deliberately NOT
  # pulled — only runtime model weights, samples, and served retrieval media.
  mapfile -d '' dvc_pointers < <(
    {
      if [ -d model/production ]; then
        find model/production -name '*.dvc' -type f -print0
      fi

      if [ -d model/candidates ]; then
        find model/candidates -name '*.dvc' -type f -print0
      fi

      if [ -d acoustic_ai/layers ]; then
        find acoustic_ai/layers -name '*.dvc' -type f \
          \( -path '*/expected/*' -o -path '*/showcase/*' \) \
          -print0
      fi

      if [ -f acoustic_ai/registry.yaml ]; then
        "$venv_python" - <<'PY'
from pathlib import Path
import sys

import yaml

registry = yaml.safe_load(Path("acoustic_ai/registry.yaml").read_text(encoding="utf-8")) or {}
for layer in (registry.get("layers") or {}).values():
    for attempt in (layer.get("attempts") or {}).values():
        for pointer in attempt.get("deploy_dvc") or []:
            if isinstance(pointer, str) and pointer:
                sys.stdout.write(pointer + "\0")
PY
      fi
    } | sort -z
  )

  if [ "${#dvc_pointers[@]}" -eq 0 ]; then
    log "No model, sample, or served media DVC pointers found; nothing to materialise"
    return 0
  fi

  log "Pulling ${#dvc_pointers[@]} model/sample/media DVC pointer(s)"
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
    fail "Some model/sample/media artifacts are still missing after dvc pull"
  fi

  log "Model/sample/media artifacts are materialised"
}

warn_large_media_banks() {
  # Deploy policy is "pull ALL candidate banks" (simple and deliberate — see
  # cicd_design.md). But retrieval media-asset banks are audio and can grow
  # large, and every candidate bank is pulled on every deploy. Warn (never
  # fail) when a single bank crosses the threshold so a dev can decide whether
  # to prune superseded banks or revisit the pull policy. Per-bank threshold;
  # override with SERVER_B_BANK_WARN_BYTES (default 2 GiB).
  local threshold="${SERVER_B_BANK_WARN_BYTES:-2147483648}"
  local bank=""
  local bytes=0
  local human_bank=""
  local human_thr=""

  [ -d model ] || return 0

  while IFS= read -r -d '' bank; do
    bytes="$(du -sb "$bank" 2>/dev/null | cut -f1 || echo 0)"
    [ "${bytes:-0}" -gt "$threshold" ] || continue
    human_bank="$(numfmt --to=iec "$bytes" 2>/dev/null || printf '%sB' "$bytes")"
    human_thr="$(numfmt --to=iec "$threshold" 2>/dev/null || printf '%sB' "$threshold")"
    warn "Large media asset bank: $bank is $human_bank (threshold $human_thr). Deploy still pulls ALL candidate banks; consider pruning superseded banks, or scoping pulls to registry-served attempts if serverB disk/deploy-time starts to hurt."
  done < <(find model -type d -name 'media_asset_bank' -print0 2>/dev/null)
}

find_listener_pid() {
  local pid=""
  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnp 2>/dev/null \
      | awk -v port=":${service_port}" '$0 ~ port { if (match($0, /pid=[0-9]+/)) { print substr($0, RSTART + 4, RLENGTH - 4); exit } }' \
      || true)"
  fi
  if [ -z "$pid" ] && command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -tiTCP:"$service_port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  fi
  printf '%s' "$pid"
}

systemd_unit_installed() {
  command -v systemctl >/dev/null 2>&1 || return 1
  systemctl list-unit-files "$service_unit" >/dev/null 2>&1
}

restart_service_systemd() {
  log "Restarting AI service via systemd unit '$service_unit' (interrupts any in-flight job)"
  sudo systemctl restart "$service_unit" || fail "systemctl restart $service_unit failed"

  sleep 3
  if systemctl is-active --quiet "$service_unit"; then
    log "Unit $service_unit is active; pre-warm runs in the background (journalctl -u $service_unit -f)"
  else
    warn "Unit $service_unit is not active after restart; check 'systemctl status $service_unit'"
  fi

  if command -v curl >/dev/null 2>&1; then
    log "Health probe (best effort):"
    curl -fsS --max-time 10 "http://${service_host}:${service_port}/health" \
      || warn "Health probe failed (service may still be starting / warming models)"
    printf '\n'
  fi
}

restart_service() {
  if [ "$restart_enabled" = "0" ]; then
    log "SERVER_B_RESTART=0 — skipping restart (sync-only). New code/models load on the next manual start."
    return 0
  fi

  # Prefer the systemd unit when it is installed: it is the boot-time owner of
  # the service, so a nohup relaunch here would race it for port
  # ${service_port}. Fall back to the legacy nohup path only when no unit
  # exists (e.g. an older serverB that has not been migrated yet).
  if systemd_unit_installed; then
    restart_service_systemd
    return 0
  fi

  [ -x "$venv_python" ] || fail "venv python not found: $venv_python"
  log "No systemd unit '$service_unit' found; using legacy nohup restart on ${service_host}:${service_port}"

  # Stop the current listener (pidfile first, then a port lookup as fallback).
  old_pid=""
  if [ -f "$service_pidfile" ]; then
    old_pid="$(cat "$service_pidfile" 2>/dev/null || true)"
  fi
  if [ -z "$old_pid" ] || ! kill -0 "$old_pid" 2>/dev/null; then
    old_pid="$(find_listener_pid)"
  fi

  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    log "Stopping running service (pid $old_pid) — interrupts any in-flight job"
    kill "$old_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$old_pid" 2>/dev/null || true
  else
    log "No running service found; starting fresh"
  fi

  # Relaunch detached in a new session so it survives this SSH connection
  # closing (setsid + nohup + fds off the SSH channel). The server pre-warms
  # its default models in a background thread, so /health answers immediately.
  cd "$deploy_dir"
  setsid bash -c \
    "nohup '$venv_python' -m uvicorn '$service_app' --host '$service_host' --port '$service_port' >> '$service_log' 2>&1 & echo \$! > '$service_pidfile'" \
    < /dev/null > /dev/null 2>&1 || true

  sleep 3
  new_pid="$(cat "$service_pidfile" 2>/dev/null || true)"
  if [ -n "$new_pid" ] && kill -0 "$new_pid" 2>/dev/null; then
    log "Service relaunched (pid $new_pid); pre-warm runs in the background (tail $service_log)"
  else
    warn "Could not confirm the new service pid; check $service_log on serverB"
  fi

  if command -v curl >/dev/null 2>&1; then
    log "Health probe (best effort):"
    curl -fsS --max-time 10 "http://${service_host}:${service_port}/health" \
      || warn "Health probe failed (service may still be starting / warming models)"
    printf '\n'
  fi
}

main() {
  log "Starting Server B deploy"
  log "Deploy checkout: $deploy_dir"
  ensure_git_checkout
  sync_git
  install_python_deps
  pull_dvc_artifacts
  warn_large_media_banks
  restart_service
  log "Server B deploy complete"
}

main "$@"
