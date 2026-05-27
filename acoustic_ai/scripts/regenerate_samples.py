#!/usr/bin/env python3
"""Regenerate canonical reference samples for one or more attempts.

Policy: .claude/context/dev/artifact_policy.md
Naming: .claude/context/dev/attempt_naming.md

For each selected attempt this script:
  1. Reads canonical_seed from registry.yaml (defaults to 42).
  2. Calls the attempt's handler.load() + handler.generate(seed=...).
  3. Writes the triplet:
        <attempt>/samples/reference/seed_<N>.png            (git-tracked)
        <attempt>/samples/reference/seed_<N>.metadata.json  (git-tracked)
        <attempt>/samples/reference/seed_<N>.wav            (then `dvc add`)

The user (not this script) runs `dvc add` + `git add` afterwards — the script
deliberately stays out of git/DVC plumbing so a failed regeneration doesn't
leave the tree in a weird half-staged state.

Usage:
    # one attempt:
    acoustic_ai/.venv/bin/python acoustic_ai/scripts/regenerate_samples.py \\
        layer_a lucas__smoke_1__audioldm2_spring_night

    # every active attempt (skips placeholders/superseded):
    acoustic_ai/.venv/bin/python acoustic_ai/scripts/regenerate_samples.py --all-active

    # everything in the registry, including placeholders (will error out):
    acoustic_ai/.venv/bin/python acoustic_ai/scripts/regenerate_samples.py --all
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Make `server.*` and `layers.*` importable.
_AI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_AI_ROOT))

from server import registry  # noqa: E402


SKIP_STATUSES = {"placeholder", "partial", "superseded"}


# ---------------------------------------------------------------------------


def _git_short_sha(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "log", "-n", "1", "--pretty=format:%h", "--", str(path)],
            cwd=path.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or "<unknown>"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "<unknown>"


def _checkpoint_dvc_hash(checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    # Look for any `*.dvc` file inside the checkpoint dir.
    if not checkpoint_path.exists():
        return None
    for dvc_file in checkpoint_path.glob("*.dvc"):
        try:
            text = dvc_file.read_text(encoding="utf-8")
        except OSError:
            continue
        # naive parse: find the first `md5:` line
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
    return None


def _render_png_bytes(layer_id: str, attempt_id: str, mel_db, duration_s: float) -> bytes | None:
    """Use the attempt-local layer_a_visualization helper if present;
    otherwise fall back to a generic matplotlib render."""
    if mel_db is None:
        return None
    import importlib
    try:
        viz = importlib.import_module(
            f"layers.{layer_id}.attempts.{attempt_id}.layer_a_visualization"
        )
        return viz.render_layer_a_mel_png_bytes(mel_db, duration_s)
    except ModuleNotFoundError:
        pass

    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")
        ax.set_title(f"{layer_id} / {attempt_id} / seed_42")
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except ImportError:
        return None


# ---------------------------------------------------------------------------


def regenerate(layer_id: str, attempt_id: str) -> dict:
    spec = registry.get_attempt(layer_id, attempt_id)

    if spec.status in SKIP_STATUSES:
        return {"layer": layer_id, "attempt": attempt_id, "skipped": spec.status}

    # Canonical seed: registry override, else 42.
    canonical_seed = int(
        spec.params.get("canonical_seed")
        or spec.params.get("samples", {}).get("canonical_seed", 42)
    )

    out_dir = (
        _AI_ROOT / "layers" / layer_id / "attempts" / attempt_id / "samples" / "reference"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{canonical_seed}"

    print(f"[INFO] {layer_id}/{attempt_id} — seed {canonical_seed}")
    result = registry.generate(layer_id, attempt_id, seed=canonical_seed)

    wav_bytes = result.get("wav_bytes", b"")
    mel_db    = result.get("mel_db")
    metadata  = dict(result.get("metadata", {}))

    sr = int(metadata.get("audio", {}).get("sample_rate", 0))
    duration_s = float(metadata.get("audio", {}).get("duration_s", 0.0))

    # Enrich metadata with traceability info.
    handler_path = (
        _AI_ROOT / "layers" / layer_id / "attempts" / attempt_id / "handler.py"
    )
    metadata["seed"] = canonical_seed
    metadata["checkpoint"] = str(spec.checkpoint) if spec.checkpoint else None
    metadata["checkpoint_dvc_hash"] = _checkpoint_dvc_hash(spec.checkpoint)
    metadata["handler_git_sha"] = _git_short_sha(handler_path)
    metadata["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()

    # Write triplet.
    wav_path = out_dir / f"{stem}.wav"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.metadata.json"

    if wav_bytes:
        wav_path.write_bytes(wav_bytes)
    png_bytes = _render_png_bytes(layer_id, attempt_id, mel_db, duration_s)
    if png_bytes:
        png_path.write_bytes(png_bytes)
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[OK]   wrote {wav_path.name}, {png_path.name}, {json_path.name}")
    print(f"       NEXT: dvc add {wav_path}  &&  git add {png_path} {json_path} {wav_path}.dvc")
    return {
        "layer": layer_id, "attempt": attempt_id, "seed": canonical_seed,
        "wav": str(wav_path), "png": str(png_path), "json": str(json_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer", nargs="?", help="Layer id (e.g. layer_a)")
    ap.add_argument("attempt", nargs="?", help="Attempt id (e.g. lucas__smoke_1__...)")
    ap.add_argument("--all", action="store_true", help="Regenerate every registered attempt.")
    ap.add_argument("--all-active", action="store_true",
                    help="Regenerate every attempt except placeholder/partial/superseded.")
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []
    if args.all or args.all_active:
        for layer in registry.list_layers():
            for att in layer["attempts"]:
                if args.all_active and att["status"] in SKIP_STATUSES:
                    print(f"[SKIP] {layer['id']}/{att['id']}  (status={att['status']})")
                    continue
                targets.append((layer["id"], att["id"]))
    elif args.layer and args.attempt:
        targets.append((args.layer, args.attempt))
    else:
        ap.print_help()
        return 2

    failures: list[tuple[str, str, str]] = []
    for layer_id, attempt_id in targets:
        try:
            regenerate(layer_id, attempt_id)
        except NotImplementedError as e:
            print(f"[SKIP] {layer_id}/{attempt_id}  (handler not implemented: {e})")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {layer_id}/{attempt_id}  ({type(e).__name__}: {e})")
            failures.append((layer_id, attempt_id, str(e)))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for l, a, msg in failures:
            print(f"  {l}/{a}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
