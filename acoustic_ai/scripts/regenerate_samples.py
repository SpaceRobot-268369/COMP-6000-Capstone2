#!/usr/bin/env python3
"""Regenerate a `showcase/` sample (model output) for one attempt.

Policy: .claude/context/dev/artifact_policy.md
Naming: .claude/context/dev/attempt_naming.md

This script writes ONLY to `<attempt>/showcase/`. The `expected/` tier is
real-audio ground truth — populated by extract_expected_samples.py, never
by this script.

For the selected attempt this script:
  1. Calls handler.load() + handler.generate(seed=<your --seed>).
  2. Writes the triplet:
        <attempt>/showcase/seed_<N>_<label>.png
        <attempt>/showcase/seed_<N>_<label>.metadata.json
        <attempt>/showcase/seed_<N>_<label>.wav

  All three are then `dvc add`'d (showcase is fully DVC-tracked).

Usage:
    acoustic_ai/.venv/bin/python acoustic_ai/scripts/regenerate_samples.py \\
        layer_a lucas__smoke_1__audioldm2_spring_night --seed 7 --label low_noise
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


def regenerate(
    layer_id: str,
    attempt_id: str,
    *,
    tier: str = "showcase",
    seed: int | None = None,
    label: str | None = None,
) -> dict:
    """Regenerate a generated sample triplet into `<attempt>/showcase/`.

    Only the `showcase` tier is writable here. `expected/` is reserved for
    real-audio ground truth (see acoustic_ai/scripts/extract_expected_samples.py
    and the artifact policy doc).

    Required args:
      seed  — explicit non-canonical seed for the variation.
      label — short snake_case label (e.g. 'low_noise', 'variation_a').
    Filename stem becomes `seed_<seed>_<label>`. All three artifacts go to DVC.
    """
    if tier != "showcase":
        raise ValueError(
            f"regenerate() only writes to the `showcase` tier; got tier={tier!r}. "
            "`expected/` is real-audio ground truth — see "
            "acoustic_ai/scripts/extract_expected_samples.py."
        )

    spec = registry.get_attempt(layer_id, attempt_id)

    if spec.status in SKIP_STATUSES:
        return {"layer": layer_id, "attempt": attempt_id, "skipped": spec.status}

    # Pick the seed:
    #   expected → canonical (registry override > project default 42)
    #   showcase → explicit --seed required
    if seed is None:
        raise ValueError("regenerate() requires --seed (showcase always uses an explicit seed)")
    if not label:
        raise ValueError("regenerate() requires --label (short snake_case)")
    run_seed = int(seed)
    stem = f"seed_{run_seed}_{label}"

    attempt_root = _AI_ROOT / "layers" / layer_id / "attempts" / attempt_id
    out_dir = attempt_root / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {layer_id}/{attempt_id} [{tier}] — seed {run_seed}")
    result = registry.generate(layer_id, attempt_id, seed=run_seed)

    wav_bytes = result.get("wav_bytes", b"")
    mel_db    = result.get("mel_db")
    metadata  = dict(result.get("metadata", {}))

    duration_s = float(metadata.get("audio", {}).get("duration_s", 0.0))

    # Enrich metadata with traceability info.
    handler_path = attempt_root / "code" / "handler.py"
    metadata["seed"] = run_seed
    metadata["tier"] = "showcase"
    metadata["showcase_label"] = label
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
    print(f"       NEXT: dvc add {wav_path} {png_path} {json_path}  &&  git add {wav_path}.dvc {png_path}.dvc {json_path}.dvc")
    return {
        "layer": layer_id, "attempt": attempt_id, "tier": "showcase", "seed": run_seed,
        "wav": str(wav_path), "png": str(png_path), "json": str(json_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer", help="Layer id (e.g. layer_a)")
    ap.add_argument("attempt", help="Attempt id (e.g. lucas__smoke_1__...)")
    ap.add_argument("--seed", type=int, required=True,
                    help="Seed for this showcase variation.")
    ap.add_argument("--label", required=True,
                    help="Short snake_case label for showcase stem (e.g. 'low_noise').")
    args = ap.parse_args()

    failures: list[tuple[str, str, str]] = []
    try:
        regenerate(args.layer, args.attempt, tier="showcase", seed=args.seed, label=args.label)
    except NotImplementedError as e:
        print(f"[SKIP] {args.layer}/{args.attempt}  (handler not implemented: {e})")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {args.layer}/{args.attempt}  ({type(e).__name__}: {e})")
        failures.append((args.layer, args.attempt, str(e)))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for l, a, msg in failures:
            print(f"  {l}/{a}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
