#!/usr/bin/env python3
"""Build a same-seed listen HTML page for an intensity eval folder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTECT_SEEDS = {48}
LABELS = {
    "light": ("Light (v3 light_c)", "derived · hybrid A/B"),
    "medium": ("Medium (v3)", "stronger denoise"),
    "heavy": ("Heavy (v2 locked)", "regression guard"),
}


def cell(root: Path, intensity: str, seed: int) -> str:
    case = root / intensity / f"seed_{seed}_generated"
    meta = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    pp = meta.get("postprocess", {})
    den = pp.get("denoise") or {}
    tag = '<span class="tag">回归保护</span>' if seed in PROTECT_SEEDS else ""
    derived = '<span class="tag">derived</span>' if meta.get("derived") else ""
    title, hint = LABELS.get(intensity, (intensity, ""))
    sub = (
        f"{hint}<br>guidance={meta.get('guidance_scale')} · rms={pp.get('output_target_rms')} · "
        f"hp={pp.get('highpass_hz')} · lp={pp.get('lowpass_hz')} · "
        f"denoise={den.get('strength')}/{den.get('floor_ratio')}"
    )
    audio_abs = (case / "audio.wav").resolve()
    return f"""<td>
  <div class="meta"><b>{title}</b> {derived}{tag}</div>
  <div class="sub">{sub}</div>
  <audio controls preload="none" src="file://{audio_abs}"></audio>
</td>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Layer B wind intensity compare")
    args = parser.parse_args()
    root = args.eval_dir.resolve()
    out = args.out_html or (root / "listen_intensity_compare.html")
    seeds = list(range(42, 52))
    intensities = [p.name for p in sorted(root.iterdir()) if p.is_dir() and p.name in LABELS]
    if not intensities:
        intensities = ["light", "medium", "heavy"]

    rows = []
    for seed in seeds:
        cells = "".join(cell(root, i, seed) for i in intensities)
        rows.append(f"<tr><th>seed {seed}</th>{cells}</tr>")

    headers = "".join(f"<th>{LABELS.get(i, (i,))[0]}</th>" for i in intensities)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>{args.title}</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 20px; background:#0f1115; color:#e6e6e6; }}
 h2 {{ margin: 0 0 8px; }}
 p.note {{ color:#9aa4b2; margin:0 0 14px; line-height:1.5; }}
 table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
 th, td {{ border: 1px solid #2a2f3a; vertical-align: top; padding: 10px; }}
 th {{ background:#1b2029; text-align:left; }}
 td {{ background:#141922; font-size: 13px; }}
 .meta {{ margin-bottom: 4px; }}
 .sub {{ font-size:11px; color:#9aa4b2; margin-bottom:8px; line-height:1.35; }}
 .tag {{ font-size:11px; color:#a9e07f; border:1px solid #3d5731; border-radius:10px; padding:1px 6px; margin-left:4px; }}
 audio {{ width:100%; }}
</style></head><body>
<h2>{args.title}</h2>
<p class="note">同 seed 横向对比。重点：light 42–44、medium 噪感、seed 48 三档回归。</p>
<table>
  <thead><tr><th>Seed</th>{headers}</tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
