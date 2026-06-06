"""Build the S3a.4 final listen page from ranking.json (computed on Server B).

Machine ranking is advisory only — final pick is by human audition.
Score (lower = likely cleaner): hiss_ratio + spectral_flatness - 0.5 * lowmid_ratio
Standard library only (json + pathlib), so it runs with any python3.
"""
import json
from pathlib import Path

ATTEMPT = Path(__file__).resolve().parents[1]
SHOWCASE = ATTEMPT / "showcase_s3a4_final"
PROTECT = {45, 48, 50}  # regression-protection seeds carried from the earlier Variant A batch

rows = json.loads((SHOWCASE / "ranking.json").read_text(encoding="utf-8"))

cards = []
for rank, r in enumerate(rows, start=1):
    seed = r["seed"]
    tag = '<span class="prot">回归保护(已优)</span>' if seed in PROTECT else ""
    audio_abs = (SHOWCASE / f"seed_{seed}_generated" / "audio.wav").resolve()
    cards.append(f"""<div class="card">
  <div class="meta"><b>排名 #{rank:02d}</b> · <code>seed {seed}</code> {tag}</div>
  <div class="score">score={r['score']:.4f} · hiss={r['hiss_ratio']:.3f} · flat={r['spectral_flatness']:.4f} · lowmid={r['lowmid_ratio']:.3f}</div>
  <audio controls preload="none" src="file://{audio_abs}"></audio>
</div>""")

html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>S3a.4 Final — 40 seed 扫描(机器预排序)</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}}
 h1{{font-size:20px}} .note{{color:#9aa4b2;font-size:13px;margin-bottom:16px;line-height:1.5}}
 .card{{background:#171a21;border:1px solid #262b36;border-radius:10px;padding:12px 14px;margin:10px 0}}
 .meta{{font-size:14px;margin-bottom:4px}} code{{color:#8ad}}
 .score{{font-size:12px;color:#8b94a3;margin-bottom:8px;font-family:monospace}}
 .prot{{background:#2a3a22;color:#a9e07f;padding:2px 8px;border-radius:6px;font-size:12px;margin-left:6px}}
 audio{{width:100%}}
</style></head><body>
<h1>S3a.4 Final — 40 seed 扫描(Variant A: strength 0.15 / floor 0.40)</h1>
<p class="note">机器预排序仅作参考(按 <code>hiss_ratio + spectral_flatness − 0.5·lowmid_ratio</code> 升序,越靠前可能越干净),<b>最终以你的听感为准</b>。<br>
绿色标签为回归保护样本(seed 45/48/50,来自早期 Variant A 批次,参数完全一致)。请先确认它们没有退化。<br>
目标:从中挑出 4–6 条黄金 seed。</p>
{''.join(cards)}
</body></html>"""

(SHOWCASE / "listen_generated.html").write_text(html, encoding="utf-8")
print("wrote", SHOWCASE / "listen_generated.html")
print("rank order:", [r["seed"] for r in rows])
