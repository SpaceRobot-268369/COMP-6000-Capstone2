#!/usr/bin/env python
"""Bake the ai-mock fixture set. OFFLINE, ONE-SHOT — never runs in Docker.

The demo `ai-mock` service does no audio processing at request time: it looks a
request up and replays a file. This script produces those files, once, on a
machine that has the real (DVC-pulled) artefacts:

    ./acoustic_ai/.venv/bin/python services/dev/ai-mock/tools/build_fixtures.py

Outputs, all committed to the `demo` branch as plain git files:

  fixtures/layers/...          sample tiers the Express backend serves directly
                               (AI_LAYERS_ROOT points here) — layer A per-cell
                               ambient bank, layer B weather stems, layer C
                               species references
  fixtures/events/<slug>.wav   one reference call per Layer C species
  fixtures/generation/<preset>/{audio.wav,spectrogram.png,metadata.json}
                               pre-baked Layer D mixes, one per demo prompt
  fixtures/analysis/<cell>.json  per-cell canned analysis reports
  fixtures/analysis/presets.json MD5(ambient wav) -> cell, so an uploaded preset
                               recording is recognised by content

Re-run only to change the preset set. Requires numpy + soundfile + matplotlib
(all present in acoustic_ai/.venv).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

HERE = Path(__file__).resolve().parent
MOCK_ROOT = HERE.parent
REPO = MOCK_ROOT.parents[2]
FIXTURES = MOCK_ROOT / "fixtures"

LAYERS = REPO / "acoustic_ai" / "layers"
BANK = REPO / "model" / "candidates" / "burger" / "mvp_2__retrieval_v2_library"

SR = 22050
MIX_SECONDS = 30.0
CANONICAL_SEED = 42

SEASONS = ("spring", "summer", "autumn", "winter")
DIELS = ("dawn", "morning", "afternoon", "night")

# --- what gets copied verbatim -------------------------------------------------
# (layer, attempt, tier) -> None copies every case, or a set of case names.
SAMPLE_SETS = [
    ("layer_a", "lucas__prod_1__per_cell_loras", "expected", "one-per-cell"),
    ("layer_b", "murphy__mvp_1__weather_stem_selector", "expected", None),
    ("layer_b", "murphy__mvp_1__rain_intensity_seed_pool", "expected", None),
    ("layer_c", "burger__mvp_2__retrieval_v2_library", "expected", None),
    ("layer_c", "burger__mvp_2__retrieval_v2_library", "showcase", None),
    ("layer_c", "lucas__smoke_1__audiogen_boobook", "expected", None),
]

# --- the demo prompts ----------------------------------------------------------
# Each entry bakes one Layer D mix. `match` is what the mock scores an incoming
# /generation/render request against. Keep these aligned with the preset chips
# in frontend/src/components/PromptChat.jsx.
PRESETS = [
    {
        "id": "summer_night_nightjar",
        "prompt": "Summer night, a lone Spotted Nightjar churring every so often",
        "cell": "summer_night",
        "weather": None,
        "species": [("spotted_nightjar", "Spotted Nightjar")],
        "event_count": 4,
    },
    {
        "id": "summer_morning_bronze_cuckoo",
        "prompt": "Summer morning, restless Horsfield's Bronze-cuckoo whistling over and over",
        "cell": "summer_morning",
        "weather": None,
        "species": [("horsfields_bronze_cuckoo", "Horsfield's Bronze-cuckoo")],
        "event_count": 9,
    },
    {
        "id": "spring_afternoon_wind_birdsong",
        "prompt": "Windy spring afternoon, distant birdsong",
        "cell": "spring_afternoon",
        "weather": ("wind", "medium"),
        "species": [
            ("yellow_throated_miner", "Yellow-throated Miner"),
            ("willie_wagtail", "Willie Wagtail"),
        ],
        "event_count": 5,
    },
    {
        "id": "winter_dawn_rain_light",
        "prompt": "Cold winter dawn, light drizzle",
        "cell": "winter_dawn",
        "weather": ("rain", "light"),
        "species": [],
        "event_count": 0,
    },
    {
        "id": "autumn_morning_rain_wind_heavy",
        "prompt": "Gusty autumn morning, rain on the wind",
        "cell": "autumn_morning",
        "weather": ("rain+wind", "heavy"),
        "species": [],
        "event_count": 0,
    },
    {
        "id": "spring_night_wind_light",
        "prompt": "Warm spring dusk, insects and a breeze",
        "cell": "spring_night",
        "weather": ("wind", "light"),
        "species": [],
        "event_count": 0,
    },
]

# (weather_type, intensity) -> layer B expected case used as the stem.
WEATHER_STEMS = {
    ("wind", "light"): ("murphy__mvp_1__weather_stem_selector", "wind_medium_site_example"),
    ("wind", "medium"): ("murphy__mvp_1__weather_stem_selector", "wind_medium_site_example"),
    ("wind", "heavy"): ("murphy__mvp_1__weather_stem_selector", "wind_medium_site_example"),
    ("rain", "light"): ("murphy__mvp_1__rain_intensity_seed_pool", "site_pool_rain_light"),
    ("rain", "medium"): ("murphy__mvp_1__weather_stem_selector", "rain_medium_site_example"),
    ("rain", "heavy"): ("murphy__mvp_1__rain_intensity_seed_pool", "site_pool_rain_heavy"),
    ("rain+wind", "light"): ("murphy__mvp_1__weather_stem_selector", "rain_wind_medium_site_example"),
    ("rain+wind", "medium"): ("murphy__mvp_1__weather_stem_selector", "rain_wind_medium_site_example"),
    ("rain+wind", "heavy"): ("murphy__mvp_1__weather_stem_selector", "rain_wind_medium_site_example"),
}

WEATHER_GAIN = {"light": 0.18, "medium": 0.34, "heavy": 0.55}

ATTEMPTS = {
    "layer_a": "lucas__prod_1__per_cell_loras",
    "layer_b": "murphy__mvp_1__weather_stem_selector",
    "layer_c": "burger__mvp_2__retrieval_v2_library",
    "layer_d": "songke__mvp_2__multi_clip_mix",
}


# ---------------------------------------------------------------- audio helpers
def load_mono(path: Path) -> np.ndarray:
    """Read a WAV as mono float32 at SR."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(mono) * SR / sr))
        mono = np.interp(
            np.linspace(0.0, len(mono) - 1, n_out, dtype=np.float64),
            np.arange(len(mono), dtype=np.float64),
            mono.astype(np.float64),
        ).astype(np.float32)
    return mono


def tile_to(sig: np.ndarray, n: int) -> np.ndarray:
    """Loop `sig` to exactly n samples, cross-fading each seam."""
    if len(sig) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(sig) >= n:
        return sig[:n].copy()
    fade = min(int(0.25 * SR), len(sig) // 4)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        chunk = sig.copy()
        if pos > 0 and fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            chunk[:fade] *= ramp
            out[pos - fade : pos] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            pos -= fade
        end = min(pos + len(chunk), n)
        out[pos:end] += chunk[: end - pos]
        pos = end
    return out


def set_rms(sig: np.ndarray, target: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(sig)))) if len(sig) else 0.0
    if rms < 1e-8:
        return sig
    return (sig * (target / rms)).astype(np.float32)


def soft_limit(sig: np.ndarray, ceiling: float = 0.6) -> np.ndarray:
    """Tanh knee. Hard peak-normalising made the rain presets quiet, because
    their transients set the scale for the whole clip; this tames the peaks
    while leaving the loudness the RMS pass just set."""
    return (ceiling * np.tanh(sig / ceiling)).astype(np.float32)


def envelope(sig: np.ndarray, fade_s: float = 0.6) -> np.ndarray:
    fade = min(int(fade_s * SR), len(sig) // 2)
    if fade <= 0:
        return sig
    out = sig.copy()
    out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def audio_stats(sig: np.ndarray) -> dict:
    peak = float(np.max(np.abs(sig))) if len(sig) else 0.0
    return {
        "sample_rate": SR,
        "duration_s": round(len(sig) / SR, 4),
        "peak": round(peak, 6),
        "rms": round(float(np.sqrt(np.mean(np.square(sig)))) if len(sig) else 0.0, 6),
        "max": round(float(np.max(sig)) if len(sig) else 0.0, 6),
        "min": round(float(np.min(sig)) if len(sig) else 0.0, 6),
        "clip_pct": round(float(np.mean(np.abs(sig) >= 0.999) * 100.0) if len(sig) else 0.0, 4),
    }


def write_wav(path: Path, sig: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sig, SR, subtype="PCM_16")


def write_spectrogram(
    path: Path,
    sig: np.ndarray,
    title: str,
    subtitle: str,
    figsize: tuple[float, float] = (7.0, 3.0),
    dpi: int = 85,
) -> None:
    """Log-magnitude STFT, styled close enough to the repo's baked spectrograms."""
    n_fft, hop = 1024, 256
    if len(sig) < n_fft:
        sig = np.pad(sig, (0, n_fft - len(sig)))
    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(sig) - n_fft) // hop
    stft = np.empty((n_fft // 2 + 1, frames), dtype=np.float32)
    for i in range(frames):
        seg = sig[i * hop : i * hop + n_fft] * window
        stft[:, i] = np.abs(np.fft.rfft(seg)).astype(np.float32)
    db = 20.0 * np.log10(np.maximum(stft, 1e-6))
    db = np.maximum(db, db.max() - 80.0)

    # Kept deliberately small: these PNGs are committed to git, and a noisy
    # spectrogram is close to incompressible, so pixel count is the whole cost.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, len(sig) / SR, 0.0, SR / 2000.0],
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title(title, fontsize=10)
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


# ---------------------------------------------------------------- source lookup
def case_dir(layer: str, attempt: str, tier: str, *parts: str) -> Path:
    return LAYERS / layer / "attempts" / attempt / tier / Path(*parts)


def ambient_case(cell: str) -> Path:
    """First case dir under the layer A per-cell bank that has a real WAV."""
    root = case_dir("layer_a", ATTEMPTS["layer_a"], "expected", cell)
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        if (case / "audio.wav").exists():
            return case
    raise SystemExit(f"no materialised ambient WAV for cell {cell} — run `dvc pull` first")


def weather_case(weather_type: str, intensity: str) -> Path:
    attempt, case = WEATHER_STEMS[(weather_type, intensity)]
    path = case_dir("layer_b", attempt, "expected", case)
    if not (path / "audio.wav").exists():
        raise SystemExit(f"missing weather stem {path} — run `dvc pull` first")
    return path


# ---------------------------------------------------------------- copy fixtures
def copy_sample_sets() -> None:
    for layer, attempt, tier, mode in SAMPLE_SETS:
        src = LAYERS / layer / "attempts" / attempt / tier
        if not src.is_dir():
            print(f"  ! skip missing {src}")
            continue
        dst = FIXTURES / "layers" / layer / "attempts" / attempt / tier
        dst.mkdir(parents=True, exist_ok=True)
        if mode == "one-per-cell":
            for cell in sorted(p for p in src.iterdir() if p.is_dir()):
                chosen = ambient_case(cell.name)
                copy_case(chosen, dst / cell.name / chosen.name)
        else:
            for case in sorted(p for p in src.iterdir() if p.is_dir()):
                if (case / "audio.wav").exists():
                    copy_case(case, dst / case.name)
                else:
                    print(f"  ! skip DVC-only case {case.name}")


def copy_case(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("audio.wav", "spectrogram.png", "metadata.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def copy_event_clips() -> dict:
    """One reference call per Layer C species, from the retrieval asset bank."""
    index_path = BANK / "index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    assets = json.loads(index_path.read_text())["assets"]

    best: dict[str, dict] = {}
    for asset in assets:
        attrs = asset["attributes"]
        slug = attrs["event_type"]
        score = float(attrs.get("quality_score") or attrs.get("score") or 0.0)
        dur = float(attrs.get("duration_s") or 99.0)
        if dur > 6.0:
            continue
        cur = best.get(slug)
        if cur is None or score > cur["_score"]:
            best[slug] = {**asset, "_score": score}

    out_dir = FIXTURES / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = {}
    for slug, asset in sorted(best.items()):
        src = BANK / asset["audio_path"]
        if not src.exists():
            print(f"  ! skip {slug}: {src.name} not materialised")
            continue
        shutil.copy2(src, out_dir / f"{slug}.wav")
        attrs = asset["attributes"]
        # Small spectrogram per species: /dev/layers shows one for every Layer C
        # generate, and these clips have no PNG sibling in the asset bank.
        write_spectrogram(
            out_dir / f"{slug}.png",
            load_mono(src),
            f"{attrs['species_common_name']} — reference call",
            f"demo fixture · {slug} · replayed, not generated",
            figsize=(5.0, 2.4),
            dpi=80,
        )
        catalog[slug] = {
            "asset_id": asset["id"],
            "species_common_name": attrs["species_common_name"],
            "species_scientific_name": attrs.get("species_scientific_name", ""),
            "audio_event_id": attrs.get("audio_event_id", ""),
            "duration_s": round(float(attrs.get("duration_s") or 0.0), 3),
            "diel_bin": attrs.get("diel_bin", ""),
            "season": attrs.get("season", ""),
            "file": f"{slug}.wav",
        }
    (out_dir / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"  events: {len(catalog)} species")
    return catalog


# ---------------------------------------------------------------- bake the mixes
def bake_preset(preset: dict, catalog: dict) -> None:
    n = int(MIX_SECONDS * SR)
    rng = np.random.default_rng(CANONICAL_SEED)

    amb_case = ambient_case(preset["cell"])
    ambient = envelope(set_rms(tile_to(load_mono(amb_case / "audio.wav"), n), 0.055))
    mix = ambient.copy()

    stems = {"layer_a": ambient}
    events_meta = []

    if preset["weather"]:
        wtype, intensity = preset["weather"]
        w_case = weather_case(wtype, intensity)
        weather = envelope(set_rms(tile_to(load_mono(w_case / "audio.wav"), n), WEATHER_GAIN[intensity]))
        mix = mix + weather
        stems["layer_b"] = weather

    if preset["event_count"] and preset["species"]:
        events = np.zeros(n, dtype=np.float32)
        slots = np.linspace(1.5 * SR, n - 4.0 * SR, preset["event_count"])
        for i, slot in enumerate(slots):
            slug, common = preset["species"][i % len(preset["species"])]
            clip_path = FIXTURES / "events" / f"{slug}.wav"
            if not clip_path.exists():
                continue
            clip = set_rms(load_mono(clip_path), 0.09)
            onset = int(slot + rng.integers(-int(0.8 * SR), int(0.8 * SR)))
            onset = max(0, min(onset, n - len(clip) - 1))
            events[onset : onset + len(clip)] += clip
            events_meta.append(
                {
                    "event_index": len(events_meta),
                    "species_common_name": common,
                    "species_slug": slug,
                    "asset_id": catalog.get(slug, {}).get("asset_id", slug),
                    "onset_s": round(onset / SR, 3),
                    "offset_s": round((onset + len(clip)) / SR, 3),
                    "gain": 1.0,
                }
            )
        mix = mix + events
        stems["layer_c"] = events

    # Loudness-match the presets to each other (RMS), then only peak-limit —
    # peak-normalising alone left the rain preset ~6x quieter than the others
    # because its transients dominate.
    mix = soft_limit(set_rms(mix, 0.085))

    out = FIXTURES / "generation" / preset["id"]
    write_wav(out / "audio.wav", mix)
    season, diel = preset["cell"].split("_")
    wtype = preset["weather"][0] if preset["weather"] else "none"
    intensity = preset["weather"][1] if preset["weather"] else "none"
    write_spectrogram(
        out / "spectrogram.png",
        mix,
        f"Layer D mix — {season} {diel}"
        + (f", {intensity} {wtype}" if preset["weather"] else "")
        + (f", {len(events_meta)} events" if events_meta else ""),
        f"demo fixture · {preset['id']} · pre-baked, not live inference",
    )

    metadata = {
        "mock": True,
        "mock_note": "Pre-baked demo fixture replayed by services/dev/ai-mock. Not model output.",
        "preset_id": preset["id"],
        "prompt": preset["prompt"],
        "prompt_locked": True,
        "generator": "ai_mock_replay",
        "cell": preset["cell"],
        "seed": CANONICAL_SEED,
        "audio": audio_stats(mix),
        "audio_length_in_s": MIX_SECONDS,
        "orchestration": {
            "seed": CANONICAL_SEED,
            "duration_s": MIX_SECONDS,
            "season": season,
            "diel": diel,
            "weather_type": wtype,
            "intensity": intensity,
            "include_weather": bool(preset["weather"]),
            "include_events": bool(events_meta),
            "attempts": dict(ATTEMPTS),
            "parameter_routing": {
                "layer_a": ["season", "diel", "seed"],
                "layer_b": ["weather_type", "intensity", "duration_s"],
                "layer_c": ["species_common_name", "seed", "duration_s"],
                "layer_d": ["duration_s"],
            },
            "events": events_meta,
        },
        "attempt": {
            "layer": "layer_d",
            "id": ATTEMPTS["layer_d"],
            "label": "Multi-clip mix",
            "stage": "mvp_2",
            "author": "songke",
            "status": "implemented_not_default",
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for layer, sig in stems.items():
        stem_dir = out / "stems" / layer
        stem_sig = soft_limit(set_rms(sig, 0.085))
        write_wav(stem_dir / "audio.wav", stem_sig)
        write_spectrogram(
            stem_dir / "spectrogram.png",
            stem_sig,
            f"{layer} stem — {preset['id']}",
            "demo fixture · pre-baked stem",
        )
        (stem_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "mock": True,
                    "preset_id": preset["id"],
                    "prompt": preset["prompt"],
                    "prompt_locked": True,
                    "cell": preset["cell"],
                    "seed": CANONICAL_SEED,
                    "generator": "ai_mock_replay",
                    "audio": audio_stats(stem_sig),
                    "attempt": {
                        "layer": layer,
                        "id": ATTEMPTS[layer],
                        "label": f"{layer} demo stem",
                        "stage": "mvp",
                        "author": "demo",
                        "status": "demo_fixture",
                    },
                },
                indent=2,
            )
            + "\n"
        )
    print(f"  mix {preset['id']}: {len(mix) / SR:.1f}s, {len(events_meta)} events, stems {sorted(stems)}")


# ---------------------------------------------------------------- preset hashes
def write_preset_hashes() -> None:
    out = {}
    for cell_dir in sorted((FIXTURES / "layers" / "layer_a" / "attempts" / ATTEMPTS["layer_a"] / "expected").iterdir()):
        if not cell_dir.is_dir():
            continue
        for case in sorted(cell_dir.iterdir()):
            wav = case / "audio.wav"
            if wav.exists():
                digest = hashlib.md5(wav.read_bytes()).hexdigest()
                out[digest] = {"cell": cell_dir.name, "case": case.name}
    path = FIXTURES / "analysis" / "presets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  preset hashes: {len(out)}")


def main() -> int:
    if not LAYERS.is_dir():
        raise SystemExit(f"cannot find {LAYERS}")
    print(f"repo:     {REPO}")
    print(f"fixtures: {FIXTURES}")

    print("copying sample tiers...")
    copy_sample_sets()

    print("copying event clips...")
    catalog = copy_event_clips()

    print("baking preset mixes...")
    for preset in PRESETS:
        bake_preset(preset, catalog)

    print("hashing preset recordings...")
    write_preset_hashes()

    print("building analysis reports...")
    from build_reports import build_all  # local import: shares this venv

    build_all(FIXTURES)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
