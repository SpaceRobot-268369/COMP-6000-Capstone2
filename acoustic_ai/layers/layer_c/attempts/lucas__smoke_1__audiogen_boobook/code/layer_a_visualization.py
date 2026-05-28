"""Layer A mel-spectrogram rendering utilities (attempt-local copy).

The dev API and AudioLDM2 CLI use this module so the same waveform is
visualized with identical mel parameters and PNG rendering settings.

Self-contained — duplicates SPEC_CFG + waveform_to_melspec from the
project's historical preprocess module so this attempt has no cross-attempt
imports. Keep in sync manually with sibling copies in other attempts if
the mel config needs to change.
"""

from __future__ import annotations

import io

import numpy as np

# Mel config — duplicated from the original modules.ambient.preprocess.
SPEC_CFG = {
    "sample_rate": 22_050,
    "n_fft":       1024,
    "hop_length":  512,
    "n_mels":      128,
    "fmin":        50,
    "fmax":        11_000,
    "top_db":      80,
}

LAYER_A_SPEC_TITLE = "Layer C - Event Spectrogram"
LAYER_A_SPEC_FIGSIZE = (12, 4)
LAYER_A_SPEC_DPI = 110
LAYER_A_SPEC_CMAP = "magma"


def waveform_to_melspec(waveform: np.ndarray, cfg: dict = SPEC_CFG) -> np.ndarray:
    """Convert a mono waveform to a log-mel spectrogram (n_mels, T) in dB."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        fmin=cfg["fmin"],
        fmax=cfg["fmax"],
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=cfg["top_db"]).astype(np.float32)


def waveform_to_layer_a_mel_db(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert an AudioLDM2 waveform to the project Layer A log-mel format."""
    cfg = {
        **SPEC_CFG,
        "sample_rate": int(sample_rate),
        "fmax": min(float(SPEC_CFG["fmax"]), float(sample_rate) / 2.0),
    }
    return waveform_to_melspec(waveform, cfg=cfg)


# ---------------------------------------------------------------------------
# PNG metadata baking
# ---------------------------------------------------------------------------
#
# Two channels, both optional:
#   overlay  -> small human-readable text drawn on the figure (header/subline
#              /footer). Survives screenshots and renders inline on GitHub.
#   png_text -> PNG tEXt key/value chunks written after savefig via Pillow.
#              Pixels are unchanged; readable with `exiftool` or
#              `PIL.Image.open(p).text`. Use for full traceability fields the
#              JSON sidecar also carries.
#
# With both args left None, output is byte-identical to the pre-overlay
# implementation (modulo matplotlib determinism).


def _draw_overlay(ax, overlay: dict) -> None:
    """Draw header/subline/footer inside the spectrogram axes with a
    translucent bbox so they don't collide with the title or colorbar.

    Keys (all optional, all str): "header", "subline", "footer".
    """
    bbox = {
        "facecolor": "black",
        "alpha":     0.60,
        "edgecolor": "none",
        "pad":       3.0,
    }
    common = {
        "color":        "white",
        "fontsize":     8,
        "family":       "monospace",
        "transform":    ax.transAxes,
        "bbox":         bbox,
        "zorder":       5,
    }
    header  = overlay.get("header")
    subline = overlay.get("subline")
    footer  = overlay.get("footer")

    # Inside the axes: top-left stack, bottom-right footer.
    if header:
        ax.text(0.01, 0.97, header, ha="left", va="top", **common)
    if subline:
        ax.text(0.01, 0.86, subline, ha="left", va="top", **common)
    if footer:
        ax.text(0.99, 0.04, footer, ha="right", va="bottom", **common)


def _inject_png_text(png_bytes: bytes, png_text: dict) -> bytes:
    """Re-encode the PNG with tEXt chunks. Pixel data is unchanged."""
    from PIL import Image, PngImagePlugin

    img = Image.open(io.BytesIO(png_bytes))
    img.load()
    info = PngImagePlugin.PngInfo()
    for k, v in png_text.items():
        if v is None:
            continue
        info.add_text(str(k), str(v))
    out = io.BytesIO()
    img.save(out, format="PNG", pnginfo=info)
    out.seek(0)
    return out.read()


def render_layer_a_mel_png_bytes(
    mel_db: np.ndarray,
    duration_s: float,
    *,
    overlay: dict | None = None,
    png_text: dict | None = None,
) -> bytes:
    """Render a Layer A mel-spectrogram PNG with shared visual settings.

    Optional kwargs:
        overlay  - dict with optional "header"/"subline"/"footer" strings,
                   drawn on the figure for at-a-glance review.
        png_text - dict[str, str] written as PNG tEXt chunks (lossless,
                   readable via Pillow/exiftool).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=LAYER_A_SPEC_FIGSIZE)
    img = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration_s, 0, SPEC_CFG["n_mels"]],
        cmap=LAYER_A_SPEC_CMAP,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(LAYER_A_SPEC_TITLE)
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()

    if overlay:
        _draw_overlay(ax, overlay)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=LAYER_A_SPEC_DPI)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    if png_text:
        png_bytes = _inject_png_text(png_bytes, png_text)
    return png_bytes


# ---------------------------------------------------------------------------
# Overlay / tEXt builders — tier-specific
# ---------------------------------------------------------------------------


def _fmt_float(v, fmt: str, default: str = "?") -> str:
    try:
        if v is None or v == "":
            return default
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return default


def build_expected_overlay(metadata: dict) -> dict:
    """Compact 3-line overlay for an expected (real-audio) sample."""
    sm     = metadata.get("source_metadata", {}) or {}
    env    = sm.get("env", {}) or {}
    audio  = metadata.get("audio", {}) or {}

    season    = sm.get("season") or env.get("season") or "?"
    diel      = sm.get("diel_bin") or env.get("sample_bin") or "?"
    rec_date  = sm.get("recording_date") or (env.get("sample_local_date") or "?")
    temp_c    = _fmt_float(env.get("temperature_c"), ".1f")
    wind_ms   = _fmt_float(env.get("wind_speed_ms"), ".1f")
    dur_s     = _fmt_float(audio.get("duration_s"), ".1f")
    rms       = _fmt_float(audio.get("rms"), ".4f")
    peak      = _fmt_float(audio.get("peak"), ".4f")
    sr        = audio.get("sample_rate", "?")
    clip_id   = metadata.get("source_clip_id", "?")

    header = (
        f"EXPECTED · {season} · {diel} · {rec_date} "
        f"· {temp_c}°C · wind {wind_ms} m/s · {dur_s} s"
    )
    subline = f"clip {clip_id}"
    footer  = f"rms={rms}  peak={peak}  sr={sr}"
    return {"header": header, "subline": subline, "footer": footer}


def build_expected_png_text(metadata: dict) -> dict:
    """Flatten expected metadata into PNG tEXt key/value pairs."""
    sm    = metadata.get("source_metadata", {}) or {}
    env   = sm.get("env", {}) or {}
    audio = metadata.get("audio", {}) or {}

    out: dict[str, str] = {
        "tier":             "expected",
        "source":           str(metadata.get("source", "")),
        "source_kind":      str(metadata.get("source_kind", "")),
        "source_clip_id":   str(metadata.get("source_clip_id", "")),
        "source_manifest":  str(metadata.get("source_manifest", "")),
        "selection_reason": str(metadata.get("selection_reason", "")),
        "caption":          str(sm.get("caption", "")),
        "recording_date":   str(sm.get("recording_date", "")),
        "season":           str(sm.get("season", "")),
        "diel_bin":         str(sm.get("diel_bin", "")),
        "audio.sample_rate": str(audio.get("sample_rate", "")),
        "audio.duration_s":  str(audio.get("duration_s", "")),
        "audio.rms":         str(audio.get("rms", "")),
        "audio.peak":        str(audio.get("peak", "")),
    }
    for k, v in env.items():
        out[f"env.{k}"] = str(v)
    return {k: v for k, v in out.items() if v not in ("", "None")}


def _truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def build_showcase_overlay(metadata: dict) -> dict:
    """Compact 3-line overlay for a showcase (generated) sample."""
    audio = metadata.get("audio", {}) or {}

    seed       = metadata.get("seed", "?")
    steps      = metadata.get("num_inference_steps", "?")
    guidance   = _fmt_float(metadata.get("guidance_scale"), ".1f")
    dur_s      = _fmt_float(audio.get("duration_s"), ".1f")
    rms        = _fmt_float(audio.get("rms"), ".4f")
    peak       = _fmt_float(audio.get("peak"), ".4f")
    prompt     = _truncate(metadata.get("prompt", ""), 110)
    ckpt       = metadata.get("checkpoint") or ""
    ckpt_short = ckpt.rsplit("/", 1)[-1] if ckpt else "?"
    sha        = metadata.get("handler_git_sha", "?")

    header = (
        f"SHOWCASE · seed={seed} · steps={steps} "
        f"· guidance={guidance} · {dur_s} s"
    )
    subline = f"prompt: {prompt}"
    footer  = f"rms={rms}  peak={peak}  ckpt={ckpt_short}  sha={sha}"
    return {"header": header, "subline": subline, "footer": footer}


def build_showcase_png_text(metadata: dict) -> dict:
    """Flatten showcase metadata into PNG tEXt key/value pairs."""
    audio = metadata.get("audio", {}) or {}
    post  = metadata.get("postprocess", {}) or {}

    out: dict[str, str] = {
        "tier":                "showcase",
        "showcase_label":      str(metadata.get("showcase_label", "")),
        "seed":                str(metadata.get("seed", "")),
        "prompt":              str(metadata.get("prompt", "")),
        "prompt_locked":       str(metadata.get("prompt_locked", "")),
        "generator":           str(metadata.get("generator", "")),
        "base_model":          str(metadata.get("base_model", "")),
        "checkpoint":          str(metadata.get("checkpoint", "")),
        "checkpoint_dvc_hash": str(metadata.get("checkpoint_dvc_hash", "")),
        "handler_git_sha":     str(metadata.get("handler_git_sha", "")),
        "generated_at":        str(metadata.get("generated_at", "")),
        "num_inference_steps": str(metadata.get("num_inference_steps", "")),
        "guidance_scale":      str(metadata.get("guidance_scale", "")),
        "audio_length_in_s":   str(metadata.get("audio_length_in_s", "")),
        "audio.sample_rate":   str(audio.get("sample_rate", "")),
        "audio.duration_s":    str(audio.get("duration_s", "")),
        "audio.rms":           str(audio.get("rms", "")),
        "audio.peak":          str(audio.get("peak", "")),
        "postprocess.highpass_hz":       str(post.get("highpass_hz", "")),
        "postprocess.output_target_rms": str(post.get("output_target_rms", "")),
    }
    return {k: v for k, v in out.items() if v not in ("", "None")}
