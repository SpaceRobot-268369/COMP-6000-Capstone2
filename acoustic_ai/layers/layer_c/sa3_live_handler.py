"""Layer C live Stable Audio 3 LoRA handler.

This handler runs SA3 generation at request time, then applies the same
species-specific post-processing used for the audited sample pools. It is meant
for GPU worker validation, not for low-latency local demo use.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SpeciesLiveConfig:
    key: str
    species_common_name: str
    event_type: str
    checkpoint: Path
    prompt: str
    duration_s: float
    sampling_steps: int
    cfg_scale: float
    postprocess: dict[str, Any]


@dataclass(frozen=True)
class LiveState:
    params: dict[str, Any]
    default_species: str
    species_configs: dict[str, SpeciesLiveConfig]
    output_root: Path


def load(checkpoint_dir: Path | None, params: dict, extra: dict | None = None) -> LiveState:
    del checkpoint_dir, extra
    species_configs: dict[str, SpeciesLiveConfig] = {}
    for key, cfg in (params.get("species_pools") or {}).items():
        checkpoint = _repo_path(cfg["source_checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"SA3 LoRA checkpoint not found: {checkpoint}")
        species_common_name = str(cfg.get("species_common_name", key))
        config = SpeciesLiveConfig(
            key=str(key),
            species_common_name=species_common_name,
            event_type=str(cfg.get("event_type", key)),
            checkpoint=checkpoint,
            prompt=str(cfg["prompt"]),
            duration_s=float(cfg.get("duration_s", params.get("duration_s", 3.0))),
            sampling_steps=int(cfg.get("sampling_steps", params.get("sampling_steps", 50))),
            cfg_scale=float(cfg.get("sampling_cfg_scale", params.get("sampling_cfg_scale", 4.0))),
            postprocess=dict(cfg.get("postprocess") or {}),
        )
        aliases = {
            str(key),
            str(key).replace("_", " ").lower(),
            species_common_name,
            species_common_name.lower(),
        }
        for alias in aliases:
            species_configs[alias] = config
    if not species_configs:
        raise ValueError("Live SA3 Layer C attempt requires params.species_pools")

    output_root = _repo_path(params.get("live_output_root", "debug/layer_c/sa3_live_api"))
    output_root.mkdir(parents=True, exist_ok=True)
    return LiveState(
        params=dict(params),
        default_species=str(params.get("default_species_common_name") or next(iter(species_configs))),
        species_configs=species_configs,
        output_root=output_root,
    )


def generate(
    state: LiveState,
    seed: int | None = None,
    season: str | None = None,
    diel: str | None = None,
    species_common_name: str | None = None,
    **_: object,
) -> dict[str, object]:
    run_seed = int(seed if seed is not None else state.params.get("seed", 42))
    config = _select_species(state, species_common_name)
    run_id = f"{config.key}_seed_{run_seed}_{int(time.time())}"
    run_root = state.output_root / run_id
    raw_dir = run_root / "raw"
    processed_dir = run_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    _run_sa3_sample(config, raw_dir, run_seed)
    output_audio, source_metadata = _postprocess(config, raw_dir, processed_dir, run_seed)
    audio, sample_rate = sf.read(output_audio, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

    metadata = {
        "layer": "layer_c",
        "method": "sa3_lora_generative_live",
        "species": config.species_common_name,
        "species_common_name": config.species_common_name,
        "event_type": config.event_type,
        "prompt": config.prompt,
        "prompt_locked": True,
        "request": {
            "seed": run_seed,
            "season": season or state.params.get("default_season", "summer"),
            "diel": diel or state.params.get("default_diel", "morning"),
            "species_common_name": config.species_common_name,
        },
        "audio": {
            "sample_rate": int(sample_rate),
            "duration_s": float(len(audio) / sample_rate) if sample_rate else 0.0,
            "peak": peak,
            "rms": rms,
            "contains_ambient_bed": False,
        },
        "generation": {
            "base_model": state.params.get("base_model", "Stable Audio 3 small-sfx-base"),
            "checkpoint": str(config.checkpoint.relative_to(REPO_ROOT)),
            "prompt": config.prompt,
            "duration_s": config.duration_s,
            "sampling_cfg_scale": config.cfg_scale,
            "sampling_steps": config.sampling_steps,
            "postprocess": config.postprocess,
            "run_dir": str(run_root.relative_to(REPO_ROOT)),
            "selected_sample": str(output_audio.relative_to(REPO_ROOT)),
            "elapsed_s": round(time.time() - started, 3),
            "note": "Live SA3 LoRA generation executed at request time on the GPU worker.",
        },
        "source_metadata": source_metadata,
    }

    (run_root / "response_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "wav_bytes": _wav_bytes(audio, int(sample_rate)),
        "mel_db": _mel_db(audio, int(sample_rate)),
        "metadata": metadata,
    }


def _select_species(state: LiveState, species_common_name: str | None) -> SpeciesLiveConfig:
    requested = species_common_name or state.default_species
    config = state.species_configs.get(str(requested)) or state.species_configs.get(str(requested).lower())
    if config is None:
        available = sorted({cfg.species_common_name for cfg in state.species_configs.values()})
        raise ValueError(f"Unknown Layer C live generative species {requested!r}; available: {available}")
    return config


def _run_sa3_sample(config: SpeciesLiveConfig, raw_dir: Path, seed: int) -> None:
    cmd = [
        sys.executable,
        "script/events/sample_sa3_lora_horsfields_cuckoo_pass35.py",
        "--checkpoint",
        str(config.checkpoint),
        "--out-dir",
        str(raw_dir),
        "--prompt",
        config.prompt,
        "--duration",
        str(config.duration_s),
        "--steps",
        str(config.sampling_steps),
        "--cfg-scale",
        str(config.cfg_scale),
        "--seed-start",
        str(seed),
        "--num-seeds",
        "1",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _postprocess(
    config: SpeciesLiveConfig,
    raw_dir: Path,
    processed_dir: Path,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    mode = str(config.postprocess.get("mode", "none"))
    seed_dir = raw_dir / f"seed_{seed:04d}"
    raw_wav = seed_dir / "generated_event.wav"
    if not raw_wav.exists():
        raise FileNotFoundError(f"SA3 generated WAV missing: {raw_wav}")

    if mode == "target_call_detection_timeline":
        s3a_dir = processed_dir / "_s3a_tight"
        s3a_cmd = [
            sys.executable,
            "script/events/postprocess_sa3_cuckoo_s3a.py",
            "--input-dir",
            str(raw_dir),
            "--out-dir",
            str(s3a_dir),
            "--mode",
            str(config.postprocess.get("s3a_mode", "tight")),
            "--low-hz",
            str((config.postprocess.get("bandpass_hz") or [2100, 4100])[0]),
            "--high-hz",
            str((config.postprocess.get("bandpass_hz") or [2100, 4100])[1]),
            "--gate-strength",
            str(config.postprocess.get("s3a_gate_strength", 0.40)),
            "--target-rms",
            str(config.postprocess.get("s3a_target_rms", 0.03)),
            "--fade-ms",
            str(config.postprocess.get("s3a_fade_ms", 100)),
        ]
        subprocess.run(s3a_cmd, cwd=REPO_ROOT, check=True)
        cmd = [
            sys.executable,
            "script/events/detect_cuckoo_target_calls_tmp.py",
            "--input-dir",
            str(s3a_dir),
            "--out-dir",
            str(processed_dir),
            "--seeds",
            f"seed_{seed:04d}",
            "--pre-s",
            str(config.postprocess.get("pre_buffer_s", 0.14)),
            "--post-s",
            str(config.postprocess.get("post_buffer_s", 0.18)),
            "--min-score",
            str(config.postprocess.get("min_score", 0.68)),
            "--min-contrast-score",
            str(config.postprocess.get("min_contrast_score", 0.70)),
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        out_dir = processed_dir / f"seed_{seed:04d}"
        return out_dir / "target_timeline.wav", _read_json(out_dir / "target_call_metadata.json")

    if mode == "s3a_tight_tailcrop335":
        cmd = [
            sys.executable,
            "script/events/postprocess_sa3_spotted_nightjar_autocrop.py",
            "--input-dir",
            str(raw_dir),
            "--out-dir",
            str(processed_dir),
            "--low-hz",
            str((config.postprocess.get("bandpass_hz") or [550, 950])[0]),
            "--high-hz",
            str((config.postprocess.get("bandpass_hz") or [550, 950])[1]),
            "--gate-strength",
            str(config.postprocess.get("spectral_gate_strength", 0.3)),
            "--target-rms",
            str(config.postprocess.get("target_rms", 0.035)),
            "--fade-ms",
            str(config.postprocess.get("fade_ms", 120)),
            "--max-end-s",
            str(config.postprocess.get("crop_end_s", 3.35)),
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        out_dir = processed_dir / f"seed_{seed:04d}"
        return out_dir / "generated_event_s3a_autocrop.wav", _read_json(out_dir / "generated_event_s3a_autocrop_metadata.json")

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_wav = processed_dir / f"seed_{seed:04d}" / "generated_event.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_wav, out_wav)
    return out_wav, _read_json(seed_dir / "generated_event_metadata.json")


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _mel_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=min(sample_rate / 2, 11025),
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)
