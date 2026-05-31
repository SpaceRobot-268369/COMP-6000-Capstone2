"""Frozen LAION-CLAP backbone shared by Layer E ambient-analysis attempts.

Contract: 48 kHz mono input, 10 s windowing with mean-pool, L2-normalised
512-d embeddings. The model is `laion/clap-htsat-unfused` — the same CLAP
that lives inside `cvssp/audioldm2`, so the analysis embedding space matches
the generation conditioning space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"
TARGET_SR = 48_000
WINDOW_SAMPLES = 10 * TARGET_SR


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _load_48k_mono(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample(audio, sr, TARGET_SR)
    return audio.astype(np.float32, copy=False)


def _window_10s(audio: np.ndarray) -> list[np.ndarray]:
    if audio.size <= WINDOW_SAMPLES:
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
        return [audio]
    windows: list[np.ndarray] = []
    n = (audio.size + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    for i in range(n):
        start = i * WINDOW_SAMPLES
        chunk = audio[start : start + WINDOW_SAMPLES]
        if chunk.size < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - chunk.size))
        windows.append(chunk)
    return windows


class CLAPBackbone:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or _pick_device()
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()

    @torch.no_grad()
    def embed_audio(self, paths: Iterable[str | Path], verbose: bool = False) -> np.ndarray:
        paths = list(paths)
        out = np.empty((len(paths), self.model.config.projection_dim), dtype=np.float32)
        for i, p in enumerate(paths):
            audio = _load_48k_mono(p)
            windows = _window_10s(audio)
            inputs = self.processor(
                audio=windows, sampling_rate=TARGET_SR, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_audio_features(**inputs)
            pooled = _l2_norm(feats.mean(dim=0, keepdim=True)).squeeze(0)
            out[i] = pooled.float().cpu().numpy()
            if verbose and (i + 1) % 100 == 0:
                print(f"  embedded {i + 1}/{len(paths)}")
        return out

    @torch.no_grad()
    def embed_text(self, strs: Iterable[str]) -> np.ndarray:
        strs = list(strs)
        inputs = self.processor(text=strs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return _l2_norm(feats).float().cpu().numpy()
