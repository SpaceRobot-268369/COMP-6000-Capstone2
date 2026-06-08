"""Frozen LAION-CLAP audio backbone for E-C known-species clips."""

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


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def load_48k_mono(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def window_10s(audio: np.ndarray) -> list[np.ndarray]:
    if audio.size <= WINDOW_SAMPLES:
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
        return [audio]
    windows: list[np.ndarray] = []
    n = (audio.size + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    for idx in range(n):
        start = idx * WINDOW_SAMPLES
        chunk = audio[start:start + WINDOW_SAMPLES]
        if chunk.size < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - chunk.size))
        windows.append(chunk)
    return windows


class CLAPBackbone:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or pick_device()
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()

    @torch.no_grad()
    def embed_audio(self, paths: Iterable[str | Path], verbose: bool = False) -> np.ndarray:
        paths = list(paths)
        rows = np.empty((len(paths), self.model.config.projection_dim), dtype=np.float32)
        for idx, path in enumerate(paths):
            audio = load_48k_mono(path)
            windows = window_10s(audio)
            inputs = self.processor(audio=windows, sampling_rate=TARGET_SR, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            out = self.model.get_audio_features(**inputs)
            feats = out.pooler_output
            pooled = l2_norm(feats.mean(dim=0, keepdim=True)).squeeze(0)
            rows[idx] = pooled.float().cpu().numpy()
            if verbose and (idx + 1) % 50 == 0:
                print(f"embedded {idx + 1}/{len(paths)}")
        return rows
