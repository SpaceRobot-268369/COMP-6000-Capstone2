"""Frozen LAION-CLAP backbone for E-B weather zero-shot analysis.

Contract: mono audio is resampled to 48 kHz for CLAP, embedded with
`laion/clap-htsat-unfused`, and L2-normalised before cosine scoring against text
prompts.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

MODEL_ID = "laion/clap-htsat-unfused"
TARGET_SR = 48_000


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    duration_s = len(samples) / float(source_rate)
    target_len = max(1, int(round(duration_s * target_rate)))
    source_x = np.linspace(0.0, duration_s, num=len(samples), endpoint=False)
    target_x = np.linspace(0.0, duration_s, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def _pick_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLAPBackbone:
    def __init__(self, device: str | None = None) -> None:
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"CLAP dependencies unavailable: {exc}") from exc

        self.torch = torch
        self.device = device or _pick_device(torch)
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()

    def _prepare_audio(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = samples.astype(np.float32, copy=False)
        return _resample_linear(mono, sample_rate, TARGET_SR)

    def embed_audio_array(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = self._prepare_audio(samples, sample_rate)
        inputs = self.processor(audio=[audio], sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            out = self.model.get_audio_features(**inputs)
        vector = out.pooler_output.float().cpu().numpy()
        return _l2_normalize_np(vector)[0]

    def embed_text(self, prompts: Iterable[str]) -> np.ndarray:
        prompts = list(prompts)
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            out = self.model.get_text_features(**inputs)
        return _l2_normalize_np(out.pooler_output.float().cpu().numpy())

