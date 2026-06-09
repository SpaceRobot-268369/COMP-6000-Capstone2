"""Dataset loader for fine-tuning AudioLDM2 on ecoacoustic soundscapes.

The training loop asks the AudioLDM2 pipeline to encode captions, so this
dataset only loads/crops/pads waveforms and returns the raw caption string.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioLDM2Dataset(Dataset):
    """
    Loads audio and captions for AudioLDM2 training.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        base_dir: str | Path,
        target_sample_rate: int = 16000,
        max_duration_s: float = 10.0,
        normalize_audio: bool = False,
        target_rms: float = 0.005,
        min_rms: float = 1e-4,
    ):
        """
        Args:
            manifest_path: Path to the dataset manifest.csv
            base_dir: Base directory for resolving relative audio paths
            target_sample_rate: Sample rate to return before model feature extraction
            max_duration_s: Crop/pad audio to this length
            normalize_audio: RMS-normalize quiet field recordings before training
            target_rms: Target RMS for non-silent clips after normalization
            min_rms: Clips below this RMS are treated as silence and not amplified
        """
        self.manifest_path = Path(manifest_path)
        self.base_dir = Path(base_dir)
        self.target_sample_rate = target_sample_rate
        self.max_length = int(self.target_sample_rate * max_duration_s)
        self.normalize_audio = normalize_audio
        self.target_rms = target_rms
        self.min_rms = min_rms

        # Load manifest
        self.items = []
        with open(self.manifest_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'ok':
                    self.items.append({
                        'audio_path': self.base_dir / row['audio_path'],
                        'caption': row['caption']
                    })

        if not self.items:
            raise ValueError(f"No valid items found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # 1. Load and process audio
        waveform_np, sr = sf.read(str(item['audio_path']), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(waveform_np.T))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)

        # Crop or pad to exact length
        if waveform.shape[1] > self.max_length:
            # Random crop
            start = torch.randint(0, waveform.shape[1] - self.max_length + 1, (1,)).item()
            waveform = waveform[:, start : start + self.max_length]
        elif waveform.shape[1] < self.max_length:
            # Pad with zeros
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        if self.normalize_audio:
            waveform = self._normalize_waveform(waveform)

        # Flatten waveform to 1D
        waveform = waveform.squeeze(0)

        return {
            "audio": waveform,
            "caption": item['caption']
        }

    def _normalize_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Bring very quiet ecoacoustic clips into AudioLDM2's expected range."""
        rms = torch.sqrt(torch.mean(waveform.square()))
        if not torch.isfinite(rms) or rms < self.min_rms:
            return waveform

        waveform = waveform * (self.target_rms / rms)
        peak = waveform.abs().max()
        if torch.isfinite(peak) and peak > 0.95:
            waveform = waveform * (0.95 / peak)
        return waveform.clamp(-1.0, 1.0)
