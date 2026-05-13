"""CLAP text encoder wrapper.

Wraps laion/clap-htsat-unfused (via HuggingFace transformers) to produce
normalised 512-dim text embeddings for diffusion conditioning.

The model is loaded lazily on first call and stays frozen — we never update
its weights. All text encoding happens in float32 on CPU by default (the
embeddings are tiny); move them to your training device afterwards.

Usage:
    encoder = ClapTextEncoder()
    embs = encoder(["spring night, ambient soundscape, ..."])  # (1, 512)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_EMB_DIM  = 512


class ClapTextEncoder:
    """Stateless CLAP text encoder (lazy-loaded, always frozen).

    Args:
        model_id:    HuggingFace model ID (default: laion/clap-htsat-unfused).
        device:      torch device for inference. Defaults to CPU — embeddings
                     are small; move the result tensors to GPU yourself.
        normalize:   L2-normalise embeddings before returning (default True).
    """

    def __init__(
        self,
        model_id: str = CLAP_MODEL_ID,
        device: str | torch.device = "cpu",
        normalize: bool = True,
    ):
        self.model_id  = model_id
        self.device    = torch.device(device)
        self.normalize = normalize
        self._model    = None
        self._tokenizer = None

    # ------------------------------------------------------------------ lazy load

    def _load(self) -> None:
        from transformers import ClapTextModelWithProjection, AutoTokenizer  # noqa: PLC0415

        print(f"[clap] loading {self.model_id} …")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = ClapTextModelWithProjection.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        ).to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        print(f"[clap] loaded — text projection dim: {CLAP_EMB_DIM}")

    # ------------------------------------------------------------------ public

    @property
    def emb_dim(self) -> int:
        return CLAP_EMB_DIM

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings → (N, 512) float32 tensor.

        Returns the tensor on `self.device`.
        """
        if self._model is None:
            self._load()

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        outputs = self._model(**inputs)
        embs = outputs.text_embeds                       # (N, 512)

        if self.normalize:
            embs = F.normalize(embs, p=2, dim=-1)

        return embs.float()
