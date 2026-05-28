"""Fine-tune AudioCraft AudioGen with LoRA for Layer C bird events.

This script keeps the base model as ``facebook/audiogen-medium`` and uses
Meta's AudioCraft API, not Hugging Face Transformers. It trains LoRA adapters
on the AudioCraft language model while the compression model and non-LoRA
weights remain frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = "facebook/audiogen-medium"
DEFAULT_TARGET_MODULES = ("out_proj", "linear1", "linear2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune AudioCraft AudioGen with LoRA for Layer C events."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision",
        choices=("no", "fp16", "bf16"),
        default="no",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--max_duration_s", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--target_modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help=(
            "Comma-separated AudioCraft LM module names for LoRA injection. "
            "Defaults avoid AudioCraft's parameter-only in_proj weights."
        ),
    )
    return parser.parse_args()


def resolve_path(path: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return base_dir / path


def choose_device(value: str | None) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def install_xformers_import_stub() -> None:
    """Allow AudioCraft import on macOS without the CUDA-only xformers package.

    AudioCraft imports ``from xformers import ops`` at module import time even
    when the loaded model is configured for standard PyTorch attention. The
    smoke-test path does not request memory-efficient xformers attention, so a
    minimal import shim is enough to keep the non-xformers code path available.
    """
    import sys
    import types

    if "xformers" in sys.modules:
        return

    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")

    def _missing(*_args, **_kwargs):
        raise ImportError(
            "xformers is not installed; this macOS AudioGen smoke path must use "
            "AudioCraft configs with memory_efficient/checkpointing xformers disabled."
        )

    class LowerTriangularMask:
        pass

    ops.unbind = torch.unbind
    ops.memory_efficient_attention = _missing
    ops.LowerTriangularMask = LowerTriangularMask
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops


def patch_audiocraft_mps_autocast() -> None:
    """Disable AudioCraft autocast wrappers on MPS.

    AudioCraft 1.4 uses ``torch.autocast(device_type="mps")`` for the T5
    conditioner and generation wrapper. PyTorch 2.1's MPS backend does not
    support that autocast device type, so on Apple Silicon we run those blocks
    in regular float32 instead.
    """
    if not torch.backends.mps.is_available():
        return

    from audiocraft.modules import conditioners
    from audiocraft.models import genmodel
    from audiocraft.utils import autocast as autocast_module

    original_torch_autocast = autocast_module.TorchAutocast

    class MpsSafeTorchAutocast:
        def __init__(self, *args, **kwargs):
            device_type = kwargs.get("device_type")
            if device_type is None and args:
                device_type = args[0]
            if kwargs.get("enabled", True) and device_type == "mps":
                self.autocast = None
            else:
                self.autocast = original_torch_autocast(*args, **kwargs)

        def __enter__(self):
            if self.autocast is None:
                return None
            return self.autocast.__enter__()

        def __exit__(self, *args, **kwargs):
            if self.autocast is None:
                return None
            return self.autocast.__exit__(*args, **kwargs)

    autocast_module.TorchAutocast = MpsSafeTorchAutocast
    conditioners.TorchAutocast = MpsSafeTorchAutocast
    genmodel.TorchAutocast = MpsSafeTorchAutocast


def get_lr_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if name != "constant":
        raise ValueError(
            f"unsupported lr scheduler {name!r}; this lightweight trainer supports only 'constant'"
        )

    def lr_lambda(step: int) -> float:
        if num_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(num_warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LayerCAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        target_sample_rate: int,
        max_duration_s: float,
    ) -> None:
        self.manifest_path = resolve_path(manifest_path)
        self.target_sample_rate = int(target_sample_rate)
        self.max_samples = int(round(max_duration_s * self.target_sample_rate))
        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"manifest contains no rows: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        audio_path = resolve_path(row["audio_path"])
        caption = (row.get("caption") or row.get("species_common_name") or "").strip()
        if not caption:
            raise ValueError(f"missing caption for row {index}: {audio_path}")

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if sample_rate != self.target_sample_rate:
            waveform = AF.resample(
                waveform,
                orig_freq=int(sample_rate),
                new_freq=self.target_sample_rate,
            )
        if waveform.numel() > self.max_samples:
            waveform = waveform[: self.max_samples]

        return {
            "caption": caption,
            "audio": waveform,
            "audio_path": str(audio_path),
            "audio_event_id": row.get("audio_event_id", ""),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    max_len = max(int(item["audio"].numel()) for item in batch)
    audio = torch.zeros((len(batch), 1, max_len), dtype=torch.float32)
    captions: list[str] = []
    audio_paths: list[str] = []
    event_ids: list[str] = []

    for index, item in enumerate(batch):
        waveform = item["audio"]
        audio[index, 0, : waveform.numel()] = waveform
        captions.append(str(item["caption"]))
        audio_paths.append(str(item["audio_path"]))
        event_ids.append(str(item["audio_event_id"]))

    return {
        "audio": audio,
        "caption": captions,
        "audio_path": audio_paths,
        "audio_event_id": event_ids,
    }


def compute_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """AudioCraft-style CE over codebooks.

    ``logits`` has shape ``[B, K, T, card]`` and ``targets`` / ``mask`` have
    shape ``[B, K, T]``.
    """
    _, num_codebooks, _, _ = logits.shape
    loss = torch.zeros((), device=logits.device)
    for codebook_idx in range(num_codebooks):
        logits_k = logits[:, codebook_idx].contiguous().view(-1, logits.size(-1))
        targets_k = targets[:, codebook_idx].contiguous().view(-1)
        mask_k = mask[:, codebook_idx].contiguous().view(-1)
        loss = loss + F.cross_entropy(logits_k[mask_k], targets_k[mask_k])
    return loss / num_codebooks


def trainable_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def main() -> int:
    args = parse_args()

    install_xformers_import_stub()
    try:
        from audiocraft.models import AudioGen
        from audiocraft.modules.conditioners import ConditioningAttributes
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "AudioCraft is required for facebook/audiogen-medium. "
            "Install/use the AudioGen environment first, e.g. "
            "`cd acoustic_ai && python3 -m venv .venv-audiogen && "
            "./.venv-audiogen/bin/python -m pip install "
            "git+https://github.com/facebookresearch/audiocraft.git`."
        ) from exc
    patch_audiocraft_mps_autocast()

    device = choose_device(args.device)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    logger.info("Loading AudioCraft AudioGen from %s", args.pretrained_model_name)
    audiogen = AudioGen.get_pretrained(args.pretrained_model_name, device=device)
    compression_model = audiogen.compression_model
    lm = audiogen.lm

    # AudioGen ships with fp16 weights, which crash MPS graph compilation
    # ("expected element type f32 but received f16"). Force everything to fp32
    # on MPS / CPU; CUDA can keep its native dtype.
    if str(device) in {"mps", "cpu"}:
        compression_model.to(torch.float32)
        lm.to(torch.float32)

    compression_model.eval()
    compression_model.requires_grad_(False)
    lm.requires_grad_(False)

    target_modules = [
        value.strip() for value in args.target_modules.split(",") if value.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    lm = inject_adapter_in_model(lora_config, lm)
    audiogen.lm = lm

    trainable, total = trainable_parameter_count(lm)
    logger.info(
        "LoRA trainable parameters: %s / %s (%.4f%%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    dataset = LayerCAudioDataset(
        manifest_path=args.manifest_path,
        target_sample_rate=audiogen.sample_rate,
        max_duration_s=args.max_duration_s,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        [param for param in lm.parameters() if param.requires_grad],
        lr=args.learning_rate,
    )
    steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = args.num_epochs * steps_per_epoch
    lr_scheduler = get_lr_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    lm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        lm,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    audiogen.lm = lm

    logger.info("***** Running Layer C AudioGen LoRA training *****")
    logger.info("  Num examples = %s", len(dataset))
    logger.info("  Num epochs = %s", args.num_epochs)
    logger.info("  Batch size = %s", args.batch_size)
    logger.info("  AudioGen sample rate = %s", audiogen.sample_rate)
    logger.info("  AudioGen frame rate = %s", audiogen.frame_rate)
    logger.info("  LoRA target modules = %s", target_modules)

    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for _epoch in range(args.num_epochs):
        lm.train()
        for batch in dataloader:
            audio = batch["audio"].to(accelerator.device)
            with torch.no_grad():
                audio_tokens, scale = compression_model.encode(audio)
                if scale is not None:
                    raise RuntimeError("scaled compression model is not supported")

            attributes = [
                ConditioningAttributes(text={"description": caption})
                for caption in batch["caption"]
            ]
            attributes = lm.cfg_dropout(attributes)
            attributes = lm.att_dropout(attributes)
            tokenized = lm.condition_provider.tokenize(attributes)
            condition_tensors = lm.condition_provider(tokenized)

            model_output = lm.compute_predictions(
                audio_tokens,
                [],
                condition_tensors,
            )
            mask = torch.ones_like(audio_tokens, dtype=torch.bool) & model_output.mask
            loss = compute_cross_entropy(
                logits=model_output.logits,
                targets=audio_tokens,
                mask=mask,
            )

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.detach().float().item()})

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        unwrapped_lm = accelerator.unwrap_model(lm)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lora_config.save_pretrained(args.output_dir)
        save_file(
            get_peft_model_state_dict(unwrapped_lm),
            args.output_dir / "adapter_model.safetensors",
        )
        (args.output_dir / "training_metadata.json").write_text(
            json.dumps(
                {
                    "pretrained_model_name": args.pretrained_model_name,
                    "manifest_path": str(args.manifest_path),
                    "num_examples": len(dataset),
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "sample_rate": int(audiogen.sample_rate),
                    "frame_rate": int(audiogen.frame_rate),
                    "target_modules": target_modules,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved LoRA adapter to %s", args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
