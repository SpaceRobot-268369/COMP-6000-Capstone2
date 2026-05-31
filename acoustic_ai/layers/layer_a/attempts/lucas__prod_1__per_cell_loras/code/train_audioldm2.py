"""Fine-tuning script for AudioLDM2 using LoRA.

This script fine-tunes the UNet of AudioLDM2 using Low-Rank Adaptation (LoRA),
while keeping the text encoders and VAE frozen. This allows for efficient
training on standard hardware.
"""


import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AudioLDM2Pipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from audioldm2_dataset import AudioLDM2Dataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AudioLDM2 using LoRA")
    parser.add_argument("--pretrained_model_name", type=str, default="cvssp/audioldm2")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Candidate dir, e.g. model/candidates/<member>/layer-a-audioldm2-<run-id>")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_duration_s", type=float, default=10.0)
    parser.add_argument("--input_sample_rate", type=int, default=16000)
    parser.add_argument("--model_sample_rate", type=int, default=None)
    parser.add_argument(
        "--normalize_audio",
        action="store_true",
        help="Mildly RMS-normalize training clips before AudioLDM2 feature extraction.",
    )
    parser.add_argument(
        "--no_normalize_audio",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--target_rms", type=float, default=0.005)
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank. Default 8 matches smoke/MVP-1.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha. Default 32 matches smoke/MVP-1.")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--val_eval_seed",
        type=int,
        default=12345,
        help="RNG seed for diffusion timesteps/noise during validation. Fixed "
             "so val loss is comparable across epochs.",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Skip val dataset / per-epoch val loss. Useful for manifests "
             "without a 'split' column (e.g. the smoke datasets).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    from transformers import GPT2LMHeadModel

    logger.info(f"Loading pipeline from {args.pretrained_model_name}")

    pipeline = AudioLDM2Pipeline.from_pretrained(
        args.pretrained_model_name,
        torch_dtype=weight_dtype,
    )
    model_sample_rate = args.model_sample_rate or pipeline.feature_extractor.sampling_rate

    pipeline.language_model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model_name,
        subfolder="language_model",
        torch_dtype=weight_dtype,
    )

    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    unet = pipeline.unet
    scheduler = pipeline.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    logger.info("Injecting LoRA into UNet...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=args.lora_dropout,
    )
    logger.info(f"LoRA config: r={args.lora_r} alpha={args.lora_alpha} "
                f"dropout={args.lora_dropout}")

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    pipeline.projection_model.to(accelerator.device, dtype=weight_dtype)
    pipeline.language_model.to(accelerator.device, dtype=weight_dtype)

    manifest_path = Path(args.manifest_path).resolve()
    project_root = manifest_path.parent.parent.parent.parent

    normalize_audio = args.normalize_audio and not args.no_normalize_audio

    train_split = None if args.no_val else "train"
    dataset = AudioLDM2Dataset(
        manifest_path=args.manifest_path,
        base_dir=project_root,
        max_duration_s=args.max_duration_s,
        target_sample_rate=args.input_sample_rate,
        normalize_audio=normalize_audio,
        target_rms=args.target_rms,
        split=train_split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_dataset = None
    val_dataloader = None
    if not args.no_val:
        try:
            val_dataset = AudioLDM2Dataset(
                manifest_path=args.manifest_path,
                base_dir=project_root,
                max_duration_s=args.max_duration_s,
                target_sample_rate=args.input_sample_rate,
                normalize_audio=normalize_audio,
                target_rms=args.target_rms,
                split="val",
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
            )
            logger.info(f"  Val dataset: {len(val_dataset)} clips")
        except ValueError as exc:
            logger.warning(
                f"No 'val' split found in manifest — skipping per-epoch val loss. "
                f"({exc})"
            )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    resampler = torchaudio.transforms.Resample(
        orig_freq=args.input_sample_rate,
        new_freq=model_sample_rate,
    ).to(accelerator.device)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Dataset sample rate = {args.input_sample_rate}")
    logger.info(f"  AudioLDM2 feature sample rate = {model_sample_rate}")
    logger.info(f"  Audio RMS normalization = {normalize_audio} (target={args.target_rms})")

    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    global_step = 0

    def step_loss(batch, generator=None, log_first_audio=False):
        """Run one diffusion training step on a batch and return the MSE loss.

        Shared between the training loop (no generator → fresh randomness +
        backprop) and the validation loop (generator with fixed seed →
        comparable val loss across epochs, no backprop).
        """
        with torch.no_grad():
            prompt_embeds, attention_mask, generated_prompt_embeds = pipeline.encode_prompt(
                batch["caption"],
                device=accelerator.device,
                num_waveforms_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        audio = batch["audio"].to(accelerator.device)
        if log_first_audio and accelerator.is_local_main_process:
            audio_rms = torch.sqrt(torch.mean(audio.float().square(), dim=1))
            audio_peak = audio.float().abs().amax(dim=1)
            logger.info(
                "  First batch audio stats after dataset preprocessing: "
                f"rms={audio_rms.detach().cpu().tolist()} "
                f"peak={audio_peak.detach().cpu().tolist()}"
            )

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        audio = resampler(audio)
        audio_arrays = audio.squeeze(1).detach().cpu().numpy()

        inputs = pipeline.feature_extractor(
            audio_arrays,
            sampling_rate=model_sample_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            accelerator.device,
            dtype=weight_dtype,
        )

        with torch.no_grad():
            latents = vae.encode(input_features).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        if generator is None:
            noise = torch.randn_like(latents)
        else:
            noise = torch.randn(
                latents.shape, generator=generator,
                device=latents.device, dtype=latents.dtype,
            )

        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            generator=generator,
        ).long()

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=generated_prompt_embeds,
            encoder_hidden_states_1=prompt_embeds,
            encoder_attention_mask_1=attention_mask,
            return_dict=False,
        )[0]

        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {scheduler.config.prediction_type}"
            )

        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    for epoch in range(args.num_epochs):
        unet.train()
        train_losses = []

        for step, batch in enumerate(dataloader):
            loss = step_loss(batch, log_first_audio=(global_step == 0))

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_losses.append(loss.detach().item())
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix({"loss": loss.detach().item()})

        mean_train = sum(train_losses) / max(1, len(train_losses))

        # Validation pass — fixed seed so the val_loss column is comparable
        # epoch-to-epoch.
        if val_dataloader is not None:
            unet.eval()
            val_generator = torch.Generator(device=accelerator.device).manual_seed(
                args.val_eval_seed
            )
            val_losses = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_loss = step_loss(val_batch, generator=val_generator)
                    val_losses.append(val_loss.detach().item())
            mean_val = sum(val_losses) / max(1, len(val_losses))
            logger.info(
                f"Epoch {epoch + 1}/{args.num_epochs}: "
                f"train_loss={mean_train:.4f}  val_loss={mean_val:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{args.num_epochs}: train_loss={mean_train:.4f}"
            )

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        logger.info(f"Saved LoRA weights to {args.output_dir}")


if __name__ == "__main__":
    main()
