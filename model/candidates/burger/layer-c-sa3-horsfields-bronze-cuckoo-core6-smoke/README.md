# Layer C SA3 Horsfield's Bronze-cuckoo Core6 Smoke

## Status

Prepared for LoRA training. This folder records the first Stable Audio 3 LoRA
smoke candidate dataset after the AudioGen LoRA route failed to produce stable
species-specific calls.

## Goal

Train a small Stable Audio 3 LoRA for Layer C event generation using verified
real Horsfield's Bronze-cuckoo clips, then evaluate whether SA3 LoRA plus
reference-conditioned variation produces more natural diversity than base SA3
reference variation alone.

## Base Model

- Model family: Stable Audio 3
- Base checkpoint: `small-sfx-base`
- Current environment: `acoustic_ai/.venv-audiogen`

Training uses the official Stable Audio 3 `scripts/train_lora.py` entrypoint
from the upstream repository. The local package exposes generation and LoRA
loading, but not the training script itself.

## Training Data

Manifest:

`acoustic_ai/data/events/layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke/sa3_lora_smoke_core6_train_manifest.csv`

Metadata JSONL:

`acoustic_ai/data/events/layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke/sa3_lora_smoke_core6_metadata.jsonl`

Official SA3 LoRA data directory:

`acoustic_ai/data/events/layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke/sa3_lora_core6_data`

This directory contains six `.wav` files, each with a matching
same-name `.txt` caption required by `train_lora.py`.

The six clips were selected from real audited source crops, not generated
audio. They first passed a reference-conditioned SA3 probe:

- Candidate pool: top 10 human-pass natural-core reference clips
- Probe: `small-sfx-base`, `init_noise_level=0.40`, `steps=8`, `cfg_scale=1.0`
- Retained criterion: each reference produced `3/3 Pass`
- Confirmation run: core6 references produced `30/30 Pass` at 5 seeds each

## Proposed Training Settings

Initial smoke settings are intentionally small:

- LoRA rank: 8
- LoRA alpha: 8
- LoRA dropout: 0.05
- Adapter: `dora-rows`
- Exclude: `seconds_total`
- Steps: smoke run starts at 100-300 steps, then scale if useful
- Batch size: 1
- Learning rate: 1e-4
- Target duration: 8 seconds

These are proposed parameters only. No LoRA weights have been produced yet.

## Training Command

Run on a CUDA GPU machine:

```bash
env MPLCONFIGDIR=/private/tmp/mpl ./acoustic_ai/.venv-audiogen/bin/python /private/tmp/stable-audio-3/scripts/train_lora.py \
  --model small-sfx-base \
  --data_dir acoustic_ai/data/events/layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke/sa3_lora_core6_data \
  --save_dir model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints \
  --name layer-c-sa3-bronze-cuckoo-core6-smoke \
  --adapter_type dora-rows \
  --rank 8 \
  --lora_alpha 8 \
  --dropout 0.05 \
  --lr 0.0001 \
  --steps 300 \
  --batch_size 1 \
  --duration 8 \
  --checkpoint_every 100 \
  --demo_every 999999 \
  --log_every 10 \
  --num_workers 0 \
  --logger none \
  --exclude seconds_total
```

Local CPU dry-run status: the script successfully loaded six clips, created
192 LoRA layers, and reported 5.6M trainable parameters. It was stopped during
CPU demo generation before a checkpoint was saved, because this machine has no
CUDA or MPS accelerator.

## Evaluation Plan

After training:

1. Generate 30 samples with the core6 reference bank.
2. Compare against base SA3 core6 generation at the same seeds/noise.
3. Manual audit for species correctness, foreground clarity, and natural
   variation.
4. Check whether outputs are less copy-like than base SA3 reference variation.

## Known Risk

Six clips are enough for a smoke LoRA, not a robust species model. If the smoke
run overfits or produces copy-like outputs, expand the verified reference bank
to 15-20 clips before the next LoRA attempt.
