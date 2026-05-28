# Layer C SA3 LoRA Cloud Handoff

## Summary

This package contains the minimum files needed to run the Layer C Stable Audio 3
LoRA smoke training job for Horsfield's Bronze-cuckoo on a CUDA GPU worker.

## Integration Notes

- This is a Layer C-specific training entrypoint. It should not change the
  generic Server A/B job API or worker loop used by other layers.
- The handoff uses an isolated `acoustic_ai/.venv-audiogen` environment to avoid
  changing dependencies for other layers. The repository's general AI
  convention is `acoustic_ai/.venv`, so confirm the environment choice before
  installing Stable Audio 3 dependencies on a shared machine.
- The handoff's runtime estimate references an NVIDIA A10G. The current Server B
  GPU must be verified with `nvidia-smi` before training, because an earlier
  `shinypokemon` check showed a Tesla T4 with 15 GB VRAM.
- Audio `.wav` inputs are dataset artifacts. They are tracked through DVC
  pointers in git and stored in the shared S3 DVC cache.

## Layer

Layer C - Event / bird-call generation.

## Owner

burger / burgeryang.

## Training Script

Repository script:

```text
script/events/train_sa3_lora_core6_smoke.sh
```

The shell script does not call a project-local Python training file. It calls
the official Stable Audio 3 upstream training script:

```text
${SA3_REPO}/scripts/train_lora.py
```

All project-local files needed by the shell script are included in this
handoff package:

- `script/events/train_sa3_lora_core6_smoke.sh`
- `script/events/requirements_sa3_lora.txt`
- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/sa3_lora_core6_data/`
- `model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/README.md`
- `model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/params.yaml`

## Stable Audio 3 Upstream Dependency

Clone the upstream repo on the GPU machine:

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git /home/ubuntu/stable-audio-3
cd /home/ubuntu/stable-audio-3
git checkout fa5ee841dd49bae0fa361fac26904adc27fd400e
```

Set:

```bash
export SA3_REPO=/home/ubuntu/stable-audio-3
```

Install the Python package and training dependencies:

```bash
cd /home/ubuntu/COMP-6000-Capstone2
python3 -m venv acoustic_ai/.venv-audiogen
acoustic_ai/.venv-audiogen/bin/python -m pip install --upgrade pip setuptools wheel
acoustic_ai/.venv-audiogen/bin/python -m pip install -r script/events/requirements_sa3_lora.txt
```

If the environment already has PyTorch/CUDA installed, keep the working torch
build as long as `torch.cuda.is_available()` returns `True`.

Verify GPU:

```bash
acoustic_ai/.venv-audiogen/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Hugging Face Access

Stable Audio 3 requires Hugging Face access to the Stability AI model weights.

Required access:

- Stability AI Stable Audio 3 model family, including `small-sfx-base`.

Do not share the Hugging Face token in chat or commit it to the repo. Log in on
the GPU worker:

```bash
acoustic_ai/.venv-audiogen/bin/hf auth login
```

Then verify:

```bash
acoustic_ai/.venv-audiogen/bin/hf auth whoami
```

## Dataset Input

Training data directory:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/sa3_lora_core6_data
```

Expected contents:

- six `.wav` files;
- six matching `.txt` caption files with identical basenames;
- `README.md`.

The `.wav` files are real audited Horsfield's Bronze-cuckoo clips. They are not
generated audio. In this handoff package the audio files are packed as real WAV
files, not symlinks.

The official `train_lora.py` reads raw audio plus matching `.txt` captions from
this directory.

## Base Model

```text
Stable Audio 3 small-sfx-base
```

## Foreground Command

Use this in worker/job-runner mode. It runs in the foreground and exits when
training finishes:

```bash
cd /home/ubuntu/COMP-6000-Capstone2

SA3_REPO=/home/ubuntu/stable-audio-3 \
MPLCONFIGDIR=/tmp/mpl \
SA3_STEPS=300 \
SA3_CHECKPOINT_EVERY=100 \
SA3_DEMO_EVERY=999999 \
bash script/events/train_sa3_lora_core6_smoke.sh
```

## Background Command

Use this only for manual SSH sessions where the process must survive disconnect:

```bash
cd /home/ubuntu/COMP-6000-Capstone2
mkdir -p logs

SA3_REPO=/home/ubuntu/stable-audio-3 \
MPLCONFIGDIR=/tmp/mpl \
SA3_STEPS=300 \
SA3_CHECKPOINT_EVERY=100 \
SA3_DEMO_EVERY=999999 \
nohup script/events/train_sa3_lora_core6_smoke.sh \
  > logs/sa3_lora_core6_train_300.log 2>&1 &
```

## Required Args / Environment Variables

- `SA3_REPO=/home/ubuntu/stable-audio-3`
- `MPLCONFIGDIR=/tmp/mpl`
- `SA3_STEPS=300`
- `SA3_CHECKPOINT_EVERY=100`
- `SA3_DEMO_EVERY=999999`
- optional: `SA3_NUM_WORKERS=0`
- optional: `PYTHON=/home/ubuntu/COMP-6000-Capstone2/acoustic_ai/.venv-audiogen/bin/python`

## Training Output

Checkpoint directory:

```text
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/
```

Expected checkpoint files:

```text
epoch=*-step=100.ckpt
epoch=*-step=200.ckpt
epoch=*-step=300.ckpt
```

The official training wrapper saves LoRA-only state in Lightning `.ckpt`
format. Optional conversion to `.safetensors` can be done after training using
Stable Audio 3's LoRA utility if the inference path requires safetensors.

Log file for the background command:

```text
logs/sa3_lora_core6_train_300.log
```

Expected metrics file:

```text
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/metrics.json
```

This is not generated by the training command because current training uses
`--logger none`. Create it after evaluation from manual audit and/or automatic
similarity checks.

## Verified 10-Step Server B Smoke

Verified on 2026-05-28 on Server B `shinypokemon`.

This was a real Stable Audio 3 LoRA training run, not the fake worker training
adapter. It proves that Server B can pull Layer C data through DVC/S3, access
the Stable Audio 3 base model through Hugging Face, use CUDA, and write a LoRA
checkpoint.

Environment:

```text
GPU: Tesla T4, 15360 MiB VRAM
torch: 2.7.1+cu126
CUDA available: true
SA3 repo: /home/ubuntu/stable-audio-3
SA3 commit: fa5ee841dd49bae0fa361fac26904adc27fd400e
Python env: acoustic_ai/.venv-audiogen
```

Command:

```bash
SA3_REPO=/home/ubuntu/stable-audio-3 \
PYTHON=/home/ubuntu/COMP-6000-Capstone2-worker/acoustic_ai/.venv-audiogen/bin/python \
MPLCONFIGDIR=/tmp/mpl \
SA3_STEPS=10 \
SA3_CHECKPOINT_EVERY=10 \
SA3_DEMO_EVERY=999999 \
SA3_NUM_WORKERS=0 \
bash script/events/train_sa3_lora_core6_smoke.sh \
  2>&1 | tee logs/sa3_lora_core6_train_10.log
```

Observed output:

```text
GPU available: True (cuda), used: True
Found 6 files
lora layers: 192
5.6 M trainable params
Trainer.fit stopped: max_steps=10 reached
```

Artifact:

```text
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/epoch=1-step=10.ckpt
model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/epoch=1-step=10.ckpt.dvc
```

The real checkpoint was uploaded with DVC. The `.dvc` pointer is committed on
`infra/songke/server-a-deployment` at commit:

```text
8d81066 data: track layer c sa3 10-step smoke checkpoint
```

The 300-step run was intentionally not started in this handoff. Treat 300-step
training as Layer C model-quality validation, not as required infrastructure
evidence for Server A/B.

## Verified Automatic Server A/B 10-Step Smoke

Verified on 2026-05-28 through the Server A job queue and Server B worker.

This was a real Stable Audio 3 LoRA training job started from Server A and
executed by Server B, not a manual SSH foreground command.

Server B commit:

```text
49aed64 fix: run sa3 training with configured python
```

Server A job:

```text
job id: 12
type: training
status: completed
payload.layer: C
payload.training_backend: sa3_lora
payload.run_id: layer-c-sa3-worker-smoke-12
payload.steps: 10
```

Observed worker result:

```text
GPU available: True (cuda), used: True
Trainer.fit stopped: max_steps=10 reached
job 12: completed checkpoint_uri=model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints/epoch=1-step=10-v1.ckpt.dvc
```

Server A API check returned `status = completed` for job 12.

Server B was shut down after verification to avoid GPU budget usage.

## Success Criteria

Training success:

- command exits with code `0`;
- checkpoint directory contains `step=300` checkpoint;
- log contains no traceback or CUDA OOM;
- GPU was used during training.

Model-selection rule:

- evaluate `step=300` first;
- if output quality is worse or copy-like, compare `step=100` and `step=200`;
- keep the checkpoint with best manual audit quality, not necessarily the last
  checkpoint.

Generation/evaluation success:

- generate a 10-30 sample audit set after training;
- compare against the validated base SA3 core6 reference-conditioned samples;
- pass requires target-like Horsfield's Bronze-cuckoo call morphology and
  natural variation without drifting into generic bird sound.

## Runtime And VRAM

Expected runtime on NVIDIA A10G:

- first run with model download: about 15-45 minutes;
- cached rerun: roughly 10-30 minutes for 300 steps.

Expected VRAM:

- minimum: about 4-8 GB;
- recommended: 12 GB+;
- validated worker: NVIDIA A10G, about 23 GB VRAM.

Observed Server B smoke hardware:

- `shinypokemon` currently reported Tesla T4 with 15 GB VRAM;
- 10-step SA3 LoRA smoke completed successfully on T4;
- 300-step runtime and quality are not yet validated on T4.

## Important Notes

- Do not commit Hugging Face tokens or AWS credentials.
- Model binaries (`*.ckpt`, `*.safetensors`) should be DVC-tracked before they
  are pushed back to GitHub.
- The current data package is a smoke dataset with six real reference clips. If
  LoRA overfits or copies too strongly, expand the manually audited reference
  bank to 15-20 clips before the next training run.
