# Server B training commands (wind intensity bank)

## 0) Precondition

- Worktree on Server B: `~/murphy/COMP-6000-Capstone2`
- Python env: `acoustic_ai/.venv`

## 1) Build heavy manifest

```bash
cd ~/murphy/COMP-6000-Capstone2
acoustic_ai/.venv/bin/python \
  acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/code/build_wind_manifest.py \
  --intensity heavy \
  --max-contamination-score 0.28 \
  --max-per-recording 3 \
  --out acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/data/wind_manifest_heavy.csv
```

## 2) Train heavy adapter

```bash
cd ~/murphy/COMP-6000-Capstone2
acoustic_ai/.venv/bin/accelerate launch \
  acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/code/train_audioldm2.py \
  --pretrained_model_name cvssp/audioldm2 \
  --manifest_path acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/data/wind_manifest_heavy.csv \
  --output_dir model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/heavy \
  --batch_size 4 \
  --num_epochs 6 \
  --learning_rate 1e-5 \
  --lr_scheduler constant \
  --lr_warmup_steps 100 \
  --seed 42 \
  --mixed_precision fp16 \
  --max_duration_s 10.0 \
  --input_sample_rate 16000
```

## 3) Medium adapter source

Copy smoke_2 adapter into bank layout:

```bash
cd ~/murphy/COMP-6000-Capstone2
mkdir -p model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/medium
cp model/candidates/murphy/smoke_2__audioldm2_wind/adapter_config.json \
  model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/medium/
cp model/candidates/murphy/smoke_2__audioldm2_wind/adapter_model.safetensors \
  model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/medium/
```

Then `dvc add` heavy+medium adapter weights in the new checkpoint bank.
