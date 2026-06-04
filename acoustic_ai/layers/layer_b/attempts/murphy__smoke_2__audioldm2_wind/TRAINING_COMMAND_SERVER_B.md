# smoke_2 Server B 训练命令

在 Server B 仓库根目录执行：

```bash
cd ~/murphy/COMP-6000-Capstone2
acoustic_ai/.venv/bin/python   acoustic_ai/layers/layer_b/attempts/murphy__smoke_2__audioldm2_wind/code/train_audioldm2.py   --pretrained_model_name cvssp/audioldm2   --manifest_path acoustic_ai/layers/layer_b/attempts/murphy__smoke_2__audioldm2_wind/data/wind_manifest.csv   --output_dir model/candidates/murphy/smoke_2__audioldm2_wind   --batch_size 4   --num_epochs 6   --learning_rate 1.0e-5   --lr_scheduler constant   --lr_warmup_steps 100   --seed 42   --mixed_precision fp16   --max_duration_s 10.0   --input_sample_rate 16000   --target_rms 0.005
```

## manifest 筛选策略（已落地）

- contamination <= 0.28
- 仅保留 wind_intensity=medium
- 排除 quality_flags 包含 `nov2019_storm_scout001`
- 每个 `source_recording_id` 最多保留 3 条

当前 manifest 统计：
- 样本数：35
- 强度分布：100% medium
- 录音来源：24
- contamination min/median/max: 0.203 / 0.252 / 0.279
