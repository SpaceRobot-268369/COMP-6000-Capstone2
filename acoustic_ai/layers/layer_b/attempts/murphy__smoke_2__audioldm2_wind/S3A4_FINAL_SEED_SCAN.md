# S3a.4 最终轮执行记录（锁定 Variant A + 40 seed 扫描 + MVP 收口）

> 本文件是 `S3A4_FINAL_ROUND_PLAN.md` 的执行落地记录。
> 决策依据：人工对比 `showcase_s3a4a`（Variant A）与 `showcase_s3a4b`（Variant B）后，
> 用户拍板采用 **Variant A**，并按建议把最终挑选的 seed 扫描数量定为 **40**。

## 1) 锁定参数（Variant A）

降噪只取较温和的一档（floor 保持 0.40，避免压风声）：

| 参数 | 值 |
|---|---|
| `denoise_enabled` | `true` |
| `denoise_strength` | **0.15** |
| `denoise_floor_ratio` | **0.40** |
| `denoise_noise_quantile` | 0.2 |
| `denoise_hop_length` | 512 |

冻结的主参数（全部不动）：

| 参数 | 值 |
|---|---|
| `guidance_scale` | 3.0 |
| `num_inference_steps` | 200 |
| `audio_length_in_s` | 8.0 |
| `output_target_rms` | 0.06 |
| `highpass_hz` | 80 |
| `fade_ms` | 80 |
| `prompt` | steady wind through dry eucalyptus woodland, gentle natural breeze, Bowra, Australia |
| `negative_prompt` | hiss, static noise, background hum, tape noise, insects, low quality, distortion |

`params.yaml` 的 `inference.denoise_strength` 已由 `0.12` 改为 `0.15` 以固化此决策。

## 2) seed 数量决策

- 实测命中率：seed 42–51 中约 2–3 条达 MVP（48 / 50，45 接近）→ 约 20–30%。
- 目标黄金种子：4–6 条。
- 按 ~25% 命中率 + 安全裕度反推 → **扫描 40 个 seed**（42–81）。
- 预期可得约 8–10 条候选，足以挑出 4–6 条最佳且有余量。

## 3) 执行步骤

1. **锁参**：`params.yaml` → Variant A（已完成）。
2. **复用确定性结果**：生成是确定性的（同 seed + 同参数 + 同代码 = 同音频），
   且 Variant A 与 `showcase_s3a4a` 参数完全一致，故 **seed 42–51 直接复用** `showcase_s3a4a`，
   仅在 Server B 新生成 **seed 52–81（30 条）**。
3. **汇总目录**：所有 40 条（42–81）汇入
   `showcase_s3a4_final/seed_<N>_generated/{audio.wav, metadata.json}`。
4. **回传**：`scp` 新样本到本地。
5. **机器预排序**：对 40 条本地计算噪声/嘶声特征并打分排序（见下）。
6. **试听页**：按预排序生成 `showcase_s3a4_final/listen_generated.html`，供人工挑选。

## 4) 机器预排序指标（仅供排序参考，不替代人工）

对每条 `audio.wav` 计算：

- `hiss_ratio` = 4 kHz 以上能量 / 总能量（越低越好 —— 嘶声少）。
- `spectral_flatness`（频谱平坦度均值，越低越好 —— 越不像白噪声）。
- `lowmid_ratio` = 80 Hz–2 kHz 能量 / 总能量（风的主要能量区，适中偏高较好）。

综合分（越小越靠前）：`score = hiss_ratio + spectral_flatness − 0.5 * lowmid_ratio`。
此分仅用于把"可能更干净"的样本排到前面，最终以人工听感为准。

## 5) 回归保护

40 条里包含 `seed 45 / 48 / 50`（来自 `showcase_s3a4a`）。
人工挑选时先确认这 3 条没有退化（风声不变弱、无边界卡顿）。
因参数与 `showcase_s3a4a` 完全一致，理论上它们与该批完全相同。

## 6) 通过标准（与计划一致）

- 新增 ≥1–2 条达到 48/50 水平；
- 45/48/50 不退化；
- 总计 ≥4 条达到 MVP 可展示标准。

满足后进入收口：人工选定 4–6 条黄金 seed → 写入正式 `showcase/` → 更新 README/metadata → 提升至 MVP。

## 7) 人工审计最终入选（2026-06-04）

用户确认的好种子（6条）：

- `48`
- `50`
- `52`
- `55`
- `59`
- `72`

对应落盘记录见：
`S3A4_GOOD_SEEDS_SELECTION.md`
