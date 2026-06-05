# Stage A 简化 + S3a.3 执行方案记录

## 关键反馈确认

用户反馈：S3a（无降噪）批次无卡顿；卡顿仅出现在带降噪版本。

结论：卡顿来自降噪重建边界，而非生成本身不稳定。

## 阶段 A（简化版）

不再排查生成/高通来源，直接修复降噪重建：

1. STFT/ISTFT 使用可重建配置（COLA 友好）
2. 采用 padded 边界并避免末尾硬补零
3. 首尾添加短淡入淡出（50-80ms）作为 click 兜底

## S3a.3 目标

在 S3a（风声明显但噪音偏大）与 S3a.2（噪音低但风被压）之间找平衡。

### 参数扫描（仅推理后处理，不重训）

- denoise_strength: 0.08 / 0.12 / 0.16
- denoise_floor_ratio: 0.40（保护风声主体）
- 其他保持 S3a 基线：
  - guidance_scale: 3.0
  - num_inference_steps: 200
  - output_target_rms: 0.06
  - highpass_hz: 80
  - negative_prompt 保持启用

## S3a.3 门控

10 条中 >=4 条满足：
- 风声主体清晰
- 底噪可接受
- 无明显开头/结尾卡顿

若未达标，按已授权进入 S3b；S3c/S3d 仍需用户确认。
