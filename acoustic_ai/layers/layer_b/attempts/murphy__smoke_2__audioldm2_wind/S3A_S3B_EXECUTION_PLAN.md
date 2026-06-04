# Layer B Wind: S3a/S3b 执行记录与门控计划

## 背景

人工反馈结论（smoke_2 第一轮）：
- 电子感已明显消失（正向）
- 底噪偏重、风声主体不够突出（核心问题）
- 少量鸟/生物音可接受（无需强制清零）

## 用户授权边界

- 立即执行：S3a
- 若 S3a 不达标：自动执行 S3b
- S3c / S3d：需要用户再次确认，当前不执行

## S3a（已启动执行）

### 目标
在不重训前提下，先改善听感可分辨度并压制噪声感。

### 改动
- 增加 `negative_prompt` 支持（handler + sample）
- inference 参数调整：
  - `output_target_rms: 0.003 -> 0.06`
  - `highpass_hz: 60 -> 80`
  - `negative_prompt: hiss/static/background hum/tape noise/insects/low quality/distortion`

### 生成评估集
- 固定 checkpoint: `model/candidates/murphy/smoke_2__audioldm2_wind`
- 生成 10 条：seed 42-51
- 人工复听后判定

### S3a 通过标准
- 10 条中 >= 4 条“风声主体明确，底噪退后，可直接展示”
- 若未达标，进入 S3b

## S3b（条件触发）

### 触发条件
- S3a 人工反馈仍认为底噪偏重或风声不够突出

### 执行策略
- 新建 `murphy__smoke_3__audioldm2_wind`
- 对训练样本做温和降噪后重训（仅限本生成任务副本）
- 延续 S3a 中已验证的推理参数再评估

### S3b 通过标准
- 10 条中 >= 4 条达到可展示标准
- 仍不达标则暂停，等待用户确认是否进入 S3c/S3d

## 说明

本文件仅记录阶段策略与授权边界，不替代训练/审计日志。
