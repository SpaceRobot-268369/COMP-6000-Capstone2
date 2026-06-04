# Layer B Wind: S3a.2 方案调整与落地记录

## 新的人工审计结论（基于 S3a）

- 04/07/09 样本改善明显（底噪降低、风声明确）
- 其余样本噪声虽下降，但仍偏大，风声主体不够突出
- 电子感问题已不再是主矛盾

## 方案调整（相对原 S3a/S3b）

1. S3a 判定改为“部分通过”，参数作为后续固定基线保留。
2. 新增 S3a.2：输出端轻量降噪（零重训）作为 S3b 之前的低成本尝试。
3. 若 S3a.2 仍不达标，自动进入 S3b（你已授权）。
4. S3c/S3d 仍需你再次确认，不自动执行。

## S3a.2 实施细节（本次落地）

- 在 `handler.py` 增加 STFT 轻量降噪后处理 `_spectral_denoise`：
  - 先高通，再降噪，再做目标 RMS 匹配
  - 降噪强度可配置、可关闭
- 在 `params.yaml` inference 增加：
  - `denoise_enabled: true`
  - `denoise_strength: 0.25`
  - `denoise_noise_quantile: 0.2`
  - `denoise_floor_ratio: 0.15`

## 门控标准（S3a.2）

- 10 条中 >=4 条达到“风声主体明确、底噪退后、可展示” -> 通过
- 若出现水声/抽气/musical noise 或整体仍底噪偏重 -> 进入 S3b

## 执行边界

- 当前自动执行范围：S3a.2 -> S3b
- S3c/S3d：暂停并等待用户确认
