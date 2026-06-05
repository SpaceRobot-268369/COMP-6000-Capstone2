# S3a.4 最后一轮方案（平衡降噪 + MVP 收口）

## 当前结论

- seed 48（第7条）与 seed 50（第9条）已接近/达到 MVP 水平。
- seed 45（第4条）噪声也明显更低。
- S3a.2 证明激进降噪会压风声，S3a.3 已回到较平衡状态。

## 最后一轮目标

在不破坏已优样本（48/50/45）的前提下，再小幅降低其余样本噪声。

## 执行方案

### 1) 微调两档（只动 denoise，不动其余主参数）

- Variant A:
  - denoise_strength: 0.15
  - denoise_floor_ratio: 0.40
- Variant B:
  - denoise_strength: 0.15
  - denoise_floor_ratio: 0.35

固定不变：
- guidance_scale: 3.0
- num_inference_steps: 200
- output_target_rms: 0.06
- highpass_hz: 80
- negative_prompt 保持
- fade_ms: 80

### 2) 保护门（回归保护）

每个变体都必须检查 seeds 45/48/50：
- 风声不得明显变弱
- 不得出现边界卡顿回归

### 3) 扩展 seed 筛选

在优胜变体上扩展到 seed 42-61（20条）进行人工挑选。

## 通过标准

- 新增至少 1-2 条达到 48/50 类似水平；
- 且 45/48/50 不退化；
- 总计 >=4 条达到 MVP 可展示标准。

## 失败分支

若 Variant A/B 都导致 45/48/50 退化：
- 回退到 S3a.3（strength=0.12, floor=0.40）作为最终版本；
- 直接做扩展 seed 筛选收口，不再继续加重降噪。
