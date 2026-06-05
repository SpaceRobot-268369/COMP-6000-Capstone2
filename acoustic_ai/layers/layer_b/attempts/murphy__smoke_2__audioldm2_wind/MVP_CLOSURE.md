# Layer B smoke_2 MVP 收口记录

日期：2026-06-04
尝试：`murphy__smoke_2__audioldm2_wind`

## 收口目标

基于 `showcase_s3a4_final` 的 40-seed 扫描结果，落地正式 MVP 展示集并固定试听入口。

## 最终入选（用户确认）

- `48`
- `50`
- `52`
- `55`
- `59`
- `72`

## 收口动作

1. 将上述 6 条在其 `metadata.json` 中写入 `human_audit: approved_good_seed`。
2. 新增人工审计记录：`S3A4_GOOD_SEEDS_SELECTION.md`。
3. 更新正式试听入口：`showcase/listen_generated.html`
   - 从旧的 10 条 smoke 初样，切换为 6 条 MVP 好种子。
   - 音频源统一指向 `showcase_s3a4_final/seed_<N>_generated/audio.wav`。
4. 更新 attempt 说明：`README.md`
   - 标记该轮已收口；
   - 固化最终参数与好种子列表；
   - 保留 40-seed 扫描目录作为追溯档案。

## 当前状态

- MVP showcase 已可直接试听（6条）：
  `showcase/listen_generated.html`
- 完整扫描档案仍保留：
  `showcase_s3a4_final/`
