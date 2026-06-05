# S3a.4 人工审计结果：好种子确认

日期：2026-06-04
轮次：`showcase_s3a4_final`（Variant A, strength=0.15, floor=0.40）

## 用户确认的好种子

- `seed_48`
- `seed_50`
- `seed_52`
- `seed_55`
- `seed_59`
- `seed_72`

## 落盘标记

以上 6 条样本均已在各自 `metadata.json` 中写入：

```json
"human_audit": {
  "status": "approved_good_seed",
  "round": "s3a4_final",
  "selected_by_user": true,
  "notes": "User-selected good seed"
}
```

对应路径：

- `showcase_s3a4_final/seed_48_generated/metadata.json`
- `showcase_s3a4_final/seed_50_generated/metadata.json`
- `showcase_s3a4_final/seed_52_generated/metadata.json`
- `showcase_s3a4_final/seed_55_generated/metadata.json`
- `showcase_s3a4_final/seed_59_generated/metadata.json`
- `showcase_s3a4_final/seed_72_generated/metadata.json`
