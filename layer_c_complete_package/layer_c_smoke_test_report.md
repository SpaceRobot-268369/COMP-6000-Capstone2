# Layer C Smoke Test Report

## 1. Goal

生成符合 **季节 + 天气 + 时间 / 场景** 的 30 秒纯自然 / 动物环境声。要求真实、干净、符合场景，并且不包含人声、音乐、机械声、交通声或明显杂音。

---

## 2. Scenes

| Scene ID | Prompt | 中文定义 |
|---|---|---|
| summer_rain | summer rain afternoon | 夏日午后下雨 |
| winter_snow | winter snow night | 冬夜下雪 |
| forest_bird | forest birds morning | 清晨森林鸟鸣 |

---

## 3. Data Collection

| Scene | Raw Clips Target | Audited Clips Target | Actual Raw | Actual Passed |
|---|---:|---:|---:|---:|
| summer_rain | 20–30 | >= 16–24 | TBD | TBD |
| winter_snow | 20–30 | >= 16–24 | TBD | TBD |
| forest_bird | 20–30 | >= 16–24 | TBD | TBD |

Filtering policy: `layer_c_filter_policy_v1.md`

Raw folder:

```text
smoke_test/layer_c/raw_clips/
```

Audited folder:

```text
smoke_test/layer_c/audited_clips/
```

---

## 4. Human Audit Result

Audit log: `layer_c_audit_log.xlsx`

Pass condition for data audit:

> Passed clips / raw clips >= 80%

| Scene | Raw Count | Passed Count | Pass Rate | Result |
|---|---:|---:|---:|---|
| summer_rain | TBD | TBD | TBD | TBD |
| winter_snow | TBD | TBD | TBD | TBD |
| forest_bird | TBD | TBD | TBD | TBD |
| Total | TBD | TBD | TBD | TBD |

---

## 5. Candidate Models

Candidate model document: `layer_c_candidate_models.md`

| Model | Type | Result |
|---|---|---|
| AudioCraft AudioGen | Text-to-audio environmental sound | TBD |
| AudioLDM2 | Latent diffusion text-to-audio | TBD |
| Project Ambient Pipeline | Internal VAE / vocoder pipeline | TBD |

Selected model:

```text
TBD
```

Reason:

```text
TBD
```

---

## 6. Fine-tuning / Small Test Config

Input:

```text
scene text prompt
```

Output:

```text
30-second environmental audio
```

Training data:

```text
smoke_test/layer_c/audited_clips/
```

Model output:

```text
smoke_test/layer_c/fine_tuned_model/
```

Suggested small test config:

```text
epochs: 5–10
sample_rate: 22050
duration: 30 sec
batch_size: depends on GPU memory
```

---

## 7. Generation Test

Generated output folder:

```text
smoke_test/layer_c/generated/
```

Target:

- 5–10 generated clips per scene
- Total 15–30 generated clips

| Scene | Generated Count | Passed Count | Pass Rate |
|---|---:|---:|---:|
| summer_rain | TBD | TBD | TBD |
| winter_snow | TBD | TBD | TBD |
| forest_bird | TBD | TBD | TBD |
| Total | TBD | TBD | TBD |

---

## 8. Smoke Test Decision

Smoke test pass condition:

> Generated clip pass rate >= 70%

Decision:

```text
TBD: PASS / FAIL
```

Main problems found:

```text
TBD
```

Next action:

```text
TBD
```

---

## 9. MVP Plan

MVP target:

> Stable generation for 3 scenes: summer_rain, winter_snow, forest_bird.

Required interface:

```text
input: scene_id or scene prompt
output: generated .wav file path
```

Suggested API behavior:

```json
{
  "scene": "summer_rain",
  "prompt": "summer rain afternoon",
  "duration_sec": 30,
  "output_path": "smoke_test/layer_c/generated/summer_rain/gen_summer_rain_001.wav"
}
```
