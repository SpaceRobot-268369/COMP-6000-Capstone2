# Layer C Filtering Policy v1

## 1. Goal

生成符合 **[季节 + 天气 + 时间 / 场景]** 的纯自然声 / 动物声。输出音频必须真实、干净、可辨认，并且不能包含语音、音乐、机械声、交通声或明显杂音。

Layer C 只负责这一条线：

> 输入：天气 / 季节 / 场景文本  
> 输出：匹配场景的纯环境声（动物声 / 自然声）

---

## 2. Scope

### 允许的声音类型

只允许以下自然 / 动物环境声音：

- 动物声：鸟鸣、昆虫声、蛙鸣、远处自然动物声
- 天气声：雨声、风声、雷声、雪地风声
- 自然环境声：溪流、水流、树叶声、森林环境声

### 禁止的声音类型

出现以下任意内容，直接判为不合格：

- 人声、说话声、笑声、咳嗽声、脚步声
- 音乐、旋律、节拍、鼓点、唱歌
- 机械声、发动机、空调、电器噪声
- 交通声、车声、飞机声、火车声、船声
- 枪声、爆炸声、警笛、施工声
- 明显电流声、爆音、削波、失真、底噪过大
- 标注缺失、场景不清楚、文件损坏、波形异常

---

## 3. Scene Definitions

### Scene 1: `summer_rain`

- Prompt: `summer rain afternoon`
- 中文定义：夏日午后下雨
- 允许声音：稳定雨声、树叶被雨打声、远处自然雷声
- 不允许声音：人声、车声、室内电器声、音乐、城市街道声
- 理想效果：真实户外夏季雨声，30 秒内稳定、无突兀干扰

### Scene 2: `winter_snow`

- Prompt: `winter snow night`
- 中文定义：冬夜下雪
- 允许声音：冷风声、轻微风雪声、雪地自然环境声
- 不允许声音：人走雪地脚步声、车声、铲雪机、室内炉火、人声、音乐
- 理想效果：安静、寒冷、夜晚感明显，不能像机械白噪声

### Scene 3: `forest_bird`

- Prompt: `forest birds morning`
- 中文定义：清晨森林鸟鸣
- 允许声音：鸟鸣、树叶声、轻风、远处溪流
- 不允许声音：人声、狗叫、交通声、音乐、过强昆虫噪声
- 理想效果：清晨森林氛围明显，鸟声自然，背景干净

---

## 4. Clip Requirements

每条音频必须满足：

- 格式：`.wav` 优先
- 时长：25–35 秒，统一裁剪为 30 秒
- 采样率：优先统一为 22050 Hz
- 声道：mono 或 stereo 均可，但训练前建议统一 mono
- 单场景数量：20–50 条
- smoke test 推荐数量：每个场景 20–30 条，总计 60–90 条

---

## 5. Filtering Rules

### Hard Pass 条件

音频进入 `raw_clips` 前必须满足：

1. 文件能正常读取。
2. 裁剪后长度为 30 秒。
3. 内容属于目标场景。
4. 没有人声、音乐、机械、交通、枪声等禁止内容。
5. 没有明显削波、爆音、强电流声或损坏。
6. 文件名能看出场景和编号。

### Hard Reject 条件

出现以下任意情况，直接删除或放入 rejected：

1. 任何人声或类似人声。
2. 任何音乐或节奏性旋律。
3. 任何车辆、机器、电器、施工声。
4. 录音太脏，底噪掩盖主要自然声。
5. 场景不匹配，例如 forest_bird 里只有雨声。
6. 时长不足 25 秒，无法裁剪 30 秒。
7. 波形明显异常，例如全静音、严重削波、文件损坏。

---

## 6. Human Audit Standard

人工审核必须逐条听完整 30 秒。

审核表字段：

- filename
- scene
- duration_sec
- pass_fail
- reason
- notes
- auditor
- audit_date

### 合格标准

一条音频合格必须同时满足：

- 真实：像真实录音或高质量自然声
- 干净：没有明显杂音、人声、音乐、机械声
- 符合场景：能听出目标天气 / 季节 / 场景
- 稳定：30 秒内没有突兀干扰

### smoke test 数据通过线

- 原始筛选后人工 Audit 合格率 ≥ 80%
- 生成测试后二次 Audit 合格率 ≥ 70%

---

## 7. Folder Convention

```text
smoke_test/layer_c/
├── raw_clips/
│   ├── summer_rain/
│   ├── winter_snow/
│   └── forest_bird/
├── audited_clips/
│   ├── summer_rain/
│   ├── winter_snow/
│   └── forest_bird/
├── generated/
│   ├── summer_rain/
│   ├── winter_snow/
│   └── forest_bird/
└── fine_tuned_model/
```

---

## 8. Naming Convention

原始筛选片段：

```text
summer_rain_001.wav
winter_snow_001.wav
forest_bird_001.wav
```

生成片段：

```text
gen_summer_rain_001.wav
gen_winter_snow_001.wav
gen_forest_bird_001.wav
```
