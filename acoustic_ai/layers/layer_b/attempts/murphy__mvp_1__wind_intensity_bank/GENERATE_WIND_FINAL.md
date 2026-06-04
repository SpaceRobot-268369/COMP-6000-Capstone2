# Layer B — Generate Wind（最终封板版）

日期：2026-06-04  
作者：murphy  
状态：**SEALED** — 生产候选运行时配置，人工试听已签收  

本文档是 Layer B **风场生成（generate · wind）** 的单页留存：从 smoke 迭代到强度分档 bank 的完整结论、参数、产物路径与复现方式。  
运行时权威来源：`params.yaml` + `acoustic_ai/registry.yaml`（`intensity_profile_version: sealed`）。

---

## 1. 系统角色

| 项 | 值 |
|---|---|
| 模式 | **Generation** — 合成风层 stem，非检索池 |
| 层 | `layer_b`（Weather） |
| 注册 attempt | `murphy__mvp_1__wind_intensity_bank` |
| 基座模型 | `cvssp/audioldm2` |
| 路由 | `weather_type=wind` + `wind_intensity ∈ {light, medium, heavy}` |
| 用户可调（Dev UI / API） | **仅 `seed`**（0–2147483647）+ **强度三档** |
| 服务端锁定 | prompt、步数、guidance、后处理、adapter 选择 |

下游 Layer D 混音时，本层输出为 **8 s · 16 kHz** 风 stem（经 handler 后处理）。

---

## 2. 演进 lineage（简表）

```
murphy__smoke_1__audioldm2_wind     → 首训 LoRA（失败：lr 过高等，见 SMOKE_1_FAILURE_AND_SMOKE_2_PLAN.md）
murphy__smoke_2__audioldm2_wind     → 收口单档风；Variant A 后处理；6 条 MVP 好种子
murphy__mvp_1__wind_intensity_bank → 三档强度 bank（medium/heavy LoRA + derived light）
```

| 阶段 | 目录 / checkpoint | 结果 |
|------|-------------------|------|
| smoke_2 | `model/candidates/murphy/smoke_2__audioldm2_wind/` | 单 adapter；`showcase/` 6 好种子 |
| mvp_1 bank | `model/candidates/murphy/mvp_1__wind_intensity_bank/adapters/{medium,heavy}/` | medium 源自 smoke_2；heavy 单独训 |

---

## 3. 封板强度配置（运行时）

### 3.1 汇总

| 档位 | 来源 | Adapter | guidance | target RMS | highpass | lowpass | denoise (strength / floor) |
|------|------|---------|----------|------------|----------|---------|---------------------------|
| **light** | v2 **light_a** | derived → `medium` | 2.5 | 0.045 | 80 Hz | **6000 Hz** | 0.22 / 0.30 |
| **medium** | v3 | `medium` | 3.0 | 0.048 | 80 Hz | — | 0.19 / 0.38 |
| **heavy** | v2 冻结 | `heavy` | 3.4 | 0.09 | 80 Hz | — | 0.15 / 0.40 |

共享推理默认值：

- `num_inference_steps`: **200**
- `audio_length_in_s`: **8.0**
- `fade_ms`: **80**
- 共享 **negative_prompt**：`hiss, static noise, background hum, tape noise, insects, low quality, distortion`
- light **不**使用 v3 light_c 的「排斥强风」扩展 negative

### 3.2 锁定 prompt

**medium**

```text
steady wind through dry eucalyptus woodland, gentle natural breeze, Bowra, Australia
```

**heavy**

```text
loud roaring blustery wind through dry eucalyptus woodland, intense strong gusts, Bowra, Australia
```

**light**（derived，同 v2 轻风措辞）

```text
gentle light breeze through dry eucalyptus woodland, soft faint airflow, Bowra, Australia
```

### 3.3 封板决策记录（听感）

- **light**：v2 A/B 后选用 **light_a**（风可辨；接受相对闷/差麦感，优于 light_b 的噪与「像 medium」）
- **medium**：v3 加强 denoise，缓解部分 seed 噪感明显
- **heavy**：v2 eval 好样本占比高，**不再调参**

详细迭代见：`INTENSITY_V2_AB_PLAN.md`、`INTENSITY_V3_CLOSURE_PLAN.md`、`INTENSITY_SEALED.md`。

---

## 4. 后处理链（handler）

实现：`code/handler.py`（`audioldm2_lora_layer_b_wind_intensity_bank`）

顺序（每档按 profile 覆盖默认）：

1. AudioLDM2 + LoRA 扩散生成（`seed` → `torch.Generator`）
2. High-pass（sub-bass 清理）
3. 可选 Low-pass（**仅 light**：6 kHz）
4. 谱减 denoise（strength / floor / quantile 0.2）
5. Fade 80 ms
6. RMS 归一化到 `output_target_rms`

元数据字段：`intensity_profile_version: sealed`（及完整 `postprocess` 统计）。

---

## 5. Checkpoint 与数据

```
model/candidates/murphy/mvp_1__wind_intensity_bank/
├── README.md
├── params.yaml                    # 与 attempt 同步快照
└── adapters/
    ├── medium/
    │   ├── adapter_config.json    # git
    │   └── adapter_model.safetensors  # DVC / Server B 本地，不进 git
    └── heavy/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

- **medium** 权重：自 `smoke_2__audioldm2_wind` 迁入 bank 布局  
- **heavy** 权重：`wind_manifest_heavy.csv` 训练（见 `TRAINING_COMMAND_SERVER_B.md`）  
- **light**：无独立 LoRA；`derived_from: medium`

训练超参见 attempt `params.yaml` 的 `training:` 段（heavy 侧 manifest）。

---

## 6. 注册与 API 透传

**Registry**（`acoustic_ai/registry.yaml` → `layer_b` → `murphy__mvp_1__wind_intensity_bank`）：

- `uses_seed: true`
- `default_intensity: medium`
- `intensity_profiles`: 上表三档

**调用链**

```
Frontend (LayerATestPage)  →  wind_intensity + seed
Backend (Express)          →  转发 wind_intensity
FastAPI (acoustic_ai/server) → registry.generate(layer_b, attempt_id, ...)
Handler                    →  load(bank_root) + generate(...)
```

无效或缺失强度 → 回退 `medium`。

---

## 7. 如何生成

### 7.1 Dev UI（本地）

1. Docker：`frontend` + `backend` + `ai-tunnel`（见 `.claude/context/setup/local/services.md`）
2. Layer 测试页：选 **layer_b** → **Wind intensity bank**
3. 选 `light | medium | heavy`，输入 `seed`，Generate

### 7.2 Server B 批量（eval 脚本）

```bash
cd ~/murphy/COMP-6000-Capstone2
./acoustic_ai/.venv/bin/python \
  acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/dev-artifacts-self-testing/run_intensity_v3_eval.py
```

（脚本按 registry 当前 sealed profile 生成；改 profile 后需重新跑批。）

### 7.3 单条 Python（registry）

```python
from server import registry
out = registry.generate(
    "layer_b",
    "murphy__mvp_1__wind_intensity_bank",
    seed=48,
    wind_intensity="heavy",
    weather_type="wind",
)
# out["wav_bytes"], out["metadata"]
```

---

## 8. 评估与试听档案（非运行时）

| 目录 | 内容 |
|------|------|
| `showcase_intensity_eval/` | v1 三档初评（seed 42–51） |
| `showcase_intensity_eval_v2/` | light_a / light_b + v2 medium/heavy |
| `showcase_intensity_eval_v3/` | v3 medium/heavy + light_c（**light 已弃用，封板用 v2 light_a**） |

**封板试听对照（已生成 wav）**

- light → `showcase_intensity_eval_v2/light_a/`
- medium → `showcase_intensity_eval_v3/medium/`
- heavy → `showcase_intensity_eval_v3/heavy/`

本地 HTML：

- v3 三列页：`showcase_intensity_eval_v3/listen_intensity_compare_v3.html`（light 列为 light_c，仅作历史）
- v2 四列页：`showcase_intensity_eval_v2/listen_intensity_compare_v2.html`

**smoke_2 单档 MVP（与 medium adapter 同源）**

- 6 好种子：`48, 50, 52, 55, 59, 72`
- 入口：`murphy__smoke_2__audioldm2_wind/showcase/listen_generated.html`
- 档案：`showcase_s3a4_final/`（seed 42–81 扫描）

---

## 9. smoke_2 后处理（medium 训练基底，供追溯）

`murphy__smoke_2__audioldm2_wind` 锁定 **Variant A**：

- `denoise_strength: 0.15`
- `denoise_floor_ratio: 0.40`
- `highpass_hz: 80`
- `output_target_rms: 0.06`（bank 封板后 medium 改为 0.048）

---

## 10. 已知限制与后续

| 限制 | 说明 |
|------|------|
| light 无真 LoRA | 站点干净 light clip 极少；用 medium + light_a 后处理 |
| 16 kHz | AudioLDM2 原生率；宽带风噪受声码器约束 |
| seed 敏感 | 坏 seed 仍会噪/闷；smoke_2 有 6 条好种子，bank 侧以强度分档为主 |
| 权重 DVC | `adapter_model.safetensors` 需 `dvc pull`，勿提交 git |

可选后续（**不在封板范围**）：

- 收集 light 数据 → 真 `light` adapter  
- 将 sealed 三档 + 精选 seed 写入正式 `showcase/`  
- Layer D 混音端到端验收  

---

## 11. 相关文档索引

| 文档 | 路径 |
|------|------|
| 封板参数简表 | `INTENSITY_SEALED.md` |
| v2 A/B 计划 | `INTENSITY_V2_AB_PLAN.md` |
| v3 收口计划 | `INTENSITY_V3_CLOSURE_PLAN.md` |
| 实现日志 | `IMPLEMENTATION_LOG.md` |
| Attempt README | `README.md` |
| smoke_2 MVP 收口 | `../murphy__smoke_2__audioldm2_wind/MVP_CLOSURE.md` |
| smoke_2 → bank 计划 | `../murphy__smoke_2__audioldm2_wind/WIND_INTENSITY_BANK_PLAN.md` |
| Server B 训练 | `TRAINING_COMMAND_SERVER_B.md` |
| 模型 card | `model/candidates/murphy/mvp_1__wind_intensity_bank/README.md` |

---

## 12. 变更日志

| 日期 | 事件 |
|------|------|
| 2026-06-04 | smoke_2 MVP 6 好种子收口 |
| 2026-06-04 | mvp_1 intensity bank；v1/v2/v3 eval |
| 2026-06-04 | **SEALED**：light=light_a，medium=v3，heavy=v2；本文档发布 |

---

*Generate-wind 最终封板 — 后续调参以新 attempt 或显式 promotion 为准，勿静默覆盖 `sealed` profile。*
