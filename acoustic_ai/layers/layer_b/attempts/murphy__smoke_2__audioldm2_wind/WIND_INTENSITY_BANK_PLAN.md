# Layer B 风强度 bank 设计与执行方案(强/中/弱)

> 决策(2026-06-04):采用 **C+A** —— `heavy`/`medium` 真训 LoRA bank;
> `light`(弱)因数据几乎为零,先用 `medium` adapter 运行时派生(带 caveat),
> 后续采够 light 站点数据再升级为真 adapter。
> 范式完全复用 Layer A 的 cell-bank(`uses_cells` + 每 cell 一个 adapter + PEFT `set_adapter`)。

## 0) 数据现实(纯净站点风:非 reject、无 rain/thunder)

| 档 | 纯净片段 | contamination≤0.28 | 处理 |
|---|---|---|---|
| `heavy`(强) | 31 | 24 | 真训新 LoRA |
| `medium`(中) | 59 | 38 | 复用 smoke_2 已训 adapter |
| `light`(弱) | 3 | 1 | 数据不足 → 由 medium 派生(caveat) |

## 1) 目标 attempt 与 checkpoint

- 新 attempt:`acoustic_ai/layers/layer_b/attempts/murphy__mvp_1__wind_intensity_bank/`
  - 由 `murphy__smoke_2__audioldm2_wind` 复制 code/(train/dataset/sample/handler/visualization)。
- 新 checkpoint bank:`model/candidates/murphy/mvp_1__wind_intensity_bank/`
  - `adapters/medium/`(= smoke_2 权重,复制或软链)
  - `adapters/heavy/`(新训)
  - 每个 adapter 目录含 `adapter_model.safetensors`(DVC) + `adapter_config.json`(git)。

## 2) 锁定 caption(每档)

- `medium`(沿用 smoke_2):
  `steady wind through dry eucalyptus woodland, gentle natural breeze, Bowra, Australia`
- `heavy`(新):
  `strong gusty wind through dry eucalyptus woodland, powerful blustery gusts, Bowra, Australia`
- `light`(派生,不训):
  `gentle light breeze through dry eucalyptus woodland, soft faint airflow, Bowra, Australia`

## 3) 训练配方(冻结,= smoke_2 已验证的 Variant A)

`lr 1e-5 · 200 steps(推理)· guidance 3.0 · rms 0.06 · highpass 80 · denoise 0.15/0.40 · fade 80 · lora r8/alpha32`

- `heavy`:`build_wind_manifest.py --only-intensity heavy --max-contamination-score 0.28`(≈24 条)→ 在 Server B 训练。
- `medium`:复用 smoke_2 adapter,不重训。

## 4) 弱档(light)派生规则(从 medium adapter,带 caveat)

运行时在 medium adapter 上改一组推理参数得到“弱”:
- `guidance_scale`: 3.0 → **2.0**(更柔、能量更低)
- `output_target_rms`: 0.06 → **0.03**(更轻)
- 追加 **低通 ~3.5 kHz**(削高频阵风/嘶声,营造“微风”)
- denoise/fade 保持
> 标注:`light` 为 parametric-derived,非独立学习档;质量为 known caveat,待数据补齐后替换为真 adapter。

## 5) 契约改动(复用 Layer A “validate-or-drop” 纪律)

1. **registry.yaml**:bank attempt 声明
   - `uses_cells: true`
   - `cells: [light, medium, heavy]`(一维强度维度)
   - `default_cell: medium`
   - 每档内联锁定 prompt + 推理参数 + adapter 名;`light` 标 `derived_from: medium`。
2. **handler**:
   - `load()`:把 medium/heavy 两个 adapter 都载入同一 PeftModel(`load_adapter(name=...)`)。
   - `generate(state, seed, wind_intensity)`:校验档位 → `set_adapter` 路由(light 用 medium adapter + 派生参数)→ 用该档锁定 prompt → 响应回显 resolved `wind_intensity`。
3. **Express 后端**:只透传校验通过的 `wind_intensity ∈ {light,medium,heavy}`,非法/缺失丢弃回落 `default_cell`(与现有 season/diel 同纪律)。
4. **前端**:bank attempt 多出 `wind_intensity` 下拉(3 项,来自 `cells`)+ 现有 `seed`。
   - ⚠️ 现契约只硬编码了 `(season, diel)`;需把 cell 选择器**泛化为按 attempt 声明的命名维度**,或为 Layer B 增一个并行 `wind_intensity` 维度。这是唯一需要动到跨层契约的点,实施时单独评审。

## 6) 出料与审计(复用 S3a.4 流程)

- 对 `heavy` 做 seed 扫描(建议同样 40 个)→ 机器预排序 → 人工挑 4–6 黄金 seed。
- `light` 派生档也出几条样审听,确认“弱”听感成立且无 artifact。
- 三档各自 showcase + `human_audit` 标记 → 收口。

## 7) 分阶段落地顺序

1. scaffold 新 attempt + 复制 code(无训练)。
2. 生成 heavy manifest,Server B 训 heavy LoRA。
3. handler/registry 改多 adapter 路由 + light 派生。
4. 跨层契约:前端强度下拉 + Express 透传(单独评审)。
5. heavy seed 扫描 + light 派生抽样 → 审计挑种子。
6. 文档 + light caveat,收口为 `mvp_1__wind_intensity_bank`。

## 待办前置确认(实施前)

- adapter 命名/目录布局是否沿用 Layer A bank 完全一致(便于复用其加载代码)?
- 契约泛化:是“通用命名 cell 维度”还是“Layer B 专用 wind_intensity 维度”?(影响前端/Express 改动面)
