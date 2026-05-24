# Layer C Complete Execution Guide

这个包用于完成 Layer C：输入天气 / 季节 / 场景，输出纯自然 / 动物环境声。

## Step 0: 放到项目根目录

```bash
git clone https://github.com/SpaceRobot-268369/COMP-6000-Capstone2.git
cd COMP-6000-Capstone2
unzip layer_c_complete_package.zip
```

如果你已经在项目根目录，直接解压即可。

## Step 1: 安装基础依赖

```bash
pip install librosa soundfile numpy pandas openpyxl tqdm
```

如果要用 AudioLDM2 baseline：

```bash
pip install torch diffusers transformers accelerate scipy
```

## Step 2: 准备原始音频库

把 Object Storage 下载来的原始音频放到：

```text
data/object_storage_audio/
```

支持 `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`。

## Step 3: 按规则筛 30 秒片段

示例：

```bash
python scripts/layer_c/prepare_raw_clips.py   --input_dir data/object_storage_audio   --output_dir smoke_test/layer_c/raw_clips   --scene summer_rain   --limit 30
```

三个场景都运行：

```bash
python scripts/layer_c/prepare_raw_clips.py --input_dir data/object_storage_audio --output_dir smoke_test/layer_c/raw_clips --scene summer_rain --limit 30
python scripts/layer_c/prepare_raw_clips.py --input_dir data/object_storage_audio --output_dir smoke_test/layer_c/raw_clips --scene winter_snow --limit 30
python scripts/layer_c/prepare_raw_clips.py --input_dir data/object_storage_audio --output_dir smoke_test/layer_c/raw_clips --scene forest_bird --limit 30
```

注意：脚本只能做格式、时长、裁剪、重命名，不能替代人工听音。人声/音乐/机械声必须人工 Audit。

## Step 4: 人工 Audit

打开：

```text
layer_c_audit_log.xlsx
```

逐条听 `raw_clips` 里的 30 秒音频，填：

- pass_fail: PASS 或 FAIL
- reason: 合格 / 人声 / 音乐 / 机械 / 交通 / 杂音 / 场景不符 / 文件异常
- notes: 简短备注

合格率必须 >= 80%。

## Step 5: 复制合格音频到 audited_clips

Audit 表填好后运行：

```bash
python scripts/layer_c/copy_passed_clips.py   --audit_xlsx layer_c_audit_log.xlsx   --raw_dir smoke_test/layer_c/raw_clips   --output_dir smoke_test/layer_c/audited_clips
```

## Step 6: 生成 metadata

```bash
python scripts/layer_c/build_metadata.py   --clips_dir smoke_test/layer_c/audited_clips   --output_csv smoke_test/layer_c/metadata.csv
```

## Step 7: 先跑 baseline generation

```bash
python scripts/layer_c/generate_audioldm2_baseline.py   --output_dir smoke_test/layer_c/generated   --num_per_scene 5
```

如果没有 GPU，也可以先跳过，或在服务器上运行。

## Step 8: 二次 Audit

听 `generated` 里的音频，填 `layer_c_generation_audit_log.xlsx`。如果 pass rate >= 70%，smoke test 通过。

## Step 9: MVP 接口

先用：

```bash
python scripts/layer_c/layer_c_generate.py --scene summer_rain --output output.wav
```

之后再让 backend 调这个脚本或封装成 API。
