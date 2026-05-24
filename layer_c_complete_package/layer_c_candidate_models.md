# Layer C Candidate Models

## Layer C Requirement

输入天气 / 季节 / 场景文本，输出 30 秒纯自然 / 动物环境声。不能生成语音、音乐、机械声或交通声。模型优先级：开源、轻量、可本地或服务器运行、支持或容易适配微调。

---

## Summary Recommendation

Smoke test 阶段建议优先顺序：

1. **AudioCraft AudioGen**：最贴近环境声 / sound effect 生成。
2. **AudioLDM2**：成熟 text-to-audio diffusion，可快速做 baseline generation。
3. **项目内 Ambient VAE / Vocoder pipeline**：最容易和团队仓库对接，但需要确认文本条件输入是否已经完整支持。

最终建议：

- 如果目标是最快生成可听样本：先用 **AudioLDM2** 做 baseline。
- 如果目标是最贴近纯环境声任务：优先尝试 **AudioGen**。
- 如果目标是项目集成和 MVP：保留项目内 pipeline 作为最终对接方案。

---

## Candidate 1: Meta AudioCraft / AudioGen

- 类型：Text-to-audio / environmental sound generation
- 链接：https://github.com/facebookresearch/audiocraft
- 适合原因：
  - AudioCraft 官方包含 AudioGen，AudioGen 面向 sound effects / environmental audio。
  - PyTorch 项目，便于服务器训练和推理。
  - 比 MusicGen 更符合 Layer C，因为本任务明确不要音乐。
- 风险：
  - 官方 fine-tune 文档对 AudioGen 不一定像 MusicGen 那样完整。
  - 小数据 60–90 条可能更适合作为 smoke test adapter / prompt tuning / 小轮次实验，不保证高质量完全微调。
- 适合阶段：
  - smoke test 生成
  - 后续 MVP 候选

---

## Candidate 2: AudioLDM2

- 类型：Latent diffusion text-to-audio
- 链接：https://github.com/haoheliu/audioldm2
- Hugging Face Diffusers 文档：https://huggingface.co/docs/diffusers/en/api/pipelines/audioldm2
- 适合原因：
  - 支持 text-to-audio。
  - Diffusers pipeline 容易写推理脚本。
  - 可以先不微调，直接测试三个 prompt 的 baseline。
- 风险：
  - 可能生成非自然声、音乐感或不稳定噪声，需要强 prompt 和人工 Audit。
  - 真实 fine-tune 成本可能比想象高。
- 适合阶段：
  - baseline generation
  - smoke test 对比模型

---

## Candidate 3: Project Ambient VAE / Vocoder Pipeline

- 类型：项目内已有 acoustic / ambient 训练流程
- 链接：团队 GitHub 仓库内 `acoustic_ai`、`dvc.yaml`、`params.yaml`
- 适合原因：
  - 最容易和现有 backend / frontend / DVC 流程对接。
  - 如果已有 spectrogram、VAE、latent、vocoder 训练 stage，可以直接接入 Layer C 数据。
  - 适合 MVP 固化。
- 风险：
  - 需要确认是否已经支持文本条件输入。
  - 如果只是无条件 VAE/vocoder，需要额外做 prompt-to-latent 或 scene label conditioning。
- 适合阶段：
  - MVP integration
  - 项目内 smoke test reproduction

---

## Selection for Smoke Test

本 smoke test 推荐先选：

> AudioLDM2 as baseline generator + 项目内 pipeline for integration check

原因：

1. AudioLDM2 最快能通过文本 prompt 生成音频。
2. Layer C 的三个场景文本非常简单，适合做 baseline。
3. 生成后可以立即做二次 Audit。
4. 如果 AudioLDM2 效果不稳定，再切换 AudioGen。

---

## Fixed Prompts

```text
summer rain afternoon, realistic clean outdoor nature ambience, only rain and leaves, no speech, no music, no traffic, no machines
winter snow night, realistic quiet cold wind and snowfall ambience, no footsteps, no speech, no music, no machines
forest birds morning, realistic clean forest birds ambience, leaves and light wind, no speech, no music, no traffic, no machines
```
