# Layer C Generation Process Log

This document records the Layer C bird-event generation attempts made during the
smoke-development cycle. It focuses on what was tried, what failed, what worked,
and why the current direction moved from AudioGen LoRA to Stable Audio 3
reference-conditioned generation and SA3 LoRA.

## Final Working Goal

Layer C needs a generative bird-call event model/workflow that can produce
short target-like bird-call clips from real training audio, with enough
accuracy for manual audit and later integration into the layered soundscape
pipeline.

The practical target evolved into:

- generated clips should sound like the target species, not just "generic bird";
- outputs should preserve call morphology visible in spectrograms;
- samples should be auditable by human listening plus automatic similarity
  checks;
- the final route should use real training/reference audio, not generated audio
  as the source of truth.

## Early Dataset And Filter Work

The first workstream built filtered Layer C candidate sets from site 257
annotations and S3-hosted audio.

Important scripts:

- `script/dataset/build_layer_c_filtered_manifest.py`
- `script/dataset/build_layer_c_quality_manifest.py`
- `script/dataset/build_layer_c_birdnet_core_manifest.py`
- `script/dataset/build_layer_c_anchor_motif_manifest.py`
- `script/dataset/extract_layer_c_segments_from_s3_clips.py`
- `script/dataset/prepare_layer_c_smoke_segments.py`

Core filtering criteria included:

- target species annotation score;
- duration window;
- event-to-background ratio;
- active/silence ratio;
- avoidance of source-clip boundaries;
- later: BirdNET embedding similarity and motif/core clustering;
- later: manual audit feedback was used to reverse-search cleaner candidates.

Main lesson: high BirdNET/classifier score alone was not enough. Many high-score
segments were not clean enough for generation training because they contained
background calls, weak foreground, or inconsistent motifs.

## Manual Audit Method

Manual audit was used throughout because automatic scores did not reliably
predict whether a generated sample sounded correct.

Audit labels:

- `Pass`: clear target species call, usable.
- `Borderline`: plausible but noisy, mixed, truncated, or slightly off.
- `Fail`: wrong species, generic bird, background only, or unusable.

Later automatic checks were added:

- BirdNET embedding/species-centroid similarity;
- target-rank style checks against other species clusters;
- spectrogram local similarity against real training examples;
- best-match spectrogram similarity thresholds, tested around `0.50` and
  `0.65`.

Important conclusion: automatic scores are useful for ranking and triage, but
they cannot replace human audit for this task. Spectrogram similarity especially
can reward copy-like or visually similar outputs that still sound wrong.

## AudioGen LoRA Route

Initial selected model route:

- Base model: `facebook/audiogen-medium`
- Training: per-species LoRA
- Inference: fixed target prompt + seed sweep

Reason for trying AudioGen:

- native short audio event window;
- stronger transient preservation than AudioLDM-style mel/vocoder paths;
- easy per-species LoRA candidate structure;
- expected to learn bird-call priors from the base model.

### Splendid Fairywren

Initial Fairywren runs were the most promising AudioGen results.

Representative candidate folders included:

- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-1epoch/`
- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-5epoch/`
- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-birdnet-core50-10epoch/`
- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-natural-core36-5epoch/`
- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-motif-core22-8epoch/`
- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-balanced-motif23-8epoch/`
- `model/candidates/burger/layer-c-audiogen-splendid-fairywren-anchor8506782-core10-8epoch/`

What was tried:

- 1 epoch, 5 epoch, 8 epoch, 10 epoch, 12 epoch, 15 epoch variants;
- 10-seed and 50-seed generation sweeps;
- BirdNET-core and natural-core training sets;
- narrower motif sets;
- anchor-based training around the better-sounding `8506782` motif;
- spectrogram-similarity ranking against real training examples.

Observed behavior:

- some seed groups sounded passable, especially the earlier `700-709` style
  run;
- increasing strictness of automatic filtering did not consistently improve
  human-perceived quality;
- very narrow motif sets often became less natural or less bird-like;
- spectrogram similarity sometimes disagreed with listening quality.

Conclusion:

Fairywren proved AudioGen LoRA could sometimes produce plausible bird-like
events, but the success was not stable enough across seeds or dataset choices.
The early "44/50 Pass" historical audit was later treated as too optimistic.

### Red-capped Robin

Candidate folders included:

- `model/candidates/burger/layer-c-audiogen-red-capped-robin-smoke-5epoch/`
- `model/candidates/burger/layer-c-audiogen-red-capped-robin-strict-pass24-8epoch/`
- `model/candidates/burger/layer-c-audiogen-red-capped-robin-multi-motif-core24-5epoch/`

What was tried:

- strict manually audited training clips;
- 5 epoch and 8 epoch routes;
- 10-seed and 50-seed generated audits;
- comparison with Fairywren, because both have high-frequency small-bird calls.

Observed behavior:

- human audit found many outputs generic or not clearly Robin;
- Fairywren and Robin were hard to separate perceptually and acoustically;
- the training set could not force AudioGen to reliably distinguish these
  small-bird call shapes.

Conclusion:

Robin did not provide a stable second-species success. The small-bird call
family was too easily collapsed into generic high-frequency bird sounds.

### Crested Bellbird

Candidate folders included:

- `model/candidates/burger/layer-c-audiogen-crested-bellbird-smoke-5epoch/`
- `model/candidates/burger/layer-c-audiogen-crested-bellbird-multi-motif-core30-5epoch/`

What was tried:

- relaxed and strict candidate filtering;
- manual audit of top candidates;
- 5 epoch training;
- 10-seed and 50-seed generation sweeps.

Observed behavior:

- training data contained some distinctive calls, but also background and motif
  inconsistency;
- generated outputs did not reliably capture the intended bell-like call shape.

Conclusion:

Bellbird did not become a reliable target under AudioGen LoRA.

### Southern Boobook

Candidate folders included:

- `model/candidates/burger/layer-c-audiogen-boobook-smoke-5epoch/`
- `model/candidates/burger/layer-c-audiogen-southern-boobook-natural-core17-8epoch/`

What was tried:

- initial smoke dataset;
- natural-core re-filtering;
- 5 epoch and 8 epoch training;
- reference to the stereotyped two-note "boobook" target pattern.

Observed behavior:

- despite the species being human-distinctive, AudioGen often missed the
  two-note shape;
- outputs were wrong in both spectrogram morphology and listening quality;
- the model tended toward generic night/bird textures rather than the call.

Conclusion:

Boobook showed that "human-distinctive species" does not automatically mean
AudioGen LoRA can learn it from a small dataset. This pushed the work away from
pure text-to-audio LoRA.

### Chestnut-rumped Thornbill

Candidate folders included:

- `model/candidates/burger/layer-c-audiogen-chestnut-rumped-thornbill-smoke-5epoch/`
- `model/candidates/burger/layer-c-audiogen-chestnut-rumped-thornbill-combined-core21-8epoch/`

What was tried:

- original strict candidates;
- supplement candidate set;
- combined core set;
- 5 epoch and 8 epoch training;
- spectrogram review.

Observed behavior:

- only a small number of clean training clips survived manual audit;
- the best-looking generated sample by spectrogram still sounded off;
- most outputs were not convincing Thornbill.

Conclusion:

Thornbill was not worth continuing under the AudioGen route because training
data quality and generated accuracy were both weak.

### Horsfield's Bronze-cuckoo

Candidate folders and data included:

- `model/candidates/burger/layer-c-audiogen-horsfields-bronze-cuckoo-pass24-8epoch/`
- `model/candidates/burger/layer-c-audiogen-horsfields-bronze-cuckoo-seed9005-core11-8epoch/`
- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/`

What was tried:

- natural-core filtering;
- reverse supplement from human-pass examples;
- pass20/pass24 training manifests;
- seed9005-inspired narrower core set;
- 8 epoch AudioGen runs;
- spectrogram matching and listening audit.

Observed behavior:

- a few individual seeds, such as `9005` and later `9102`, sounded closer;
- most seeds still failed or sounded generic/wrong;
- individual good seeds were not enough to claim a trained model works.

Conclusion:

Bronze-cuckoo became the best candidate for moving away from AudioGen because
it had a clearer repeated whistle motif and some good real reference clips.

## Why AudioGen Was Deprioritized

Across species, the same failure pattern appeared:

- seed luck mattered too much;
- generated outputs often became "generic bird" instead of target species;
- increasing epochs did not reliably fix morphology;
- very narrow datasets could overfit or become less natural;
- automatic similarity could rank samples that humans still rejected;
- text prompts were too weak to force exact call shape.

The key technical issue was that text-conditioned AudioGen LoRA did not have a
strong enough mechanism to preserve exact spectrogram/call morphology from real
audio. This led to the next route: audio-to-audio / reference-conditioned
generation.

## Stable Audio 3 Reference-Conditioned Route

The next route used Stable Audio 3 as an audio-to-audio / variation model.

Main script:

- `acoustic_ai/modules/events/sample_stable_audio_inpaint.py`

Base model:

- `small-sfx-base`

Important idea:

Instead of asking a text-only model to invent the species call from a prompt,
condition generation on real target audio and allow controlled variation with
`init_noise_level`.

### Reference Bank Search

Initial real reference bank:

- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/stable_audio_real_reference_bank_top5.csv`

Best individual references by human audit:

- `ref_4_13395066`: best match to desired Bronze-cuckoo result.
- `ref_2_18696395`: second-best.

Early experiments:

- `noise=0.18`
- `noise=0.22`
- `noise=0.32`
- `noise=0.40`

Observed behavior:

- lower noise preserved the reference but looked/sounded copy-like;
- `noise=0.40` still sounded acceptable while introducing more variation;
- using only one reference made outputs too visually similar.

### Top10 Real-Pass Reference Probe

Input reference bank:

- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/stable_audio_natural_pass_reference_bank_top10.csv`

Probe output:

- `debug/layer_c/stable_audio/samples/horsfields_bronze_cuckoo_stable_audio_realpass_top10_noise040_3seed/`

Manual audit result:

- 6 of 10 references produced `3/3 Pass` at `noise=0.40`.
- Passing references were kept as the core set.

Core reference bank:

- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/stable_audio_core_reference_bank_pass6.csv`

### Core6 Confirmation

Output playlist:

- `debug/layer_c/stable_audio/samples/horsfields_bronze_cuckoo_stable_audio_core6_noise040_5seed/horsfields_bronze_cuckoo_stable_audio_core6_noise040_5seed_sample_audit_absolute.m3u`

Output audit CSV:

- `debug/layer_c/stable_audio/samples/horsfields_bronze_cuckoo_stable_audio_core6_noise040_5seed/horsfields_bronze_cuckoo_stable_audio_core6_noise040_5seed_sample_audit.csv`

Manual audit result:

- `30/30 Pass`

Conclusion:

This is the strongest validated Layer C result so far. It is not yet a trained
SA3 LoRA model, but it proves that real-audio-conditioned Stable Audio 3 can
produce target-like Bronze-cuckoo variations far more reliably than AudioGen
text-to-audio LoRA.

## Stable Audio 3 LoRA Preparation

Because the final requirement is still a trainable model/workflow from real
audio, the next step prepared a Stable Audio 3 LoRA smoke candidate using the
validated core6 real references.

Candidate folder:

- `model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/`

Training data:

- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/sa3_lora_core6_data`

Source manifests:

- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/sa3_lora_smoke_core6_train_manifest.csv`
- `resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/sa3_lora_smoke_core6_metadata.jsonl`

Training script wrapper:

- `script/events/train_sa3_lora_core6_smoke.sh`

Official upstream training script:

- `/home/ubuntu/stable-audio-3/scripts/train_lora.py` on the RONIN VM
- `/private/tmp/stable-audio-3/scripts/train_lora.py` on local setup

Proposed smoke settings:

- base model: `small-sfx-base`
- adapter: `dora-rows`
- rank: `8`
- alpha: `8`
- dropout: `0.05`
- excluded conditioner: `seconds_total`
- steps: `300`
- batch size: `1`
- duration: `8`
- checkpoint every `100` steps

Local dry-run:

- confirmed official `train_lora.py` imports after installing missing deps;
- confirmed the dataset loads 6 files;
- confirmed 192 LoRA layers and 5.6M trainable parameters;
- stopped before checkpoint because local Mac had no CUDA/MPS accelerator.

RONIN cloud setup:

- host: `ubuntu@sa3-lora-layer-c.adelaideuni.cloud`
- GPU: NVIDIA A10G, about 23GB VRAM
- CUDA PyTorch verified: `torch 2.7.1+cu126`, CUDA available
- Hugging Face login verified as `burgeryjh`
- Stable Audio 3 installed
- official repo cloned to `/home/ubuntu/stable-audio-3`
- 300-step LoRA run was started with:

```bash
SA3_REPO=/home/ubuntu/stable-audio-3 \
MPLCONFIGDIR=/tmp/mpl \
SA3_STEPS=300 \
SA3_CHECKPOINT_EVERY=100 \
SA3_DEMO_EVERY=999999 \
nohup script/events/train_sa3_lora_core6_smoke.sh \
  > logs/sa3_lora_core6_train_300.log 2>&1 &
```

Status note: the run was confirmed to start on GPU and enter the Lightning
training flow, but final checkpoint completion still needs to be checked.

## Current Best Result

The best validated result before SA3 LoRA completion is:

- model/workflow: Stable Audio 3 reference-conditioned variation
- species: Horsfield's Bronze-cuckoo
- reference set: core6 real audited clips
- noise: `0.40`
- generation result: `30/30 Pass` by manual audit

This route is more reliable than AudioGen LoRA because it conditions on real
audio morphology rather than relying on text prompt plus LoRA to rediscover the
call shape.

## Main Lessons

1. Text-only LoRA is not enough for small bird-event morphology.
   AudioGen frequently produced plausible generic bird sounds instead of the
   target species.

2. More epochs or more seeds do not solve the core problem.
   They can improve isolated samples but do not reliably raise the true pass
   probability.

3. Training data quality matters, but even very strict sets can fail.
   The model still needs an audio-conditioned mechanism to preserve motif shape.

4. Human audit remains necessary.
   BirdNET embedding, classifier-like rank, and spectrogram similarity are all
   helpful filters, not final judges.

5. Stable Audio 3 reference-conditioned generation is the strongest direction.
   It preserves real call morphology and gives controlled variation.

6. SA3 LoRA is the next model-training proof.
   It should be evaluated by comparing LoRA outputs against the already
   validated base SA3 core6 reference-conditioned outputs.

## Next Recommended Steps

1. Check the RONIN 300-step SA3 LoRA run:

```bash
ssh -i ~/.ssh/burger-layer-c-key.pem ubuntu@sa3-lora-layer-c.adelaideuni.cloud
cd /home/ubuntu/COMP-6000-Capstone2
tail -n 80 logs/sa3_lora_core6_train_300.log
find model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints -type f
```

2. If checkpoints exist, generate a small 10-seed LoRA audit set.

3. Compare three groups:

- base SA3 core6 reference-conditioned samples;
- SA3 LoRA samples with the same core6 references;
- SA3 LoRA text-only or weaker-reference samples if supported.

4. Judge whether the LoRA adds useful natural variation without losing the
   Bronze-cuckoo call shape.

5. If LoRA overfits or copies too much, expand the real reference bank from 6
   to 15-20 manually audited clips before the next SA3 LoRA run.
