# Layer C 2-3 Species Smoke Plan

Goal: finish only the Layer C smoke-test stage in ~1.5 days.

## Scope

Use 2-3 bird species from the existing site 257 annotation index:

1. Splendid Fairywren
2. Chestnut-rumped Thornbill
3. Southern Boobook

The smoke target is not a production model. The target is a small, auditable
Layer C event set with 30-50 usable bird-call outputs after filtering,
retrieval fallback, and generation/sample selection.

## Step 1 - Build Filter Manifest

Use the shared S3 resources instead of downloading from A2O. The annotation
index is mirrored as:

```text
s3://eco-acoustic-data.store.adelaideuni.cloud/dataset/original/site_257_bowra-dry-a/all_items_annotation.zip
```

Download and unpack it into:

```text
resources/site_257_bowra-dry-a/all_items_annotation/
```

```bash
python3 script/dataset/build_layer_c_filtered_manifest.py \
  --species splendid_fairywren \
  --species chestnut_rumped_thornbill \
  --species boobook \
  --segments-per-species 50 \
  --min-score 0.9 \
  --min-duration 1.0 \
  --max-duration 8.0
```

Outputs:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/manifest.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/rejected_manifest.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/filter_report.md
```

Current run result:

```text
Annotation CSVs scanned: 1655
Candidates after hard filters: 7235
Selected rows: 150
Splendid Fairywren: 50
Chestnut-rumped Thornbill: 50
Southern Boobook: 50
```

## Step 2 - Extract Exact Segments From S3

```bash
python3 script/dataset/extract_layer_c_segments_from_s3_clips.py
```

This downloads only the required 300 s source chunks from the shared S3
`downloaded_clips/` tree into `/private/tmp/layer_c_s3_clip_cache`, then trims
each buffered event segment into:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/segments/
```

Current run result: `150/150` event webm segments extracted.

## Step 3 - Prepare WAVs And Spectrograms

```bash
./acoustic_ai/.venv-audiogen/bin/python script/dataset/prepare_layer_c_smoke_segments.py \
  --dataset-dir resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species
```

Current run result:

```text
prepared=150
errors=0
audio.wav files=150
mel_spectrogram.png files=150
prepared_manifest.csv rows=150
```

## Step 4 - Quick Manual Audit

Audit 10-20 prepared clips per species first. Mark each as:

- Pass: clear target bird-call event
- Borderline: plausible but noisy, truncated, or mixed
- Fail: wrong species, mostly background, or unusable

Keep at least 30-50 total pass clips across the 2-3 species before training or
sampling.

## Step 5 - Smoke Generation

Use the audited retrieval set as a baseline and data source, but judge Layer C
model feasibility with generated AudioGen LoRA samples. The selected smoke
model route is `facebook/audiogen-medium` plus LoRA, trained first on the
cleanest species subset.

Smoke pass target: at least 90% usable selected Layer C events in the final
30-50 sample set.

Current status:

```text
Manual audit complete for:
- Splendid Fairywren rows 1-35: 24 Pass, 8 Borderline, 3 Fail
- Chestnut-rumped Thornbill rows 51-100: 25 Pass, 23 Borderline, 2 Fail
- Southern Boobook rows 101-150: 24 Pass, 26 Borderline, 0 Fail
```

Generated smoke deliverables:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/smoke_retrieval_pass_set.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_splendid_fairywren_pass.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_sample_audit.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_5epoch_sample_audit.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_5epoch_50seed_sample_audit.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_boobook_pass.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_boobook_5epoch_sample_audit.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/train_manifest_chestnut_rumped_thornbill_pass.csv
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_chestnut_rumped_thornbill_5epoch_sample_audit.csv
model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-1epoch/
model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-5epoch/
model/candidates/burger/layer-c-audiogen-boobook-smoke-5epoch/
model/candidates/burger/layer-c-audiogen-chestnut-rumped-thornbill-smoke-5epoch/
```

The Splendid Fairywren AudioGen LoRA smoke run trained for 1 epoch on 24
audited Pass clips. Two samples have been generated under:

```text
debug/layer_c/audiogen/samples/layer_c_audiogen_splendid_fairywren_smoke_1epoch/
```

Generated sample audit:

```text
- seed 42: Borderline — usable fairywren-like sample, but includes one non-target call in the middle.
- seed 43: Fail — not usable as a target bird-call sample.
```

Conclusion: the 1-epoch Fairywren LoRA is a valid smoke attempt, but it does
not meet the 90% usable target.

Follow-up 5-epoch run:

```text
model/candidates/burger/layer-c-audiogen-splendid-fairywren-smoke-5epoch/
debug/layer_c/audiogen/samples/layer_c_audiogen_splendid_fairywren_smoke_5epoch/
```

Training completed on CPU: 120 steps, 5 epochs, 24 manually audited Pass clips.
Generated 8 fixed-seed samples (`100`-`107`) at 3 seconds each.

Manual audit result:

```text
Pass: 7
Borderline: 1
Fail: 0
Clean pass rate: 87.5%
Usable rate (Pass + Borderline): 100.0%
```

Conclusion: `facebook/audiogen-medium + LoRA` is feasible as the selected
Layer C smoke model route for Splendid Fairywren event generation. This is the
main generative smoke proof; the retrieval pass set remains the audited source
data baseline and fallback sample set.

50-seed stability expansion:

```text
Generated seeds: 100-149
Samples generated: 50
Audit table: resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_5epoch_50seed_sample_audit.csv
Pass: 45
Borderline: 5
Fail: 0
Clean pass rate: 90.0%
Usable rate (Pass + Borderline): 100.0%
```

Final smoke decision: the 50-seed audit meets the Layer C smoke target. Report
`facebook/audiogen-medium + LoRA` as the selected Layer C generative model
route, proven on Splendid Fairywren with 24 audited Pass training clips and 50
audited generated samples.

Second-species smoke expansion:

```text
model/candidates/burger/layer-c-audiogen-boobook-smoke-5epoch/
debug/layer_c/audiogen/samples/layer_c_audiogen_boobook_smoke_5epoch/
```

Training completed on CPU: 120 steps, 5 epochs, 24 manually audited Southern
Boobook Pass clips. Generated 8 fixed-seed samples (`200`-`207`) at 3 seconds
each.

Generated sample audit table:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_boobook_5epoch_sample_audit.csv
```

Manual audit result:

```text
Samples audited: 8
Pass: 0
Borderline: 0
Fail: 8
Clean pass rate: 0.0%
Usable rate (Pass + Borderline): 0.0%
```

Conclusion: do not expand the Boobook run. The training pipeline completed,
but the generated audio did not produce usable Southern Boobook calls under
the current 5-epoch LoRA setup and prompt. Treat this as a negative smoke
result and move second-species effort to another species rather than spending
more seed budget here.

Third-species smoke attempt:

```text
model/candidates/burger/layer-c-audiogen-chestnut-rumped-thornbill-smoke-5epoch/
debug/layer_c/audiogen/samples/layer_c_audiogen_chestnut_rumped_thornbill_smoke_5epoch/
```

Training completed on CPU: 125 steps, 5 epochs, 25 manually audited
Chestnut-rumped Thornbill Pass clips. Training loss was noisier than the
successful Fairywren run, so this remains a candidate pending generated audio
audit.

Generated 8 fixed-seed samples (`300`-`307`) at 3 seconds each.

Generated sample audit table:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/generated_lora_chestnut_rumped_thornbill_5epoch_sample_audit.csv
```

Manual audit result after expanding to 30 fixed seeds (`300`-`329`):

```text
Samples audited: 30
Pass: 16
Borderline: 0
Fail: 14
Clean pass rate: 53.3%
Usable rate (Pass + Borderline): 53.3%
```

Conclusion: do not report Thornbill as a second successful generative smoke
species. The model learned some target-like structure and can produce usable
examples, but the success rate is far below the 90% smoke target. Keep the
final Layer C smoke claim limited to the successful Splendid Fairywren 50-seed
run, and report Boobook/Thornbill as cross-species negative or partial attempts
that motivate stronger filtering, more data, or species-specific training
changes in MVP/product work.

DVC note: `dvc add` for the LoRA adapter is still blocked locally by
`unable to open database file`; do not git-add `adapter_model.safetensors`
until DVC is fixed.

## Recovery Analysis - Filter V2

After Boobook and Thornbill failed to meet the generated-audio target, a second
audio-quality filter was added:

```text
script/dataset/build_layer_c_quality_manifest.py
```

This pass joins `prepared_manifest.csv` with `manual_audit_grouped.csv` when
manual audit exists. For recovery pools it can also run with
`--include-unaudited`, crops the foreground event window to
`event_start/event_end +/- 0.5s`, computes audio-quality features, and writes
quality-ranked training manifests.

Key features:

```text
event_to_background_db
active_ratio
silence_ratio
spectral_centroid_hz
high_freq_ratio_3khz
quality_score
```

Strict Pass-only quality run:

```bash
acoustic_ai/.venv-audiogen/bin/python script/dataset/build_layer_c_quality_manifest.py \
  --dataset-dir resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species \
  --crop-buffer-s 0.5 \
  --min-event-bg-db 3.0 \
  --min-active-ratio 0.03 \
  --max-silence-ratio 0.95 \
  --overwrite-crops
```

Result:

```text
Splendid Fairywren: 13 / 24 quality-pass clips
Chestnut-rumped Thornbill: 4 / 25 quality-pass clips
Southern Boobook: 0 / 24 quality-pass clips
```

Relaxed Pass+Borderline quality run:

```bash
acoustic_ai/.venv-audiogen/bin/python script/dataset/build_layer_c_quality_manifest.py \
  --dataset-dir resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species \
  --include-verdict Pass \
  --include-verdict Borderline \
  --crop-buffer-s 0.5 \
  --min-event-bg-db 0.0 \
  --min-active-ratio 0.02 \
  --max-silence-ratio 0.98 \
  --output-dir resources/site_257_bowra-dry-a/layer_c_smoke_2_3_species/quality_v2_relaxed
```

Result:

```text
Splendid Fairywren: 20 quality-pass clips
Chestnut-rumped Thornbill: 17 quality-pass clips
Southern Boobook: 10 quality-pass clips
```

Interpretation:

```text
The original metadata filter is not enough for Layer C training.
BirdNET score confirms species presence, but not foreground dominance.
Boobook and Thornbill failed mostly because the available training clips are
too background-heavy, distant, overlapped, or inconsistent for AudioGen LoRA.
```

Next best move:

```text
1. Do not switch models yet.
2. Build a much larger metadata candidate pool for Boobook and Thornbill.
3. Extract and prepare those candidates.
4. Run quality_v2 before manual audit.
5. Train only on short foreground crops that pass quality_v2 and manual audit.
6. If Thornbill still fails after 80-150 quality-pass crops, then test a
   higher-sample-rate model route.
```

## Recovery Candidate Pool

A larger v2 metadata candidate pool was created without overwriting the smoke
dataset:

```bash
acoustic_ai/.venv-audiogen/bin/python script/dataset/build_layer_c_filtered_manifest.py \
  --species splendid_fairywren \
  --species chestnut_rumped_thornbill \
  --species boobook \
  --segments-per-species 250 \
  --min-score 0.8 \
  --min-duration 1.0 \
  --max-duration 8.0 \
  --max-per-recording 2 \
  --output resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/manifest.csv \
  --rejected-output resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/rejected_manifest.csv \
  --report-output resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/filter_report.md
```

Result:

```text
Annotation CSVs scanned: 1655
Rows matching target species: 52372
Candidates after hard filters: 14328
Selected rows: 750
Splendid Fairywren: 250 / 7040 candidates selected
Chestnut-rumped Thornbill: 250 / 3866 candidates selected
Southern Boobook: 250 / 3422 candidates selected
```

The combined manifest was split into per-species manifests so recovery can run
incrementally:

```text
resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/manifest_boobook.csv
resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/manifest_chestnut_rumped_thornbill.csv
resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/manifest_splendid_fairywren.csv
```

Recovery extraction status:

```text
Southern Boobook: 245 / 250 extracted, 5 S3 404s
Chestnut-rumped Thornbill: 232 / 250 extracted, 18 S3 404s
Prepared WAV rows: 477
```

Quality V2 run on the extracted recovery pool:

```bash
acoustic_ai/.venv-audiogen/bin/python script/dataset/prepare_layer_c_smoke_segments.py \
  --dataset-dir resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool

acoustic_ai/.venv-audiogen/bin/python script/dataset/build_layer_c_quality_manifest.py \
  --dataset-dir resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool \
  --include-unaudited \
  --crop-buffer-s 0.5 \
  --min-event-bg-db 0.0 \
  --min-active-ratio 0.02 \
  --max-silence-ratio 0.98 \
  --output-dir resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/quality_v2_relaxed
```

Result:

```text
Strict gate:
- Southern Boobook: 0 / 245 quality-pass
- Chestnut-rumped Thornbill: 38 / 232 quality-pass

Relaxed gate:
- Southern Boobook: 39 / 245 quality-pass
- Chestnut-rumped Thornbill: 80 / 232 quality-pass
```

Generated audit tables:

```text
resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/quality_v2_relaxed/manual_audit_boobook_quality_v2_candidates.csv
resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/quality_v2_relaxed/manual_audit_chestnut_rumped_thornbill_quality_v2_candidates.csv
resources/site_257_bowra-dry-a/layer_c_v2_candidate_pool/quality_v2_relaxed/manual_audit_quality_v2_candidates_combined.csv
```

Updated recovery decision:

```text
1. Train the next Thornbill LoRA from quality_v2 candidates after manual audit.
   Thornbill now has enough candidate volume for a real v2 retry.
2. Do not train Boobook blindly. The 0 strict-pass result means Boobook still
   lacks strong foreground data in this pool. Audit the 39 relaxed candidates;
   if fewer than ~25 are clear Pass, rebuild Boobook with a larger pool or a
   call-pattern-specific filter before training.
3. Keep the AudioGen LoRA route for now because Fairywren proved the model can
   work when the data is clean.
```
