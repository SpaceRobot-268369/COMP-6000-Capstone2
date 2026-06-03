# Layer E-C Smoke 1 — Tiny Event Classifier

## Purpose / Hypothesis

Train a small analysis-mode event classifier that can pass a local smoke test
without BirdNET, CLAP, AudioGen, or frontend/backend wiring.

The model recognises three event classes already present as audited local WAVs:

- `horsfields_bronze_cuckoo`
- `splendid_fairywren`
- `southern_boobook`

## Method

The classifier uses fixed audio features rather than a neural backbone:

1. Load each audited event snippet at 22,050 Hz mono.
2. Compute a 64-bin log-mel spectrogram.
3. Collapse time into robust statistics: mean, std, min, max, p10, p50, p90.
4. Add class-relevant band energy features:
   - Southern Boobook: 450-1200 Hz
   - Horsfield's Bronze-cuckoo: 2100-4100 Hz
   - Splendid Fairywren: 3500-7600 Hz
5. Add simple acoustic descriptors: RMS, zero-crossing rate, centroid,
   bandwidth, rolloff, and flatness statistics.
6. Train one `StandardScaler + LogisticRegression(class_weight="balanced")`
   binary classifier per species.

This is deliberately small and smoke-oriented. It is not a replacement for the
BirdNET-first E-C detector, but it gives the project a local model path that can
be trained, saved, loaded, and evaluated quickly.

For complex audio detection, the upgraded smoke path runs class-specific
bandpass one-vs-rest classifiers per window and emits all matching species
candidates. This turns the model from forced single-label classification into
a detector with calibrated per-species rejection.

## Data

Local materialised manifests:

- `resources/site_257_bowra-dry-a/layer_c_retrieval_event_library_split_v1/final_pass_library_v1/layer_c_retrieval_final_pass_event_index.csv`
- `resources/site_257_bowra-dry-a/layer_c_retrieval_event_library_split_v1/boobook_audit_v1/refined_call_units_v1/boobook_pass24_refined_call_units_index.csv`

## Smoke Test

Original snippet split pass criteria:

- macro F1 >= 0.85
- accuracy >= 0.85

Current upgraded smoke uses all 86 clean snippets for training, then evaluates
against less-cropped complex counterparts (`audio.wav`, `audio_full.wav`, or
similar local complex files).

Run:

```bash
./acoustic_ai/.venv-audiogen/bin/python \
  acoustic_ai/layers/layer_e/attempts/burger__smoke_1__tiny_event_classifier/code/tiny_event_classifier.py \
  train \
  --out-dir model/candidates/burger/smoke_1__tiny_event_classifier
```

Outputs:

- `model.joblib`
- `metrics.json`
- `predictions.csv`

## Results analysis / audit

Smoke run completed with:

- 86 examples
- fit-all training on clean snippets
- one binary classifier per species/frequency band
- 36 complex test clips (12/class)
- whole-clip accuracy 0.917
- target detection recall 0.917
- mean non-target detections 0.194 per complex clip
- complex smoke pass true

This passes the complex smoke threshold: target recall >= 0.90 and mean
non-target detections <= 0.25. It is still a small site/species-specific smoke
model, not a general detector.
