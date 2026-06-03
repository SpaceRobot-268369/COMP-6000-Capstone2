# Tiny Layer E-C Event Classifier

## Role

Smoke checkpoint for `burger__smoke_1__tiny_event_classifier`, a lightweight
analysis-mode E-C event classifier.

## Model

- Feature extractor: 64-bin log-mel statistics plus simple acoustic descriptors.
- Event-band features: 450-1200 Hz, 2100-4100 Hz, 3500-7600 Hz.
- Classifier: one `StandardScaler + LogisticRegression(class_weight="balanced")`
  binary classifier per species.
- Complex detection: class-specific bandpass one-vs-rest candidates per
  sliding window.
- Classes:
  - `horsfields_bronze_cuckoo`
  - `southern_boobook`
  - `splendid_fairywren`

## Training Data

86 local audited event snippets from site 257:

- 30 Horsfield's Bronze-cuckoo
- 24 Southern Boobook
- 32 Splendid Fairywren

## Training

Fit-all training on all 86 clean snippets. Each one-vs-rest classifier sees all
86 snippets through its own species bandpass:

- `horsfields_bronze_cuckoo`: 30
- `southern_boobook`: 24
- `splendid_fairywren`: 32
- rows per one-vs-rest classifier: 86

## Complex Smoke Results

Complex counterpart audio (`audio.wav`, `audio_full.wav`, or similar) with
12 examples per class:

- Whole-clip accuracy: 0.917
- Target detection recall: 0.917
- Mean non-target detections per clip: 0.194
- Smoke pass: true

Smoke threshold: target recall >= 0.90 and mean non-target detections <= 0.25.
The next step is broader validation on unrelated full recordings and background
negative windows.

## Files

- `model.joblib`: trained sklearn pipeline.
- `metrics.json`: evaluation report.
- `complex_eval_metrics.json`: complex-audio detector metrics.
- `complex_eval_predictions.csv`: complex-audio detector outputs.
- `complex_test_manifest.csv`: selected less-cropped test clips.
