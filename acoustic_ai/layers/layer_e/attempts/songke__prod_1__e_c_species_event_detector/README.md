# E-C Species Event Detector - Production

## Status

- Layer/head: Layer E, E-C events analysis
- Attempt id: `songke__prod_1__e_c_species_event_detector`
- Stage: `prod_1`
- Registry status: `production`
- Production checkpoint: `model/production/layer_e_c_species_event_detector/`
- Source attempt: `songke__smoke_2__known_species_clap_probe`
- Supported labels: 13 known Australian species

This production attempt serves the online E-C Analysis Mode demo. It detects
known species events in uploaded audio and returns event time ranges,
confidence values, species match scores, phenology metadata, and an
aggregator-ready `analysis_report`.

## Production Artifact

The production checkpoint is DVC-managed:

```text
model/production/layer_e_c_species_event_detector/
```

Restore it locally with:

```powershell
dvc pull model/production/layer_e_c_species_event_detector/best_probe.pt.dvc
```

The Server B production sync workflow automatically materialises DVC pointers
under `model/production/`.

## Runtime

Pipeline:

```text
uploaded audio -> overlapping 5 s windows -> frozen LAION-CLAP encoder
               -> MLP probe -> merged species events -> phenology/report adapter
```

Default inference settings:

```text
threshold=0.55
window_s=5.0
hop_s=1.0
merge_gap_s=1.0
min_event_windows=7
```

Registry-facing handler check:

```powershell
acoustic_ai\.venv\Scripts\python.exe acoustic_ai\layers\layer_e\attempts\songke__prod_1__e_c_species_event_detector\code\handler.py "C:\path\to\recording.mp3"
```

## Outputs

- `events`: merged species detections with onset, offset, and confidence.
- `species_matches`: ranked known-species scores for each event.
- `phenology`: ecological metadata for detected species.
- `analysis_report.observations`: direct E-C detections.
- `analysis_report.inferred_context`: conservative diel, season, and habitat cues.
- `analysis_report.disagreements`: reserved for future E-A/E-B/E-C aggregation.
- `diagnostics.detected_windows`: supporting window evidence.

## Metrics And Sign-Off

- Test accuracy: `0.817`
- Test macro-F1: `0.811`
- Demo sign-off: accepted by the team on 2026-06-03
- Promotion date: 2026-06-04

See the production model card and metrics:

```text
model/production/layer_e_c_species_event_detector/
```

## Known Limitations

- This is a known-species detector for the configured 13 labels.
- It does not explicitly reject unknown species or background-only audio.
- Unseen species may be assigned to the nearest known label.
- Confidence values are classifier scores, not ecological certainty.
- E-C does not yet fuse disagreements with E-A or E-B.
