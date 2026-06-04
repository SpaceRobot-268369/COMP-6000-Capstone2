# Layer E MVP 5 — CLAP Weather Probe

Owner: `liting`

## Purpose

Test whether a frozen LAION-CLAP audio embedding separates the E-B weather
classes better than the PANNs/DSP heads from MVP2-MVP4.

This attempt is intentionally audited-only:

- no pseudo-labelled expansion
- no sound-library assets
- no CLAP backbone fine-tuning
- same audited Site257 validation anchor as MVP2-MVP4

## Output Contract

```json
{
  "wind_intensity": "none | light | moderate | strong",
  "rain_intensity": "none | light | moderate | heavy",
  "thunder_intensity": "none",
  "confidence": 0.0
}
```

Thunder remains suppressed until validated Site257 thunder examples exist.

## Model

- Frozen backbone: `laion/clap-htsat-unfused`
- Audio embedding: 512-d L2-normalised CLAP vector over 10 s windows
- Supporting features: existing DSP weather features
- Trainable part: small rain and wind classifier heads

## Why This Attempt Exists

MVP4 showed that noisy pseudo-labelled data expansion can hurt the wind
boundary. MVP5 changes the representation instead of expanding data. If CLAP
separates wind/rain better, it can become the next integration candidate. If it
does not, the next useful path is manually audited data expansion.

## Server B Target

- Feature extraction: CLAP over 101 audited clips
- Head fitting: under 5 minutes
- Validation: audited-only deterministic split

## Pass Bar

- Rain validation accuracy >= 0.70
- Wind validation accuracy >= 0.70
- Single-component joint accuracy >= 0.65
- Compare directly against:
  - MVP2: rain 0.769, wind 0.731, joint 0.615
  - MVP3: rain 0.885, wind 0.692, joint 0.654
  - MVP4: rain 0.846, wind 0.615, joint 0.615

## Server B Result

Best audited-only CLAP run:

- Rain validation accuracy: 0.808
- Wind validation accuracy: 0.692
- Joint validation accuracy: 0.615
- Actual head fitting time: 1.43 seconds
- Total runtime including CLAP feature extraction: 32.81 seconds
- Gate status: `needs_iteration`

Interpretation: CLAP is a viable representation and recovers from MVP4's noisy
pseudo-label failure, but it still does not beat MVP2 as the safest candidate.
The next attempt should not be another tiny-head sweep; it should expand the
audited Site257 weather labels or train on a more balanced manual set.

## Outputs

```text
model/candidates/liting/mvp_5__clap_weather_probe/weather_head.pt
model/candidates/liting/mvp_5__clap_weather_probe/metrics.json
debug/e_b_weather_mvp5/
```
