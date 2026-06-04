# Plan — E-B MVP 5 CLAP Weather Probe

## Hypothesis

Frozen CLAP embeddings may separate audible weather texture differently from
PANNs CNN14. Since MVP2-MVP4 struggled mostly on wind intensity boundaries,
MVP5 tests representation change before spending time on larger data curation.

## Steps

1. Load 101 audited Site257 weather assets from the Layer B weather asset index.
2. Embed each clip using frozen LAION-CLAP.
3. Append transparent DSP weather features.
4. Train small rain and wind heads on the deterministic train split.
5. Evaluate only on audited validation rows.
6. Write checkpoint, metrics, validation predictions, and report.

## Non-Goals

- No CLAP fine-tuning.
- No pseudo-labelled extra clips.
- No thunder claim without Site257 evidence.
- No sound-library assets.

## Decision Rule

If MVP5 beats MVP2/MVP3 on wind or joint accuracy, keep it as a candidate for
integration. If it does not, record the result and move to manually audited data
expansion.
