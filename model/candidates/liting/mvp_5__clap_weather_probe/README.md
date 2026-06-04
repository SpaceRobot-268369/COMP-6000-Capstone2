# E-B MVP 5 — CLAP Weather Probe Candidate

Owner: `liting`

## Role

Layer E-B weather analysis checkpoint candidate using frozen LAION-CLAP audio
embeddings plus DSP features.

## Expected Artifacts

```text
weather_head.pt
metrics.json
```

Metrics are git-tracked. The checkpoint is not promoted because the Server B
run did not pass the weather gate.

## Server B Result

- Rain validation accuracy: 0.808
- Wind validation accuracy: 0.692
- Joint validation accuracy: 0.615
- Gate: `needs_iteration`

MVP5 is useful evidence that CLAP is competitive with PANNs/DSP, but MVP2
remains the safer current integration candidate.
