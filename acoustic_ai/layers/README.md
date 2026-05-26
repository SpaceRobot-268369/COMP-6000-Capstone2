# `acoustic_ai/layers/`

One folder per **layer code** (`layer-a` … `layer-e`). Each layer has an
`attempts/` subfolder; each attempt is a self-contained Python package.

```
layers/
└── layer_<X>/                          # snake_case for Python validity
    └── attempts/
        └── <member>__<stage>__<slug>/  # all underscores
            ├── README.md       # model card + run log
            ├── handler.py      # required: load() + generate(seed, **kw)
            ├── train.py / sample.py / preprocess.py / dataset.py …
            ├── params.yaml     # per-attempt hyperparameters
            ├── data/           # attempt-local derived data (DVC-tracked)
            ├── precompute/     # attempt-local scripts
            └── debug/          # attempt-local diagnostics
```

Authoritative rules: **[../../.claude/context/dev/attempt_naming.md](../../.claude/context/dev/attempt_naming.md)**.

Currently-registered attempts live in **[../registry.yaml](../registry.yaml)**.
The FastAPI server reads it to drive `GET /layers` (frontend dropdown) and
`POST /layers/<layer>/attempts/<id>/generate`.

## Current attempts

| Layer | Attempt | Status |
|---|---|---|
| layer_a | `lucas__smoke_1__audioldm2_spring_night` | ✓ active |
| layer_a | `lucas__smoke_2__audioldm2_insects` | ✓ active (duplicates smoke_1 code with different prompt/LoRA) |
| layer_a | `lucas__smoke_4__vae_baseline` | superseded, but still backs `/analysis` |
| layer_b | `lucas__smoke_1__curated_assets` | placeholder |
| layer_c | `lucas__smoke_1__audiogen_boobook` | ✓ active |
| layer_d | `lucas__smoke_1__layered_mix` | placeholder |
| layer_e | `lucas__smoke_1__detectors` | partial |

The earlier CLAP diffusion experiment (would have been `smoke_3`) was dropped
during the restructure — see git history `modules/ambient/diffusion/*_clap.py`.
