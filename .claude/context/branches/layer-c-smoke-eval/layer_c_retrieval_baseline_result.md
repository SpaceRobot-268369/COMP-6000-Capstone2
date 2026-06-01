# Layer C Retrieval Baseline Result

Date: 2026-05-31

## Scope

This is a Layer C retrieval baseline, not a from-scratch generative model.

It uses real, dataset-derived bird-call snippets:

```text
real audited snippets -> retrieval index -> species/time selector -> variation scheduler -> 60s event layer
```

Current species:

- Horsfield's Bronze-cuckoo
- Splendid Fairywren

Human listening status:

- Horsfield's Bronze-cuckoo retrieval timeline: acceptable for smoke baseline
- Splendid Fairywren retrieval timeline: acceptable for smoke baseline

## Source Pools

Horsfield's Bronze-cuckoo:

```text
resources/site_257_bowra-dry-a/layer_c_smoke_fairywren_robin_bellbird/bronze_cuckoo_natural_core_v1/manual_audit_horsfields_bronze_cuckoo_pass24_trainset.csv
```

This pool contains 24 real snippets with human Pass labels. The earlier
six-snippet core remains the strict reference set, but the default retrieval
index now uses the expanded 24-snippet pool.

Splendid Fairywren:

```text
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/retrieval_pool_v3_target30/manual_audit_splendid_fairywren_retrieval_pass23.csv
```

This pool contains 23 real snippets with explicit human Pass labels. It combines
16 previous Pass snippets with 7 Pass snippets from the target-30 supplement
audit.

## Implemented Files

Policy:

```text
.claude/context/branches/layer-c-smoke-eval/layer_c_retrieval_policy.md
```

Retrieval code:

```text
acoustic_ai/modules/events/event_index.py
acoustic_ai/modules/events/retriever.py
acoustic_ai/modules/events/scheduler.py
acoustic_ai/modules/events/run_layer_c_retrieval.py
```

Index:

```text
acoustic_ai/data/events/retrieval/layer_c_event_index.csv
```

Index contents:

- 24 Horsfield's Bronze-cuckoo snippets
- 23 Splendid Fairywren snippets
- 47 total rows
- all rows have `verdict=Pass`

Fresh 80-candidate expansion manifests were also built:

```text
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/cuckoo/manifest.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/manifest.csv
```

Expansion status files:

```text
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/pool_expansion_summary.md
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/cuckoo/retrieval_pool_v2/candidate_manifest_80_status.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/retrieval_pool_v2/candidate_manifest_80_status.csv
resources/site_257_bowra-dry-a/layer_c_retrieval_cuckoo_fairywren/fairywren/retrieval_pool_v3_target30/manual_audit_splendid_fairywren_retrieval_pass23.csv
```

## Runtime Processing

The final smoke baseline keeps real retrieval snippets acoustically intact.
To preserve listening quality, pitch shift and time stretch are disabled by
default. The scheduler still applies conservative gain and short fade in/out
for headroom and click prevention.

Optional pitch/time variation remains available behind `--enable-variation`, but
it is not used for the final smoke bundles.

## Repro Commands

Build index:

```bash
./acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/event_index.py
```

Run final Cuckoo retrieval:

```bash
MPLCONFIGDIR=/private/tmp/mpl \
./acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/run_layer_c_retrieval.py \
  --species "Horsfield's Bronze-cuckoo" \
  --diel-bin morning \
  --season summer \
  --duration 60 \
  --seed 42 \
  --count 5 \
  --out-dir debug/layer_c/retrieval/cuckoo_final
```

Run final Fairywren retrieval:

```bash
MPLCONFIGDIR=/private/tmp/mpl \
./acoustic_ai/.venv-audiogen/bin/python acoustic_ai/modules/events/run_layer_c_retrieval.py \
  --species "Splendid Fairywren" \
  --diel-bin dawn \
  --season summer \
  --duration 60 \
  --seed 42 \
  --count 6 \
  --out-dir debug/layer_c/retrieval/fairywren_final
```

## Output Bundles

Cuckoo final:

```text
debug/layer_c/retrieval/cuckoo_final/layer_c_events.wav
debug/layer_c/retrieval/cuckoo_final/layer_c_timeline.json
debug/layer_c/retrieval/cuckoo_final/layer_c_timeline.png
debug/layer_c/retrieval/cuckoo_final/layer_c_spectrogram.png
```

Scheduled events:

| onset | event id | diel | season | score |
|---:|---|---|---|---:|
| 4.558s | 6066203 | morning | autumn | 0.9985 |
| 19.157s | 1519148 | morning | winter | 0.9696 |
| 44.874s | 9480305 | morning | winter | 0.9995 |

Fairywren final:

```text
debug/layer_c/retrieval/fairywren_final/layer_c_events.wav
debug/layer_c/retrieval/fairywren_final/layer_c_timeline.json
debug/layer_c/retrieval/fairywren_final/layer_c_timeline.png
debug/layer_c/retrieval/fairywren_final/layer_c_spectrogram.png
```

Scheduled events:

| onset | event id | diel | season | score |
|---:|---|---|---|---:|
| 4.558s | 24116609 | dawn | spring | 0.9964 |
| 18.157s | 3797024 | dawn | autumn | 0.9967 |
| 39.874s | 2247215 | dawn | summer | 0.9902 |
| 51.286s | 16425656 | dawn | summer | 0.9443 |

## Verification

Both final output WAV files were verified locally:

| Bundle | Sample rate | Duration | Peak | Events |
|---|---:|---:|---:|---:|
| cuckoo_final | 22050 Hz | 60.0s | 0.07037 | 3 |
| fairywren_final | 22050 Hz | 60.0s | 0.58215 | 4 |

Current human audit conclusion:

- Cuckoo sounds acceptable for the retrieval smoke baseline.
- Fairywren sounds acceptable for the retrieval smoke baseline.

## Limitations

- This is retrieval, not a trained generative model.
- Species correctness is high because output events are real audited snippets.
- Diversity depends on the size and quality of the retrieval library.
- Future work can add more species, richer environment matching, background-bed
  handoff, or a generation fallback.
