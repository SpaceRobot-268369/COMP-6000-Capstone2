# lucas__mvp_2__clap_knn_probe_enlarged

Layer E ambient-analysis retry using the enlarged Bowra ambient pool.
It keeps the `mvp_1` engine and changes the data source from the original
1,982-segment Layer A pool to `ambient_pool_v2` (6,093 indexed segments).

## Engine

- **k-NN** over a source-clip-disjoint index -> diel, hour, month, and
  `similar_clips` evidence.
- **Season probe** over frozen CLAP embeddings -> 4-way season.
- **Agreement / OOD gate** from cell-match anchors vs k-NN cell vote.

## Data

- Index CSV:
  `resources/site_257_bowra-dry-a/ambient_pool_v2/ambient_index.csv`
- WAV segments:
  `resources/site_257_bowra-dry-a/ambient_pool_v2/ambient_segments/`
- Candidate checkpoint:
  `model/candidates/lucas/mvp_2__clap_knn_probe_enlarged/season_probe.pt`

## Reproduce

From this attempt root, with `acoustic_ai/.venv` available and
`ambient_pool_v2` materialised by DVC:

```bash
../../../../../acoustic_ai/.venv/bin/python code/embed_segments.py --force
../../../../../acoustic_ai/.venv/bin/python code/build_split.py
../../../../../acoustic_ai/.venv/bin/python code/train_probe.py
../../../../../acoustic_ai/.venv/bin/python code/eval.py
```

Single-clip analysis:

```bash
../../../../../acoustic_ai/.venv/bin/python code/handler.py /path/to/upload.wav
```

## MVP Bar

- season acc > 0.60 and better than k-NN season
- diel acc >= 0.683
- hour MAE < 2.5 h
- month MAE < 2.0 mo

## Results

Retrained on serverB, seed 42, source-clip-disjoint split:

| metric | result | bar |
|---|---:|---|
| probe season acc | 0.553 | > 0.60 |
| k-NN season acc | 0.512 | baseline |
| diel acc | 0.706 | >= 0.683 |
| hour MAE | 1.97 h | < 2.5 h |
| month MAE | 1.80 mo | < 2.0 mo |

Verdict: the MLP season probe improves over k-NN by `+0.041`, but it still
misses the `0.60` season bar. The environmental k-NN heads pass their bars.
