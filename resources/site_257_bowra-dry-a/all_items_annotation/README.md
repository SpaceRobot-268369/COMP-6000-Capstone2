# All-Items Annotation Dataset

This directory holds annotation CSVs downloaded for recordings in
`resources/site_257_bowra-dry-a/site_257_all_items.csv`.

It does **not** hold the full per-event segment audio archive. Any previously
downloaded event segment audio was intentionally deleted because the complete
segment set is too large for the current stage. Keep this directory as the
annotation index, then download only the small event subset needed for a smoke
test or a later curated training run.

## Purpose

Training data source for **Layer C — Event layer**: event snippet extraction,
event index construction, and event scheduler training.

Layer C generates or retrieves short acoustic event clips (bird calls, insect
choruses, frogs, wind gusts, anthropogenic sounds, etc.) and schedules them
over the ambient bed produced by Layer A. To do that, it needs:

1. A **snippet bank** — short audio clips of real events extracted from field
   recordings, each tagged with event class and environmental context.
2. An **event index** — a lookup table mapping event class × time-of-day ×
   season × environmental conditions to expected occurrence probability.
3. **Training data** for fine-tuning a generative model (e.g. AudioLDM2 LoRA)
   to synthesise events when no real snippet is available for a given class.

These annotation CSVs are the index into all three. Each row records an
annotated acoustic event with its start/end time within the recording, so the
pipeline can extract exactly the right audio segment later if that event is
selected. Segment audio is not required just to build filters, manifests, or
smoke-test plans.

### Why the full 12,251-item archive rather than just the 287 MVP items

The 287-item MVP sample (`site_257_filtered_items.csv`) was designed for
ambient (Layer A) training — diel-balanced, seasonally spread. Event coverage
is a different problem: rare species and infrequent event types may appear only
a handful of times across the whole archive. Using the full 12,251-item
annotation set maximises the number of distinct event classes and examples
available for the snippet bank and generative fine-tuning, without needing to
download all 125 GB of audio upfront.

### Why not all 12,251 items are usable

Many recordings have no human or automated annotations at all — the API returns
an empty CSV. Items with empty annotation files are excluded from snippet
extraction and the event index. Only recordings with at least one annotation
row contribute to Layer C data. The exact count of annotated vs empty items is
determined at index-build time.

## Source

Downloaded via:

```bash
python3 script/download/download_site_257_annotations.py \
  --csv-path resources/site_257_bowra-dry-a/site_257_all_items.csv \
  --output-dir resources/site_257_bowra-dry-a/all_items_annotation \
  --start-item 1 --end-item 12251 --workers 10
```

One CSV per annotated recording is kept under `site_257_item_<recording_id>/`.
Empty files or folders may exist as markers so the download does not retry
recordings with no usable annotation rows.

### Segment audio status

Per-event segment audio is **not currently retained** for the full dataset.
Do not run a broad event-segment download for all annotations at this stage.
The estimated sizes below are included to explain why: even filtered candidates
can require tens of GB before deduplication.

For the current Layer C smoke test, generate a tiny exact-event manifest and
download only those one or two event clips:

```bash
python3 script/dataset/build_layer_c_smoke_manifest.py

python3 script/download/download_site_257_event_segments.py \
  --event-manifest resources/site_257_bowra-dry-a/layer_c_smoke_test/manifest.csv \
  --output-dir resources/site_257_bowra-dry-a/layer_c_smoke_test/segments \
  --min-score 0.9 \
  --min-duration 1.0 \
  --max-duration 10.0 \
  --workers 2
```

## Key annotation fields

| Field | Description |
|---|---|
| `audio_recording_id` | Links back to the recording in `site_257_all_items.csv` |
| `event_start_seconds` | Start offset of the event within the recording |
| `event_end_seconds` | End offset of the event within the recording |
| `event_duration_seconds` | Derived duration (end − start) |
| `common_name_tags` | Human-readable event label (e.g. "Superb Fairywren") — populated for a subset of A2O-verified taxa |
| `species_name_tags` | Scientific name where available |
| `other_tags` | In this dataset, contains the BirdNET-predicted species (common name + Latin binomial) for the bulk of events. Despite the name, **no non-species labels were observed** in the actual data (see Dataset Analysis below). |
| `score` | BirdNET confidence (range 0.500–1.000; hard threshold at 0.5) |
| `low_frequency_hertz` / `high_frequency_hertz` | Frequency band of the event — populated only for the 470 manual Raven annotations; blank for BirdNET imports |
| `verification_consensus` | Annotation confidence from human verification — currently unpopulated for all events |
| `audio_event_import_file_name` | Source of the event: `BirdNET.results.csv` or a Raven `*.selections.txt` file |

---

## Dataset Analysis

Analysis run on the full set of downloaded annotation CSVs (as of 2026-05-06).

### [1] Coverage
- **1,655 item folders**, all non-empty
- **126,915 annotation events** across 1,655 unique recordings (avg 76.7 events/recording)

### [2] Event Source

| Source | Events | Share |
|---|---|---|
| BirdNET (automated classifier) | 126,445 | 99.6% |
| Manual Raven selection tables (5 files) | 470 | 0.4% |

The 470 manual annotations are the only events with frequency-box data populated — humans drew boxes around them in Raven Pro.

### [3] Event Type
**100% bird species detections.** No insects, no weather, no mammals, no anthropogenic sounds.

- BirdNET (99.6% of events) is a bird-only classifier — it cannot output non-bird classes.
- The 5 manual Raven selection files (0.4% of events) were also tagged exclusively with bird species, suggesting an annotation pass targeting bird calls specifically.
- Only 2 events in the entire dataset are tagged `"Unknown"` — a placeholder for unidentifiable bird calls, not a non-bird category.

> **Implication for Layer C:** if the soundscape needs insects/cicadas, weather events, or anthropogenic sounds, this annotation set will not provide them — a separate detector or annotation pass is required.

### [4] Species Distribution
- **~158 unique bird species** (most species sit in the `other_tags` column, paired as common name + scientific name)
- Heavy long-tail distribution

**Top 10 species by event count:**

| Common name | Events |
|---|---|
| Splendid Fairywren | 22,458 |
| Chestnut-rumped Thornbill | 15,862 |
| Southern Boobook | 14,052 |
| Crested Bellbird | 12,026 |
| White-browed Woodswallow | 10,363 |
| Red-capped Robin | 8,337 |
| Superb Fairywren | 3,940 |
| Horsfield's Bronze-cuckoo | 3,772 |
| Rufous Whistler | 3,211 |
| Australian Owlet-nightjar | 2,766 |

### [5] Categorization Potential
The data only provides species names — no built-in category field. Useful groupings can be derived externally:

| Category type | How | Example bins |
|---|---|---|
| Taxonomic family | Species → family lookup | owls, honeyeaters, parrots, cuckoos, robins, fairywrens, woodswallows |
| Diel activity | External knowledge | nocturnal (Boobook, Owlet-nightjar, Spotted Nightjar) vs diurnal (rest) |
| Body size / loudness proxy | External lookup | small passerines vs large corvids/cockatoos |
| Vocalization type (call/song/alarm) | **Not in data** — needs ML or manual labeling |
| Source / quality | `audio_event_import_file_name` | BirdNET (auto, no freq box) vs Raven (manual, with freq box) |

### [6] BirdNET Confidence Scores
N=126,445 — min **0.500** (hard threshold), median **0.761**, mean **0.762**, max **1.000**

| Bucket | Count | Share |
|---|---|---|
| 0.5–0.7 | 49,670 | 39.3% |
| 0.7–0.9 | 44,029 | 34.8% |
| ≥0.9 | 32,746 | 25.9% |

Filtering at score ≥ 0.7 → **~76,775** high-confidence events.

### [7] Event Duration
N=126,915 — min **0.27 s**, median **3.00 s**, mean **6.02 s**, max **21.23 s**

| Bucket | Count | Share |
|---|---|---|
| ≤1 s | 168 | 0.1% |
| 1–3 s | 69,405 | 54.7% |
| 3–5 s | 21 | 0.0% |
| 5–10 s | 35,099 | 27.7% |
| 10–30 s | 22,222 | 17.5% |
| >30 s | 0 | 0.0% |

**82.4% (104,525)** fall in the 1–10 s window — usable for Layer C snippets.
The bimodal pattern (3 s and ~6 s clusters) reflects BirdNET's native 3 s detection windows plus merged contiguous detections.

### [8] Frequency Range
- Only **470 / 126,915 events (0.4%)** have `low_frequency_hertz` / `high_frequency_hertz` populated — these are the manual Raven annotations.
- BirdNET imports leave these blank (it only marks time + species, not a frequency box).
- Layer C cannot rely on this column; if frequency-band info is needed (e.g. for band-aware mixing), it must be computed at snippet-extraction time from the spectrogram.

### [9] Verification Status
- **0 events** have human verification — these are raw BirdNET predictions.
- Treat the confidence score as the only quality signal; consider a manual audit for top species before using snippets verbatim.

### [10] Temporal Distribution (AEST)

**By diel bin:**

| Bin | Events |
|---|---|
| dawn (05–08) | 37,015 |
| morning (08–11) | 22,248 |
| midday (11–13) | 13,379 |
| afternoon (13–16) | 17,501 |
| dusk (16–19) | 11,889 |
| evening (19–22) | 7,135 |
| night (22–05) | 17,748 |

Dawn dominates (expected for bird activity); evening is lowest.

**By year:**

| Year | Events |
|---|---|
| 2019 | 19,525 |
| 2020 | 18,367 |
| 2023 | 17,735 |
| 2024 | 55,373 |
| 2025 | 15,915 |

2024 dominates; **2021–2022 absent** (matches the known A2O archive gap noted in `CLAUDE.md`); 2025 is partial.

### Key Takeaways for Layer C
1. **Bird-only event library.** If the soundscape needs insect/cicada, weather, or anthropogenic events, a different detector or annotation pass is required.
2. **Score ≥ 0.7 filter** → ~77k high-quality candidates.
3. **Combined "usable" filter (1–10 s + score ≥ 0.7)** → roughly **~63k candidate snippets** — comfortably enough for a rich event library.
4. **No verification**, so de-duplication / manual audit recommended for top species before using snippets verbatim.
5. **Frequency bands missing** for 99.6% of events — compute on-the-fly from spectrograms if needed.

## Layer C smoke-test filtering policy

The full annotation index is useful, but the full segment-audio extraction is
intentionally too large for smoke testing. A Layer C smoke dataset should prove
the event path end-to-end with a small number of species-level event types and
enough clips per type to make a tiny LoRA run meaningful.

### Event-type scope for the smoke stage

Layer C smoke testing should train or validate **bird vocal events only**,
because this annotation dataset contains only bird species detections. For the
current stage, treat each event class as a species-level bird-call class, not a
generic sound category.

| Stage | Event types to use | Event types to exclude |
|---|---|---|
| Smoke test | One or two species-level bird call classes. Default: `Southern Boobook` / `Ninox boobook` and `Splendid Fairywren` / `Malurus splendens` | Insects/cicadas, frogs, mammals, rain, wind gusts, human/vehicle/anthropogenic sounds, generic `Unknown` |
| Next small pilot | 2-4 high-count bird species with clear diel contrast, e.g. nocturnal owl/nightjar plus common diurnal passerines | Same exclusions unless a separate detector/annotation source is added |
| Full Layer C training | Curated bird species subset from the usable filter, after manual audit and de-duplication | Non-bird events from this annotation set, because they are not present |

For the smoke stage, the goal is not broad ecological coverage. The goal is to
verify that Layer C can:

- select annotated events deterministically for each event type
- download only the selected buffered audio clips
- attach species/event metadata to the snippet
- schedule a sparse event layer over an ambient bed
- pass the event layer into the mixer without needing the full archive

Do not train smoke-stage Layer C on insects, weather, or anthropogenic events
from this dataset. Those sounds must come from separate Layer B assets, a
separate detector, or a future annotation pass.

Default smoke-test target:

| Policy item | Value |
|---|---|
| Event types | `Southern Boobook` / `Ninox boobook`; `Splendid Fairywren` / `Malurus splendens` |
| Event source | `BirdNET.results.csv` |
| Score | `>= 0.9` |
| Raw event duration | `1.0–10.0 s` |
| Diel preference | Boobook: AEST `night`, then `evening`, then `dusk`; Fairywren: `dawn`, then `morning`, then `afternoon`, then `dusk` |
| Diversity | Prefer distinct recordings before taking multiple events from one recording |
| Clip buffer | `±3.0 s`, clamped to recording bounds |
| Target size | `50 segments per event type` |

Why this policy:

- Southern Boobook is the current Layer C smoke-test species noted in the
  project architecture.
- Splendid Fairywren is the highest-count species in this annotation set and
  provides a contrasting common dawn/diurnal passerine event type.
- Both species are common enough in the archive to select high-confidence
  examples without touching the heavy long-tail dataset.
- Event-type-specific diel filters match expected activity and reduce the
  chance of selecting obvious temporal false positives.
- The `1–10 s` window matches the usable Layer C snippet window described above.
- Fifty buffered clips per type is still small enough for smoke testing but
  large enough to test a tiny AudioGen LoRA training run more realistically than
  a two-clip wiring check.

Generate the manifest:

```bash
python3 script/dataset/build_layer_c_smoke_manifest.py
```

This writes:

```text
resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/manifest.csv
```

Download only those exact event audio segments:

```bash
python3 script/download/download_site_257_event_segments.py \
  --event-manifest resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/manifest.csv \
  --output-dir resources/site_257_bowra-dry-a/smoking_test_1_layer_C_dataset_1/segments \
  --min-score 0.9 \
  --min-duration 1.0 \
  --max-duration 10.0 \
  --workers 2
```

The smoke manifest is a filter artifact, not the training dataset. Full Layer C
training should later use a curated subset from the broader usable filter
(`score >= 0.7`, `1–10 s`) plus manual audit/de-duplication before any larger
segment download or model fine-tuning.

## Segment extraction — buffer policy

When extracting selected audio snippets for the smoke set, snippet bank, or
generative training, use a **3-second buffer on each side** of the annotated
event window:

```
extracted_start = max(0, event_start_seconds − 3.0)
extracted_end   = min(recording_duration, event_end_seconds + 3.0)
```

The 3-second default is chosen because event durations are not known in
advance at this stage — short calls (~0.5 s) and longer bursts (~5–8 s) both
exist in the data. A fixed 3 s buffer:

- Captures natural acoustic onset and offset for all expected event lengths
- Keeps total clip length within AudioLDM2's practical generation window (~5–12 s)
- Preserves enough surrounding ambient for the generative model to learn
  event-in-context rather than event-in-silence

Do not zero-pad or fade the buffer — the surrounding ambient should be the
real background from that recording.

### Estimated full-download size

Assuming a reference rate of **13.2 MB per 5 min** (≈ 2.64 MB/min, ≈ 352 kbps —
matches the project's mono FLAC encoding) and the ±3 s buffer policy above,
each extracted clip is `event_duration + 6 s`:

| Filter | Events | Audio time | Estimated size |
|---|---|---|---|
| All events | 126,915 | ~25,425 min (~424 h) | **~65.5 GB** |
| Score ≥ 0.7 | ~76,775 | ~15,380 min | **~39.6 GB** |
| Score ≥ 0.7 + duration 1–10 s | ~63,000 | ~10,500 min | **~27 GB** |

> **Real on-disk size will likely be smaller** — many BirdNET events on the
> same recording sit close together in time, so their ±3 s buffered windows
> overlap. If the extractor merges overlapping windows into single clips,
> actual storage may shrink by ~20–40% depending on event density.

> **Current policy:** these are estimates only, not a required download plan.
> The full segment archive should stay absent unless the project explicitly
> moves from smoke testing to curated Layer C training data preparation.

## Scope

- **Layer C only.** Layer A (ambient) uses `site_257_filtered_items.csv` and
  the DVC-tracked clip archive. Layer B (weather) uses its own asset index.
  Neither should consume from this directory.
- Annotation rows here feed event filtering, manifest generation, the event
  index, selected snippet extraction, and later generative fine-tuning. They are
  not used for ambient or weather modeling.
