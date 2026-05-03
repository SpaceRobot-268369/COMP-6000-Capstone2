# Site 257 Annotation Audit

Scope: downloaded A2O annotation CSVs only. No event index, snippet manifest, classifier, or Layer C runtime was built in this step.

## Coverage Summary

- Annotation CSV files found: 287
- Files with at least one event row: 29
- Empty annotation files: 258
- Parse errors: 0
- Total event rows: 3252
- Recordings with any label tags: 29
- Recordings with common-name tags: 18
- Recordings with species-name tags: 1
- Recordings with score/activity rows: 29

## Row-Level Signals

- Rows with usable start/end timing: 3252 / 3252 (100.0%)
- Rows with score: 3252 / 3252 (100.0%)
- Rows with common-name tags: 577 / 3252 (17.7%)
- Rows with species-name tags: 439 / 3252 (13.5%)
- Rows with other tags: 2813 / 3252 (86.5%)
- Rows with any label tags: 3252 / 3252 (100.0%)
- Score-only rows without common/species/other labels: 0 / 3252 (0.0%)
- Rows with verification fields populated: 0 / 3252 (0.0%)
- Rows with import/source fields populated: 3252 / 3252 (100.0%)
- Rows with frequency bounds: 0 / 3252 (0.0%)

## Schema

- Unique schemas found: 1
- Dominant schema file count: 287
- Dominant schema columns: 46

All expected audit fields are present in every parsed CSV schema.

## Top Common Name Tags

| Common name | Rows |
|---|---:|
| 107:Southern Boobook | 439 |
| 1:Rufous Whistler | 42 |
| 15:Pied Butcherbird | 37 |
| 57:Australian Owlet-nightjar | 8 |
| 46:Noisy Miner | 8 |
| 53:Rainbow Bee-eater | 7 |
| 35:Little Friarbird | 6 |
| 41:Galah | 5 |
| 29:Australian Magpie | 5 |
| 50:Brown Honeyeater | 4 |
| 72:Mistletoebird | 4 |
| 48:White-throated Treecreeper | 2 |
| 6:Sacred Kingfisher | 2 |
| 60:Striped Honeyeater | 2 |
| 64:Channel-billed Cuckoo | 1 |
| 62:Tawny Grassbird | 1 |
| 9:Olive-backed Oriole | 1 |
| 47:Varied Sittella | 1 |
| 77:Spotted Pardalote | 1 |
| 20:White-browed Scrubwren | 1 |

## Top Species Name Tags

| Species name | Rows |
|---|---:|
| 103:Ninox boobook | 439 |

## Top Other Tags

| Other tag | Rows |
|---|---:|
| 26670:White-browed Woodswallow:general | 666 |
| 26669:Artamus superciliosus:general | 666 |
| 26887:Acanthiza uropygialis:general | 332 |
| 153:Chestnut-rumped Thornbill:general | 332 |
| 26679:Oreoica gutturalis:general | 324 |
| 164:Crested Bellbird:general | 324 |
| 26703:Malurus splendens:general | 192 |
| 160:Splendid Fairywren:general | 192 |
| 26872:Masked Woodswallow:general | 190 |
| 26871:Artamus personatus:general | 190 |
| 26657:Coracina novaehollandiae:general | 125 |
| 243:Black-faced Cuckooshrike:general | 125 |
| 26660:Corvus coronoides:general | 124 |
| 152:Australian Raven:general | 124 |
| 26764:Cincloramphus mathewsi:general | 120 |
| 166:Rufous Songlark:general | 120 |
| 26769:Petroica goodenovii:general | 109 |
| 168:Red-capped Robin:general | 109 |
| 26737:Chrysococcyx basalis:general | 81 |
| 211:Horsfield's Bronze-cuckoo:general | 81 |
| 26819:Black Honeyeater:general | 70 |
| 26818:Sugomel nigrum:general | 70 |
| 26692:Cracticus torquatus:general | 60 |
| 26693:Gray Butcherbird:general | 60 |
| 26665:Pachycephala rufiventris:general | 42 |
| 26694:Cracticus nigrogularis:general | 37 |
| 26683:Willie-wagtail:general | 36 |
| 26682:Rhipidura leucophrys:general | 36 |
| 26721:Melopsittacus undulatus:general | 33 |
| 242:Budgerigar:general | 33 |

## Import Sources

| Import file | Rows |
|---|---:|
| BirdNET.results.csv | 3252 |

### Import Names

| Import name | Rows |
|---|---:|
| Import for BirdNET classifier for A2O | 3252 |

## Score Bands

| Score band | Rows |
|---|---:|
| 0.5-0.75 | 1545 |
| >=0.9 | 892 |
| 0.75-0.9 | 815 |

## Duration Bands

| Duration band | Rows |
|---|---:|
| 5-30s | 1658 |
| 1-5s | 1594 |

## Verification Consensus

| Consensus | Rows |
|---|---:|
| blank | 3252 |

## Event Rows By Recording

| Recording ID | Event rows |
|---|---:|
| 1402652 | 452 |
| 5296 | 443 |
| 1669291 | 413 |
| 1401632 | 222 |
| 215683 | 217 |
| 1672079 | 203 |
| 214654 | 201 |
| 1539179 | 192 |
| 445582 | 148 |
| 1680295 | 129 |
| 1313384 | 113 |
| 1401247 | 63 |
| 1676441 | 60 |
| 5493 | 55 |
| 1680188 | 50 |
| 214508 | 50 |
| 1313285 | 42 |
| 445765 | 42 |
| 1678750 | 32 |
| 215727 | 31 |
| 1534156 | 30 |
| 215192 | 15 |
| 1401305 | 12 |
| 1401311 | 11 |
| 1679611 | 8 |
| 215905 | 8 |
| 1314118 | 5 |
| 216095 | 4 |
| 215918 | 1 |
| 1312975 | 0 |

## Audit Interpretation

- Species-name labels exist, but recording coverage is sparse; retrieval or pseudo-label assistance is safer than supervised multiclass training for MVP.
- Import/source metadata is available and should be retained in later indexes for reliability filtering.
- Event timing fields are consistently usable, so the next preprocessing step can map events to 300-second clips.
- Next step should be event-index construction only after accepting this audit result.
