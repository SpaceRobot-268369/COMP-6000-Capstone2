# Analysis Page — Test Cases

28 clips sampled across all 16 season × sample_bin groups, with varied temperature,
humidity, wind, and precipitation. Use these to verify that the analysis page returns
estimated environmental conditions that are plausible given the ground truth.

All files are at:
`resources/site_257_bowra-dry-a/downloaded_clips/site_257_item_<id>/<filename>`

---

## How to Interpret Results

- **Season / sample_bin** should match exactly (or be adjacent) — these are the strongest
  signal since the model was trained with stratified groups.
- **Temperature** estimate within ±5–8 °C of ground truth is good.
- **Humidity**, **wind speed**, **precipitation** will have higher variance — the model
  captures broad patterns, not precise meteorology.
- **Confidence** > 0.6 = strong nearest-neighbour match; < 0.4 = ambiguous clip.

---

## Test Cases

| # | File | Season | Bin | Date | Hour | Temp °C | Humidity % | Wind m/s | Wind Gust | Precip mm | Solar W/m² | Pressure kPa | Temp Max | Temp Min | Days Since Rain | Daylight hrs |
|---|------|--------|-----|------|------|---------|-----------|---------|-----------|-----------|-----------|-------------|---------|---------|----------------|-------------|
| 1 | `site_257_item_215467_clip_003.webm.wav` | summer | dawn | 2020-01-10 | 06:00 | 33.36 | 21.87 | 4.05 | 3.62 | 0.0 | 143.38 | 99.42 | 46.92 | 30.25 | 66 | 13.95 |
| 2 | `site_257_item_1401611_clip_023.webm.wav` | spring | night | 2023-11-21 | 22:00 | 28.10 | 37.39 | 2.78 | 4.09 | 0.0 | 0.0 | 100.09 | 37.03 | 19.25 | 1 | 13.74 |
| 3 | `site_257_item_1402288_clip_019.webm.wav` | summer | afternoon | 2023-12-25 | 14:00 | 32.05 | 12.05 | 6.75 | 6.91 | 0.0 | 968.28 | 99.00 | 32.05 | 20.64 | 1 | 14.07 |
| 4 | `site_257_item_216610_clip_014.webm.wav` | autumn | night | 2020-04-15 | 22:00 | 19.06 | 62.69 | 2.03 | 4.60 | 0.0 | 0.0 | 100.46 | 31.07 | 12.26 | 4 | 11.35 |
| 5 | `site_257_item_5421_clip_012.webm.wav` | spring | night | 2019-09-23 | 22:00 | 13.07 | 26.51 | 2.80 | 4.95 | 0.0 | 0.0 | 101.45 | 24.27 | 9.12 | 77 | 12.11 |
| 6 | `site_257_item_1313286_clip_001.webm.wav` | autumn | night | 2023-05-10 | 22:00 | 10.90 | 53.11 | 1.40 | 3.86 | 0.0 | 0.0 | 101.17 | 21.95 | 5.52 | 10 | 10.73 |
| 7 | `site_257_item_1402117_clip_006.webm.wav` | summer | morning | 2024-02-07 | 08:00 | 24.21 | 37.07 | 5.82 | 6.48 | 0.0 | 545.55 | 100.30 | 34.29 | 21.02 | 0 | 13.36 |
| 8 | `site_257_item_1672819_clip_024.webm.wav` | winter | dawn | 2025-06-20 | 06:00 | 1.84 | 100.00 | 1.64 | 2.68 | 0.0 | 0.0 | 101.31 | 15.54 | 2.37 | 15 | 10.20 |
| 9 | `site_257_item_214659_clip_008.webm.wav` | spring | afternoon | 2019-11-03 | 14:00 | 22.30 | 84.44 | 4.79 | 7.10 | 220.58 | 71.00 | 99.88 | 24.91 | 17.71 | 0 | 13.31 |
| 10 | `site_257_item_1313484_clip_002.webm.wav` | autumn | dawn | 2023-05-23 | 05:00 | 5.31 | 62.60 | 1.30 | 2.21 | 0.0 | 0.0 | 101.43 | 20.25 | 5.05 | 7 | 10.47 |
| 11 | `site_257_item_215473_clip_003.webm.wav` | summer | night | 2020-01-10 | 22:00 | 38.68 | 17.65 | 5.25 | 8.53 | 0.0 | 0.0 | 99.17 | 47.62 | 30.72 | 67 | 13.95 |
| 12 | `site_257_item_1313184_clip_009.webm.wav` | autumn | dawn | 2023-04-29 | 06:00 | 14.53 | 81.07 | 6.01 | 4.88 | 5.88 | 0.0 | 99.85 | 32.19 | 19.92 | 29 | 10.99 |
| 13 | `site_257_item_1539179_clip_003.webm.wav` | winter | morning | 2024-07-18 | 09:00 | 10.54 | 77.90 | 3.18 | 5.12 | 0.03 | 259.20 | 100.71 | 13.21 | 6.10 | 8 | 10.43 |
| 14 | `site_257_item_215469_clip_004.webm.wav` | summer | afternoon | 2020-01-10 | 14:00 | 47.62 | 10.41 | 4.81 | 8.53 | 0.0 | 744.28 | 99.01 | 47.62 | 30.72 | 67 | 13.95 |
| 15 | `site_257_item_1676387_clip_024.webm.wav` | autumn | night | 2025-03-02 | 22:00 | 31.59 | 23.36 | 3.95 | 5.04 | 0.0 | 0.0 | 99.78 | 42.29 | 27.33 | 19 | 12.68 |
| 16 | `site_257_item_1402286_clip_003.webm.wav` | summer | night | 2023-12-25 | 22:00 | 21.76 | 26.76 | 3.16 | 6.91 | 0.0 | 0.0 | 99.32 | 32.05 | 20.64 | 1 | 14.07 |
| 17 | `site_257_item_1401151_clip_005.webm.wav` | summer | dawn | 2024-01-02 | 05:00 | 26.70 | 54.07 | 3.40 | 7.69 | 0.30 | 14.02 | 99.96 | 39.71 | 23.50 | 8 | 14.03 |
| 18 | `site_257_item_5495_clip_002.webm.wav` | winter | dawn | 2019-08-18 | 06:00 | 9.52 | 38.00 | 2.95 | 2.24 | 0.0 | 0.0 | 100.22 | 26.42 | 8.30 | 40 | 11.08 |
| 19 | `site_257_item_5380_clip_014.webm.wav` | spring | morning | 2019-09-01 | 08:00 | 13.91 | 53.29 | 2.41 | 3.37 | 0.0 | 368.38 | 100.87 | 23.57 | 6.23 | 54 | 11.47 |
| 20 | `site_257_item_1676791_clip_018.webm.wav` | summer | morning | 2024-12-17 | 08:00 | 32.87 | 43.25 | 4.17 | 5.37 | 0.0 | 646.83 | 99.47 | 41.59 | 25.63 | 8 | 14.07 |
| 21 | `site_257_item_1680712_clip_005.webm.wav` | winter | night | 2025-06-08 | 22:00 | 9.47 | 70.68 | 3.02 | 5.62 | 0.0 | 0.0 | 99.95 | 15.38 | 7.79 | 4 | 10.26 |
| 22 | `site_257_item_215474_clip_012.webm.wav` | summer | morning | 2020-01-10 | 08:00 | 39.46 | 16.96 | 6.50 | 3.62 | 0.0 | 596.42 | 99.41 | 46.92 | 30.25 | 66 | 13.95 |
| 23 | `site_257_item_1675748_clip_011.webm.wav` | winter | afternoon | 2025-06-08 | 12:00 | 15.06 | 49.67 | 5.21 | 5.62 | 0.0 | 465.92 | 99.76 | 15.38 | 7.79 | 4 | 10.26 |
| 24 | `site_257_item_1401633_clip_005.webm.wav` | summer | afternoon | 2023-12-13 | 12:00 | 40.23 | 26.20 | 4.83 | 5.96 | 0.48 | 810.88 | 99.48 | 40.23 | 25.16 | 5 | 14.05 |
| 25 | `site_257_item_1401100_clip_001.webm.wav` | spring | morning | 2023-11-03 | 09:00 | 29.04 | 15.20 | 2.63 | 3.64 | 0.0 | 846.25 | 100.21 | 32.64 | 12.72 | 28 | 13.31 |
| 26 | `site_257_item_1677392_clip_022.webm.wav` | spring | night | 2024-10-26 | 22:00 | 19.95 | 29.31 | 1.35 | 3.68 | 0.0 | 0.0 | 100.29 | 30.12 | 12.17 | 7 | 13.11 |
| 27 | `site_257_item_1534143_clip_019.webm.wav` | winter | night | 2024-06-29 | 22:00 | 17.37 | 67.74 | 3.80 | 5.04 | 0.0 | 0.0 | 100.04 | 25.71 | 7.87 | 12 | 10.22 |
| 28 | `site_257_item_1313796_clip_001.webm.wav` | winter | morning | 2023-06-13 | 09:00 | 16.15 | 41.19 | 3.04 | 4.80 | 0.0 | 333.33 | 100.18 | 25.85 | 10.84 | 5 | 10.23 |

---

## Notable Edge Cases

| # | Why Interesting |
|---|----------------|
| 8 | Coldest clip — winter dawn, 1.84 °C, 100% humidity (frost conditions) |
| 9 | Heavy rain — 220 mm precipitation, high humidity (84%), unusual for Bowra |
| 11 | Hottest night — 38.68 °C at 22:00, very dry (17% humidity) |
| 14 | Extreme heat — 47.62 °C summer afternoon, driest clip (10% humidity) |
| 1 | Long dry spell — 66 days since rain |
| 5 | Very long dry spell — 77 days since rain |
| 12 | Post-rain autumn dawn — 5.88 mm recent precip, high humidity |
| 17 | Recent rain summer dawn — 0.3 mm, high humidity for summer |

---

## Coverage Summary

| Season | Dawn | Morning | Afternoon | Night |
|--------|------|---------|-----------|-------|
| Summer | #1, #17 | #7, #20, #22 | #3, #14, #24 | #11, #16 |
| Autumn | #10, #12 | — | — | #4, #6, #15 |
| Winter | #8, #18 | #13, #28 | #23 | #21, #27 |
| Spring | — | #19, #25 | #9 | #2, #5, #26 |

---

## File Locations

All files are under:
```
resources/site_257_bowra-dry-a/downloaded_clips/site_257_item_<recording_id>/
```

Example full path:
```
resources/site_257_bowra-dry-a/downloaded_clips/site_257_item_215467/site_257_item_215467_clip_003.webm.wav
```
