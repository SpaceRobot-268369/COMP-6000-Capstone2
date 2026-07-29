# ai-mock — the demo branch's fake AI service

**Nothing here is model output.** This service exists so the `demo` branch can be
walked through end to end without serverB, a GPU, or any model weights. It speaks
the same HTTP contract as the real FastAPI worker (`acoustic_ai/server/server.py`),
so **no backend or frontend code changes for the demo** — only
`services/dev/docker-compose.yml` points at it instead of the ssh tunnel.

## What it actually does

It replays pre-baked files. There is **no audio processing at request time**: the
container has no numpy, no torch, no matplotlib — just FastAPI, uvicorn and PyYAML.
A request is scored against the baked preset set, the winner's WAV/PNG/JSON are
read off disk and base64'd into the response envelope.

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Always healthy, `statusKey: "online"` (so the reconnect path never fires) |
| `GET /layers` | Real `acoustic_ai/registry.yaml`, parsed, with `available: true` forced on every attempt — nothing is loadable here, and the UI disables its controls on `available: false` |
| `POST /generation/parse` | Keyword parser mirroring `frontend/src/demo/resolvePrompt.js`, plus a validity gate that produces `ok` / `corrected` / `rejected` |
| `POST /generation/render` | Replays the nearest pre-baked Layer D mix; `include_stems` replays its A/B/C stems |
| `POST /layers/{l}/attempts/{a}/generate` | Replays a single vendored stem: layer A per cell, layer B per weather control, layer C per species |
| `POST /analysis/run` | Identifies the upload by MD5 against the preset recordings, returns that cell's pre-authored report |
| `POST /analysis/narrative` | Template narrator, immersive + analytical registers |
| `POST /layers/{l}/attempts/{a}/analyze` | Slices the same canned bundle per Layer E head |

Every response carries `mock: true` somewhere in its metadata or report.

## Known limits — read before demoing

- **Only the preset prompts and preset recordings are faithful.** Off-preset input
  resolves to the nearest bake rather than erroring, so no page dead-ends — but the
  answer is not about *that* input. The report says so: an unrecognised upload gets
  `mock_source: "unrecognised_upload"` and an extra line in `limitations`.
- **Seed is cosmetic.** It is echoed back in the response metadata, but the audio for
  a given preset is a fixed file. Same seed and different seed sound identical.
- **Confidences, similarity scores and detection windows are invented**, not measured.

## Fixtures

Baked by `tools/build_fixtures.py`, committed as plain git files (not DVC), so the
demo runs from a fresh clone with no `dvc pull`:

```
fixtures/
  layers/      sample tiers the Express backend serves directly (AI_LAYERS_ROOT
               points here) — 16 layer A ambient cells, layer B weather stems,
               layer C species references
  events/      one reference call + spectrogram per Layer C species (63)
  generation/  six pre-baked Layer D mixes, one per demo prompt, each with stems
  analysis/    16 per-cell canned reports + presets.json (MD5 → cell)
```

Roughly 68 MB in total. This deliberately breaks the repo's usual
no-binaries-in-git rule (`CLAUDE.md` → pre-commit audit); it is scoped to this
branch, which is never merged to `main`.

### Rebuilding

Needs a checkout with the real artefacts pulled (`dvc pull`) and the project venv:

```bash
./acoustic_ai/.venv/bin/python services/dev/ai-mock/tools/build_fixtures.py
```

Edit `PRESETS` in that script to change the demo prompts, and `build_reports.py`
to change what the analysis reports claim. Keep the preset list in step with the
chips in `frontend/src/components/PromptChat.jsx` — a chip with no matching bake
still returns audio, just not the audio it describes.

## Config

| Env | Default | Meaning |
|---|---|---|
| `MOCK_FIXTURES_ROOT` | `/fixtures` | Mounted fixture tree |
| `MOCK_REGISTRY_PATH` | `/registry.yaml` | Mounted `acoustic_ai/registry.yaml` |
| `MOCK_LATENCY_MS_PARSE` | `800` | Fake think-time |
| `MOCK_LATENCY_MS_GENERATE` | `3000` | Fake think-time — the staged progress UI needs room to read as real work |
| `MOCK_LATENCY_MS_ANALYSIS` | `1500` | Fake think-time |
| `MOCK_LATENCY_MS_NARRATIVE` | `400` | Fake think-time |
