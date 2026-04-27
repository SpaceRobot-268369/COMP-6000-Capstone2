# Workflow Diagrams — Stage 3 (Implementation)

---

## Pipeline 1 — Data Collection & Preprocessing

```
Australian Acoustic Observatory (A2O)
         │
         │  script/sample_mvp_dataset.py
         │  (diel stratified sampling, seed=42)
         ▼
resources/site_257_bowra-dry-a/
  site_257_filtered_items.csv         ← 287 recordings, 4 diel bins
         │
         │  script/download_site_257_clips.py
         │  (parallel HTTP download, max 10 retries)
         ▼
resources/.../downloaded_clips/
  site_257_item_{id}/
    clip_001.webm  clip_002.webm ...  ← 6,148 .webm files (12 bad = 422 errors)
         │
         │  ffmpeg (manual step)
         │  ffmpeg -i clip_001.webm clip_001.webm.wav
         ▼
    clip_001.webm.wav  ...            ← 6,148 .wav files at 22,050 Hz mono
         │
         │  acoustic_ai/precompute_spectrograms.py
         │  (parallel ProcessPoolExecutor, librosa mel)
         ▼
    clip_001.webm.npy  ...            ← 6,148 .npy files, shape (1, 128, 12919)
                                         float32, dB scale [-80, 0]

NASA POWER API (hourly + daily)
         │
         │  script/fetch_nasa_env_data.py
         │  (per-recording UTC timestamp lookup)
         ▼
  site_257_env_data.csv               ← 287 rows × 20 env columns

         ┌──────────────────────┐
         │ filtered_items.csv   │
         │ env_data.csv         │──► script/build_training_manifest.py
         │ downloaded_clips/    │
         └──────────────────────┘
                    │
                    ▼
  site_257_training_manifest.csv      ← 6,148 rows (clips × env features)
                                         one row per clip, 29 conditioning features
```

**Trade-offs:**
- Webm → wav appended naming (`clip.webm.wav`) avoids overwriting originals but is non-standard; tools that infer format from extension may fail
- 300 s clips are long — full spectrograms are 12,919 frames (1.65 M values); solved by crop during training
- NASA POWER gives 1-hourly data; actual recording timestamps are matched to nearest hour — ±30 min error in env features
- 12 clips (0.2%) are permanently missing from A2O server (422 errors); excluded at manifest build time

**Potential improvements:**
- Replace ffmpeg manual step with an automated conversion script integrated into the download pipeline
- Use TERN EcoPlots for higher-resolution env data (15-min intervals)
- Add NDVI (vegetation index) from MODIS as ecological conditioning signal
- Validate spectrogram shapes at precompute time (currently checked manually)

---

## Pipeline 2 — Model Training (VAE, Stage 3)

```
site_257_training_manifest.csv
         │
         │  acoustic_ai/dataset.py  (SoundscapeDataset)
         │  ├─ stratified train/val split by sample_bin (val_frac=0.15)
         │  ├─ normalise mel: (dB + 80) / 80  →  [0, 1]
         │  ├─ random 30 s crop (1,291 frames) per step
         │  └─ build 29-dim env feature vector (z-score + sin/cos + one-hot)
         ▼
  (mel tensor, env tensor, meta)  ← DataLoader, batch_size=16, num_workers=0
         │
         │  acoustic_ai/model.py  (SoundscapeModel — VAE)
         │
         │  ┌─────────────────────────────────────────────────┐
         │  │  AudioEncoder (CNN)                             │
         │  │  (1,128,T) → 4× ConvBlock+MaxPool → AvgPool    │
         │  │  → Linear → (512,)  audio embedding            │
         │  └──────────────────┬──────────────────────────────┘
         │                     │
         │  ┌──────────────────▼──────────────────────────────┐
         │  │  FusionMLP (VAE heads)                          │
         │  │  concat([audio_emb(512), env(29)]) → (541,)     │
         │  │  → shared trunk: Linear(512)→GELU→Dropout(0.1) │
         │  │                  → Linear(256)→GELU            │
         │  │  → fc_mu(256)  and  fc_log_var(256)            │
         │  │  reparameterise: z = mu + ε·σ                  │
         │  └──────────────────┬──────────────────────────────┘
         │                     │
         │  ┌──────────────────▼──────────────────────────────┐
         │  │  MelDecoder (fixed architecture)                │
         │  │  z(256) → Linear → (256, 4, 64) seed           │
         │  │  → 4× TransposeBlock(×2 upsample)              │
         │  │  → ConvTranspose2d → (1, 128, 2048)            │
         │  │  → crop to (1, 128, 1291)  [no interpolation]  │
         │  └─────────────────────────────────────────────────┘
         │
         │  Loss: MSE(recon, input) + β × KL(q(z|x) ‖ N(0,I))
         │  β = 0.01  (KL weight)
         │  Optimiser: AdamW lr=1e-4, weight_decay=1e-4
         │  Scheduler: CosineAnnealingLR T_max=30 epochs
         │  Grad clip: max_norm=1.0
         │
         │  acoustic_ai/train.py  --beta-kl 0.01
         ▼
acoustic_ai/checkpoints/
  epoch_001.pt ... epoch_030.pt       ← per-epoch checkpoints
  best.pt                             ← lowest val loss checkpoint
                                         (val loss ≈ 0.003580, KL ≈ 0.05/element)
```

**Trade-offs:**
- VAE with β=0.01 — reconstruction loss still ~6× stronger than KL; latent space is partially but not fully regularised toward N(0,1)
- MSE loss encourages slightly blurry spectrograms; perceptual/adversarial loss would sharpen outputs
- 30 s crop per step — model misses slow temporal evolution (dawn chorus builds over minutes)

**Potential improvements (Stage 4):**
- Multi-scale spectral loss to encourage sharper spectrograms
- Increase β or use β-annealing schedule for better latent regularisation
- Longer crops (60–120 s) or hierarchical temporal modelling

---

## Pipeline 3 — Latent Database Precomputation

```
best.pt  +  site_257_training_manifest.csv
         │
         │  acoustic_ai/precompute_latents.py
         │  ├─ load model, freeze weights
         │  ├─ encode all 5,318 training clips → mu (256,) each
         │  ├─ save per-clip: latents (5318, 256) + env_vecs (5318, 29)
         │  └─ group by (season × sample_bin), average per group
         ▼
acoustic_ai/latent_clips.npy       ← 5,318 per-clip latent + env vectors
                                      used for nearest-neighbour generation
acoustic_ai/latent_templates.npy   ← 16 group mean latents (fallback only)

Groups (season × sample_bin):
  autumn×afternoon (364)  autumn×dawn (347)  autumn×morning (307)  autumn×night (366)
  spring×afternoon (417)  spring×dawn (360)  spring×morning (410)  spring×night (413)
  summer×afternoon (312)  summer×dawn (303)  summer×morning (346)  summer×night (384)
  winter×afternoon (236)  winter×dawn (293)  winter×morning (207)  winter×night (253)
```

**Trade-offs:**
- Per-clip database uses the dataset-normalised env vectors; user input is raw/unnormalised — cosine similarity is directionally meaningful but not numerically exact
- top-k=10 neighbours averaged — smooths within-group variation; single nearest would be noisier

**Potential improvements (Stage 4):**
- Normalise user env input using saved dataset stats for more accurate cosine similarity
- k-means clustering of latents per group for faster approximate nearest-neighbour

---

## Pipeline 4 — Inference & Generation (Serving)

```
Browser (GenerationPage.jsx)
    │  POST /api/generation  { season, sample_bin, temperature_c, humidity_pct, ... }
    │  credentials: include (session cookie via Vite proxy)
    ▼
frontend Vite proxy  :5173/api/* → :4000/api/*
    ▼
backend/src/index.js  (Express, port 4000)
    │  requireAuth middleware  (session check)
    │  POST /api/generation → fetch → AI_SERVER_URL/generation
    ▼
acoustic_ai/server.py  (FastAPI, port 8000)
    │
    │  acoustic_ai/inference.py  generate_spectrogram()
    │  ├─ load latent_clips.npy (5,318 per-clip latents + env vectors)
    │  ├─ build 29-dim env tensor from request env dict
    │  ├─ cosine similarity → top-10 nearest clips → average latents
    │  ├─ add Gaussian noise (std=0.3)
    │  └─ model.decoder(z) → crop → denormalise → mel_db (128, 1291)
    │
    │  acoustic_ai/inference.py  mel_db_to_wav_ecoacoustic()
    │  ├─ normalise dB [-80,0] → [0,1]
    │  ├─ HiFiGANGenerator (vocoder_checkpoints/best.pt, 128-bin, 22,050 Hz)
    │  └─ peak normalise → WAV bytes
    │     [fallback: speech SpeechT5 HiFi-GAN → Griffin-Lim if vocoder missing]
    │
    │  acoustic_ai/server.py  _mel_to_png_b64()
    │  └─ matplotlib magma colourmap → PNG → base64
    │
    │  returns { ok, shape, image_b64, audio_b64 }
    ▼
Express  →  frontend
    ├─ image_b64  →  <img> in generation-canvas (full spectrogram)
    ├─ audio_b64  →  AudioPlayer (browser playback)
    └─ download buttons (PNG + WAV)
```

**Trade-offs:**
- Two-hop proxy (Vite → Express → FastAPI) adds latency; acceptable for demo but not production
- Vocoder and VAE model weights are reloaded on every request — slow for high traffic
- Nearest-neighbour search is O(N) over 5,318 clips — fast enough (~5 ms) but would need indexing at scale
- Base64 audio (~1.2 MB) is large for JSON; should use streaming or blob URL for production

**Potential improvements (Stage 4):**
- Cache model instances at FastAPI startup (lifespan event) — load once, reuse per request
- Return audio as `audio/wav` stream instead of base64 JSON
- Add `/api/transformation` endpoint (audio upload + new env conditions → transformed audio)
- Approximate nearest-neighbour index (FAISS) for faster latent lookup at scale

---

## Pipeline 5 — Analysis (Partially Implemented)

```
Browser (AnalysisPage — not yet wired to live backend)
    │  POST /api/analysis  multipart: { file: .wav, env params }
    ▼
Express  →  FastAPI  acoustic_ai/server.py  /analysis
    │
    │  acoustic_ai/inference.py  encode_clip()
    │  ├─ librosa.load → waveform
    │  ├─ waveform_to_melspec (acoustic_ai/preprocess.py)
    │  ├─ normalise + pad_or_crop
    │  └─ model.encode(mel, env) → latent z (256,)
    ▼
  { ok, latent_dim: 256, latent: [...] }   ← raw latent vector

  [NOT YET IMPLEMENTED]
  latent → acoustic indices
    ├─ ACI (Acoustic Complexity Index)
    ├─ BIO (Bioacoustic Index)
    ├─ dominant frequency band
    └─ comparison to training distribution
```

**Trade-offs:**
- Currently returns a raw 256-dim latent — not human-interpretable
- No acoustic indices computed yet; frontend shows mock data

**Potential improvements:**
- Add BirdNET inference on the uploaded clip for species-level analysis
- Compute standard acoustic indices (soundecology / librosa)
- Compare latent to template centroids → "this clip sounds most like summer/morning"
- Visualise latent position relative to training distribution (PCA/UMAP projection)

---

## Stage Comparison Summary

| Aspect | Stage 2 — Pilot | Stage 3 — Implementation (current) | Stage 4 — Refinement (next) |
|--------|----------------|-------------------------------------|------------------------------|
| Latent space | Unregularised autoencoder | VAE with KL loss (β=0.01) | Better regularisation (β-annealing) |
| Generation strategy | 16 group mean latents + noise | Nearest-neighbour (5,318 clips, all 29 env features) | Latent diffusion conditioned on env |
| Decoder | 4×4 seed + 10× bilinear stretch | 4×64 seed → native 2048 frames, crop | Same + perceptual loss |
| Vocoder | SpeechT5 HiFi-GAN (80 mel, 16 kHz) + interp | Ecoacoustic HiFi-GAN (128 mel, 22,050 Hz) | + discriminator for finer detail |
| Annotation use | None | None | Module B classifier on frozen Module A |
| Analysis output | Raw latent vector | Raw latent vector | Acoustic indices + species probabilities |
| Transformation | Not implemented | Not implemented | Audio upload → encode → new env → decode |
| Model serving | Load on request | Load on request | Cached at startup |
| Audio quality | Machine noise / robotic | Ecoacoustic texture, no temporal narrative | Natural, with temporal structure |
