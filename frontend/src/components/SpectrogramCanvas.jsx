import { useEffect, useRef, useState } from "react";

const FFT_SIZE  = 1024;  // must be power-of-2
const HOP_SIZE  = 512;
const N_MELS    = 128;
const MAX_COLS  = 520;   // time frames to draw
const MIN_DB    = -120;  // clamp floor before normalisation
const DB_RANGE  = 80;    // visible dynamic range below the per-clip peak

const HANN = Float32Array.from({ length: FFT_SIZE }, (_, i) =>
  0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)))
);

const melFilterCache = new Map();

// ─── Radix-2 FFT (in-place) ───────────────────────────────────────────────────
function fft(re, im) {
  const N = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // Butterfly passes
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const ang  = -Math.PI / half;
    for (let i = 0; i < N; i += len) {
      let wr = 1, wi = 0;
      const wdr = Math.cos(ang), wdi = Math.sin(ang);
      for (let k = 0; k < half; k++) {
        const ur = re[i + k],         ui = im[i + k];
        const vr = re[i+k+half]*wr - im[i+k+half]*wi;
        const vi = re[i+k+half]*wi + im[i+k+half]*wr;
        re[i+k]      = ur + vr;  im[i+k]      = ui + vi;
        re[i+k+half] = ur - vr;  im[i+k+half] = ui - vi;
        const nwr = wr*wdr - wi*wdi;
        wi = wr*wdi + wi*wdr;
        wr = nwr;
      }
    }
  }
}

// ─── Colour map (dark → cyan → white, matching design palette) ───────────────
// `t` is a normalised intensity in [0, 1] (0 = floor, 1 = per-clip peak).
function intensityToRgb(t) {
  if (t < 0.45) {
    const s = t / 0.45;
    return [0, Math.round(s * 80), Math.round(s * 100)];
  }
  if (t < 0.75) {
    const s = (t - 0.45) / 0.3;
    return [0, Math.round(80 + s * 160), Math.round(100 + s * 130)];
  }
  const s = (t - 0.75) / 0.25;
  return [Math.round(s * 230), Math.round(240 + s * 15), Math.round(230 + s * 25)];
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilters(sampleRate) {
  const cacheKey = `${sampleRate}:${FFT_SIZE}:${N_MELS}`;
  if (melFilterCache.has(cacheKey)) return melFilterCache.get(cacheKey);

  const numBins = FFT_SIZE / 2 + 1;
  const maxMel = hzToMel(sampleRate / 2);
  const melPoints = Array.from({ length: N_MELS + 2 }, (_, i) =>
    (i / (N_MELS + 1)) * maxMel
  );
  const binPoints = melPoints.map((mel) =>
    Math.max(0, Math.min(numBins - 1, Math.floor((FFT_SIZE + 1) * melToHz(mel) / sampleRate)))
  );

  const filters = Array.from({ length: N_MELS }, () => new Float32Array(numBins));

  for (let m = 1; m <= N_MELS; m++) {
    const left   = binPoints[m - 1];
    const center = binPoints[m];
    const right  = binPoints[m + 1];
    const filter = filters[m - 1];
    let weightSum = 0;

    if (center > left) {
      for (let k = left; k < center; k++) {
        filter[k] = (k - left) / (center - left);
        weightSum += filter[k];
      }
    }
    if (right > center) {
      for (let k = center; k < right; k++) {
        filter[k] = (right - k) / (right - center);
        weightSum += filter[k];
      }
    }
    if (weightSum > 0) {
      for (let k = left; k < right; k++) filter[k] /= weightSum;
    }
  }

  melFilterCache.set(cacheKey, filters);
  return filters;
}

function mixToMono(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) return audioBuffer.getChannelData(0);

  const samples = new Float32Array(audioBuffer.length);
  for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
    const channel = audioBuffer.getChannelData(ch);
    for (let i = 0; i < samples.length; i++) {
      samples[i] += channel[i] / audioBuffer.numberOfChannels;
    }
  }
  return samples;
}

// ─── Build log-mel spectrogram matrix from raw PCM ────────────────────────────
function buildMelSpectrogram(samples, sampleRate) {
  const totalFrames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP_SIZE) + 1);
  const frameStride = Math.max(1, Math.ceil(totalFrames / MAX_COLS));
  const numCols     = Math.ceil(totalFrames / frameStride);
  const numRows     = N_MELS;
  const numBins     = FFT_SIZE / 2 + 1;
  const filters     = buildMelFilters(sampleRate);

  const matrix = new Float32Array(numCols * numRows);
  const re     = new Float32Array(FFT_SIZE);
  const im     = new Float32Array(FFT_SIZE);
  const power  = new Float32Array(numBins);
  let maxDb = MIN_DB;

  for (let col = 0; col < numCols; col++) {
    const offset = col * frameStride * HOP_SIZE;
    re.fill(0);
    im.fill(0);

    for (let k = 0; k < FFT_SIZE; k++) {
      re[k] = (samples[offset + k] ?? 0) * HANN[k];
    }

    fft(re, im);

    for (let bin = 0; bin < numBins; bin++) {
      power[bin] = (re[bin] * re[bin] + im[bin] * im[bin]) / FFT_SIZE;
    }

    for (let mel = 0; mel < N_MELS; mel++) {
      const filter = filters[mel];
      let energy = 0;
      for (let bin = 0; bin < numBins; bin++) energy += power[bin] * filter[bin];

      const db = energy > 1e-20 ? Math.max(MIN_DB, 10 * Math.log10(energy)) : MIN_DB;
      if (db > maxDb) maxDb = db;
      // Store rows bottom-to-top so low mel bands render at the bottom.
      matrix[(numRows - 1 - mel) * numCols + col] = db;
    }
  }

  return { matrix, numCols, numRows, maxDb };
}

// ─── Component ────────────────────────────────────────────────────────────────
/**
 * Renders a browser-side log-mel spectrogram from an uploaded audio File.
 * Falls back to the decorative static view when no file is provided.
 */
export default function SpectrogramCanvas({ file }) {
  const canvasRef = useRef(null);
  const [status,  setStatus]  = useState("idle"); // idle | computing | done | error

  useEffect(() => {
    if (!file) { setStatus("idle"); return; }

    let cancelled = false;
    setStatus("computing");

    (async () => {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) throw new Error("Web Audio API unavailable");

        const audioCtx = new AudioContextCtor();
        let audioBuffer;
        try {
          audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        } finally {
          await audioCtx.close().catch(() => {});
        }

        if (cancelled) return;

        const samples = mixToMono(audioBuffer);
        const { matrix, numCols, numRows, maxDb } = buildMelSpectrogram(samples, audioBuffer.sampleRate);

        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width  = numCols;
        canvas.height = numRows;

        const ctx   = canvas.getContext("2d");
        const imgData = ctx.createImageData(numCols, numRows);
        const px    = imgData.data;

        // Normalise against the loudest bin in this clip so broadband field
        // recordings use the full colour range instead of rendering near-black.
        const peakDb = maxDb > MIN_DB ? maxDb : MIN_DB + DB_RANGE;
        for (let i = 0; i < numCols * numRows; i++) {
          const t = Math.max(0, Math.min(1, (matrix[i] - peakDb + DB_RANGE) / DB_RANGE));
          const [r, g, b] = intensityToRgb(t);
          const base = i * 4;
          px[base]     = r;
          px[base + 1] = g;
          px[base + 2] = b;
          px[base + 3] = 255;
        }

        ctx.putImageData(imgData, 0, 0);
        setStatus("done");
      } catch (err) {
        if (!cancelled) { console.error(err); setStatus("error"); }
      }
    })();

    return () => { cancelled = true; };
  }, [file]);

  // Static decorative view (no file loaded)
  if (!file || status === "idle") {
    return (
      <div className="spectrogram">
        <div className="scan-line" />
      </div>
    );
  }

  return (
    <div className="spectrogram spectrogram-live" style={{ position: "relative" }}>
      {status === "computing" && (
        <div className="spectrogram-overlay">
          <span className="spectrogram-status">Computing spectrum…</span>
        </div>
      )}
      {status === "error" && (
        <div className="spectrogram-overlay">
          <span className="spectrogram-status">Could not decode audio</span>
        </div>
      )}
      <canvas
        ref={canvasRef}
        className="spectrogram-canvas"
        aria-label="Mel spectrogram"
        style={{ opacity: status === "done" ? 1 : 0 }}
      />
    </div>
  );
}
