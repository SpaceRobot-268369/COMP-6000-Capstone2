import { useEffect, useRef, useState } from "react";

const FFT_SIZE  = 1024;
const HOP_SIZE  = 512;
const MEL_BANDS = 128;
const MAX_COLS  = 520;
const DYNAMIC_RANGE_DB = 80;

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

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilters(sampleRate) {
  const nyquist = sampleRate / 2;
  const minMel = hzToMel(50);
  const maxMel = hzToMel(Math.min(11000, nyquist));
  const melPoints = Array.from({ length: MEL_BANDS + 2 }, (_, i) =>
    minMel + (i / (MEL_BANDS + 1)) * (maxMel - minMel)
  );
  const binPoints = melPoints.map(mel => {
    const hz = melToHz(mel);
    return Math.max(0, Math.min(FFT_SIZE / 2, Math.floor((hz / sampleRate) * FFT_SIZE)));
  });

  return Array.from({ length: MEL_BANDS }, (_, band) => {
    const left = binPoints[band];
    const center = Math.max(left + 1, binPoints[band + 1]);
    const right = Math.max(center + 1, binPoints[band + 2]);
    const weights = [];

    for (let bin = left; bin <= right && bin <= FFT_SIZE / 2; bin += 1) {
      let weight = 0;
      if (bin <= center) weight = (bin - left) / Math.max(1, center - left);
      else weight = (right - bin) / Math.max(1, right - center);
      if (weight > 0) weights.push([bin, weight]);
    }

    return weights;
  });
}

// ─── Colour map (dark → cyan → white, matching design palette) ───────────────
function energyToRgb(norm) {
  const t = Math.max(0, Math.min(1, norm));
  if (t < 0.45) {
    const s = t / 0.45;
    return [0, Math.round(s * 72), Math.round(s * 95)];
  }
  if (t < 0.75) {
    const s = (t - 0.45) / 0.3;
    return [0, Math.round(72 + s * 155), Math.round(95 + s * 128)];
  }
  const s = (t - 0.75) / 0.25;
  return [Math.round(s * 225), Math.round(227 + s * 28), Math.round(223 + s * 32)];
}

// ─── Build mel-style log-energy spectrogram from raw PCM ──────────────────────
function buildMelSpectrogram(samples, sampleRate) {
  const totalFrames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP_SIZE));
  const step        = Math.max(1, Math.floor(totalFrames / MAX_COLS));
  const numCols     = Math.min(MAX_COLS, Math.ceil(totalFrames / step));
  const numRows     = MEL_BANDS;
  const filters     = buildMelFilters(sampleRate);

  const hann = Float32Array.from({ length: FFT_SIZE }, (_, i) =>
    0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)))
  );

  const matrix = new Float32Array(numCols * numRows);
  let maxDb = -Infinity;

  for (let col = 0; col < numCols; col++) {
    const offset = col * step * HOP_SIZE;
    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);

    for (let k = 0; k < FFT_SIZE; k++) {
      re[k] = (samples[offset + k] ?? 0) * hann[k];
    }

    fft(re, im);

    const power = new Float32Array(FFT_SIZE / 2 + 1);
    for (let bin = 0; bin <= FFT_SIZE / 2; bin += 1) {
      power[bin] = (re[bin] * re[bin] + im[bin] * im[bin]) / FFT_SIZE;
    }

    for (let band = 0; band < numRows; band += 1) {
      let energy = 0;
      let weightSum = 0;
      for (const [bin, weight] of filters[band]) {
        energy += power[bin] * weight;
        weightSum += weight;
      }
      const db = 10 * Math.log10(energy / Math.max(weightSum, 1e-8) + 1e-12);
      matrix[(numRows - 1 - band) * numCols + col] = db;
      if (db > maxDb) maxDb = db;
    }
  }

  const floorDb = maxDb - DYNAMIC_RANGE_DB;
  for (let i = 0; i < matrix.length; i += 1) {
    matrix[i] = Math.max(0, Math.min(1, (matrix[i] - floorDb) / DYNAMIC_RANGE_DB));
  }

  return { matrix, numCols, numRows };
}

// ─── Component ────────────────────────────────────────────────────────────────
/**
 * Renders a real mel-style log-energy spectrogram from an uploaded audio File.
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
        const audioCtx    = new AudioContext();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        await audioCtx.close();

        if (cancelled) return;

        const samples = audioBuffer.getChannelData(0);
        const { matrix, numCols, numRows } = buildMelSpectrogram(samples, audioBuffer.sampleRate);

        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width  = numCols;
        canvas.height = numRows;

        const ctx   = canvas.getContext("2d");
        const imgData = ctx.createImageData(numCols, numRows);
        const px    = imgData.data;

        for (let i = 0; i < numCols * numRows; i++) {
          const [r, g, b] = energyToRgb(matrix[i]);
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
        aria-label="Frequency spectrogram"
        style={{ opacity: status === "done" ? 1 : 0 }}
      />
    </div>
  );
}
