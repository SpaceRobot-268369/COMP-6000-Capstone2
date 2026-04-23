import { useEffect, useRef, useState } from "react";

const FFT_SIZE  = 256;   // must be power-of-2; 128 bins output
const MAX_COLS  = 400;   // time frames to draw
const MIN_DB    = -80;

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
function dbToRgb(db) {
  const t = Math.max(0, Math.min(1, (db - MIN_DB) / -MIN_DB));
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

// ─── Build spectrogram matrix from raw PCM ────────────────────────────────────
function buildSpectrogram(samples, sampleRate) {
  const totalFrames = Math.floor(samples.length / FFT_SIZE);
  const step        = Math.max(1, Math.floor(totalFrames / MAX_COLS));
  const numCols     = Math.min(MAX_COLS, Math.ceil(totalFrames / step));
  const numRows     = FFT_SIZE / 2;

  // Hann window coefficients
  const hann = Float32Array.from({ length: FFT_SIZE }, (_, i) =>
    0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)))
  );

  const matrix = new Float32Array(numCols * numRows);

  for (let col = 0; col < numCols; col++) {
    const offset = col * step * FFT_SIZE;
    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);

    for (let k = 0; k < FFT_SIZE; k++) {
      re[k] = (samples[offset + k] ?? 0) * hann[k];
    }

    fft(re, im);

    for (let row = 0; row < numRows; row++) {
      const mag = Math.sqrt(re[row] * re[row] + im[row] * im[row]) / FFT_SIZE;
      const db  = mag > 1e-10 ? Math.max(MIN_DB, 20 * Math.log10(mag)) : MIN_DB;
      // store rows bottom-to-top (low freq at bottom)
      matrix[(numRows - 1 - row) * numCols + col] = db;
    }
  }

  return { matrix, numCols, numRows };
}

// ─── Component ────────────────────────────────────────────────────────────────
/**
 * Renders a real log-mel-style spectrogram from an uploaded audio File.
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
        const { matrix, numCols, numRows } = buildSpectrogram(samples, audioBuffer.sampleRate);

        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width  = numCols;
        canvas.height = numRows;

        const ctx   = canvas.getContext("2d");
        const imgData = ctx.createImageData(numCols, numRows);
        const px    = imgData.data;

        for (let i = 0; i < numCols * numRows; i++) {
          const [r, g, b] = dbToRgb(matrix[i]);
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
