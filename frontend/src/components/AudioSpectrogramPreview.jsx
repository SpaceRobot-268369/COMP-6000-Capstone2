import { useEffect, useRef, useState } from "react";

const FFT_SIZE = 1024;
const HOP_SIZE = 512;
const MEL_BANDS = 128;
const MAX_COLS = 520;
const DYNAMIC_RANGE_DB = 80;

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = -Math.PI / half;
    const wdr = Math.cos(angle);
    const wdi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      for (let k = 0; k < half; k += 1) {
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + half] * wr - im[i + k + half] * wi;
        const vi = re[i + k + half] * wi + im[i + k + half] * wr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + half] = ur - vr;
        im[i + k + half] = ui - vi;
        const nextWr = wr * wdr - wi * wdi;
        wi = wr * wdi + wi * wdr;
        wr = nextWr;
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
      const weight = bin <= center
        ? (bin - left) / Math.max(1, center - left)
        : (right - bin) / Math.max(1, right - center);
      if (weight > 0) weights.push([bin, weight]);
    }
    return weights;
  });
}

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

function buildMelSpectrogram(samples, sampleRate) {
  const totalFrames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP_SIZE));
  const step = Math.max(1, Math.floor(totalFrames / MAX_COLS));
  const numCols = Math.min(MAX_COLS, Math.ceil(totalFrames / step));
  const filters = buildMelFilters(sampleRate);
  const hann = Float32Array.from({ length: FFT_SIZE }, (_, i) =>
    0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)))
  );
  const matrix = new Float32Array(numCols * MEL_BANDS);
  let maxDb = -Infinity;

  for (let col = 0; col < numCols; col += 1) {
    const offset = col * step * HOP_SIZE;
    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);
    for (let k = 0; k < FFT_SIZE; k += 1) re[k] = (samples[offset + k] || 0) * hann[k];
    fft(re, im);

    const power = new Float32Array(FFT_SIZE / 2 + 1);
    for (let bin = 0; bin <= FFT_SIZE / 2; bin += 1) {
      power[bin] = (re[bin] * re[bin] + im[bin] * im[bin]) / FFT_SIZE;
    }

    for (let band = 0; band < MEL_BANDS; band += 1) {
      let energy = 0;
      let weightSum = 0;
      for (const [bin, weight] of filters[band]) {
        energy += power[bin] * weight;
        weightSum += weight;
      }
      const db = 10 * Math.log10(energy / Math.max(weightSum, 1e-8) + 1e-12);
      matrix[(MEL_BANDS - 1 - band) * numCols + col] = db;
      if (db > maxDb) maxDb = db;
    }
  }

  const floorDb = maxDb - DYNAMIC_RANGE_DB;
  for (let i = 0; i < matrix.length; i += 1) {
    matrix[i] = Math.max(0, Math.min(1, (matrix[i] - floorDb) / DYNAMIC_RANGE_DB));
  }

  return { matrix, numCols, numRows: MEL_BANDS };
}

export default function AudioSpectrogramPreview({ src }) {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("computing");

  useEffect(() => {
    if (!src) return;
    let cancelled = false;
    setStatus("computing");

    (async () => {
      try {
        const response = await fetch(src);
        const audioCtx = new AudioContext();
        const audioBuffer = await audioCtx.decodeAudioData(await response.arrayBuffer());
        await audioCtx.close();
        if (cancelled) return;

        const samples = audioBuffer.getChannelData(0);
        const { matrix, numCols, numRows } = buildMelSpectrogram(samples, audioBuffer.sampleRate);
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = numCols;
        canvas.height = numRows;

        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(numCols, numRows);
        for (let i = 0; i < matrix.length; i += 1) {
          const [r, g, b] = energyToRgb(matrix[i]);
          const base = i * 4;
          imgData.data[base] = r;
          imgData.data[base + 1] = g;
          imgData.data[base + 2] = b;
          imgData.data[base + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        setStatus("done");
      } catch (err) {
        if (!cancelled) setStatus("error");
      }
    })();

    return () => { cancelled = true; };
  }, [src]);

  return (
    <div className="gen-audio-spectrogram">
      {status === "computing" && <span>Computing final audio spectrogram…</span>}
      {status === "error" && <span>Final audio spectrogram unavailable</span>}
      <canvas
        ref={canvasRef}
        aria-label="Final mixed audio mel spectrogram"
        style={{ opacity: status === "done" ? 1 : 0 }}
      />
    </div>
  );
}
