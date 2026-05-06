const FFT_SIZE = 512;
const HOP_SIZE = 512;

function clip01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function level(value, light, strong, labels) {
  if (value >= strong) return labels[2];
  if (value >= light) return labels[1];
  return labels[0];
}

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

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

export async function analyseAudioBasicFallback(file) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) throw new Error("Browser audio analysis is unavailable.");

  const audioCtx = new AudioCtx();
  const audioBuffer = await audioCtx.decodeAudioData(await file.arrayBuffer());
  await audioCtx.close();

  const sampleRate = audioBuffer.sampleRate;
  const maxSamples = Math.min(audioBuffer.length, Math.floor(sampleRate * 300));
  const source = audioBuffer.getChannelData(0).subarray(0, maxSamples);
  const durationSec = source.length / sampleRate;

  let sumSq = 0;
  let peak = 0;
  let zc = 0;
  for (let i = 0; i < source.length; i += 1) {
    const sample = source[i];
    sumSq += sample * sample;
    peak = Math.max(peak, Math.abs(sample));
    if (i > 0 && Math.sign(sample) !== Math.sign(source[i - 1])) zc += 1;
  }

  const rms = Math.sqrt(sumSq / Math.max(source.length, 1));
  const rmsDb = 20 * Math.log10(rms + 1e-8);
  const zeroCrossingRate = zc / Math.max(source.length, 1);

  const hann = Float32Array.from({ length: FFT_SIZE }, (_, i) =>
    0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)))
  );

  let lowEnergy = 0;
  let midEnergy = 0;
  let highEnergy = 0;
  let totalEnergy = 0;
  let centroidNum = 0;
  const bandwidthValues = [];
  const rolloffValues = [];
  const flatnessValues = [];
  const frameRmsValues = [];
  let onsetCount = 0;
  let previousFrameRms = 0;

  const frameCount = Math.max(1, Math.floor((source.length - FFT_SIZE) / HOP_SIZE));
  const stride = Math.max(1, Math.floor(frameCount / 700));

  for (let frame = 0; frame < frameCount; frame += stride) {
    const offset = frame * HOP_SIZE;
    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);
    let frameSq = 0;

    for (let i = 0; i < FFT_SIZE; i += 1) {
      const value = (source[offset + i] || 0) * hann[i];
      re[i] = value;
      frameSq += value * value;
    }

    const frameRms = Math.sqrt(frameSq / FFT_SIZE);
    frameRmsValues.push(frameRms);
    if (previousFrameRms > 0 && frameRms > previousFrameRms * 1.55 + 0.004) onsetCount += 1;
    previousFrameRms = previousFrameRms * 0.72 + frameRms * 0.28;

    fft(re, im);

    const mags = [];
    let frameEnergy = 0;
    let frameCentroidNum = 0;
    for (let bin = 1; bin < FFT_SIZE / 2; bin += 1) {
      const freq = (bin * sampleRate) / FFT_SIZE;
      const mag = Math.sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
      const energy = mag * mag;
      mags.push(mag + 1e-10);
      frameEnergy += energy;
      frameCentroidNum += freq * energy;
      if (freq < 500) lowEnergy += energy;
      else if (freq < 4000) midEnergy += energy;
      else highEnergy += energy;
    }

    if (frameEnergy > 0) {
      const centroid = frameCentroidNum / frameEnergy;
      centroidNum += frameCentroidNum;
      totalEnergy += frameEnergy;

      let spread = 0;
      let cumulative = 0;
      let rolloff = 0;
      for (let bin = 1; bin < FFT_SIZE / 2; bin += 1) {
        const freq = (bin * sampleRate) / FFT_SIZE;
        const mag = mags[bin - 1];
        const energy = mag * mag;
        spread += energy * (freq - centroid) * (freq - centroid);
        cumulative += energy;
        if (!rolloff && cumulative >= frameEnergy * 0.85) rolloff = freq;
      }
      bandwidthValues.push(Math.sqrt(spread / frameEnergy));
      rolloffValues.push(rolloff);

      const arithmetic = mean(mags);
      const geometric = Math.exp(mean(mags.map(value => Math.log(value))));
      flatnessValues.push(arithmetic > 0 ? geometric / arithmetic : 0);
    }
  }

  const total = totalEnergy + 1e-8;
  const lowRatio = lowEnergy / total;
  const midRatio = midEnergy / total;
  const highRatio = highEnergy / total;
  const centroid = centroidNum / total;
  const flatness = mean(flatnessValues);
  const transientRate = onsetCount / Math.max(durationSec, 1e-6);
  const onsetDensity = clip01(transientRate / 3.0);
  const loudnessNorm = clip01((rmsDb + 55) / 45);
  const brightness = clip01((centroid - 600) / 4200);
  const soundDensity = clip01(loudnessNorm * 0.45 + flatness * 1.65 + onsetDensity * 0.35);
  const activityIndex = clip01(onsetDensity * 0.55 + highRatio * 0.65 + midRatio * 0.25 + loudnessNorm * 0.15);
  const windIndex = clip01((lowRatio * 1.9 + midRatio * 0.25 + (1 - onsetDensity) * 0.25) * (0.35 + loudnessNorm));
  const rainIndex = clip01(highRatio * 1.35 + flatness * 2.6 + zeroCrossingRate * 0.25 + soundDensity * 0.45);
  const bioIndex = clip01(onsetDensity * 0.70 + highRatio * 0.60 + midRatio * 0.20 - flatness * 0.35);
  const timeHintScore = clip01(bioIndex * 0.65 + brightness * 0.25 + onsetDensity * 0.25);

  const wind = level(windIndex, 0.35, 0.62, ["none", "light", "strong"]);
  const rain = level(rainIndex, 0.38, 0.68, ["none", "light", "dense"]);
  const activity = level(activityIndex, 0.35, 0.65, ["low", "moderate", "high"]);
  const brightnessLabel = level(brightness, 0.35, 0.65, ["dark", "balanced", "bright"]);
  const timeHint = timeHintScore >= 0.55
    ? "dawn/morning"
    : brightness < 0.22 && activityIndex < 0.35
      ? "night"
      : "day/afternoon";

  const acousticFeatures = {
    duration_sec: Number(durationSec.toFixed(2)),
    sample_rate: sampleRate,
    rms_db: Number(rmsDb.toFixed(2)),
    peak_amplitude: Number(peak.toFixed(4)),
    zero_crossing_rate: Number(zeroCrossingRate.toFixed(4)),
    spectral_centroid_hz: Number(centroid.toFixed(2)),
    spectral_bandwidth_hz: Number(mean(bandwidthValues).toFixed(2)),
    spectral_rolloff_hz: Number(mean(rolloffValues).toFixed(2)),
    spectral_flatness: Number(flatness.toFixed(4)),
    transient_rate_per_sec: Number(transientRate.toFixed(4)),
    onset_density: Number(onsetDensity.toFixed(4)),
    low_energy_ratio: Number(lowRatio.toFixed(4)),
    mid_energy_ratio: Number(midRatio.toFixed(4)),
    high_energy_ratio: Number(highRatio.toFixed(4)),
    low_high_energy_ratio: Number((lowRatio / (highRatio + 1e-8)).toFixed(4)),
    sound_density: Number(soundDensity.toFixed(3)),
    brightness: Number(brightness.toFixed(3)),
    brightness_label: brightnessLabel,
    activity_level: activity,
    activity_score: Number(activityIndex.toFixed(3)),
    wind_texture_proxy: Number(windIndex.toFixed(3)),
    rain_texture_proxy: Number(rainIndex.toFixed(3)),
  };

  const heuristicEnvironment = {
    wind: {
      level: wind,
      confidence: Number(windIndex.toFixed(3)),
      explanation: `${wind} wind likelihood from low-frequency sustained energy and low transient density.`,
    },
    rain: {
      level: rain,
      confidence: Number(rainIndex.toFixed(3)),
      explanation: `${rain} rain likelihood from high-frequency energy, spectral flatness, and dense texture.`,
    },
    activity: {
      level: activity,
      confidence: Number(activityIndex.toFixed(3)),
      biological_activity_score: Number(bioIndex.toFixed(3)),
      explanation: `${activity} activity from onset density plus mid/high-frequency burst energy.`,
    },
    time_of_day_hint: {
      label: timeHint,
      confidence: Number(timeHintScore.toFixed(3)),
      explanation: "Estimated from brightness and biological activity patterns only; not derived from recording metadata.",
    },
  };

  return {
    ok: true,
    analysis_mode: "basic_fallback",
    checkpoint_available: false,
    latent_dim: 0,
    latent: [],
    acoustic_features: acousticFeatures,
    heuristic_environment: heuristicEnvironment,
    estimated_conditions: {
      wind_speed_ms: Number((0.5 + windIndex * 5.5).toFixed(2)),
      wind_max_ms: Number((1 + windIndex * 8).toFixed(2)),
      precipitation_mm: Number((Math.max(0, rainIndex - 0.28) * 7).toFixed(2)),
      precipitation_daily_mm: Number((Math.max(0, rainIndex - 0.22) * 10).toFixed(2)),
      humidity_pct: Number((42 + rainIndex * 38).toFixed(2)),
      days_since_rain: Number((Math.max(0, 7 * (1 - rainIndex))).toFixed(2)),
      confidence: 0.25,
      inference_method: "browser_basic_audio_feature_fallback",
      wind,
      rain,
      activity,
      time_of_day_hint: timeHint,
    },
    limitations: [
      "Backend VAE checkpoint is unavailable, so the browser computed a basic fallback analysis.",
      "Environmental values are low-confidence audio-feature proxies, not NASA-aligned nearest-neighbour estimates.",
    ],
    summary: "Browser Basic Analysis Mode completed without VAE latent analysis.",
  };
}
