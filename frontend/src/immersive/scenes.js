/* scenes.js — the 16-scene mood matrix.
   Season drives vegetation/particles/saturation; time-of-day drives light + sky.
   They compose into a flat parameter object the renderer can cross-fade.
   Ported verbatim from the eco-acoustic artifact; only added `export`.        */

export const SEASONS = ['spring', 'summer', 'autumn', 'winter'];
export const TIMES   = ['dawn', 'morning', 'afternoon', 'night'];

export const SEASON_LABEL = { spring: 'Spring', summer: 'Summer', autumn: 'Autumn', winter: 'Winter' };
export const TIME_LABEL   = { dawn: 'Dawn', morning: 'Morning', afternoon: 'Afternoon', night: 'Night' };

// ---- TIME OF DAY: light geometry, sky gradient, exposure ---------------------
// sunAz ~150-210 keeps the disc IN FRAME (camera looks toward -Z); elevation sets
// its screen height so the key light visibly rises across the four times.
const TIME_CFG = {
  dawn: {
    sunEl: 5, sunAz: 158,
    sunColor:    [1.00, 0.52, 0.34],
    skyTop:      [0.18, 0.15, 0.33],
    skyHorizon:  [1.00, 0.55, 0.45],
    haze: 0.62, brightness: 0.94, ambient: [0.22, 0.18, 0.25],
    exposure: 1.02, temp: 0.16, contrast: 1.00, shaft: 0.95,
    stars: 0.0, moon: 0.0, sunSize: 1.0,
  },
  morning: {
    sunEl: 30, sunAz: 150,
    sunColor:    [1.00, 0.95, 0.83],
    skyTop:      [0.27, 0.51, 0.84],
    skyHorizon:  [0.90, 0.94, 0.99],
    haze: 0.26, brightness: 1.32, ambient: [0.58, 0.63, 0.68],
    exposure: 1.20, temp: 0.05, contrast: 1.02, shaft: 0.62,
    stars: 0.0, moon: 0.0, sunSize: 0.6,
  },
  afternoon: {
    sunEl: 47, sunAz: 207,
    sunColor:    [1.00, 0.84, 0.56],
    skyTop:      [0.19, 0.39, 0.66],
    skyHorizon:  [0.96, 0.82, 0.60],
    haze: 0.46, brightness: 1.14, ambient: [0.44, 0.39, 0.33],
    exposure: 1.14, temp: 0.11, contrast: 1.05, shaft: 0.72,
    stars: 0.0, moon: 0.0, sunSize: 0.7,
  },
  night: {
    sunEl: 52, sunAz: 168,
    sunColor:    [0.60, 0.70, 1.00],
    skyTop:      [0.010, 0.018, 0.052],
    skyHorizon:  [0.050, 0.085, 0.165],
    haze: 0.30, brightness: 0.46, ambient: [0.07, 0.09, 0.16],
    exposure: 0.92, temp: -0.24, contrast: 1.18, shaft: 0.16,
    stars: 1.0, moon: 1.0, sunSize: 0.5,
  },
};

// ---- SEASON: vegetation, particles, grade shift ------------------------------
const SEASON_CFG = {
  spring: {
    particle: 'pollen', pCount: 200, pColor: [1.00, 0.96, 0.82], pSize: 3.0,
    fogTint: [0.62, 0.72, 0.66], foliage: 0.72, vegColor: [0.30, 0.46, 0.22],
    dBright: 0.02, dTemp: 0.00, dSat: 0.06,
  },
  summer: {
    particle: 'dust', pCount: 170, pColor: [1.00, 0.92, 0.72], pSize: 2.4, shimmer: 1.0,
    fogTint: [0.74, 0.74, 0.58], foliage: 1.00, vegColor: [0.24, 0.40, 0.18],
    dBright: 0.06, dTemp: 0.07, dSat: 0.09,
  },
  autumn: {
    particle: 'leaves', pCount: 140, pColor: [0.90, 0.52, 0.24], pSize: 4.2,
    fogTint: [0.66, 0.56, 0.44], foliage: 0.48, vegColor: [0.42, 0.30, 0.15],
    dBright: -0.02, dTemp: 0.06, dSat: 0.00,
  },
  winter: {
    particle: 'snow', pCount: 360, pColor: [0.92, 0.96, 1.00], pSize: 3.4,
    fogTint: [0.70, 0.74, 0.82], foliage: 0.04, vegColor: [0.36, 0.38, 0.40],
    dBright: -0.11, dTemp: -0.13, dSat: -0.46,
  },
};

function deg2rad(d) { return d * Math.PI / 180; }

// direction the light COMES from (pointing toward scene), unit vector
function sunDirection(elDeg, azDeg) {
  const el = deg2rad(elDeg), az = deg2rad(azDeg);
  return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
}

/* Compose a flat params object for (season,time). Pure data — the renderer maps
   these onto uniforms and cross-fades between two of them on transition. */
export function buildScene(season, time) {
  const t = TIME_CFG[time], s = SEASON_CFG[season];
  return {
    season, time,
    skyTop: t.skyTop.slice(),
    skyHorizon: t.skyHorizon.slice(),
    sunColor: t.sunColor.slice(),
    sunDir: sunDirection(t.sunEl, t.sunAz),
    sunSize: t.sunSize,
    haze: t.haze,
    brightness: t.brightness + s.dBright,
    ambient: t.ambient.slice(),
    stars: t.stars,
    moon: t.moon,
    // fog colour = horizon tinted toward the season
    fog: mix3(t.skyHorizon, s.fogTint, 0.45),
    // grade
    exposure: t.exposure,
    temp: t.temp + s.dTemp,
    saturation: 1.0 + s.dSat,
    contrast: t.contrast,
    shaft: t.shaft,
    sunEl: t.sunEl,
    // vegetation / particles
    vegColor: s.vegColor.slice(),
    particle: s.particle,
    pCount: s.pCount,
    pColor: s.pColor.slice(),
    pSize: s.pSize,
    shimmer: s.shimmer || 0.0,
    foliage: s.foliage,
  };
}

function mix3(a, b, k) {
  return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}
