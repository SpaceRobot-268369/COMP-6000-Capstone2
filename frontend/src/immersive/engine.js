/* engine.js — bootstrap, scene state + cross-fade, thunder, render loop.

   Ported from the artifact's `main.js` IIFE into a factory the React page can
   mount/unmount. Key changes from the original:
     - takes its DOM nodes (scene, bolt, title, audio) as refs instead of
       reading document.getElementById,
     - threads the per-engine WORLD object through the world helpers,
     - returns a control API plus a dispose() that cancels the rAF loop, removes
       listeners, and frees all GPU + WebAudio resources (the IIFE had no teardown).
*/

import * as THREE from 'three';
import { PostPipeline } from './post.js';
import {
  buildWorld, setupParticles, updateParticles, setRain, updateRain, setWind, disposeWorld,
} from './world.js';
import { buildScene, SEASON_LABEL, TIME_LABEL } from './scenes.js';
import { createTypography } from './typography.js';
import { createAudio } from './audio.js';

const DEFAULT_INITIAL = {
  season: 'autumn', time: 'dawn', rain: false, rainAmount: 0.6, wind: 0, thunder: false, narration: null,
};

export function createImmersive({ sceneEl, boltEl, titleWordsEl, titleScrimEl, audioEl, initial }) {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const container = sceneEl;
  const init = { ...DEFAULT_INITIAL, ...(initial || {}) };

  // ---- renderer / scene / camera ----
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(dpr);
  // size from the container's real box, not window — robust to panel resizes
  const sizeOf = () => {
    const r = container.getBoundingClientRect();
    return { w: Math.max(1, Math.round(r.width || window.innerWidth)),
             h: Math.max(1, Math.round(r.height || window.innerHeight)) };
  };
  let { w: VW, h: VH } = sizeOf();
  renderer.setSize(VW, VH, false);              // don't write inline px style
  const cv = renderer.domElement;
  cv.style.width = '100%'; cv.style.height = '100%'; cv.style.display = 'block';
  container.appendChild(cv);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(52, VW / VH, 0.1, 1400);
  camera.position.set(0, 2.6, 9);

  const WORLD = buildWorld(scene, { dpr });
  const post = new PostPipeline(renderer, VW, VH, { bloomScale: 0.5, bloom: true });
  const typo = createTypography(titleWordsEl, titleScrimEl);
  const audio = createAudio(audioEl);

  // ---- state ----
  const APP = {
    state: { season: init.season, time: init.time, rain: init.rain, rainAmount: init.rainAmount, wind: init.wind },
    narration: init.narration,
    cur: null, from: null, to: null, tx: 1, txDur: 2.6,
    flashT: -10, boltShown: false, pendingParticle: false, particleSwapped: true,
  };

  // ---- param interpolation ----
  const A = ['skyTop', 'skyHorizon', 'sunColor', 'sunDir', 'fog', 'ambient', 'vegColor'];
  const S = ['haze', 'brightness', 'stars', 'moon', 'sunSize', 'exposure', 'temp', 'saturation', 'foliage', 'contrast', 'shaft', 'sunEl'];
  function lerpParams(a, b, k) {
    const o = {
      particle: b.particle,
      pColor: a.pColor.map((v, i) => v + (b.pColor[i] - v) * k),
      pSize: b.pSize, pCount: b.pCount,
    };
    A.forEach(key => { o[key] = a[key].map((v, i) => v + (b[key][i] - v) * k); });
    S.forEach(key => { o[key] = a[key] + (b[key] - a[key]) * k; });
    // keep the sun a unit vector through the blend so its disc/glow travels
    // along an arc instead of dipping in magnitude (which warped the falloff)
    const sd = o.sunDir, m = Math.hypot(sd[0], sd[1], sd[2]) || 1;
    o.sunDir = [sd[0] / m, sd[1] / m, sd[2] / m];
    return o;
  }

  function applyParams(p) {
    const u = WORLD.skyMat.uniforms;
    u.uSkyTop.value.setRGB(p.skyTop[0], p.skyTop[1], p.skyTop[2]);
    u.uSkyHorizon.value.setRGB(p.skyHorizon[0], p.skyHorizon[1], p.skyHorizon[2]);
    u.uSunColor.value.setRGB(p.sunColor[0], p.sunColor[1], p.sunColor[2]);
    u.uFog.value.setRGB(p.fog[0], p.fog[1], p.fog[2]);
    u.uSunDir.value.set(p.sunDir[0], p.sunDir[1], p.sunDir[2]);
    u.uHaze.value = p.haze; u.uBrightness.value = p.brightness;
    u.uStars.value = p.stars; u.uMoon.value = p.moon; u.uSunSize.value = p.sunSize;

    scene.fog.color.setRGB(p.fog[0], p.fog[1], p.fog[2]);
    scene.fog.density = 0.0058 + (1 - p.brightness) * 0.0016;

    // directional ground lit from the sun azimuth
    const gu = WORLD.groundMat.uniforms;
    // skyLift lifts the WHOLE floor on bright days (morning) without touching night
    const skyLift = Math.max(0, p.brightness - 0.95);
    // ambient lifts the floor so bright times (morning) aren't crushed; night stays dark
    gu.uBase.value.setRGB(
      p.fog[0] * 0.16 + p.ambient[0] * (0.12 + skyLift * 0.14),
      p.fog[1] * 0.16 + p.ambient[1] * (0.12 + skyLift * 0.14),
      p.fog[2] * 0.18 + p.ambient[2] * (0.12 + skyLift * 0.14));
    gu.uSkyLift.value = skyLift * 0.6;
    gu.uSunColor.value.setRGB(p.sunColor[0], p.sunColor[1], p.sunColor[2]);
    gu.uFog.value.setRGB(p.fog[0], p.fog[1], p.fog[2]);
    const azLen = Math.hypot(p.sunDir[0], p.sunDir[2]) || 1;
    gu.uSunAz.value.set(p.sunDir[0] / azLen, p.sunDir[2] / azLen);
    gu.uBrightness.value = p.brightness;
    gu.uFogDensity.value = scene.fog.density;
    // low sun (dawn) throws the longest ground streak
    gu.uShaftLow.value = p.shaft * (1 - Math.min(p.sunEl / 60, 1)) * 1.4;

    WORLD.canopyMats.forEach(m => { m.opacity = 0.18 + p.foliage * 0.5; });

    // atmospheric tint: on bright days the silhouettes lift toward the sky/fog
    // colour so the foreground isn't dead black under a luminous morning.
    // dawn / night keep brightness < 0.95 → atmo 0 → trees stay dramatically dark.
    const atmo = Math.min(0.6, Math.max(0, p.brightness - 0.95) * 0.7);
    WORLD.treeMats.forEach(o => {
      // aerial perspective: the distant treeline dissolves toward the horizon
      // fog so it never reads as a hard dark band against the bright horizon
      // (the "line"); the close hero trees keep their crushed-black drama.
      const d = -o.z;
      const aerial = Math.min(0.78, Math.max(0, (d - 46) / 82) * 0.78);
      const k = Math.max(atmo, aerial);
      o.mat.color.setRGB(
        o.base.r + (p.fog[0] - o.base.r) * k,
        o.base.g + (p.fog[1] - o.base.g) * k,
        o.base.b + (p.fog[2] - o.base.b) * k);
    });

    // undergrowth takes on the season's vegetation colour in daylight (green in
    // spring/summer, amber in autumn, sparse grey in winter); at dawn/night the
    // low light keeps it crushed to silhouette like the trees
    const vegK = Math.min(0.9, Math.max(0, p.brightness - 0.96) * 1.6);
    WORLD.bushMats.forEach(o => {
      const d = -o.z;
      const aerial = Math.min(0.58, Math.max(0, (d - 18) / 78) * 0.58);
      const k = Math.max(vegK, aerial);
      const target = vegK >= aerial ? p.vegColor : p.fog;
      o.mat.color.setRGB(
        o.base.r + (target[0] - o.base.r) * k,
        o.base.g + (target[1] - o.base.g) * k,
        o.base.b + (target[2] - o.base.b) * k);
    });

    // particles pick up the key light
    const kt = 0.5 * Math.max(p.shaft, p.moon);
    WORLD.ptMat.uniforms.uColor.value.setRGB(
      p.pColor[0] + (p.sunColor[0] - p.pColor[0]) * kt,
      p.pColor[1] + (p.sunColor[1] - p.pColor[1]) * kt,
      p.pColor[2] + (p.sunColor[2] - p.pColor[2]) * kt);

    const c = post.matComposite.uniforms;
    c.uExposure.value = p.exposure; c.uTemp.value = p.temp; c.uSaturation.value = p.saturation;
    c.uContrast.value = p.contrast;
    c.uFogColor.value.setRGB(p.fog[0], p.fog[1], p.fog[2]);
    c.uSkyColor.value.setRGB(p.skyHorizon[0], p.skyHorizon[1], p.skyHorizon[2]);
    c.uAtmosphere.value = Math.min(1, 0.55 + p.haze * 0.45);
    // night leans on bloom + vignette for the moody read
    APP.baseBloom = 0.78 + p.moon * 0.35;
    APP.shaftBase = p.shaft;
    APP.sunDirVec = new THREE.Vector3(p.sunDir[0], p.sunDir[1], p.sunDir[2]);
    c.uVignette.value = 0.28 + (1 - p.brightness) * 0.3;
    c.uGrain.value = 0.052 + (1 - p.brightness) * 0.03;
  }

  function setScene(season, time, instant) {
    APP.state.season = season; APP.state.time = time;
    const target = buildScene(season, time);
    const typeChange = WORLD.curParticle !== target.particle;
    if (instant) {
      setupParticles(WORLD, target);
      APP.pendingParticle = false; APP.particleSwapped = true;
      WORLD.ptMat.uniforms.uMaster.value = 1;
    }
    if (!APP.cur || instant) {
      APP.cur = target; APP.from = target; APP.to = target; APP.tx = 1; applyParams(target);
    } else {
      APP.from = APP.cur; APP.to = target; APP.tx = 0;
      // a season change that swaps particle TYPE dissolves the field out, swaps
      // at the midpoint (alpha ~0), then dissolves the new type in — no pop.
      // time-only changes keep their particles continuous.
      APP.pendingParticle = typeChange;
      APP.particleSwapped = !typeChange;
    }
  }

  // ---- thunder flash envelope ----
  function thunderFlash(e) {
    if (e < 0 || e > 1.5) return 0;
    const gate = Math.min(e / 0.045, 1);
    const f = 0.95 * Math.exp(-e * 4.6) + 0.5 * Math.exp(-Math.max(e - 0.13, 0) * 6.5);
    return gate * f;
  }

  // ---- lightning bolt (optional, ~half the time) ----
  const bctx = boltEl.getContext('2d');
  function sizeBolt() { boltEl.width = VW; boltEl.height = VH; }
  sizeBolt();
  function drawBolt() {
    const W = boltEl.width, H = boltEl.height;
    bctx.clearRect(0, 0, W, H);
    bctx.strokeStyle = 'rgba(220,235,255,0.92)';
    bctx.shadowColor = 'rgba(180,210,255,0.9)'; bctx.shadowBlur = 26;
    let x = W * (0.3 + Math.random() * 0.4), y = -20;
    const seg = H / 14; bctx.lineWidth = 2.4; bctx.beginPath(); bctx.moveTo(x, y);
    while (y < H * 0.62) { y += seg * (0.7 + Math.random() * 0.6); x += (Math.random() - 0.5) * W * 0.09; bctx.lineTo(x, y); }
    bctx.stroke(); bctx.shadowBlur = 0;
  }

  // ---- initial scene + title ----
  setScene(init.season, init.time, true);
  if (init.rain) setRain(WORLD, true, init.rainAmount);
  setWind(WORLD, init.wind);
  const titleTimer = setTimeout(
    () => typo.play(SEASON_LABEL[APP.state.season], TIME_LABEL[APP.state.time], APP.narration),
    600,
  );
  // a storm scene opens with a strike once the narration has settled in
  const thunderTimer = init.thunder
    ? setTimeout(() => { APP.flashT = clock.getElapsedTime(); APP.boltShown = false; }, 3200)
    : null;

  // ---- resize ----
  function onResize() {
    const { w, h } = sizeOf();
    if (w === VW && h === VH) return;
    VW = w; VH = h;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    post.setSize(w, h); sizeBolt();
  }
  window.addEventListener('resize', onResize);
  const ro = window.ResizeObserver ? new ResizeObserver(onResize) : null;
  if (ro) ro.observe(container);
  // catch late layout/font settling
  const settle1 = setTimeout(onResize, 60), settle2 = setTimeout(onResize, 400);

  // ---- render loop ----
  const clock = new THREE.Clock();
  let rafId = 0;
  let disposed = false;
  function frame() {
    if (disposed) return;
    rafId = requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.getElapsedTime();
    const amp = audio.sample();

    // scene cross-fade
    if (APP.tx < 1) {
      APP.tx = Math.min(1, APP.tx + dt / APP.txDur);
      const k = APP.tx < 0.5 ? 2 * APP.tx * APP.tx : 1 - Math.pow(-2 * APP.tx + 2, 2) / 2;
      APP.cur = lerpParams(APP.from, APP.to, k);
      applyParams(APP.cur);

      // particle type swap rides a smooth dip: alpha → 0 at the midpoint, swap
      // the field there, then alpha → 1. No instant pop on season change.
      if (APP.pendingParticle) {
        const m = Math.abs(2 * APP.tx - 1);          // 1 → 0 → 1
        WORLD.ptMat.uniforms.uMaster.value = m * m * (3 - 2 * m);
        if (APP.tx >= 0.5 && !APP.particleSwapped) {
          setupParticles(WORLD, APP.to);
          APP.particleSwapped = true;
        }
      }
      if (APP.tx >= 1) { WORLD.ptMat.uniforms.uMaster.value = 1; APP.pendingParticle = false; }
    }

    updateParticles(WORLD, dt, t, amp);
    updateRain(WORLD, dt);

    // thunder
    const e = t - APP.flashT;
    const flash = thunderFlash(e) * (1 + amp * 0.5);
    WORLD.skyMat.uniforms.uFlash.value = flash;
    WORLD.skyMat.uniforms.uTime.value = t;
    post.matComposite.uniforms.uFlash.value = flash * 0.5;
    // bolt: fire once per strike, ~55% of strikes
    if (e >= 0 && e < 0.05 && !APP.boltShown) {
      APP.boltShown = true;
      if (Math.random() < 0.55) { drawBolt(); boltEl.style.opacity = '0.9'; setTimeout(() => { boltEl.style.opacity = '0'; }, 120); }
    }

    // camera breath (+ loose audio sway)
    const m = reduceMotion ? 0.15 : 1;
    camera.position.x = Math.sin(t * 0.33) * 0.4 * m * (1 + amp * 0.5);
    camera.position.y = 2.6 + Math.sin(t * 0.21) * 0.14 * m;
    camera.position.z = 9 + Math.sin(t * 0.15) * 0.3 * m;
    camera.lookAt(Math.sin(t * 0.08) * 0.6 * m, 4.2, -40);
    WORLD.sky.position.copy(camera.position);

    // project the sun to screen for the light shafts
    let shaft = 0;
    const sunScreen = post._sunUV || (post._sunUV = new THREE.Vector2(0.5, 0.6));
    if (APP.sunDirVec && (APP.shaftBase || 0) > 0.001) {
      const sw = APP.sunDirVec.clone().multiplyScalar(400).add(camera.position).project(camera);
      const inFront = sw.z < 1;
      const su = sw.x * 0.5 + 0.5, sv = sw.y * 0.5 + 0.5;
      if (inFront) {
        sunScreen.set(su, sv);
        const edge = Math.max(Math.abs(su - 0.5), Math.abs(sv - 0.5));
        const vis = Math.max(0, Math.min(1, 1 - (edge - 0.55) / 0.45));
        shaft = APP.shaftBase * (0.25 + 0.75 * vis);
      }
    }

    // grade per-frame bits
    const c = post.matComposite.uniforms;
    c.uTime.value = t;
    c.uBloom.value = (APP.baseBloom || 0.8) + amp * 0.18;
    c.uWet.value = APP.state.rain ? (0.45 + APP.state.rainAmount * 0.4) : 0;

    renderer.setRenderTarget(post.rtScene);
    renderer.render(scene, camera);
    post.run({
      exposure: c.uExposure.value, temp: c.uTemp.value, saturation: c.uSaturation.value,
      bloom: c.uBloom.value, vignette: c.uVignette.value, grain: c.uGrain.value,
      time: t, wet: c.uWet.value, flash: flash * 0.5,
      shaft: shaft * (1 - APP.state.rainAmount * (APP.state.rain ? 0.5 : 0)), sunScreen,
    });
  }
  frame();

  // ---- control API (consumed by the React dev panel) ----
  function togglePlay() {
    audio.ensure();
    if (audio.ctx && audio.ctx.state === 'suspended') audio.ctx.resume();
    if (audioEl.paused) { audioEl.play().catch(() => {}); return true; }
    audioEl.pause(); return false;
  }
  function loadFile(file) {
    if (!file) return null;
    audioEl.src = URL.createObjectURL(file);
    audio.ensure();
    return file.name;
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(rafId);
    clearTimeout(titleTimer); clearTimeout(settle1); clearTimeout(settle2);
    if (thunderTimer) clearTimeout(thunderTimer);
    window.removeEventListener('resize', onResize);
    if (ro) ro.disconnect();
    typo.dispose();
    audio.dispose();
    post.dispose();
    disposeWorld(WORLD);
    renderer.dispose();
    if (cv.parentNode === container) container.removeChild(cv);
    renderer.forceContextLoss();
  }

  return {
    state: APP.state,
    setSeason: (s) => setScene(s, APP.state.time),
    setTime: (t) => setScene(APP.state.season, t),
    setRain: (on) => { APP.state.rain = on; setRain(WORLD, on, APP.state.rainAmount); },
    setRainAmount: (v) => { APP.state.rainAmount = v; if (APP.state.rain) setRain(WORLD, true, v); },
    setWind: (v) => { APP.state.wind = v; setWind(WORLD, v); },
    thunder: () => { APP.flashT = clock.getElapsedTime(); APP.boltShown = false; },
    replayTitle: () => typo.play(SEASON_LABEL[APP.state.season], TIME_LABEL[APP.state.time], APP.narration),
    setNarration: (text) => { APP.narration = text; },
    playNarration: (text) => { if (text != null) APP.narration = text; typo.play(SEASON_LABEL[APP.state.season], TIME_LABEL[APP.state.time], APP.narration); },
    togglePlay,
    loadFile,
    dispose,
  };
}
