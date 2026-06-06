/* world.js — procedural scene contents.
   Sky dome · fog · ground · layered tree silhouettes · ambient particles · rain.

   Ported from the eco-acoustic artifact. Change from the original: the shared
   global `WORLD` singleton is now a per-engine object that `buildWorld` creates
   and returns, threaded into the helpers as the first arg. This keeps two
   engine instances (e.g. React StrictMode's dev double-mount) fully isolated
   and lets `dispose()` free exactly its own GPU resources.                    */

import * as THREE from 'three';
import {
  SKY_VERT, SKY_FRAG, GROUND_VERT, GROUND_FRAG, PT_VERT, PT_FRAG,
} from './shaders.js';

const PT_MAX = 720;
const RAIN_MAX = 1500;

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ---- procedural tree silhouette: returns {bare, canopy} CanvasTextures --------
function makeTreePair(seed) {
  const W = 220, H = 520, baseX = 110, baseY = 512;
  const cb = document.createElement('canvas'); cb.width = W; cb.height = H;
  const cc = document.createElement('canvas'); cc.width = W; cc.height = H;
  const gb = cb.getContext('2d'), gc = cc.getContext('2d');
  const rnd = mulberry32(seed);
  const tips = [];
  // white ink → alpha carries the silhouette shape, material.color sets the tint
  gb.strokeStyle = '#ffffff'; gb.lineCap = 'round';

  function branch(x, y, ang, len, w, depth) {
    const x2 = x + Math.cos(ang) * len, y2 = y + Math.sin(ang) * len;
    gb.lineWidth = w; gb.beginPath(); gb.moveTo(x, y); gb.lineTo(x2, y2); gb.stroke();
    if (depth <= 0 || len < 7) { tips.push([x2, y2, len]); return; }
    const n = 2 + (rnd() < 0.45 ? 1 : 0);
    for (let i = 0; i < n; i++) {
      const a = ang + (rnd() - 0.5) * 1.15;
      branch(x2, y2, a, len * (0.64 + rnd() * 0.14), w * 0.66, depth - 1);
    }
  }
  branch(baseX, baseY, -Math.PI / 2 + (rnd() - 0.5) * 0.2, 118, 17, 7);

  // canopy blobs around upper tips
  for (const [x, y] of tips) {
    if (y > H * 0.6) continue;
    const r = 26 + rnd() * 34;
    const g = gc.createRadialGradient(x, y, 1, x, y, r);
    g.addColorStop(0, 'rgba(255,255,255,0.62)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    gc.fillStyle = g; gc.beginPath(); gc.arc(x, y, r, 0, 7); gc.fill();
  }
  // feather the very bottom of both canvases so trunks/foliage dissolve into the
  // ground contact instead of ending in a hard flat cut. The side feather keeps
  // foreground planes from exposing their rectangular texture bounds on desktop.
  for (const ctx of [gb, gc]) {
    ctx.globalCompositeOperation = 'destination-out';
    const fade = ctx.createLinearGradient(0, H, 0, H - 58);
    fade.addColorStop(0, 'rgba(0,0,0,1)'); fade.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = fade; ctx.fillRect(0, H - 58, W, 58);
    const side = 72;
    const fadeLeft = ctx.createLinearGradient(0, 0, side, 0);
    fadeLeft.addColorStop(0, 'rgba(0,0,0,1)'); fadeLeft.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = fadeLeft; ctx.fillRect(0, 0, side, H);
    const fadeRight = ctx.createLinearGradient(W, 0, W - side, 0);
    fadeRight.addColorStop(0, 'rgba(0,0,0,1)'); fadeRight.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = fadeRight; ctx.fillRect(W - side, 0, side, H);
    ctx.globalCompositeOperation = 'source-over';
  }
  const mk = (cv) => new THREE.CanvasTexture(cv);
  return { bare: mk(cb), canopy: mk(cc) };
}

function makeBushTexture(seed) {
  const W = 1024, H = 256, g = document.createElement('canvas');
  g.width = W; g.height = H; const ctx = g.getContext('2d');
  const rnd = mulberry32(seed);
  // soft blob clumps forming an irregular brush silhouette
  for (let i = 0; i < 120; i++) {
    const x = rnd() * W, y = H * (0.32 + rnd() * 0.62), r = 16 + rnd() * 48;
    const rg = ctx.createRadialGradient(x, y, 1, x, y, r);
    rg.addColorStop(0, 'rgba(255,255,255,0.9)'); rg.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  }
  // feather the bottom and top edges so the strip dissolves into the ground and
  // the air instead of ending in a hard horizontal cut at the plane edge
  ctx.globalCompositeOperation = 'destination-out';
  const fadeBottom = ctx.createLinearGradient(0, H, 0, H * 0.74);
  fadeBottom.addColorStop(0, 'rgba(0,0,0,1)'); fadeBottom.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = fadeBottom; ctx.fillRect(0, H * 0.74, W, H * 0.26);
  const fadeTop = ctx.createLinearGradient(0, 0, 0, H * 0.5);
  fadeTop.addColorStop(0, 'rgba(0,0,0,1)'); fadeTop.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = fadeTop; ctx.fillRect(0, 0, W, H * 0.5);
  const side = 140;
  const fadeLeft = ctx.createLinearGradient(0, 0, side, 0);
  fadeLeft.addColorStop(0, 'rgba(0,0,0,1)'); fadeLeft.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = fadeLeft; ctx.fillRect(0, 0, side, H);
  const fadeRight = ctx.createLinearGradient(W, 0, W - side, 0);
  fadeRight.addColorStop(0, 'rgba(0,0,0,1)'); fadeRight.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = fadeRight; ctx.fillRect(W - side, 0, side, H);
  ctx.globalCompositeOperation = 'source-over';
  const t = new THREE.CanvasTexture(g); return t;
}

export function buildWorld(scene, opts) {
  const WORLD = {};
  // track everything that owns GPU memory so dispose() can free it
  WORLD._textures = [];
  WORLD._geometries = [];
  WORLD._materials = [];
  const track = (obj, bucket) => { bucket.push(obj); return obj; };

  const g = new THREE.Group(); scene.add(g); WORLD.group = g;

  // SKY DOME --------------------------------------------------------------
  WORLD.skyMat = new THREE.ShaderMaterial({
    vertexShader: SKY_VERT, fragmentShader: SKY_FRAG, side: THREE.BackSide, depthWrite: false,
    uniforms: {
      uSkyTop: { value: new THREE.Color() }, uSkyHorizon: { value: new THREE.Color() },
      uSunColor: { value: new THREE.Color() }, uFog: { value: new THREE.Color() },
      uSunDir: { value: new THREE.Vector3(0, 0.2, -1) },
      uHaze: { value: 0.4 }, uBrightness: { value: 1 }, uStars: { value: 0 },
      uMoon: { value: 0 }, uSunSize: { value: 0.8 }, uFlash: { value: 0 }, uTime: { value: 0 },
    },
  });
  track(WORLD.skyMat, WORLD._materials);
  const skyGeo = track(new THREE.SphereGeometry(600, 40, 24), WORLD._geometries);
  WORLD.sky = new THREE.Mesh(skyGeo, WORLD.skyMat);
  scene.add(WORLD.sky);

  // FOG + GROUND ----------------------------------------------------------
  scene.fog = new THREE.FogExp2(0x111118, 0.0062);
  WORLD.groundMat = new THREE.ShaderMaterial({
    vertexShader: GROUND_VERT, fragmentShader: GROUND_FRAG,
    uniforms: {
      uBase: { value: new THREE.Color(0x0a0c0f) },
      uSunColor: { value: new THREE.Color(1, 0.6, 0.4) },
      uFog: { value: new THREE.Color(0x111118) },
      uSunAz: { value: new THREE.Vector2(0, -1) },
      uBrightness: { value: 1 }, uFogDensity: { value: 0.0062 }, uShaftLow: { value: 0.5 },
      uSkyLift: { value: 0.0 },
    },
  });
  track(WORLD.groundMat, WORLD._materials);
  const groundGeo = track(new THREE.PlaneGeometry(1400, 1400), WORLD._geometries);
  const ground = new THREE.Mesh(groundGeo, WORLD.groundMat);
  ground.rotation.x = -Math.PI / 2; ground.position.y = 0; scene.add(ground);

  // TREES (layered depth bands) ------------------------------------------
  const variants = [makeTreePair(11), makeTreePair(29), makeTreePair(53), makeTreePair(77)];
  variants.forEach((v) => { track(v.bare, WORLD._textures); track(v.canopy, WORLD._textures); });
  WORLD.canopyMats = [];
  WORLD.treeMats = [];   // {mat, base:Color, z} — tinted per-scene for atmosphere
  const BARE_BASE = 0x05080e, CANOPY_BASE = 0x06090a;
  const placeTree = (x, z, h, vi) => {
    const v = variants[vi % variants.length];
    const aspect = 220 / 520, w = h * aspect;
    const bm = new THREE.MeshBasicMaterial({ map: v.bare, transparent: true, alphaTest: 0.01, depthWrite: false, fog: true, color: BARE_BASE });
    const cm = new THREE.MeshBasicMaterial({ map: v.canopy, transparent: true, alphaTest: 0.01, depthWrite: false, fog: true, opacity: 0.5, color: CANOPY_BASE });
    track(bm, WORLD._materials); track(cm, WORLD._materials);
    const geo = track(new THREE.PlaneGeometry(w, h), WORLD._geometries); geo.translate(0, h / 2, 0);
    const bare = new THREE.Mesh(geo, bm); bare.position.set(x, 0, z);
    const canopy = new THREE.Mesh(geo, cm); canopy.position.set(x, 0, z);
    g.add(bare); g.add(canopy); WORLD.canopyMats.push(cm);
    WORLD.treeMats.push({ mat: bm, base: new THREE.Color(BARE_BASE), z });
    WORLD.treeMats.push({ mat: cm, base: new THREE.Color(CANOPY_BASE), z });
  };
  const R = mulberry32(7);
  // foreground hero trees framing the shot
  placeTree(-16, -22, 25, 0); placeTree(19, -26, 27, 1); placeTree(-26, -34, 22, 2);
  placeTree(33, -40, 21, 3); placeTree(48, -64, 17, 0);
  // mid band
  for (let i = 0; i < 8; i++) placeTree((R() - 0.5) * 80, -42 - R() * 36, 13 + R() * 5, i);
  // transition band — bridges the mid->far gap so depth recedes continuously
  for (let i = 0; i < 10; i++) placeTree((R() - 0.5) * 116, -74 - R() * 24, 10 + R() * 5, i);
  // far band, dissolving into fog
  for (let i = 0; i < 14; i++) placeTree((R() - 0.5) * 150, -90 - R() * 80, 9 + R() * 5, i);

  // undergrowth — low scattered clumps along the ground, each its own small
  // plane so the foreground reads as broken-up brush instead of one big slab.
  const bush = track(makeBushTexture(101), WORLD._textures);
  const BUSH_BASE = 0x04060a;
  WORLD.bushMats = [];   // tinted per-season toward the vegetation colour
  const BR = mulberry32(303);
  const placeBush = (x, z, w) => {
    const h = w * 0.34;                          // keep clumps low to the ground
    const bgeo = track(new THREE.PlaneGeometry(w, h), WORLD._geometries);
    bgeo.translate(0, h / 2, 0);
    const bmat = track(new THREE.MeshBasicMaterial({
      map: bush, transparent: true, alphaTest: 0.01, depthWrite: false, fog: true, color: BUSH_BASE,
    }), WORLD._materials);
    const m = new THREE.Mesh(bgeo, bmat); m.position.set(x, 0, z); g.add(m);
    WORLD.bushMats.push({ mat: bmat, base: new THREE.Color(BUSH_BASE), z });
  };
  // a near apron framing the lower edge, then a scattered mid layer that also
  // grounds the foreground tree bases so they don't look pasted onto the floor.
  // depth is heavily staggered so no row of clump bases ever lines up into a seam
  for (let i = 0; i < 8; i++) placeBush((BR() - 0.5) * 78, -8 - BR() * 30, 28 + BR() * 30);
  for (let i = 0; i < 10; i++) placeBush((BR() - 0.5) * 130, -30 - BR() * 38, 24 + BR() * 32);

  // AMBIENT PARTICLES -----------------------------------------------------
  const pgeo = track(new THREE.BufferGeometry(), WORLD._geometries);
  WORLD.pPos = new Float32Array(PT_MAX * 3);
  WORLD.pSize = new Float32Array(PT_MAX);
  WORLD.pAlpha = new Float32Array(PT_MAX);
  WORLD.pData = []; // per-particle motion params
  pgeo.setAttribute('position', new THREE.BufferAttribute(WORLD.pPos, 3));
  pgeo.setAttribute('aSize', new THREE.BufferAttribute(WORLD.pSize, 1));
  pgeo.setAttribute('aAlpha', new THREE.BufferAttribute(WORLD.pAlpha, 1));
  WORLD.ptMat = new THREE.ShaderMaterial({
    vertexShader: PT_VERT, fragmentShader: PT_FRAG, transparent: true, depthWrite: false,
    uniforms: { uColor: { value: new THREE.Color(1, 1, 1) }, uSoft: { value: 1.0 },
                uMaster: { value: 1.0 }, uPx: { value: opts.dpr * 46 } },
  });
  track(WORLD.ptMat, WORLD._materials);
  WORLD.points = new THREE.Points(pgeo, WORLD.ptMat); scene.add(WORLD.points);

  // RAIN ------------------------------------------------------------------
  const rgeo = track(new THREE.BufferGeometry(), WORLD._geometries);
  WORLD.rPos = new Float32Array(RAIN_MAX * 2 * 3);
  WORLD.rData = [];
  for (let i = 0; i < RAIN_MAX; i++) WORLD.rData.push(seedRainDrop());
  rgeo.setAttribute('position', new THREE.BufferAttribute(WORLD.rPos, 3));
  WORLD.rainMat = new THREE.LineBasicMaterial({ color: 0xbcd0e8, transparent: true, opacity: 0.34, fog: true });
  track(WORLD.rainMat, WORLD._materials);
  WORLD.rain = new THREE.LineSegments(rgeo, WORLD.rainMat);
  WORLD.rain.visible = false; scene.add(WORLD.rain);
  WORLD.rainGeo = rgeo;

  WORLD.wind = 0;          // 0 = calm … 1 = gale; blows toward +x
  WORLD.curParticle = null;
  return WORLD;
}

function seedRainDrop() {
  return { x: (Math.random() - 0.5) * 90, y: Math.random() * 60, z: -100 + Math.random() * 110,
           len: 1.0 + Math.random() * 1.4, sp: 38 + Math.random() * 26 };
}

// reconfigure particle field for a season's particle type
export function setupParticles(WORLD, p) {
  const count = Math.min(p.pCount, PT_MAX);
  const old = WORLD.pData || [];
  WORLD.pData = [];
  for (let i = 0; i < count; i++) {
    const o = old[i];           // reuse prior position so a season change morphs in place
    WORLD.pData.push({
      x: o ? o.x : (Math.random() - 0.5) * 110,
      y: o ? o.y : Math.random() * 40,
      z: o ? o.z : -100 + Math.random() * 88,
      ph: Math.random() * 6.28, sway: 0.4 + Math.random() * 1.2,
      fall: 0.4 + Math.random() * 0.8, drift: (Math.random() - 0.5) * 0.4,
      s: (0.6 + Math.random() * 0.7) * p.pSize, a: (0.28 + Math.random() * 0.4),
    });
  }
  WORLD.points.geometry.setDrawRange(0, count);
  WORLD.ptMat.uniforms.uColor.value.setRGB(p.pColor[0], p.pColor[1], p.pColor[2]);
  const t = p.particle;
  WORLD.ptMat.uniforms.uSoft.value = (t === 'leaves') ? 0.2 : 1.0;
  WORLD.ptMat.blending = (t === 'pollen' || t === 'dust') ? THREE.AdditiveBlending : THREE.NormalBlending;
  WORLD.ptMat.needsUpdate = true;
  WORLD.curParticle = t;
}

export function updateParticles(WORLD, dt, t, amp) {
  const d = WORLD.pData, type = WORLD.curParticle;
  const wind = WORLD.wind || 0;
  // nonlinear so a gentle breeze stays gentle but a gale really rips
  const gust = wind * (1 + wind);
  // light debris rides the wind hardest; airborne pollen/dust drifts less
  const ride = type === 'leaves' ? 16 : type === 'snow' ? 11 : 7;
  for (let i = 0; i < d.length; i++) {
    const p = d[i];
    if (type === 'snow') { p.y -= p.fall * 2.4 * dt; p.x += Math.sin(t * 0.6 + p.ph) * p.sway * dt; }
    else if (type === 'leaves') { p.y -= p.fall * 2.0 * dt; p.x += Math.sin(t * 0.9 + p.ph) * p.sway * 1.8 * dt; }
    else { // pollen / dust — hover + brownian
      p.y += Math.sin(t * 0.3 + p.ph) * 0.25 * dt + p.fall * 0.12 * dt;
      p.x += Math.cos(t * 0.4 + p.ph) * p.drift * 1.4 * dt;
    }
    if (wind > 0) p.x += gust * ride * dt;
    if (p.y < 0) { p.y = 40; p.x = (Math.random() - 0.5) * 110; }
    if (p.y > 42) p.y = 0.2;
    // wrap horizontally so the field keeps flowing instead of piling downwind
    if (p.x > 60) p.x -= 120; else if (p.x < -60) p.x += 120;
    const i3 = i * 3;
    WORLD.pPos[i3] = p.x; WORLD.pPos[i3 + 1] = p.y; WORLD.pPos[i3 + 2] = p.z;
    WORLD.pSize[i] = p.s * (1 + (type === 'leaves' ? 0.3 * Math.sin(t * 3 + p.ph) : 0));
    WORLD.pAlpha[i] = p.a * (0.85 + 0.15 * amp);
  }
  const geo = WORLD.points.geometry;
  geo.attributes.position.needsUpdate = true;
  geo.attributes.aSize.needsUpdate = true;
  geo.attributes.aAlpha.needsUpdate = true;
}

export function setRain(WORLD, on, intensity) {
  WORLD.rain.visible = on;
  WORLD.rainIntensity = intensity;
  const count = on ? Math.floor(RAIN_MAX * intensity) : 0;
  WORLD.rainGeo.setDrawRange(0, count * 2);
}

export function updateRain(WORLD, dt) {
  if (!WORLD.rain.visible) return;
  const count = Math.floor(RAIN_MAX * (WORLD.rainIntensity || 0.6));
  const pos = WORLD.rPos;
  const wind = WORLD.wind || 0;
  // nonlinear so a gale drives the rain far harder than a light breeze
  const gust = wind * (1 + wind);
  for (let i = 0; i < count; i++) {
    const r = WORLD.rData[i];
    r.y -= r.sp * dt;
    if (wind > 0) r.x += gust * 34 * dt;
    if (r.y < 0) { r.y = 55 + Math.random() * 8; r.x = (Math.random() - 0.5) * 90; r.z = -100 + Math.random() * 110; }
    else if (r.x > 55) r.x -= 110;       // wrap so wind-blown rain keeps crossing the frame
    const i6 = i * 6;
    // streak leans downwind: near-vertical when calm, raked over in a gale
    const lean = (0.2 + gust * 4.6) * r.len;
    pos[i6] = r.x; pos[i6 + 1] = r.y; pos[i6 + 2] = r.z;
    pos[i6 + 3] = r.x + lean; pos[i6 + 4] = r.y - r.len * 2.2; pos[i6 + 5] = r.z;
  }
  WORLD.rainGeo.attributes.position.needsUpdate = true;
}

export function setWind(WORLD, intensity) {
  WORLD.wind = Math.max(0, Math.min(1, intensity || 0));
}

// free all GPU resources this world created
export function disposeWorld(WORLD) {
  WORLD._textures.forEach((t) => t.dispose());
  WORLD._geometries.forEach((g) => g.dispose());
  WORLD._materials.forEach((m) => m.dispose());
}
