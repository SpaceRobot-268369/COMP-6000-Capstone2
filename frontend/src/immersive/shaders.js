/* shaders.js — GLSL source strings. Ported verbatim from the eco-acoustic
   artifact; only `const` → `export const` to make them ES-module imports. */

// ---------------- SKY DOME ----------------------------------------------------
export const SKY_VERT = `
varying vec3 vDir;
void main(){
  vDir = normalize(position);          // dome is a unit-ish sphere centred on camera
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
}`;

export const SKY_FRAG = `
precision highp float;
varying vec3 vDir;
uniform vec3  uSkyTop, uSkyHorizon, uSunColor, uFog;
uniform vec3  uSunDir;
uniform float uHaze, uBrightness, uStars, uMoon, uSunSize, uFlash, uTime;

float hash(vec2 p){ p = fract(p*vec2(443.897,441.423)); p += dot(p,p+19.19); return fract(p.x*p.y); }

void main(){
  vec3 dir = normalize(vDir);
  float up = clamp(dir.y, -0.2, 1.0);
  float t  = pow(clamp(up, 0.0, 1.0), 0.42);
  vec3 col = mix(uSkyHorizon, uSkyTop, t);

  // sun / moon
  float sd = max(dot(dir, normalize(uSunDir)), 0.0);
  float disc = smoothstep(0.9990 + 0.0006*(1.0-uSunSize), 0.99985, sd);
  float glow = pow(sd, 7.0)*0.45 + pow(sd, 90.0)*0.9;
  float aura = pow(sd, 2.0)*0.12;
  col += uSunColor * (glow + aura) * (0.6 + uHaze);
  vec3 discCol = mix(uSunColor, vec3(1.0), uMoon*0.4);
  col += discCol * disc * (uMoon > 0.5 ? 2.2 : 3.4);

  // stars (night only, above horizon)
  if(uStars > 0.5 && dir.y > 0.02){
    vec2 g = floor((dir.xz/(dir.y+0.3)) * 120.0);
    float h = hash(g);
    float star = step(0.995, h) * smoothstep(0.0, 0.05, dir.y);
    float tw = 0.6 + 0.4*sin(uTime*2.0 + h*40.0);
    col += vec3(0.9,0.95,1.0) * star * tw * (1.0 - smoothstep(0.0,0.4,sd));
  }

  // horizon haze band
  col = mix(col, uFog, (1.0 - t) * uHaze * 0.65);

  col *= uBrightness;
  col += vec3(0.9, 0.94, 1.0) * uFlash;     // thunder lifts the whole sky
  gl_FragColor = vec4(col, 1.0);
}`;

// ---------------- AMBIENT PARTICLES (Points) ----------------------------------
export const PT_VERT = `
attribute float aSize;
attribute float aAlpha;
varying float vAlpha;
uniform float uPx;
void main(){
  vAlpha = aAlpha;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = clamp(aSize * uPx * (1.0 / -mv.z), 1.0, 17.0);
  gl_Position = projectionMatrix * mv;
}`;

export const PT_FRAG = `
precision mediump float;
varying float vAlpha;
uniform vec3 uColor;
uniform float uSoft;          // 1 = soft round, 0 = harder (leaves)
uniform float uMaster;        // global fade for smooth season swaps
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c);
  float a = smoothstep(0.5, mix(0.15, 0.42, uSoft), d);
  gl_FragColor = vec4(uColor, a * vAlpha * uMaster);
}`;

// ---------------- GROUND (directional, lit from sun azimuth) ------------------
export const GROUND_VERT = `
varying vec3 vWorld;
varying float vDist;
void main(){
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDist = -mv.z;
  gl_Position = projectionMatrix * mv;
}`;

export const GROUND_FRAG = `
precision highp float;
varying vec3 vWorld;
varying float vDist;
uniform vec3  uBase, uSunColor, uFog;
uniform vec2  uSunAz;          // normalised xz toward the sun
uniform float uBrightness, uFogDensity, uShaftLow, uSkyLift;
void main(){
  vec3 col = uBase;
  vec2 p = vWorld.xz;
  float r = length(p) + 1e-3;
  vec2 pn = p / r;
  // overall skylight wash — lifts the WHOLE floor on bright days so the
  // foreground isn't dead black under a luminous morning sky
  float openBand = smoothstep(2.0, 40.0, r) * (1.0 - smoothstep(120.0, 360.0, r));
  col += uFog * uSkyLift * (0.35 + 0.45 * openBand);
  float align = max(dot(pn, uSunAz), 0.0);         // 1 toward the sun
  // mid-distance band so the wash sits in the scene, not under the camera
  float band = smoothstep(2.0, 46.0, r) * (1.0 - smoothstep(120.0, 340.0, r));
  float pool = pow(align, 2.2) * band;
  col += uSunColor * pool * (0.62 * uBrightness);
  // tight light streak under the sun, longest when the sun is low (dawn)
  float streak = pow(align, 34.0) * band;
  col += uSunColor * streak * uShaftLow * 1.1;
  // exponential fog to match the sky
  float f = 1.0 - exp(-uFogDensity * uFogDensity * vDist * vDist);
  col = mix(col, uFog, clamp(f, 0.0, 1.0));
  gl_FragColor = vec4(col, 1.0);
}`;

// ---------------- POST: fullscreen passes -------------------------------------
export const FS_VERT = `
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// bright-pass (extract highlights for bloom)
export const BRIGHT_FRAG = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform float uThreshold;
void main(){
  vec3 c = texture2D(tDiffuse, vUv).rgb;
  float l = dot(c, vec3(0.299,0.587,0.114));
  float k = max(l - uThreshold, 0.0) / max(l, 0.0001);
  gl_FragColor = vec4(c * k, 1.0);
}`;

// separable gaussian blur
export const BLUR_FRAG = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform vec2 uDir;            // texel-sized step * direction
void main(){
  vec3 sum = vec3(0.0);
  sum += texture2D(tDiffuse, vUv).rgb * 0.227;
  sum += texture2D(tDiffuse, vUv + uDir*1.0).rgb * 0.196;
  sum += texture2D(tDiffuse, vUv - uDir*1.0).rgb * 0.196;
  sum += texture2D(tDiffuse, vUv + uDir*2.0).rgb * 0.120;
  sum += texture2D(tDiffuse, vUv - uDir*2.0).rgb * 0.120;
  sum += texture2D(tDiffuse, vUv + uDir*3.0).rgb * 0.054;
  sum += texture2D(tDiffuse, vUv - uDir*3.0).rgb * 0.054;
  gl_FragColor = vec4(sum, 1.0);
}`;

// radial light-shafts (god-rays) from the sun's screen position, over the bright pass
export const GODRAY_FRAG = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D tDiffuse;   // bright-pass highlights
uniform vec2 uSun;            // sun position in UV
uniform float uDensity, uDecay, uWeight;
void main(){
  const int N = 24;
  vec2 delta = (vUv - uSun) * (uDensity / float(N));
  vec2 uv = vUv;
  vec3 acc = vec3(0.0);
  float illum = 1.0;
  for(int i = 0; i < N; i++){
    uv -= delta;
    acc += texture2D(tDiffuse, uv).rgb * illum;
    illum *= uDecay;
  }
  gl_FragColor = vec4(acc * uWeight, 1.0);
}`;

// final composite: scene + bloom, colour grade, wet, vignette, grain, flash
export const COMPOSITE_FRAG = `
precision highp float;
varying vec2 vUv;
uniform sampler2D tDiffuse;   // scene
uniform sampler2D tBloom;     // blurred highlights
uniform sampler2D tGod;       // light shafts
uniform float uExposure, uTemp, uSaturation, uContrast;
uniform float uBloom, uVignette, uGrain, uTime, uWet, uFlash, uShaft;

float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }

void main(){
  vec3 col = texture2D(tDiffuse, vUv).rgb;
  col += texture2D(tBloom, vUv).rgb * uBloom;
  col += texture2D(tGod, vUv).rgb * uShaft;

  // exposure
  col *= uExposure;

  // temperature (warm/cool)
  col.r += uTemp * 0.10;
  col.b -= uTemp * 0.10;

  // wet grade (rain) — darken + cool slightly
  col *= (1.0 - uWet * 0.28);
  col.b += uWet * 0.02;

  // saturation
  float l = dot(col, vec3(0.299,0.587,0.114));
  col = mix(vec3(l), col, uSaturation);

  // contrast (filmic-ish around 0.5 pivot, gentle shadow lift)
  col = (col - 0.5) * uContrast + 0.5;
  col = max(col, 0.0);

  // thunder flash also lifts the frame
  col += uFlash * 0.6;

  // vignette
  vec2 q = vUv - 0.5;
  float vig = smoothstep(0.98, 0.42, length(q) * (1.0 + uVignette));
  col *= mix(1.0, vig, 0.55);

  // film grain
  float g = hash(vUv * vec2(1920.0,1080.0) + uTime*60.0) - 0.5;
  col += g * uGrain;

  gl_FragColor = vec4(col, 1.0);
}`;
