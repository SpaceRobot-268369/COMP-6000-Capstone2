/* post.js — hand-rolled post pipeline.
   scene -> bright-pass -> separable blur (half res) -> composite(grade/grain/vignette).
   Quality scales: bloomScale 0.5 (rich) or 0.34 (lite); blur can be skipped.
   Ported from the eco-acoustic artifact: globals → ES-module imports, plus a
   dispose() so render targets/geometries/materials free on React unmount.      */

import * as THREE from 'three';
import { FS_VERT, BRIGHT_FRAG, BLUR_FRAG, GODRAY_FRAG, COMPOSITE_FRAG } from './shaders.js';

export class PostPipeline {
  constructor(renderer, w, h, opts) {
    this.renderer = renderer;
    this.opts = Object.assign({ bloomScale: 0.5, bloom: true }, opts || {});
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quadScene = new THREE.Scene();
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quadScene.add(this.quad);

    const rtPars = { type: THREE.UnsignedByteType, depthBuffer: true,
                     minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.rtScene = new THREE.WebGLRenderTarget(w, h, rtPars);
    const bw = Math.max(2, Math.floor(w * this.opts.bloomScale));
    const bh = Math.max(2, Math.floor(h * this.opts.bloomScale));
    const bpars = { type: THREE.UnsignedByteType, depthBuffer: false,
                    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.rtA = new THREE.WebGLRenderTarget(bw, bh, bpars);
    this.rtB = new THREE.WebGLRenderTarget(bw, bh, bpars);
    this.rtGod = new THREE.WebGLRenderTarget(bw, bh, bpars);
    this.bw = bw; this.bh = bh;

    this.matBright = new THREE.ShaderMaterial({
      vertexShader: FS_VERT, fragmentShader: BRIGHT_FRAG,
      uniforms: { tDiffuse: { value: null }, uThreshold: { value: 0.68 } },
      depthTest: false, depthWrite: false,
    });
    this.matBlur = new THREE.ShaderMaterial({
      vertexShader: FS_VERT, fragmentShader: BLUR_FRAG,
      uniforms: { tDiffuse: { value: null }, uDir: { value: new THREE.Vector2() } },
      depthTest: false, depthWrite: false,
    });
    this.matGod = new THREE.ShaderMaterial({
      vertexShader: FS_VERT, fragmentShader: GODRAY_FRAG,
      uniforms: { tDiffuse: { value: null }, uSun: { value: new THREE.Vector2(0.5, 0.6) },
                  uDensity: { value: 0.9 }, uDecay: { value: 0.94 }, uWeight: { value: 0.32 } },
      depthTest: false, depthWrite: false,
    });
    this.matComposite = new THREE.ShaderMaterial({
      vertexShader: FS_VERT, fragmentShader: COMPOSITE_FRAG,
      uniforms: {
        tDiffuse: { value: null }, tBloom: { value: null }, tGod: { value: null },
        uExposure: { value: 1.0 }, uTemp: { value: 0.0 }, uSaturation: { value: 1.0 },
        uContrast: { value: 1.06 }, uBloom: { value: 0.9 }, uVignette: { value: 0.35 },
        uGrain: { value: 0.05 }, uTime: { value: 0.0 }, uWet: { value: 0.0 }, uFlash: { value: 0.0 },
        uShaft: { value: 0.0 },
        uFogColor: { value: new THREE.Color(0x111118) },
        uSkyColor: { value: new THREE.Color(0x332840) },
        uAtmosphere: { value: 0.75 },
        uAspect: { value: w / h },
      },
      depthTest: false, depthWrite: false,
    });
  }

  setSize(w, h) {
    this.rtScene.setSize(w, h);
    const bw = Math.max(2, Math.floor(w * this.opts.bloomScale));
    const bh = Math.max(2, Math.floor(h * this.opts.bloomScale));
    this.rtA.setSize(bw, bh); this.rtB.setSize(bw, bh); this.rtGod.setSize(bw, bh);
    this.bw = bw; this.bh = bh;
    this.matComposite.uniforms.uAspect.value = w / h;
  }

  _draw(mat, target) {
    this.quad.material = mat;
    this.renderer.setRenderTarget(target || null);
    this.renderer.render(this.quadScene, this.quadCam);
  }

  // grade = { exposure, temp, saturation, bloom, vignette, grain, time, wet, flash }
  run(grade) {
    const r = this.renderer;
    if (this.opts.bloom) {
      this.matBright.uniforms.tDiffuse.value = this.rtScene.texture;
      this._draw(this.matBright, this.rtA);
      // light shafts from the bright pass (before it gets blurred for bloom)
      if (grade.shaft > 0.001) {
        this.matGod.uniforms.tDiffuse.value = this.rtA.texture;
        this.matGod.uniforms.uSun.value.copy(grade.sunScreen);
        this._draw(this.matGod, this.rtGod);
      }
      this.matBlur.uniforms.tDiffuse.value = this.rtA.texture;
      this.matBlur.uniforms.uDir.value.set(1.2 / this.bw, 0);
      this._draw(this.matBlur, this.rtB);
      this.matBlur.uniforms.tDiffuse.value = this.rtB.texture;
      this.matBlur.uniforms.uDir.value.set(0, 1.2 / this.bh);
      this._draw(this.matBlur, this.rtA);
    }
    const u = this.matComposite.uniforms;
    u.tDiffuse.value = this.rtScene.texture;
    u.tBloom.value = this.opts.bloom ? this.rtA.texture : null;
    u.tGod.value = this.rtGod.texture;
    u.uShaft.value = grade.shaft || 0.0;
    u.uBloom.value = this.opts.bloom ? grade.bloom : 0.0;
    u.uExposure.value = grade.exposure;
    u.uTemp.value = grade.temp;
    u.uSaturation.value = grade.saturation;
    u.uVignette.value = grade.vignette;
    u.uGrain.value = grade.grain;
    u.uTime.value = grade.time;
    u.uWet.value = grade.wet;
    u.uFlash.value = grade.flash;
    this._draw(this.matComposite, null);
    r.setRenderTarget(null);
  }

  // free GPU resources on teardown (React unmount / route change / HMR)
  dispose() {
    [this.rtScene, this.rtA, this.rtB, this.rtGod].forEach((rt) => rt.dispose());
    [this.matBright, this.matBlur, this.matGod, this.matComposite].forEach((m) => m.dispose());
    this.quad.geometry.dispose();
  }
}
