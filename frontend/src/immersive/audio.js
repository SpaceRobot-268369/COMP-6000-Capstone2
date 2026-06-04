/* audio.js — loose amplitude analyser feeding the scene's audio reactivity.
   Ported from the artifact's `AUDIO`; parameterized to take its <audio> element
   and given a dispose() that tears the WebAudio graph down on unmount.        */

export function createAudio(audioEl) {
  let ctx = null, analyser = null, data = null, amp = 0, src = null;

  function ensure() {
    if (ctx) return;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      src = ctx.createMediaElementSource(audioEl);
      analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      data = new Uint8Array(analyser.frequencyBinCount);
      src.connect(analyser); analyser.connect(ctx.destination);
    } catch (e) { /* file may be empty / blocked; amplitude stays 0 */ }
  }

  function sample() {
    if (!analyser) return amp;
    analyser.getByteFrequencyData(data);
    let s = 0; for (let i = 0; i < data.length; i++) s += data[i];
    const a = (s / data.length) / 255;
    amp += (a - amp) * 0.12;       // smooth
    return amp;
  }

  function dispose() {
    try { audioEl.pause(); } catch (e) { /* ignore */ }
    try { src && src.disconnect(); } catch (e) { /* ignore */ }
    try { analyser && analyser.disconnect(); } catch (e) { /* ignore */ }
    try { ctx && ctx.close(); } catch (e) { /* ignore */ }
    ctx = null; analyser = null; data = null; src = null;
  }

  return { ensure, sample, dispose, get ctx() { return ctx; } };
}
