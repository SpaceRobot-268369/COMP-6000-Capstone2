/* typography.js — the signature one-word-at-a-time title engine.
   Ported from the artifact's `TYPO`; parameterized to take its DOM nodes
   (instead of getElementById) and given a dispose() that clears timers. */

export function createTypography(titleWordsEl, titleScrimEl) {
  const cadence = 280, sentencePause = 220;
  let timers = [];

  function clear() {
    timers.forEach(clearTimeout); timers = [];
    titleWordsEl.innerHTML = '';
  }

  // play(season, time) reveals the default analysis line; pass a narration
  // string as the third arg to reveal arbitrary prose instead (the demo flow
  // feeds it a second-person description of the resolved scene).
  function play(season, time, narration) {
    clear();
    const body = 'A dry woodland wakes. Rain moves through the canopy. Somewhere, thunder. This is what the recording remembers.';
    const text = narration && narration.trim()
      ? narration.trim()
      : `${season}. ${time}. ${body}`;
    const words = text.split(/\s+/);
    titleScrimEl.classList.add('on');
    let delay = 350;
    words.forEach((w) => {
      const span = document.createElement('span');
      span.className = 'word'; span.textContent = w;
      titleWordsEl.appendChild(span);
      const d = delay;
      timers.push(setTimeout(() => span.classList.add('show'), d));
      delay += cadence + (/[.;:!?]$/.test(w) ? sentencePause : 0);
    });
    // settle marker (leaves text in place)
    timers.push(setTimeout(() => {}, delay));
  }

  return { play, clear, dispose: clear };
}
