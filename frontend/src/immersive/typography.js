/* typography.js — the signature one-word-at-a-time title engine.
   Ported from the artifact's `TYPO`; parameterized to take its DOM nodes
   (instead of getElementById) and given a dispose() that clears timers. */

export function createTypography(titleWordsEl, titleScrimEl) {
  const cadence = 280, sentencePause = 220;
  let timers = [];
  let lastSeason = '';
  let lastTime = '';
  let lastNarration = '';

  function clear() {
    timers.forEach(clearTimeout); timers = [];
    titleWordsEl.innerHTML = '';
    titleScrimEl.classList.remove('on');
  }

  // play(season, time) reveals the default analysis line; pass a narration
  // string as the third arg to reveal arbitrary prose instead (the demo flow
  // feeds it a second-person description of the resolved scene).
  function play(season, time, narration) {
    clear();
    lastSeason = season;
    lastTime = time;
    lastNarration = narration;

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

    // Stay there for 8 seconds after the animation completes, then fade out and replay.
    // The word transition duration is 0.58s (580ms).
    const holdDelay = delay + 580;
    const fadeOutDelay = holdDelay + 8000;

    timers.push(setTimeout(() => {
      // Fade out all words
      const wordSpans = titleWordsEl.querySelectorAll('.word');
      wordSpans.forEach((span) => {
        span.classList.remove('show');
      });
      // Fade out the scrim
      titleScrimEl.classList.remove('on');

      // Wait 1.2s (matching the scrim fade transition) before replaying
      timers.push(setTimeout(() => {
        play(lastSeason, lastTime, lastNarration);
      }, 1200));
    }, fadeOutDelay));
  }

  return { play, clear, dispose: clear };
}
