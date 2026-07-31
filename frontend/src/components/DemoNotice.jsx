/* DemoNotice — the standing "this is not real inference" stamp.
 *
 * On the `demo` branch every AI response comes from services/dev/ai-mock, which
 * replays pre-baked fixtures. The UI is otherwise indistinguishable from the
 * real thing: it plays audio, prints confidences, and lists the attempt IDs
 * under "Models used". This strip is the one thing that says otherwise, so it
 * renders on every route and cannot be dismissed.
 *
 * It is fixed to the top of the viewport rather than placed in the flow: every
 * full-height page (.main-panel, .demo-chat, .immersive-page) pins itself to
 * 100vh, so an in-flow strip would push them all into overflow. Everything that
 * needs to clear it is offset by --demo-notice-h in styles.css.
 */
export default function DemoNotice() {
  return (
    <aside className="demo-notice" role="note">
      <span className="demo-notice-mark" aria-hidden="true">
        ⚠
      </span>
      <span className="demo-notice-text">
        <strong>Demo build</strong> — the AI service is mocked. Audio, reports and narration are
        pre-baked fixtures replayed from disk, not live model output.
      </span>
      <span className="demo-notice-text-short">
        <strong>Demo build</strong> — mocked AI, pre-baked fixtures.
      </span>
    </aside>
  );
}
