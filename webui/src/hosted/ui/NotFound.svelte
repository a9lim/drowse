<script lang="ts">
  import { onMount } from "svelte";
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import HeroShader from "./HeroShader.svelte";
  import RecordedTokenText from "./RecordedTokenText.svelte";
  import { nextMessage, recording } from "./notFoundMessages";

  let selection = $state(nextMessage());
  let highlights = $state(true);
  let paused = $state(false);
  let plainBackground = $state(false);
  const message = $derived(recording.messages[selection]);
  let announcement = $state("");

  function another() {
    selection = nextMessage(message.id);
    announcement = message.text;
  }

  onMount(() => {
    const contrast = matchMedia("(prefers-contrast: more), (forced-colors: active)");
    const update = () => { plainBackground = contrast.matches; };
    update();
    contrast.addEventListener("change", update);
    return () => contrast.removeEventListener("change", update);
  });
</script>

<TabIdentity state="error" title="Page not found · Drowse" />
<svelte:head><meta name="robots" content="noindex, follow" /></svelte:head>

<a class="skip-link" href="#not-found-main">Skip to content</a>
<div class="not-found-shell" class:plain={plainBackground}>
  <div class="scene" aria-hidden="true">
    {#if !plainBackground}<HeroShader contained closeup {paused} />{/if}
    <div class="scene-shade"></div>
    <span class="scene-code">404</span>
  </div>
  <div class="page-layer">
    <PageHeader />
    <main id="not-found-main">
      <section class="message-stage" aria-label="Page not found">
        <h1 class="eyebrow"><span class="status-dot" aria-hidden="true"></span>Page not found</h1>
        <div class="message" data-message-id={message.id}>
          {#key message.id}<RecordedTokenText tokens={message.tokens} {highlights} />{/key}
        </div>
        <p class="explanation">A small detour. Find your way back.</p>
        <div class="recovery-actions">
          <a class="primary-action" href="/">Back to home</a>
          <a class="secondary-action" href="/app">Open Drowse <span aria-hidden="true">↗</span></a>
        </div>
      </section>

      <section class="recording-controls" aria-label="Explore the recorded message">
        <div class="recording-caption">
          <p>Written by Gemma 3 4B<span aria-hidden="true"> · </span><span>{recording.uniqueMessages} unique messages from {recording.totalRuns} takes</span></p>
          <p class="hint">Hover or tap a word to inspect its tokens. Brighter highlights mean less likely choices.</p>
        </div>
        <div class="controls">
          <button class="control highlight-control" type="button" aria-pressed={highlights} onclick={() => highlights = !highlights}>
            <span class="highlight-swatch" aria-hidden="true"></span>Surprisal
          </button>
          <button class="control" type="button" onclick={another}><FluentIcon name="refresh" />Another message</button>
          {#if !plainBackground}<button class="control motion-control" type="button" aria-pressed={paused} onclick={() => paused = !paused}>{paused ? "Resume background" : "Pause background"}</button>{/if}
        </div>
        <a class="recording-link" href="/recordings/404-gemma3-4b.json" download>View all {recording.totalRuns} recordings</a>
      </section>
      <span class="sr-only" aria-live="polite">{announcement}</span>
    </main>
    <PageFooter />
  </div>
</div>

<style>
  .not-found-shell { position: relative; min-height: 100dvh; color: var(--fg); background: var(--hero-bg); isolation: isolate; overflow-x: clip; }
  .scene { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
  .scene-shade { position: absolute; inset: 0; background: linear-gradient(90deg, color-mix(in srgb, var(--hero-bg) 76%, transparent), color-mix(in srgb, var(--hero-bg) 22%, transparent) 70%), linear-gradient(0deg, var(--hero-bg), transparent 55%, color-mix(in srgb, var(--hero-bg) 68%, transparent)); }
  .scene-code { position: absolute; top: 10%; right: -0.055em; color: color-mix(in srgb, var(--fg) 5%, transparent); font-family: var(--font-structure); font-size: var(--text-not-found-code); font-weight: var(--weight-display); line-height: 1; letter-spacing: -0.08em; user-select: none; }
  .page-layer { position: relative; display: flex; flex-direction: column; min-height: 100dvh; }
  main { flex: 1; display: flex; flex-direction: column; width: min(100%, var(--page-max)); padding-inline: max(var(--surface-padding), env(safe-area-inset-left)) max(var(--surface-padding), env(safe-area-inset-right)); margin-inline: auto; }
  .message-stage { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: start; padding-block: clamp(80px, 12vh, 160px) clamp(56px, 9vh, 112px); }
  .eyebrow { display: flex; align-items: center; gap: var(--space-sm); margin: 0 0 var(--space-lg); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-sm); }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 16px var(--accent-subtle); }
  .message { max-width: min(100%, 24ch); min-height: 3.75em; color: var(--fg-strong); font-family: var(--font-structure); font-size: var(--text-not-found-title); font-weight: var(--weight-structure); letter-spacing: -0.035em; line-height: 1.3; }
  .explanation { margin: var(--space-lg) 0 0; color: var(--fg-dim); font-family: var(--font-reading); font-size: var(--text-md); line-height: 1.6; text-wrap: pretty; }
  .recovery-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-sm) var(--space-lg); margin-top: var(--space-lg); }
  .primary-action, .secondary-action { display: inline-flex; align-items: center; justify-content: center; min-height: 48px; padding: var(--space-sm) var(--space-lg); border-radius: var(--radius); font-family: var(--font-structure); font-size: var(--text); font-weight: var(--weight-structure); text-decoration: none; }
  .primary-action { background: var(--accent); color: var(--text-on-accent); box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 60%, transparent), 0 8px 24px color-mix(in srgb, var(--hero-bg) 22%, transparent); }
  .primary-action:hover { background: var(--accent-light); }
  .secondary-action { gap: var(--space-sm); color: var(--fg); padding-inline: var(--space-xs); }
  .secondary-action:hover { color: var(--accent); }
  .recording-controls { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-xs) var(--space-lg); padding-block: var(--space-lg); align-items: center; }
  .recording-caption p { margin: 0; font-size: var(--text-sm); color: var(--fg-dim); line-height: 1.7; text-wrap: pretty; }
  .recording-caption .hint { color: var(--fg-muted); font-size: var(--text-xs); }
  .recording-caption p > span[aria-hidden] { margin-inline: var(--space-xs); }
  .controls { display: flex; flex-wrap: wrap; gap: var(--space-xs); align-items: center; justify-content: end; }
  .control { display: inline-flex; align-items: center; justify-content: center; gap: var(--space-xs); min-height: 44px; padding: var(--space-xs) var(--space-sm); color: var(--fg-dim); font-family: var(--font-structure); font-size: var(--text-sm); border: 0; border-radius: var(--radius); background: color-mix(in srgb, var(--surface-hi) 72%, transparent); cursor: pointer; }
  .control:hover { color: var(--fg); background: var(--bg-hover); }
  .highlight-control[aria-pressed="true"] { color: var(--accent); background: var(--accent-subtle); }
  .highlight-swatch { width: 12px; height: 12px; border-radius: var(--data-mark-radius); background: var(--pillar-lens); opacity: 0.35; }
  [aria-pressed="true"] .highlight-swatch { opacity: 1; }
  .recording-link { justify-self: start; display: inline-flex; align-items: center; min-height: 40px; color: var(--fg-muted); font-size: var(--text-xs); text-underline-offset: 4px; }
  .recording-link:hover { color: var(--accent); }
  .skip-link { position: fixed; z-index: 10; top: 8px; left: 8px; padding: var(--space-sm); border-radius: var(--radius); background: var(--accent); color: var(--text-on-accent); transform: translateY(-160%); }
  .skip-link:focus { transform: translateY(0); }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
  @media (prefers-reduced-motion: no-preference) {
    .control, .primary-action, .secondary-action { transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
    .control:active, .primary-action:active, .secondary-action:active { scale: 0.96; }
  }
  @media (max-width: 1000px) {
    .recording-controls { grid-template-columns: 1fr; gap: var(--space-sm); }
    .controls { justify-content: start; }
  }
  @media (max-width: 600px) {
    .message-stage { padding-block: 96px 48px; }
    .message { font-size: var(--text-not-found-compact); min-height: 5.2em; }
    .scene-code { top: 12%; font-size: var(--text-not-found-code-compact); }
    .scene-shade { background: linear-gradient(0deg, var(--hero-bg) 4%, color-mix(in srgb, var(--hero-bg) 44%, transparent) 60%, color-mix(in srgb, var(--hero-bg) 60%, transparent)); }
    .recording-caption p > span:last-child { display: block; }
    .recording-caption p > span[aria-hidden] { display: none; }
    .motion-control { font-size: var(--text-xs); }
  }
  @media (prefers-reduced-motion: reduce) { .motion-control { display: none; } }
  @media (forced-colors: active) {
    .scene { display: none; }
    .control, .primary-action { border: 1px solid ButtonText; }
  }
</style>
