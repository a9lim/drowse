<script lang="ts">
  import { onMount } from "svelte";
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import HeroShader from "./HeroShader.svelte";
  import RecordedTokenText from "./RecordedTokenText.svelte";
  import { nextMessage, recording } from "./notFoundMessages";

  const message = recording.messages[nextMessage()];
  let plainBackground = $state(false);

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
    {#if !plainBackground}<HeroShader contained closeup />{/if}
    <div class="scene-shade"></div>
  </div>
  <div class="page-layer">
    <PageHeader />
    <main id="not-found-main">
      <section class="message-stage" aria-label="Page not found">
        <div class="message" data-message-id={message.id}>
          <RecordedTokenText tokens={message.tokens} />
        </div>
        <div class="recovery-actions">
          <a class="primary-action" href="/">Back to home</a>
          <a class="secondary-action" href="/app">Open Drowse <span aria-hidden="true">↗</span></a>
        </div>
      </section>

    </main>
    <PageFooter />
  </div>
</div>

<style>
  .not-found-shell { position: relative; min-height: 100dvh; color: var(--fg); background: var(--hero-bg); isolation: isolate; overflow-x: clip; }
  .scene { position: absolute; inset: 0; overflow: hidden; pointer-events: none; }
  .scene-shade { position: absolute; inset: 0; background: linear-gradient(90deg, color-mix(in srgb, var(--hero-bg) 76%, transparent), color-mix(in srgb, var(--hero-bg) 22%, transparent) 70%), linear-gradient(0deg, var(--hero-bg), transparent 55%, color-mix(in srgb, var(--hero-bg) 68%, transparent)); }
  .page-layer { position: relative; display: flex; flex-direction: column; min-height: 100dvh; }
  main { flex: 1; display: flex; flex-direction: column; width: min(100%, var(--page-max)); padding-inline: max(var(--surface-padding), env(safe-area-inset-left)) max(var(--surface-padding), env(safe-area-inset-right)); margin-inline: auto; }
  .message-stage { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: start; padding-block: clamp(80px, 12vh, 160px) clamp(56px, 9vh, 112px); }
  .message { max-width: min(100%, 24ch); min-height: 3.75em; color: var(--fg-strong); font-family: var(--font-structure); font-size: var(--text-not-found-title); font-weight: var(--weight-structure); letter-spacing: -0.035em; line-height: 1.3; }
  .recovery-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-sm) var(--space-lg); margin-top: var(--space-lg); }
  .primary-action, .secondary-action { display: inline-flex; align-items: center; justify-content: center; min-height: 48px; padding: var(--space-sm) var(--space-lg); border-radius: var(--radius); font-family: var(--font-structure); font-size: var(--text); font-weight: var(--weight-structure); text-decoration: none; }
  .primary-action { background: var(--accent); color: var(--text-on-accent); box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 60%, transparent), 0 8px 24px color-mix(in srgb, var(--hero-bg) 22%, transparent); }
  .primary-action:hover { background: var(--accent-light); }
  .secondary-action { gap: var(--space-sm); color: var(--fg); padding-inline: var(--space-xs); }
  .secondary-action:hover { color: var(--accent); }
  .skip-link { position: fixed; z-index: 10; top: 8px; left: 8px; padding: var(--space-sm); border-radius: var(--radius); background: var(--accent); color: var(--text-on-accent); transform: translateY(-160%); }
  .skip-link:focus { transform: translateY(0); }
  @media (prefers-reduced-motion: no-preference) {
    .primary-action, .secondary-action { transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
    .primary-action:active, .secondary-action:active { scale: 0.96; }
  }
  @media (max-width: 600px) {
    .message-stage { padding-block: 96px 48px; }
    .message { font-size: var(--text-not-found-compact); min-height: 5.2em; }
    .scene-shade { background: linear-gradient(0deg, var(--hero-bg) 4%, color-mix(in srgb, var(--hero-bg) 44%, transparent) 60%, color-mix(in srgb, var(--hero-bg) 60%, transparent)); }
  }
  @media (forced-colors: active) {
    .scene { display: none; }
    .primary-action { border: 1px solid ButtonText; }
  }
</style>
