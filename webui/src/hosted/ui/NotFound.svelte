<script lang="ts">
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
</script>

<TabIdentity state="error" title="Page not found · Drowse" />
<svelte:head><meta name="robots" content="noindex, follow" /></svelte:head>

<div class="not-found-shell">
<PageHeader />
<main>
  <p>404</p>
  <h1>Page not found</h1>
  <p class="explanation">This link may be outdated. Open Drowse to return to your models and chats.</p>
  <a class="action" href="/app">Open Drowse</a>
</main>
<PageFooter />
</div>

<style>
  .not-found-shell { display: flex; flex-direction: column; min-height: 100dvh; }
  main { flex: 1; display: grid; align-content: center; width: min(100%, var(--page-max)); padding: var(--surface-padding); margin-inline: auto; }
  p { margin: 0 0 var(--space-sm); color: var(--fg-muted); font-family: var(--font-mono); }
  h1 { max-width: 650px; margin: 0; color: var(--fg); font-size: var(--text-not-found-title); font-weight: var(--weight-display); letter-spacing: -0.045em; line-height: 1; }
  .explanation { margin: var(--space-5) 0 0; max-width: 65ch; font-family: var(--font-reading); line-height: 1.5; }
  .action { width: fit-content; margin-top: var(--space-lg); padding: var(--space-sm) var(--space-sm); border-radius: var(--radius); background: var(--accent); color: var(--text-on-accent); font-family: var(--font-structure); font-weight: var(--weight-structure); text-decoration: none; }

  @media (prefers-reduced-motion: no-preference) {
    .action { transition: transform var(--dur-fast) var(--ease-out); }
    .action:active { transform: scale(var(--press-scale)); }
  }
</style>
