<script lang="ts">
  import ThemeToggle from "../../lib/ui/ThemeToggle.svelte";
  import type { Snippet } from "svelte";

  let { current, onChats, onModels, siteNavigation = true, homeHref = "/", compact = false, actions, leading }: {
    current?: "home" | "chats" | "models" | "credits" | "workbench";
    onChats?: () => void;
    onModels?: () => void;
    siteNavigation?: boolean;
    homeHref?: string;
    compact?: boolean;
    actions?: Snippet;
    leading?: Snippet;
  } = $props();

  const sourceUrl = typeof __DROWSE_SOURCE_URL__ === "string"
    ? __DROWSE_SOURCE_URL__ : "https://github.com/a9lim/polythetic";

  function navigate(event: MouseEvent, action?: () => void) {
    if (!action || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    action();
  }
</script>

<header class="page-header" class:compact class:workbench={current === "workbench"}>
  <div class="page-leading">
    {@render leading?.()}
    <a class="page-brand" href={homeHref} aria-label="Drowse home" aria-current={current === "home" ? "page" : undefined} translate="no">Drowse</a>
  </div>
  {#if current === "workbench"}
    <div class="workbench-actions">{@render actions?.()}</div>
  {:else}
  <nav aria-label="Primary navigation">
    {#if siteNavigation}
    <a href="/app" aria-current={current === "chats" ? "page" : undefined} onclick={(event) => navigate(event, onChats)}>Chats</a>
    <a href="/app?choose=1" aria-current={current === "models" ? "page" : undefined} onclick={(event) => navigate(event, onModels)}>Models</a>
    <a href="/credits" aria-current={current === "credits" ? "page" : undefined}>Credits</a>
    <a href={sourceUrl} rel="noreferrer">Contribute</a>
    {/if}
  </nav>
  <div class="appearance"><ThemeToggle /></div>
  {/if}
</header>

<style>
  .page-header {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    column-gap: var(--space-8);
    align-items: center;
  }
  nav { display: flex; flex-wrap: nowrap; align-items: center; justify-content: safe end; gap: var(--space-5); min-width: 0; overflow-x: auto; }
  nav a {
    display: inline-flex;
    flex: none;
    white-space: nowrap;
    align-items: center;
    justify-content: center;
    min-height: var(--control-target);
    min-width: 40px;
    padding-inline: var(--space-2);
    color: var(--fg-dim);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    text-decoration: none;
    border-radius: var(--radius-sm);
    transition: color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out);
  }
  nav a:hover { color: var(--fg); background: var(--bg-hover); }
  nav { scrollbar-width: none; }
  nav::-webkit-scrollbar { display: none; }
  nav a[aria-current="page"] { color: var(--accent); }
  .appearance { display: flex; align-items: center; justify-content: end; }
  .page-brand { justify-self: start; }
  .page-leading { display: flex; align-items: center; gap: var(--space-3); min-width: 0; }
  @media (max-width: 760px) {
    .page-header {
      column-gap: var(--space-2);
      padding-top: max(var(--surface-padding), env(safe-area-inset-top));
      padding-bottom: var(--surface-padding);
    }
    nav { gap: 0; }
    nav a { padding-inline: calc(var(--space-xs) / 2); }
  }
  @media (prefers-reduced-motion: reduce) { nav a { transition: none; } }
  .page-header.compact {
    width: 100%;
    min-height: var(--workbench-header-height, 56px);
    padding: var(--space-2) var(--surface-padding);
    column-gap: var(--space-6);
    row-gap: 0;
  }
  .compact .page-brand { font-size: var(--text-wordmark); }
  .compact nav { gap: var(--space-2); }
  .compact :global(.theme-toggle) { padding: 0; background: transparent; }
  @media (max-width: 760px) { .compact nav { gap: 0; } }
  @media (max-width: 480px) {
    .page-header:not(.workbench) {
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: var(--space-xs);
      padding-inline: max(var(--space-xs), env(safe-area-inset-left)) max(var(--space-xs), env(safe-area-inset-right));
    }
    .page-header:not(.workbench) nav { justify-content: start; gap: var(--space-xs); overscroll-behavior-inline: contain; }
    nav a { font-size: var(--text-sm); padding-inline: var(--space-xs); }
  }
  .page-header.workbench { grid-template-columns: minmax(0, 1fr) auto; align-items: center; min-height: var(--workbench-header-height, 56px); }
  .workbench-actions { display: flex; justify-content: end; align-items: center; }
</style>
