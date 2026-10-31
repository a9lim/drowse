<script lang="ts">
  import ThemeToggle from "./ThemeToggle.svelte";
  import type { Snippet } from "svelte";

  type Current = "home" | "chats" | "models" | "credits" | "contact" | "learn" | "privacy" | "about" | "contribute" | "workbench";

  let { current, variant, onChats, onModels, showChats = true, siteNavigation = true, homeHref = "/", compact = false, actions, leading }: {
    current?: Current;
    /** "site" for public pages, "app" inside /app. Chats and Models pages default to "app". */
    variant?: "site" | "app";
    onChats?: () => void;
    onModels?: () => void;
    /** Hide the Chats link, for example before the first chat exists. */
    showChats?: boolean;
    siteNavigation?: boolean;
    homeHref?: string;
    compact?: boolean;
    actions?: Snippet;
    leading?: Snippet;
  } = $props();

  const resolvedVariant = $derived(variant ?? (current === "chats" || current === "models" ? "app" : "site"));
  const learnCurrent = $derived(current === "learn" ? "page" : undefined);

  function navigate(event: MouseEvent, action?: () => void) {
    if (!action || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    action();
  }
</script>

<header class="page-header" class:compact class:workbench={current === "workbench"} class:site={current !== "workbench" && resolvedVariant === "site" && siteNavigation}>
  <div class="page-leading">
    {@render leading?.()}
    <a class="page-brand" href={homeHref} aria-label="Drowse home" aria-current={current === "home" ? "page" : undefined} translate="no">Drowse</a>
  </div>
  {#if current === "workbench"}
    <div class="workbench-actions">{@render actions?.()}</div>
  {:else}
  <nav aria-label="Primary navigation">
    {#if siteNavigation}
      {#if resolvedVariant === "app"}
        {#if showChats}<a href="/app" aria-current={current === "chats" ? "page" : undefined} onclick={(event) => navigate(event, onChats)}>Chats</a>{/if}
        <a href="/app?choose=1" aria-current={current === "models" ? "page" : undefined} onclick={(event) => navigate(event, onModels)}>Models</a>
      {:else}
        <a href="/learn" aria-current={learnCurrent}>Learn</a>
        <a href="/app?choose=1" aria-current={current === "models" ? "page" : undefined} onclick={(event) => navigate(event, onModels)}>Models</a>
        <a href="/credits" aria-current={current === "credits" ? "page" : undefined}>Credits</a>
        <a href="/contribute" aria-current={current === "contribute" ? "page" : undefined}>Contribute</a>
      {/if}
    {/if}
  </nav>
  <div class="header-end">
    {#if siteNavigation && resolvedVariant === "site"}<a class="open-action" href="/app">Open Drowse</a>{/if}
    <div class="appearance"><ThemeToggle /></div>
  </div>
  {/if}
</header>

<style>
  .page-header {
    --header-theme-icon-size: 1.5rem;
    --header-theme-target: max(44px, var(--control-target));
    --header-scrim: color-mix(in srgb, var(--bg) 92%, transparent);
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    grid-template-areas: "leading nav end";
    column-gap: var(--space-8);
    align-items: center;
  }
  /* Public and app headers stay in reach while the page scrolls. The scrim
     fades in once content passes underneath, so the hero stays clear at rest. */
  .page-header:not(.workbench) { position: sticky; top: 0; z-index: 5; }
  .page-header:not(.workbench)::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    border-image: linear-gradient(to top, var(--grid-line) 1px, var(--header-scrim) 1px) fill 0 / 0 / 0 100vmax;
    opacity: 0;
    animation: header-scrim linear both;
    animation-timeline: scroll(nearest block);
    animation-range: 0 48px;
  }
  @keyframes header-scrim { to { opacity: 1; } }
  @supports not (animation-timeline: scroll()) {
    .page-header:not(.workbench)::before { animation: none; opacity: 1; }
  }
  @media (prefers-contrast: more), (forced-colors: active) {
    .page-header:not(.workbench) { --header-scrim: var(--bg); }
  }
  .page-leading { grid-area: leading; }
  nav { grid-area: nav; display: flex; flex-wrap: nowrap; align-items: center; justify-content: safe end; gap: var(--space-5); min-width: 0; overflow-x: auto; }
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
  .header-end { grid-area: end; display: flex; align-items: center; justify-content: end; gap: var(--space-5); }
  .open-action {
    display: inline-flex;
    flex: none;
    align-items: center;
    justify-content: center;
    min-height: var(--control-target);
    padding-inline: var(--space-5);
    border-radius: var(--radius);
    background: var(--control-sheen), var(--action-bg);
    box-shadow: var(--shadow-control);
    color: var(--action-ink);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure-bold);
    white-space: nowrap;
    text-decoration: none;
    transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out);
  }
  .open-action:hover { background: var(--control-sheen), var(--action-hover); box-shadow: var(--shadow-control-hover); }
  .open-action:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }
  .page-header a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .appearance {
    display: flex;
    align-items: center;
    justify-content: end;
    margin-inline-end: calc((var(--header-theme-icon-size) - var(--header-theme-target)) / 2);
  }
  .page-brand { justify-self: start; }
  .page-leading { display: flex; align-items: center; gap: var(--space-3); min-width: 0; }
  @media (max-width: 800px) {
    .page-header {
      column-gap: var(--space-2);
      padding-top: max(var(--surface-padding), env(safe-area-inset-top));
      padding-bottom: var(--surface-padding);
    }
    nav { gap: 0; }
    nav a { padding-inline: calc(var(--space-xs) / 2); }
    .header-end { gap: var(--space-2); }
  }
  @media (prefers-reduced-motion: reduce) { nav a, .open-action { transition: none; } .open-action:active { transform: none; } }
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
      padding-inline: max(var(--space-sm), env(safe-area-inset-left)) max(var(--space-sm), env(safe-area-inset-right));
    }
    .page-header:not(.workbench) nav { justify-content: space-between; gap: 0; overscroll-behavior-inline: contain; }
    nav a { font-size: var(--text-sm); padding-inline: calc(var(--space-xs) / 2); }
  }
  /* Narrow pages give navigation its own wrapping row. The wordmark and
     actions can also wrap when text is enlarged, keeping every control in view. */
  @media (max-width: 640px) {
    .page-header:not(.workbench) {
      display: flex;
      flex-wrap: wrap;
      row-gap: 0;
      min-height: 0;
    }
    .page-header:not(.workbench) .page-leading { flex: 1 0 auto; }
    .page-header:not(.workbench) .header-end { flex-wrap: wrap; min-width: 0; max-width: 100%; margin-inline-start: auto; }
    .page-header:not(.workbench) nav { flex: 1 1 100%; order: 1; flex-wrap: wrap; justify-content: space-between; overflow: visible; }
    .page-header.site {
      padding-top: max(var(--space-xs), env(safe-area-inset-top));
      padding-bottom: 0;
    }
    .page-header.site nav { justify-content: space-between; margin-inline: calc(var(--space-xs) / -2); }
  }
  .page-header.workbench { grid-template-columns: minmax(0, 1fr) auto; grid-template-areas: "leading end"; align-items: center; min-height: var(--workbench-header-height, 56px); }
  .workbench-actions { grid-area: end; display: flex; justify-content: end; align-items: center; }
</style>
