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
    ? __DROWSE_SOURCE_URL__ : "https://github.com/a9lim/drowse";
  let brandCollapsed = $state(false);

  function fitNavigation(header: HTMLElement) {
    const nav = header.querySelector("nav");
    if (!nav || !siteNavigation) return;
    const leading = header.querySelector<HTMLElement>(".page-leading")!;
    const appearance = header.querySelector<HTMLElement>(".appearance")!;
    const links = Array.from(nav.querySelectorAll("a"));
    let frame = 0;
    const measure = () => {
      const style = getComputedStyle(header);
      const available = header.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      const navigationWidth = links.reduce((width, link) => width + link.getBoundingClientRect().width, 0)
        + parseFloat(getComputedStyle(nav).columnGap) * Math.max(0, links.length - 1);
      const required = leading.getBoundingClientRect().width + navigationWidth
        + appearance.getBoundingClientRect().width + 2 * parseFloat(style.columnGap);
      brandCollapsed = required > available + 1;
    };
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    });
    for (const element of [header, leading, appearance, ...links]) observer.observe(element);
    measure();
    return { destroy() { observer.disconnect(); cancelAnimationFrame(frame); } };
  }

  function navigate(event: MouseEvent, action?: () => void) {
    if (!action || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    action();
  }
</script>

<header class="page-header" class:compact class:workbench={current === "workbench"} class:brand-collapsed={brandCollapsed} use:fitNavigation>
  <div class="page-leading" aria-hidden={brandCollapsed ? "true" : undefined}>
    {@render leading?.()}
    <a class="page-brand" href={homeHref} aria-label="Drowse home" aria-current={current === "home" ? "page" : undefined} tabindex={brandCollapsed ? -1 : undefined} translate="no">Drowse</a>
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
    .page-header:not(.workbench) nav { justify-content: start; gap: 0; overscroll-behavior-inline: contain; }
    nav a { font-size: var(--text-sm); padding-inline: calc(var(--space-xs) / 2); }
  }
  .page-header.brand-collapsed { grid-template-columns: minmax(0, 1fr) auto; }
  .brand-collapsed .page-leading { position: absolute; visibility: hidden; pointer-events: none; width: max-content; }
  .brand-collapsed nav { flex-wrap: wrap; overflow: visible; justify-content: start; }
  .page-header.workbench { grid-template-columns: minmax(0, 1fr) auto; align-items: center; min-height: var(--workbench-header-height, 56px); }
  .workbench-actions { display: flex; justify-content: end; align-items: center; }
</style>
