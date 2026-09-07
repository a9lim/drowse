<script lang="ts">
  import { onMount, tick } from "svelte";
  import { dropdownMotion } from "../dropdownMotion.svelte";
  import ThemeToggle from "./ThemeToggle.svelte";
  import FluentIcon from "./FluentIcon.svelte";
  import { openDrawer } from "../stores/drawers.svelte";

  let { hosted, busy, hasChat, generating, toolsLabel, toolsVisible, onToggleTools, onChats, onModels, onDownload, onAllTools, onHelp }: {
    hosted: boolean;
    busy: boolean;
    hasChat: boolean;
    generating: boolean;
    toolsLabel: string | null;
    toolsVisible: boolean;
    onToggleTools: () => void;
    onChats: () => void;
    onModels: () => void;
    onDownload: () => void;
    onAllTools: () => void;
    onHelp: () => void;
  } = $props();

  const uid = $props.id();
  const presence = dropdownMotion();
  const sourceUrl = typeof __DROWSE_SOURCE_URL__ === "string"
    ? __DROWSE_SOURCE_URL__ : "https://github.com/a9lim/polythetic";
  let open = $state(false);
  let trigger: HTMLButtonElement | null = $state(null);
  let panel: HTMLDivElement | null = $state(null);
  let position = $state("");

  function place() {
    if (!trigger || !panel) return;
    const rect = trigger.getBoundingClientRect();
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? innerWidth;
    const height = viewport?.height ?? innerHeight;
    const panelWidth = Math.min(288, width - 32);
    const x = Math.max(left + 16, Math.min(rect.right - panelWidth, left + width - panelWidth - 16));
    const y = Math.max(top + 16, Math.min(rect.bottom + 8, top + height - 64));
    position = `left:${x}px;top:${y}px;width:${panelWidth}px;max-height:${top + height - y - 16}px;`;
  }

  async function show() {
    open = true;
    presence.mount();
    await tick();
    if (!open || !panel) return;
    panel.showPopover();
    place();
    await tick();
    if (!open || !panel) return;
    presence.show(panel);
    panel.querySelector<HTMLElement>("button:not(:disabled), a")?.focus({ preventScroll: true });
  }

  function close(restoreFocus = false) {
    open = false;
    presence.close(panel);
    if (restoreFocus) trigger?.focus({ preventScroll: true });
  }

  function run(action: () => void) {
    close(true);
    action();
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    close(true);
  }

  function outside(event: PointerEvent) {
    if (open && !panel?.contains(event.target as Node) && !trigger?.contains(event.target as Node)) close();
  }

  onMount(() => {
    const viewport = window.visualViewport;
    viewport?.addEventListener("resize", place);
    viewport?.addEventListener("scroll", place);
    return () => {
      viewport?.removeEventListener("resize", place);
      viewport?.removeEventListener("scroll", place);
      presence.destroy();
    };
  });
</script>

<svelte:window onpointerdown={outside} onresize={place} />

<button bind:this={trigger} class="menu-trigger" type="button" aria-label="Workspace menu"
  aria-haspopup="dialog" aria-expanded={open} aria-controls={uid}
  onclick={() => open ? close() : void show()} onkeydown={onKeydown}>
  <span>Menu</span><FluentIcon name="down" size={14} />
</button>

{#if presence.mounted}
  <div bind:this={panel} id={uid} class="workbench-menu t-dropdown" data-origin="top-right"
    popover="manual" role="dialog" aria-label="Workspace menu" tabindex="-1" style={position}
    onkeydown={onKeydown} onfocusout={(event) => {
      if (open && event.relatedTarget instanceof Node && !panel?.contains(event.relatedTarget) && event.relatedTarget !== trigger) close();
    }}>
    <div class="menu-group">
      <button type="button" disabled={busy} onclick={() => run(onChats)}><FluentIcon name="chats" size={18} />Your chats</button>
      {#if hosted}<button type="button" disabled={busy} onclick={() => run(onModels)}><FluentIcon name="models" size={18} />Models</button>{/if}
    </div>
    <div class="menu-group">
      {#if toolsLabel}
        <button type="button" class="view-tools" aria-pressed={toolsVisible}
          aria-controls="chat-tools-header loom-tools-header" onclick={() => run(onToggleTools)}>
          <FluentIcon name="controls" size={18} />{toolsVisible ? "Hide" : "Show"} {toolsLabel}
        </button>
      {/if}
      {#if hasChat}<button type="button" disabled={generating} onclick={() => run(onDownload)}><FluentIcon name="download" size={18} />Download chat</button>{/if}
      <button type="button" onclick={() => run(onAllTools)}><FluentIcon name="search" size={18} />All tools</button>
      <button type="button" onclick={() => run(() => openDrawer("appearance"))}><FluentIcon name="appearance" size={16} />Appearance</button>
      <button type="button" onclick={() => run(onHelp)}><FluentIcon name="help" size={18} />Help</button>
    </div>
    <div class="appearance-row"><span>Appearance</span><ThemeToggle /></div>
    {#if hosted}
      <nav class="menu-links" aria-label="About Drowse">
        <a href="/credits">Credits</a><a href={sourceUrl} rel="noreferrer">Contribute</a>
      </nav>
    {/if}
  </div>
{/if}

<style>
  .menu-trigger {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-3);
    border: 0;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-dim);
    font: inherit;
    transition: color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out);
  }
  .menu-trigger:hover, .menu-trigger[aria-expanded="true"] { background: var(--bg-hover); color: var(--fg); }
  .menu-trigger:active { transform: scale(var(--press-scale)); }
  .workbench-menu {
    position: fixed;
    inset: auto;
    margin: 0;
    padding: var(--space-2);
    box-sizing: border-box;
    overflow-y: auto;
    overscroll-behavior: contain;
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    background: var(--popup-bg);
    box-shadow: var(--popup-shadow);
    color: var(--fg);
    font-size: var(--text-sm);
  }
  .menu-group { display: grid; gap: var(--space-1); }
  .menu-group + .menu-group { margin-top: var(--space-3); }
  .menu-group button, .menu-links a {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: var(--control-target);
    min-width: 0;
    padding: var(--space-2) var(--space-3);
    border: 0;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-dim);
    font: inherit;
    text-align: start;
    text-decoration: none;
    transition: color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out);
  }
  .menu-group button:hover:not(:disabled), .menu-links a:hover { background: var(--bg-hover); color: var(--fg); }
  .menu-group button:disabled { color: var(--fg-muted); cursor: not-allowed; }
  .appearance-row { display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); padding: var(--space-3); color: var(--fg-muted); }
  .appearance-row :global(.theme-toggle) { padding: 0; background: transparent; box-shadow: none; }
  .menu-links { display: flex; flex-wrap: wrap; gap: var(--space-1); }
  .menu-links a { color: var(--fg-muted); }
  @media (prefers-reduced-motion: reduce) {
    .menu-trigger, .menu-group button, .menu-links a { transition: none; transform: none; }
  }
</style>
