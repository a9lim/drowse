<script lang="ts">
  import { RAIL_CATEGORIES } from "../drawers";
  import { drawerAvailability } from "../lib/runtime/ui-capabilities";
  import { openToolInSidebar } from "../lib/stores/drawers.svelte";
  import type { DrawerName } from "../lib/types";
  let query = $state("");
  const categories = $derived(RAIL_CATEGORIES.map(category => ({ ...category, tools: category.tools.filter(tool =>
    `${tool.label} ${tool.keywords ?? ""}`.toLowerCase().includes(query.trim().toLowerCase()),
  ) })).filter(category => category.tools.length > 0));
  const shortcuts: { label: string; drawer: DrawerName; detail: string }[] = [
    { label: "Create a concept or scale", drawer: "manifold_builder", detail: "Generate examples, use a template, or supply your own." },
    { label: "Concept library", drawer: "subspace", detail: "Fit, inspect, steer, or add a probe." },
    { label: "Scale library", drawer: "manifolds", detail: "Explore fitted moods and curved scales." },
    { label: "Sampling settings", drawer: "advanced_sampling", detail: "Control how the next response is generated." },
    { label: "System prompt", drawer: "system_prompt", detail: "Set the instructions shared by the conversation." },
  ];
  const matchingShortcuts = $derived(shortcuts.filter(tool => `${tool.label} ${tool.detail}`.toLowerCase().includes(query.trim().toLowerCase())));
</script>

<section class="tool-directory" aria-labelledby="tool-directory-title">
  <header><h1 id="tool-directory-title">Tools</h1><p>Open a tool beside your work. Switch between a side panel and a dialog from its header.</p></header>
  <label class="search"><span>Find a tool</span><input type="search" bind:value={query} placeholder="Search tools…" /></label>
  {#if matchingShortcuts.length > 0}
    <section aria-label="Frequently used tools" class="quick-tools">
      {#each matchingShortcuts as tool}
        {@const availability = drawerAvailability(tool.drawer)}
        <button type="button" disabled={!availability.available} onclick={() => openToolInSidebar(tool.drawer)}>
          <strong>{tool.label}</strong><span>{availability.available ? tool.detail : availability.reason}</span>
        </button>
      {/each}
    </section>
  {/if}
  {#each categories as category}
    <section class="tool-group" aria-labelledby={`tools-${category.key}`}>
      <h2 id={`tools-${category.key}`}>{category.label}</h2>
      <div class="tool-list">{#each category.tools as tool}<button type="button" onclick={() => openToolInSidebar(tool.drawer)}>{tool.label.replace(/…$/, "")}</button>{/each}</div>
    </section>
  {/each}
  {#if categories.length === 0 && matchingShortcuts.length === 0}<p role="status">No tools match “{query}”.</p><button type="button" onclick={() => (query = "")}>Clear search</button>{/if}
</section>

<style>
  .tool-directory { overflow-y: auto; padding: var(--surface-padding); display: flex; flex-direction: column; gap: var(--space-lg); min-height: 0; color: var(--fg); }
  header { display: grid; gap: var(--space-sm); }
  h1, h2, p { margin: 0; }
  h1, h2, strong { text-wrap: balance; }
  h1 { font-size: var(--text-lg); font-weight: var(--weight-structure); }
  h2 { font-size: var(--text); font-weight: var(--weight-medium); }
  p { max-width: 65ch; line-height: 1.5; color: var(--fg-dim); text-wrap: pretty; }
  .search { display: grid; gap: var(--space-sm); max-width: 36rem; font-size: var(--text-sm); }
  input { width: 100%; min-width: 0; min-height: var(--control-field); padding: var(--space-sm); border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); box-shadow: var(--shadow-well); color: var(--fg); font: inherit; }
  input:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .quick-tools { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr)); gap: var(--space-sm); }
  button { min-width: var(--control-target); min-height: var(--control-target); padding: var(--space-sm); background: var(--workspace-field-bg); border: 1px solid transparent; border-radius: var(--radius); text-align: start; font: inherit; color: var(--fg); overflow-wrap: anywhere; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  button:active:not(:disabled) { background: var(--workspace-neutral-hover); scale: var(--press-scale); }
  button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  button:disabled { color: var(--fg-muted); cursor: not-allowed; }
  .quick-tools button { display: grid; gap: var(--space-sm); align-content: start; padding: var(--space-md); border-radius: var(--radius-group); }
  .quick-tools span { color: var(--fg-dim); line-height: 1.5; font-size: var(--text-sm); text-wrap: pretty; }
  .tool-group { display: grid; gap: var(--space-sm); }
  .tool-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr)); gap: var(--space-sm); }
  @media (hover: hover) { button:hover:not(:disabled) { background: var(--workspace-neutral-hover); } }
  @media (max-width: 760px) { input { font-size: var(--text-input-touch); } }
</style>
