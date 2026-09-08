<script lang="ts">
  import MorphText from "../lib/ui/MorphText.svelte";
  import FluentIcon from "../lib/ui/FluentIcon.svelte";
  // The steering bar — the composed steering expression made permanently
  // visible.  Every racked term renders as a pillar-colored chip regardless
  // of which instrument tab is open: subspace white, manifold violet,
  // j-lens blue (sae gold, once the runtime lands).  Click a chip to jump
  // to its pillar; × removes the term; disabled terms show muted.  The ⧉
  // copies the canonical expression string — the exact text the WS
  // ``steering`` field carries.

  import Chip from "../lib/ui/Chip.svelte";
  import {
    steerRack,
    setInspectorTab,
    removeSubspaceFromRack,
    removeManifoldFromRack,
    atomActions,
    currentSteeringExpression,
  } from "../lib/stores.svelte";
  import type { InspectorTab } from "../lib/stores.svelte";
  import {
    formatSubspaceTerm,
    formatManifoldTerm,
    formatJLensTerm,
    formatSaeTerm,
  } from "../lib/expression";
  import type { SteerEntry } from "../lib/types";
  import { pushToast } from "../lib/stores/toasts.svelte";

  interface ChipModel {
    name: string;
    text: string;
    color: string;
    tab: InspectorTab;
    /** Serialization-order group: subspace 0, jlens 1, manifold 2. */
    order: number;
    enabled: boolean;
    remove: () => void;
  }

  function chipFor(name: string, entry: SteerEntry): ChipModel {
    switch (entry.mode) {
      case "subspace":
        return {
          name,
          text: formatSubspaceTerm(name, entry, steerRack.subspaceAlong),
          color: "var(--pillar-subspace)",
          tab: "subspace",
          order: 0,
          enabled: entry.enabled,
          remove: () => removeSubspaceFromRack(name),
        };
      case "manifold":
        return {
          name,
          text: formatManifoldTerm(name, entry),
          color: "var(--pillar-manifold)",
          tab: "manifold",
          order: 2,
          enabled: entry.enabled,
          remove: () => removeManifoldFromRack(name),
        };
      case "jlens":
        return {
          name,
          text: formatJLensTerm(name, entry),
          color: "var(--pillar-lens)",
          tab: "lens",
          order: 1,
          enabled: entry.enabled,
          remove: () => atomActions("jlens").remove(name),
        };
      case "sae":
        return {
          name,
          text: formatSaeTerm(name, entry),
          color: "var(--pillar-sae)",
          tab: "sae",
          order: 1,
          enabled: entry.enabled,
          remove: () => atomActions("sae").remove(name),
        };
    }
  }

  const chips = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].map(([name, entry]) =>
      chipFor(name, entry),
    );
    arr.sort(
      (a, b) => a.order - b.order || a.name.localeCompare(b.name),
    );
    return arr;
  });

  const expression = $derived(currentSteeringExpression());
  const custom = $derived(steerRack.customExpression !== null);

  let copied = $state(false);
  $effect(() => { if (!copied) return; const timer = setTimeout(() => copied = false, 1800); return () => clearTimeout(timer); });

  async function copyExpression(): Promise<void> {
    if (!expression) return;
    try {
      await navigator.clipboard.writeText(expression);
      copied = true;
    } catch {
      pushToast("Could not copy the recipe. Select its text and copy it manually.", { kind: "error" });
    }
  }
</script>

{#if custom ? expression.trim() : chips.length > 0}
<div class="recipe">
  <span class="lbl">Steering</span>
  {#if custom}
    <span class="custom-expression">
      <span class="custom-badge">custom</span>
      <code>{expression}</code>
    </span>
  {:else}
    <div class="chips">
      {#each chips as chip (chip.name)}
        <Chip
          color={chip.color}
          muted={!chip.enabled}
          title={chip.enabled ? undefined : "Disabled"}
          onclick={() => setInspectorTab(chip.tab)}
          onremove={chip.remove}
          removeLabel={`Remove ${chip.name} from steering recipe`}
        >
          <MorphText text={chip.text} numbers={false} />
        </Chip>
      {/each}
    </div>
  {/if}
  {#if expression}
    <button
      type="button"
      class="copy"
      data-cursor="copy"
      {...{ "aria-description": "Copy the technical response recipe" }}
      aria-label="Copy response recipe"
      onclick={copyExpression}
    ><FluentIcon name="copy" /><span role="status"><MorphText text={copied ? "Copied" : "Copy"} numbers={false} /></span></button>
  {/if}
</div>
{/if}

<style>
  /* A quiet glass well; the chips carry the meaning. */
  .recipe {
    position: relative;
    z-index: 1;
    display: flex;
    flex: none;
    align-items: center;
    gap: var(--space-3);
    padding: var(--surface-padding);
    margin: var(--surface-gutter) var(--surface-gutter) 0;
    border-radius: var(--radius-lg);
    border: 1px solid transparent;
    background: var(--surface-sheen), var(--bg-alt);
  }
  .lbl {
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--fg-muted);
    flex: none;
  }
  .custom-expression {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex: 1 1 auto;
    min-width: 0;
    text-align: start;
  }
  .custom-expression code {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .custom-badge {
    flex: none;
    color: var(--accent-amber);
    background: color-mix(in srgb, var(--accent-amber) 12%, transparent);
    border-radius: var(--radius-sm);
    padding: var(--space-xs) var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
    flex: 1;
    min-width: 0;
    /* Two chip rows before the bar scrolls — the expression stays
     * glanceable without eating the instrument column. */
    max-height: calc((var(--control-target) + var(--space-xs) * 2) * 2 + var(--space-xs));
    overflow-y: auto;
    scrollbar-gutter: stable both-edges;
  }
  .copy {
    flex: none;
    min-width: var(--control-target);
    min-height: var(--control-target);
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1;
    padding: var(--space-xs) var(--space-2);
    border-radius: var(--radius);
    transition: color var(--dur-fast) var(--ease-out);
  }
  .copy:hover {
    color: var(--fg);
    background: var(--bg-hover);
  }
  @media (max-width: 620px) {
    .recipe { flex-wrap: wrap; }
    .chips { flex-basis: 100%; }
  }
</style>
