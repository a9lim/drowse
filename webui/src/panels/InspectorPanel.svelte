<script lang="ts">
  // Inspector — the instrument stack.  Four coequal pillars over the ONE
  // steering expression + probe roster, each with the same verbs
  // (observe / steer / gate) and its own hue:
  //
  //   subspace — flat/affine fits (concept axes, personas) · white
  //   manifold — curved fits (emotions, months) · violet
  //   sae      — resident sparse-autoencoder feature space · gold
  //   lens     — the Jacobian-lens surface (JLensPanel) · blue
  //
  // The split is presentational — a lens steer chip and a subspace card
  // serialize into the same expression, which the persistent steering bar
  // above the tabs keeps visible across all four. The lens and SAE tabs expose
  // compatible sources; local fit/train controls exist only in the Python
  // runtime and are capability-hidden in the hosted app.

  import SteeringRack from "./SteeringRack.svelte";
  import ProbeRack from "./ProbeRack.svelte";
  import JLensPanel from "./JLensPanel.svelte";
  import SaePanel from "./SaePanel.svelte";
  import RecipeBar from "./RecipeBar.svelte";
  import SamplingStrip from "./SamplingStrip.svelte";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import {
    inspectorState,
    setInspectorTab,
    tokenHoverState,
    sessionState,
    type InspectorTab,
  } from "../lib/stores.svelte";

  const PILLARS: {
    value: InspectorTab;
    label: string;
    color: string;
  }[] = [
    {
      value: "subspace",
      label: "Subspace",
      color: "var(--pillar-subspace)",
    },
    {
      value: "manifold",
      label: "Manifold",
      color: "var(--pillar-manifold)",
    },
    {
      value: "sae",
      label: "SAE",
      color: "var(--pillar-sae)",
    },
    {
      value: "lens",
      label: "J-lens",
      color: "var(--pillar-lens)",
    },
  ];

  const tab = $derived(inspectorState.tab);
  const hoverToken = $derived.by(() => {
    const visible = tokenHoverState.tokenText
      .replace(/\n/g, "↵")
      .replace(/\t/g, "⇥")
      .replace(/ /g, "·");
    return visible || "∅";
  });
  const hoverProbeCount = $derived.by(() => new Set([
    ...Object.keys(tokenHoverState.probeReadings ?? {}),
    ...Object.keys(tokenHoverState.probes ?? {}),
    ...Object.keys(tokenHoverState.coordsByProbe ?? {}),
  ]).size);
  const hoverLensCount = $derived(tokenHoverState.lensAggregate?.length ?? 0);
  const hoverSaeCount = $derived(tokenHoverState.saeReadout?.length ?? 0);
</script>

<aside class="inspector" aria-label="Drowse inspector">
  <RecipeBar />

  <div class="control-grid">
    <section class="reply-panel" aria-labelledby="reply-style-title">
      <header class="section-heading">
        <div>
          <h2 id="reply-style-title">Generation settings</h2>
        </div>
      </header>
      <SamplingStrip />
    </section>

    <section class="guidance-panel" aria-labelledby="guidance-title">
      <header class="section-heading guidance-heading">
        <div>
          <h2 id="guidance-title">Instruments</h2>
        </div>
      </header>

      <div class="instrument-head">
        <nav class="tabs" aria-label={sessionState.info?.is_base_model ? "Completion guidance" : "Response guidance"}>
          <SegmentedTabs
            items={PILLARS}
            value={tab}
            onchange={(v) => setInspectorTab(v)}
            fill
            ariaLabel={sessionState.info?.is_base_model ? "Completion guidance type" : "Response guidance type"}
          />
        </nav>
        {#if tokenHoverState.active}
          <div class="token-readout" role="status" aria-live="off">
            <span class="readout-dot"></span>
            <span class="readout-label">Selected word</span>
            <code {...{ "aria-description": (tokenHoverState.tokenText) }}>{hoverToken}</code>
            <span class="readout-channels">
              {hoverProbeCount} readings
              · {tokenHoverState.lensLoading ? "predictions…" : `${hoverLensCount} predictions`}
              · {tokenHoverState.saeLoading ? "features…" : `${hoverSaeCount} features`}
            </span>
          </div>
        {/if}
      </div>

      <div class="guidance-body">
        {#if tab === "lens"}
          <JLensPanel />
        {:else if tab === "sae"}
          <SaePanel />
        {:else if tab === "manifold"}
          <div class="rack-grid">
            <SteeringRack family="manifold" />
            <ProbeRack family="manifold" />
          </div>
        {:else}
          <div class="rack-grid">
            <SteeringRack family="subspace" />
            <ProbeRack family="subspace" />
          </div>
        {/if}
      </div>
    </section>
  </div>
</aside>

<style>
  .inspector {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: clip;
    background: transparent;
  }

  .control-grid {
    display: grid;
    flex: 1;
    grid-template-columns: minmax(19rem, 0.72fr) minmax(30rem, 1.45fr);
    grid-template-rows: minmax(36rem, 1fr);
    gap: var(--surface-gutter);
    min-width: 0;
    min-height: 0;
    padding: var(--surface-gutter);
    overflow-y: auto;
    scrollbar-gutter: stable both-edges;
  }

  .reply-panel,
  .guidance-panel {
    min-width: 0;
    min-height: 0;
    overflow: clip;
    background: transparent;
  }

  .reply-panel {
    overflow-y: auto;
    scrollbar-gutter: stable both-edges;
  }

  .guidance-panel {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
  }

  .section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--panel-padding);
  }

  .section-heading h2 {
    margin: 0;
  }

  .section-heading h2 {
    color: var(--fg);
    font-size: var(--text-lg);
    font-weight: var(--weight-display);
    letter-spacing: -0.025em;
  }

  .tabs {
    padding: 0 var(--panel-padding) var(--panel-padding);
  }

  .instrument-head {
    min-width: 0;
  }

  .token-readout {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-target);
    margin: 0 var(--panel-padding) var(--space-3);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius);
    background: var(--surface-sheen), var(--glass);
    color: var(--fg-subtle);
    font-size: var(--text-xs);
  }

  .readout-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--live);
  }

  .readout-label {
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
  }

  .token-readout code {
    overflow: hidden;
    color: var(--fg-strong);
    font-family: var(--font-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .readout-channels {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    white-space: nowrap;
  }

  .guidance-body {
    min-width: 0;
    min-height: 0;
    overflow: clip;
  }

  .rack-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    height: 100%;
    max-height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: clip;
    gap: var(--space-md);
    padding: 0 var(--panel-padding) var(--panel-padding);
  }

  .rack-grid :global(.rack) { padding: 0; }

  @media (max-width: 920px) {
    .control-grid {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: max-content minmax(36rem, 1fr);
      overflow-y: auto;
      scrollbar-gutter: stable both-edges;
      padding: 0;
    }

    .reply-panel,
    .guidance-panel {
      border-radius: 0;
      background: transparent;
    }

    .reply-panel {
      overflow: visible;
      scrollbar-gutter: auto;
    }

    .guidance-panel { min-height: 36rem; }
  }

  @media (max-width: 620px) {
    .control-grid {
      gap: var(--surface-gutter);
      grid-template-rows: max-content max-content;
    }

    .guidance-panel {
      min-height: 0;
      grid-template-rows: auto auto auto;
      overflow: visible;
    }

    .guidance-body {
      overflow: visible;
    }

    .rack-grid {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: auto auto;
      height: auto;
      max-height: none;
      overflow: visible;
    }

    .rack-grid :global(.rack) {
      height: auto;
      max-height: none;
      overflow: visible;
    }

    .rack-grid :global(.strips) {
      flex: none;
      max-height: none;
      overflow: visible;
    }

    .readout-channels {
      display: none;
    }
  }
</style>
