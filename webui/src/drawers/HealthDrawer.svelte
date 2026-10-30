<script lang="ts">
  import MorphText from "../lib/ui/MorphText.svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import ResetSettings from "../lib/ui/ResetSettings.svelte";
  import {
    closeDrawer,
    genStatus,
    geometricMeanPpl,
    loomTree,
    probeRack,
    sessionState,
    steerRack,
    vectorsState,
  } from "../lib/stores.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { healthState, healthWarnings, refreshHealth } from "../lib/stores/health.svelte";

  let { params = null }: { params?: unknown } = $props();
  const embedded = $derived(
    (params as { embedded?: boolean } | null)?.embedded === true,
  );

  const busy = $derived(healthState.busy);
  const lastAudit = $derived(healthState.lastAudit ? new Date(healthState.lastAudit).toLocaleTimeString() : null);
  const errorMsg = $derived(healthState.error);

  const ppl = $derived(geometricMeanPpl(genStatus));
  const warnings = $derived(healthWarnings());

  async function audit(): Promise<void> {
    if (!healthState.busy) await refreshHealth();
  }
</script>

<section
  class="drawer-shell"
  class:embedded
  role={embedded ? "region" : undefined}
  aria-label={embedded ? "Model controls" : "Health drawer"}
>
  {#if !embedded}
    <header class="drawer-header">
      <div class="title">
        <h2 class="eyebrow">Model health</h2>
      </div>
      <DrawerCloseButton onclick={closeDrawer} />
    </header>
  {/if}

  <div class="body" aria-busy={busy}>
    <section class="hero">
      <div>
        <h2>{sessionState.info?.model_id ?? "no model"}</h2>
        <p>{sessionState.info ? `${sessionState.info.device}/${sessionState.info.dtype}` : "session offline"}</p>
      </div>
      <Button variant="solid" {busy} disabled={busy} onclick={audit}>
        <MorphText text={busy ? "checking…" : "refresh"} />
      </Button>
    </section>

    {#if errorMsg}
      <div class="error" role="alert">Health check failed: {errorMsg}</div>
    {/if}

    <section class="panel"><ResetSettings full /></section>

    <section class="grid">
      <div class="tile">
        <span>generation</span>
        <strong>{genStatus.active ? "active" : genStatus.finishReason ?? "idle"}</strong>
        <p><MorphText text={`${genStatus.tokensSoFar}/${genStatus.maxTokens || "-"} tokens`} /> · <MorphText text={genStatus.tokPerSec.toFixed(1)} /> tok/s</p>
      </div>
      <div class="tile">
        <span>Perplexity</span>
        <strong><MorphText text={ppl === null ? "-" : ppl.toFixed(2)} /></strong>
        <p>{genStatus.ppl.count} steps</p>
      </div>
      <div class="tile">
        <span>loom tree</span>
        <strong>{loomTree.nodes.size || "-"}</strong>
        <p>rev {loomTree.loaded ? loomTree.rev : "-"} · depth {loomTree.activePath.length || "-"}</p>
      </div>
      <div class="tile">
        <span>artifacts</span>
        <strong><MorphText text={steerRack.catalog.length} /></strong>
        <p>{steerRack.entries.size} racked · {vectorsState.names.length} resident</p>
      </div>
      <div class="tile">
        <span>probes</span>
        <strong><MorphText text={probeRack.active.length} /></strong>
        <p>{probeRack.entries.size} rows · {steerRack.correlation ? "matrix cached" : "no matrix"}</p>
      </div>
    </section>

    <section class="panel">
      <h3>checks</h3>
      <div class="checks">
        <div class:ok={!!sessionState.info}>Session details: {sessionState.info ? "loaded" : "unavailable"}</div>
        <div class:ok={loomTree.loaded && !loomTree.error}>Loom: {loomTree.loaded && !loomTree.error ? "loaded" : "unavailable"}</div>
        <div class:ok={steerRack.catalog.length > 0}>Response controls: {steerRack.catalog.length > 0 ? "available" : "none installed"}</div>
        <div class:ok={probeRack.active.length > 0}>Probes: {probeRack.active.length > 0 ? "active" : "none active"}</div>
        <div class:ok={steerRack.correlation !== null}>Correlation: {steerRack.correlation !== null ? "available" : "not measured"}</div>
      </div>
    </section>

    <section class="panel">
      <h3>warnings · <MorphText text={warnings.length} /></h3>
      {#if warnings.length === 0}
        <p class="good">clear</p>
      {:else}
        <ul>
          {#each warnings as warning (warning)}
            <li>{warning}</li>
          {/each}
        </ul>
      {/if}
      {#if lastAudit}
        <p class="dim">updated {lastAudit}</p>
      {/if}
    </section>
  </div>
</section>

<style>
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/identifier/number sits in mono. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .drawer-shell.embedded { height: 100%; }
  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .hero p, .tile p, .dim {
    margin: var(--space-1) 0 0;
    color: var(--fg-muted);
    line-height: 1.5;
  }
  .body {
    display: grid;
    gap: var(--drawer-section-gap);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    overflow: auto;
  }
  /* Data wells — recessed stat/summary containers. */
  .hero, .tile, .panel {
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--bg);
    padding: var(--panel-padding);
  }
  .hero {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-6);
  }
  h2, h3 { margin: 0; color: var(--fg); }
  .hero > div { min-width: 0; overflow-wrap: anywhere; }
  h2 {
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  h3 {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-5);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-5);
  }
  .tile span {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .tile strong {
    display: block;
    margin-top: var(--space-2);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-variant-numeric: tabular-nums;
  }
  .checks {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .checks div {
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--surface-padding);
    color: var(--fg-muted);
    background: var(--surface-sheen), var(--bg-elev);
    font-size: var(--text-sm);
    line-height: 1.5;
    overflow-wrap: anywhere;
    transition:
      color var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .checks div.ok {
    color: var(--accent-green);
    border-color: color-mix(in srgb, var(--accent-green) 35%, var(--glass-line));
    background: color-mix(in srgb, var(--accent-green) 8%, var(--bg-elev));
  }
  ul {
    margin: 0;
    padding-inline-start: var(--space-6);
    color: var(--accent-yellow);
    line-height: 1.5;
  }
  li + li { margin-top: var(--space-2); }
  .good { color: var(--accent-green); margin: 0; line-height: 1.5; }
  .error {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 8%, transparent);
    border-radius: var(--radius);
    padding: var(--surface-padding);
    line-height: 1.5;
  }
  @media (max-width: 600px) {
    .grid, .checks { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
</style>
