<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import { slidingSelection } from "../../lib/slidingSelection";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // Unified steer card — one row for every steering term.  Every term is a
  // position on a fitted geometry; the card branches on ``entry.mode``:
  //
  //   subspace → a flat affine fit (a 2-node bipolar axis through the rank-8
  //              personas fan).  Subspace accent (●/○, --accent).
  //              statline: glyph · name · unfitted/stale warn · trigger · ✕
  //              body:     snap-to-node Select · XYPad   (NO per-card along —
  //                        the magnitude is the rack-level "subspace along"
  //                        master shared by every subspace term)
  //   manifold → a curved fit (e.g. emotions).  Manifold accent (◆/◇,
  //              --pillar-manifold).
  //              statline: glyph · name · unfitted/stale warn · trigger · ✕
  //              body:     snap-to-node Select · XYPad · along · onto
  //
  // Rank-one subspaces expose push/ablate. Higher-rank ``~``/``|`` projection
  // and tensor variants remain advanced-expression operations.
  // ``s`` / ``m`` are the narrowed entry views the template renders behind
  // ``{#if s}`` / ``{#if m}`` so svelte-check enforces mode-correct access.

  import type { SteerEntry, SubspaceSteerEntry, ManifoldSteerEntry } from "../../lib/types";
  import {
    setSubspaceCoords,
    setSubspaceLabel,
    setSubspaceTrigger,
    setSubspaceEnabled,
    setSubspaceAblate,
    removeSubspaceFromRack,
    setManifoldBlend,
    setManifoldOnto,
    setManifoldCoords,
    setManifoldLabel,
    setManifoldTrigger,
    setManifoldEnabled,
    removeManifoldFromRack,
    manifoldByName,
  } from "../../lib/stores.svelte";
  import Slider from "../../lib/Slider.svelte";
  import Select from "../../lib/Select.svelte";
  import XYPad from "../manifold/XYPad.svelte";
  import RackCard from "./RackCard.svelte";
  import RackMarker from "./RackMarker.svelte";
  import { TRIGGER_LABEL, TRIGGER_WORD, nextTrigger } from "./triggers";

  interface Props {
    name: string;
    entry: SteerEntry;
  }

  let { name, entry }: Props = $props();

  // Narrowed views — exactly one is non-null per entry.
  const s = $derived<SubspaceSteerEntry | null>(
    entry.mode === "subspace" ? entry : null,
  );
  const m = $derived<ManifoldSteerEntry | null>(
    entry.mode === "manifold" ? entry : null,
  );

  // ---------- family chrome (accent + enable glyph) ----------

  const subspace = $derived(entry.mode === "subspace");
  const accent = $derived(subspace ? "--accent" : "--pillar-manifold");

  /** Display name — bare name with the namespace prefix stripped
   *  (``default/personas`` → ``personas``).  Full name stays in the accessible description. */
  const displayName = $derived(name.split("/").pop() ?? name);

  /** Catalog row — drives the node list, the XYPad bounds, and the
   *  fitted/stale chips. */
  const info = $derived(manifoldByName(name));
  const fitted = $derived(info?.fitted_for_session === true);
  const stale = $derived(info?.stale === true);
  const rankOne = $derived(info?.node_count === 1 || info?.node_count === 2);

  // ---------- trigger cycle (shared vocabulary in ./triggers) ----------

  function cycleTrigger(): void {
    const next = nextTrigger(entry.trigger);
    if (entry.mode === "subspace") setSubspaceTrigger(name, next);
    else setManifoldTrigger(name, next);
  }

  function toggleEnabled(): void {
    if (entry.mode === "subspace") setSubspaceEnabled(name, !entry.enabled);
    else setManifoldEnabled(name, !entry.enabled);
  }

  function removeTerm(): void {
    // Remove from the rack only — never deletes the server-side artifact
    // (the RackDrawer's delete button owns that, behind a confirm).
    if (entry.mode === "subspace") removeSubspaceFromRack(name);
    else removeManifoldFromRack(name);
  }

  // ---------- position controls (shared snap + XYPad) ----------

  function onSnapToNode(val: string): void {
    const label = val === "" ? null : val;
    if (entry.mode === "subspace") setSubspaceLabel(name, label);
    else setManifoldLabel(name, label);
  }
  function onCoordsChange(coords: number[]): void {
    if (entry.mode === "subspace") setSubspaceCoords(name, coords);
    else setManifoldCoords(name, coords);
  }

  const snapOptions = $derived.by<{ value: string; label: string }[]>(() => {
    const labels = info?.node_labels ?? [];
    const roles = info?.node_roles;
    const opts = labels.map((nl, i) => {
      const role = roles?.[i];
      return { value: nl, label: nl + (role ? ` [role=${role}]` : "") };
    });
    return [{ value: "", label: "(free position)" }, ...opts];
  });

  const activeLabel = $derived(s?.label ?? m?.label ?? null);
  const orderedNodes = $derived.by(() => {
    if (!info || info.intrinsic_dim !== 1 || info.is_discover || info.node_coords.length !== info.node_labels.length) return [];
    return info.node_labels.map((label, index) => ({ label, position: info.node_coords[index]?.[0] })).filter(node => Number.isFinite(node.position)).sort((a, b) => a.position - b.position);
  });
  const activeCoords = $derived(s?.coords ?? m?.coords ?? []);

  // ---------- manifold-only controls ----------

  function onBlendInput(val: number): void {
    if (Number.isFinite(val)) setManifoldBlend(name, val);
  }
  function onOntoInput(val: number): void {
    if (Number.isFinite(val)) setManifoldOnto(name, val);
  }
</script>

<RackCard {accent} disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={toggleEnabled}
      aria-pressed={entry.enabled}
      aria-label="Toggle steering for {name}"
    >
      <RackMarker
        shape={subspace ? "circle" : "diamond"}
        filled={entry.enabled}
      />
    </button>

    <span class="name" class:struck={!entry.enabled} {...{ "aria-description": (subspace ? `subspace ${name}` : `manifold ${name}`) }}>
      {displayName}
    </span>

    {#if !fitted && info}
      <span class="warn" {...{ "aria-description": "fit required" }}>
        unfitted
      </span>
    {:else if stale}
      <span class="warn" {...{ "aria-description": "refit required" }}>
        stale
      </span>
    {/if}

    <button
      type="button"
      class="icon remove"
      onclick={removeTerm}
      aria-label="remove {name}"
    >
      <FluentIcon name="dismiss" />
    </button>
  {/snippet}

  {#snippet body()}
    <div class="trigger-row">
      <span class="ctl-label">Trigger</span>
      <button type="button" class="trigger-pill" onclick={cycleTrigger}
        aria-label="trigger for {name}: {entry.trigger}" {...{ "aria-description": (TRIGGER_LABEL[entry.trigger]) }}>
        <MorphText text={TRIGGER_WORD[entry.trigger]} numbers={false} />
      </button>
    </div>
    {#if info}
      {#if s && rankOne}
        <div class="operation" role="group" aria-label="steering operation for {name}" use:slidingSelection>
          <button
            type="button"
            class:active={!s.ablate}
            aria-pressed={!s.ablate}
            onclick={() => setSubspaceAblate(name, false)}
          >push</button>
          <button
            type="button"
            class:active={s.ablate}
            aria-pressed={s.ablate}
            onclick={() => setSubspaceAblate(name, true)}
          >ablate</button>
        </div>
      {/if}
      {#if !s?.ablate && info.node_labels.length > 0}
        <label class="ctl-row">
          <span class="ctl-label">Target node</span>
          <span class="ctl-select">
            <Select
              value={activeLabel ?? ""}
              options={snapOptions}
              onchange={onSnapToNode}
              ariaLabel="snap to node"
            />
          </span>
        </label>
      {/if}
      {#if !s?.ablate && orderedNodes.length > 1}
        <details class="node-scrubber"><summary>Scrub named nodes</summary>
          <Slider value={Math.max(0, orderedNodes.findIndex(node => node.label === activeLabel))} min={0} max={orderedNodes.length - 1} step={1} displayValue={activeLabel ?? "Choose a node"} ariaLabel="Ordered manifold node" oninput={index => onSnapToNode(orderedNodes[index].label)} />
          <span><MorphText text={activeLabel ?? "Free position"} numbers={false} /></span>
        </details>
      {/if}
      {#if !s?.ablate}
        <XYPad
          manifold={info}
          coords={activeCoords}
          onchange={onCoordsChange}
          locked={activeLabel !== null}
        />
      {/if}
      {#if s?.ablate}
        <p class="hint">removes this direction from the current activation</p>
      {:else if m}
        <!-- Curved manifold: per-card along + onto.  Subspace terms have no
             per-card along — they share the rack-level "subspace along". -->
        <div class="along-row">
          <span class="along-label">along</span>
          <Slider
            value={m.blend}
            min={0}
            max={1}
            step={0.05}
            oninput={onBlendInput}
            ariaLabel="along fraction for {name}"
          />
          <span class="along-val"><RollingNumber value={m.blend} digits={2} /></span>
        </div>
        <div class="along-row">
          <span class="along-label">onto</span>
          <Slider
            value={m.onto}
            min={0}
            max={1}
            step={0.05}
            oninput={onOntoInput}
            ariaLabel="onto fraction for {name}"
          />
          <span class="along-val"><RollingNumber value={m.onto} digits={2} /></span>
        </div>
      {:else}
        <p class="hint">Uses shared subspace α</p>
      {/if}
    {:else}
      <p class="missing">metadata unavailable</p>
    {/if}
  {/snippet}
</RackCard>

<style>
  .node-scrubber summary { min-height: 44px; align-content: center; cursor: pointer; color: var(--fg-dim); font-size: var(--text-xs); }
  /* ----- statline pieces ----- */
  .enable {
    display: inline-grid;
    place-items: center;
    inline-size: 24px;
    block-size: 24px;
    margin: 0 calc(var(--space-xs) * -1);
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    padding: 0;
    color: var(--card-accent);
    flex: 0 0 24px;
    cursor: pointer;
  }
  .enable.off {
    color: var(--fg-muted);
  }

  .name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    flex: 1;
    overflow-wrap: anywhere;
    min-width: 0;
  }
  .name.struck {
    text-decoration: line-through;
    color: var(--fg-muted);
  }

  .warn {
    flex: 0 0 auto;
    color: var(--warning-ink);
    font-size: var(--text-2xs);
    background: var(--warning-bg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: 0 var(--space-2);
  }

  .trigger-row { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); }

  .trigger-pill {
    min-height: var(--control-target);
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius);
    font-size: var(--text-xs);
    line-height: 1.2;
    flex: 0 0 auto;
    cursor: pointer;
    transition: background var(--dur) var(--ease-out);
  }
  .trigger-pill:hover {
    background: var(--glass-strong);
  }

  .icon {
    min-width: var(--control-target);
    min-height: var(--control-target);
    background: transparent;
    border: 0;
    color: var(--fg-muted);
    font-size: var(--text);
    line-height: 1;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    flex: 0 0 auto;
    cursor: pointer;
    transition: color var(--dur) var(--ease-out),
      background var(--dur) var(--ease-out);
  }
  .icon:hover:not(:disabled) {
    color: var(--fg-strong);
    background: var(--bg-elev);
  }
  .remove:hover:not(:disabled) {
    color: var(--accent-red);
  }

  /* ----- body: position controls ----- */
  .operation {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-xs);
    padding: var(--space-xs);
    border-radius: var(--radius-group);
    background: var(--surface-sheen), var(--glass);
  }
  .operation button {
    min-height: var(--control-target);
    border: 0;
    border-radius: var(--radius-inset);
    background: transparent;
    color: var(--fg-muted);
    font-family: var(--font-structure);
    font-size: var(--text-xs);
    font-weight: var(--weight-structure);
    cursor: pointer;
  }
  .operation button.active {
    background: var(--bg-elev);
    color: var(--fg-strong);
    box-shadow: var(--shadow-rack-active);
  }
  .ctl-row {
    display: grid;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
    color: var(--fg-strong);
    font-size: var(--text-xs);
  }
  .ctl-label {
    color: var(--fg-muted);
    flex: 0 0 auto;
  }
  .ctl-select {
    flex: 1 1 auto;
    display: inline-flex;
    min-width: 0;
    max-width: 100%;
  }
  .along-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) minmax(0, 1fr) 3em;
    align-items: center;
    gap: var(--space-2);
  }
  .along-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: lowercase;
  }
  .along-val {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: end;
  }
  .hint,
  .missing {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
</style>
