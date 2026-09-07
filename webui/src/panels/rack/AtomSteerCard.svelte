<script lang="ts">
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import { slidingSelection } from "../../lib/slidingSelection";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // Atom steer card — one racked single-direction term, wearing the same
  // RackCard chrome as the concept steer cards.  Both atom families share
  // this component; the family table below is the whole difference.
  //
  //   statline : ■/□ (lens) or ▲/△ (sae) enable toggle · atom id ·
  //              trigger pill · ✕
  //   body     : one α slider row (per-card, NOT the shared subspace
  //              along — atoms run hotter than concept vectors, so each
  //              one needs its own dial; ≈0.3 is the sweet spot)

  import type { AtomMode, AtomSteerEntry } from "../../lib/types";
  import Slider from "../../lib/Slider.svelte";
  import { ATOM_PREFIX, atomActions } from "../../lib/stores.svelte";
  import RackCard from "./RackCard.svelte";
  import RackMarker from "./RackMarker.svelte";
  import type { RackMarkerShape } from "./RackMarker.svelte";
  import { TRIGGER_LABEL, TRIGGER_WORD, nextTrigger } from "./triggers";

  /** Per-family presentation — the hue ontology's lens-blue ■ and
   *  sae-gold ▲, plus what the hover title calls the atom. */
  const FAMILY: Record<
    AtomMode,
    { accent: string; marker: RackMarkerShape; noun: string }
  > = {
    jlens: { accent: "--pillar-lens", marker: "square", noun: "j-lens token atom" },
    sae: { accent: "--pillar-sae", marker: "triangle", noun: "SAE decoder-row atom" },
  };

  interface Props {
    mode: AtomMode;
    name: string;
    entry: AtomSteerEntry;
  }

  let { mode, name, entry }: Props = $props();

  const family = $derived(FAMILY[mode]);
  const actions = $derived(atomActions(mode));
  const atomId = $derived(name.slice(ATOM_PREFIX[mode].length));

  function cycleTrigger(): void {
    actions.setTrigger(name, nextTrigger(entry.trigger));
  }
</script>

<RackCard accent={family.accent} disabled={!entry.enabled}>
  {#snippet statline()}
    <button
      type="button"
      class="enable"
      class:off={!entry.enabled}
      onclick={() => actions.setEnabled(name, !entry.enabled)}
      aria-pressed={entry.enabled}
      aria-label="Toggle steering for {name}"
    >
      <RackMarker shape={family.marker} filled={entry.enabled} />
    </button>

    <span class="name" class:struck={!entry.enabled} title="{family.noun} {name}">
      {atomId}
    </span>

    <button
      type="button"
      class="icon remove"
      onclick={() => actions.remove(name)}
      aria-label="remove {name}"
    >
      <FluentIcon name="dismiss" />
    </button>
  {/snippet}

  {#snippet body()}
    <div class="trigger-row">
      <span>Trigger</span>
      <button
        type="button"
        class="trigger-pill"
        onclick={cycleTrigger}
        title={TRIGGER_LABEL[entry.trigger]}
        aria-label="trigger for {name}: {entry.trigger}"
      >{TRIGGER_WORD[entry.trigger]}</button>
    </div>
    <div class="operation" role="group" aria-label="steering operation for {name}" use:slidingSelection>
      <button
        type="button"
        class:active={!entry.ablate}
        aria-pressed={!entry.ablate}
        onclick={() => actions.setAblate(name, false)}
      >push</button>
      <button
        type="button"
        class:active={entry.ablate}
        aria-pressed={entry.ablate}
        onclick={() => actions.setAblate(name, true)}
      >ablate</button>
    </div>
    <div class="alpha-row">
      <span class="alpha-label">α</span>
      <Slider
        value={entry.alpha}
        min={0}
        max={1}
        step={0.05}
        ariaLabel="alpha for {name}"
        oninput={(v) => Number.isFinite(v) && actions.setAlpha(name, v)}
      />
      <span class="alpha-val"><RollingNumber value={entry.alpha} digits={2} /></span>
    </div>
  {/snippet}
</RackCard>

<style>
  /* ----- statline pieces (mirrors SteerCard) ----- */
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

  .trigger-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }

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

  /* ----- body: α row (the manifold card's along-row shape) ----- */
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
  .alpha-row {
    display: grid;
    grid-template-columns: minmax(3em, auto) minmax(0, 1fr) 3em;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .alpha-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .alpha-val {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: end;
  }
</style>
