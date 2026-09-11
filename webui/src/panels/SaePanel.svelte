<script lang="ts">
  import { flip } from "svelte/animate";
  import { motionDuration } from "../lib/motion";
  // SAE — the inspector column's sparse-autoencoder tab: two sections,
  // card-based and symmetric with the other three pillars (every row wears
  // RackCard; the SAE family accent is gold, marker ▲/△).
  //
  //   STEER — one card per ``α sae/<id>`` decoder-row atom in the ONE
  //           steering expression.  Per-card α slider + trigger pill.
  //   PROBE — pinned ``sae/<id>`` feature probes (persistent, gate-able)
  //           first, then the live discovery cards for the per-step top-k
  //           features not already pinned — both the same card shape,
  //           exactly like the lens workspace readout (the pinned card's ▲
  //           unpins, the unpinned card's △ pins).  The card list owns the
  //           scroll; header + add form stay anchored.  The header's live
  //           toggle is the SAE live switch: off ⇒ no per-step feature
  //           readout — pinned probes settle to the end-of-gen activation,
  //           discovery cards go quiet.
  //
  // SOURCE mirrors J-LENS exactly: one selector uses or fetches an artifact,
  // followed by a labelled custom row. Successful preparation makes the source
  // resident and turns live readout on.

  import RuntimeSaeSourceSection from "@runtime-sae-source";
  import SaeProbeCard from "./rack/SaeProbeCard.svelte";
  import { mergeInstrumentProbeRows } from "./rack/probeRows";
  import type { InstrumentProbeRow } from "./rack/probeRows";
  import AtomSteerCard from "./rack/AtomSteerCard.svelte";
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { apiInstruments } from "../lib/runtime/services";
  import {
    activeProbeNames,
    addSaeToRack,
    attachProbe,
    probeRack,
    probeEntryForDisplay,
    saeState,
    saeRawFallbackScale,
    saeReadoutForDisplay,
    sessionState,
    setLiveSae,
    setSaeSortMode,
    steerRack,
    tokenHoverState,
    instrumentFamily,
    saeLoaded,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type {
    SaeProbeInfo,
    SaeSteerEntry,
    ScalarReadingJSON,
  } from "../lib/types";
  import type { SaeSortMode } from "../lib/stores.svelte";

  const loaded = $derived(saeLoaded());
  const displayReadout = $derived(saeReadoutForDisplay());
  /** The resident SAE, read off the family block + the prepared-sources
   *  listing (the flat ``sae_info`` key is gone — the read plane has one
   *  representation now). */
  const residentLayer = $derived.by((): number | null => {
    const live = instrumentFamily("sae")?.live;
    return live && "layer" in live ? live.layer : null;
  });

  // ---------- STEER: sae-mode rack entries (by feature id) ----------
  const steerCards = $derived.by(() => {
    const rows = [...steerRack.entries.entries()].filter(
      (row): row is [string, SaeSteerEntry] => row[1].mode === "sae",
    );
    rows.sort((a, b) => Number(a[0].slice(4)) - Number(b[0].slice(4)));
    return rows;
  });

  // ---------- PROBE: pinned probe cards + unpinned discovery cards ----------
  const pinnedBase = $derived.by(() => activeProbeNames()
    .map((name) => ({ name, entry: probeEntryForDisplay(name) }))
    .filter((row) => row.entry?.info.family === "sae"));

  const discoveryBase = $derived.by(() => displayReadout
    .filter((row) => !pinnedBase.some(({ entry }) =>
      entry?.info.family === "sae" && entry.info.feature_id === row.id))
    .map((row) => {
      // Metadata merges from the row's server-cached values and the
      // between-generation backfill (saeState.meta).
      const meta = tokenHoverState.active ? undefined : saeState.meta.get(row.id);
      return {
        ...row,
        label: row.label ?? meta?.label ?? null,
        max_act: row.max_act ?? meta?.max_act ?? null,
      };
    }));

  /** Panel-shared raw scale for metadata-less cards: the max raw reading
   *  across the visible cards that lack a maxActApprox unit, so their
   *  bars and numbers rank identically.  Cards WITH the unit render on
   *  the absolute 0..1 strength scale instead. */
  const fallbackScale = $derived(saeRawFallbackScale());

  const SORT_OPTIONS: { value: SaeSortMode; label: string }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
  ];

  function visibleStrength(value: number, maxAct: number | null): number {
    return maxAct != null && maxAct > 0 ? value / maxAct : value / fallbackScale;
  }

  type VisibleProbeCard =
    | ({
        kind: "pinned";
        name: string;
        entry: NonNullable<(typeof pinnedBase)[number]["entry"]>;
      } & InstrumentProbeRow)
    | ({
        kind: "discovery";
        feature: (typeof discoveryBase)[number];
      } & InstrumentProbeRow);

  const pinnedRows = $derived.by((): VisibleProbeCard[] =>
    pinnedBase.map((row) => {
      const entry = row.entry!;
      const info = entry.info as SaeProbeInfo;
      const latest = (entry.aggregate ?? entry.reading) as
        | ScalarReadingJSON
        | null;
      const value = latest?.value ?? entry.current;
      return {
        kind: "pinned",
        key: row.name,
        name: row.name,
        entry,
        sortName: info.label || (!tokenHoverState.active && saeState.meta.get(info.feature_id)?.label) || String(info.feature_id),
        // Pinned values with max_act are already normalized server-side —
        // the reading's own ``unit`` says which one applied.
        strength: latest?.unit === "activation_over_max" ? value : value / fallbackScale,
      };
    }));

  const discoveryRows = $derived.by((): VisibleProbeCard[] =>
    discoveryBase.map((feature) => ({
      kind: "discovery",
      key: `sae/${feature.id}`,
      feature,
      sortName: feature.label || String(feature.id),
      strength: visibleStrength(feature.activation, feature.max_act ?? null),
    })));

  /** One visible list, one sort order. Pinning changes persistence/actions,
   *  never a card's position outside the selected sort. */
  const probeCards = $derived(
    mergeInstrumentProbeRows(
      pinnedRows,
      saeState.live || tokenHoverState.active ? discoveryRows : [],
      saeState.sortMode,
      { numericNames: true },
    ),
  );

  let steerInput = $state("");
  let probeInput = $state("");
  let featureBusy = $state(false);

  async function validateInput(raw: string): Promise<number | null> {
    const id = Number(raw.trim().replace(/^sae\//, ""));
    if (!Number.isInteger(id) || id < 0) {
      pushToast("Enter a feature number of 0 or greater.", { kind: "error" });
      return null;
    }
    const validated = await apiInstruments.validateSaeFeature(id);
    return validated.id;
  }

  async function addSteer(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (featureBusy) return;
    featureBusy = true;
    try {
      const id = await validateInput(steerInput);
      if (id !== null) {
        addSaeToRack(id);
        steerInput = "";
      }
    } catch (error) {
      pushToast(userFacingError(error, "Unable to add that feature control. Try again."), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  async function pin(id: number): Promise<void> {
    if (featureBusy || probeRack.active.includes(`sae/${id}`)) return;
    featureBusy = true;
    try {
      await apiInstruments.validateSaeFeature(id);
      await attachProbe(`sae/${id}`);
    } catch (error) {
      pushToast(userFacingError(error, "Unable to pin that feature. Try again."), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  async function addProbe(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (featureBusy) return;
    featureBusy = true;
    try {
      const id = await validateInput(probeInput);
      if (id !== null) {
        await attachProbe(`sae/${id}`);
        probeInput = "";
      }
    } catch (error) {
      pushToast(userFacingError(error, "Unable to add that feature reading. Try again."), { kind: "error" });
    } finally {
      featureBusy = false;
    }
  }

  function openConversation(): void {
    window.dispatchEvent(new CustomEvent("drowse:workspace", {
      detail: "conversation",
    }));
  }
</script>

<div class="sae" aria-label="Model feature controls">
  <RuntimeSaeSourceSection />

  {#if loaded}

    <!-- STEER — decoder-row atom cards in the shared steering expression. -->
    <section class="section steer">
      <RackSectionHeader
        title="SAE steering"
        count={`${steerCards.length} term${steerCards.length === 1 ? "" : "s"}`}
      />

      {#if steerCards.length > 0}
        <div class="cards steer-cards" role="list">
          {#each steerCards as [name, entry] (name)}
            <div role="listitem" animate:flip={{ duration: motionDuration(160) }}>
              <AtomSteerCard mode="sae" {name} {entry} />
            </div>
          {/each}
        </div>
      {:else}
        <p class="hint empty-copy">
          No feature direction added. Enter a feature number below, then choose Add feature.
        </p>
      {/if}

      <form class="add-form" onsubmit={addSteer}>
        <label class="add-field">
          <span class="add-label">Feature number</span>
          <input
            class="add-input"
            type="text"
            inputmode="numeric"
            placeholder="e.g. 42"
            bind:value={steerInput}
            aria-label="Add a model feature direction"
          />
        </label>
        <button
          type="submit"
          class="add-btn"
          disabled={featureBusy || !steerInput.trim()}
        >
          Add feature
        </button>
      </form>
    </section>

    <!-- PROBE — pinned feature probes + the live discovery top-k.  The
         card list owns the scroll; header + add form stay anchored (the
         other pillars' fixed-chrome / scrollable-middle shape). -->
    <section class="section probe">
      <RackSectionHeader
        title="SAE readout"
        count={`${pinnedBase.length} pinned`}
        live={saeState.live}
        liveBusy={saeState.busy}
        liveTitle={saeState.live
          ? "turn live readout off"
          : "turn live readout on"}
        liveLabel="live model-feature readings"
        liveHelp="Update feature activity while the model writes. Pinned features stay visible when it finishes."
        onLiveToggle={() => void setLiveSae(!saeState.live)}
        sortValue={saeState.sortMode}
        sortOptions={SORT_OPTIONS}
        sortAriaLabel="Sort model features by"
        onSortChange={setSaeSortMode}
      />

      <div class="scroll">
        {#if probeCards.length > 0}
          <div class="cards" role="list" aria-label="SAE feature probes">
            {#each probeCards as row (row.key)}
              <div role="listitem" animate:flip={{ duration: motionDuration(160) }}>
                {#if row.kind === "pinned"}
                  {@const reading = (row.entry.aggregate ?? row.entry.reading) as ScalarReadingJSON | null}
                  {@const probe = row.entry.info as SaeProbeInfo}
                  <!-- A pinned probe's channel (value, sparkline, gates) is
                       already normalized server-side when max_act is set. -->
                  <SaeProbeCard
                    id={probe.feature_id}
                    probeName={row.name}
                    label={probe.label || (!tokenHoverState.active ? saeState.meta.get(probe.feature_id)?.label : null)}
                    layer={residentLayer}
                    value={reading?.value ?? row.entry.current ?? 0}
                    measured={reading !== null || row.entry.sparkline.length > 0}
                    maxAct={reading?.unit === "raw_activation" ? null : probe.max_act}
                    valueIsStrength={reading?.unit === "activation_over_max"}
                    {fallbackScale}
                    series={row.entry.sparkline}
                    pinned={true}
                  />
                {:else}
                  <SaeProbeCard
                    id={row.feature.id}
                    label={row.feature.label}
                    layer={residentLayer}
                    value={row.feature.activation}
                    maxAct={row.feature.max_act}
                    {fallbackScale}
                    series={tokenHoverState.active ? [row.feature.activation] : saeState.history.get(row.feature.id) ?? []}
                    pinned={false}
                    busy={featureBusy}
                    onpin={(id) => void pin(id)}
                  />
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        {#if tokenHoverState.active}
          {#if tokenHoverState.saeLoading}
            <p class="hint">reading hovered token…</p>
          {:else if tokenHoverState.saeError}
            <p class="hint read-error" role="alert">{tokenHoverState.saeError}</p>
          {:else if probeCards.length === 0}
            <p class="hint">no SAE score for this token</p>
          {/if}
        {:else if saeState.live}
          {#if discoveryBase.length === 0}
            <div class="empty-state">
              <p class="hint">Pin a feature, then send a message to track its activity.</p>
              <button type="button" class="empty-action" onclick={openConversation}>
                Go to conversation
              </button>
            </div>
          {/if}
        {:else}
          <p class="hint">Live is off. Turn it on to see active features while the model writes, or pin one below.</p>
        {/if}
      </div>

      <form class="add-form anchored" onsubmit={addProbe}>
        <label class="add-field">
          <span class="add-label">Feature number</span>
          <input
            class="add-input"
            type="text"
            inputmode="numeric"
            placeholder="e.g. 42"
            bind:value={probeInput}
            aria-label="Watch a model feature"
          />
        </label>
        <button
          type="submit"
          class="add-btn"
          disabled={featureBusy || !probeInput.trim()}
        >
          Watch feature
        </button>
      </form>
    </section>
  {/if}
</div>

<style>
  /* Fixed-chrome column, matching the other pillar tabs: STEER sizes to
     its content up to half the inspector, PROBE takes the rest and scrolls
     internally so the header + add form stay visible. */
  .sae {
    display: flex;
    flex-direction: column;
    height: 100%;
    max-height: 100%;
    min-height: 0;
    overflow: hidden;
  }

  /* Flat borderless sections — typography + padding carry the divide,
     matching the rack chrome. */
  .section {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--surface-padding);
    min-height: 0;
  }
  .section.steer {
    flex: 0 1 auto;
    max-height: 50%;
    overflow: hidden;
  }
  /* A populated steer-card pile scrolls inside its own half-column cap
     rather than eating the probe section's share of the inspector. */
  .steer-cards {
    overflow-y: auto;
    min-height: 0;
  }
  .section.probe {
    flex: 1 1 0;
    overflow: hidden;
  }
  /* The scrollable middle — cards + hints; header and add form stay put. */
  .scroll {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    flex: 1 1 0;
    min-height: 2.4rem;
    overflow-y: auto;
    padding-inline-end: var(--space-1);
  }
  /* Anchored footer — borderless, same padding treatment as the racks'
     actions row. */
  .add-form.anchored {
    flex: 0 0 auto;
    padding-top: var(--space-3);
  }

  .hint {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .read-error { color: var(--accent-red); }
  .empty-copy {
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--surface-sheen), var(--glass);
  }
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--surface-sheen), var(--glass);
  }
  .empty-action {
    min-height: var(--control-target);
    padding: 0 var(--space-4);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
    background: var(--glass-strong);
    color: var(--fg-strong);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
  }
  .empty-action:hover { color: var(--pillar-sae); }
  /* Card stack — same rhythm as the other racks' strips. */
  .cards {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  /* ----- add / load forms ----- */
  .add-form {
    display: flex;
    align-items: flex-end;
    gap: var(--space-2);
  }
  .add-field {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .add-label {
    color: var(--fg-dim);
    font-family: var(--font-structure);
    font-size: var(--text-xs);
    font-weight: var(--weight-structure);
  }
  .add-input {
    width: 100%;
    min-height: var(--control-target);
    min-width: 0;
    /* Borderless input: recessed well fill; ring on focus only. */
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    padding: var(--space-xs) var(--space-3);
    transition: border-color var(--dur-fast) var(--ease-out);
  }
  .add-input:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 1px;
    border-color: var(--accent-strong);
  }
  .add-btn {
    min-height: var(--control-target);
    /* Family-tinted fill — the gold sibling of the racks' launchers. */
    background: color-mix(in srgb, var(--pillar-sae) 10%, transparent);
    color: var(--pillar-sae);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: var(--space-xs) var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
    transition: background var(--dur) var(--ease-out);
  }
  .add-btn:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-sae) 18%, transparent);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .add-btn:active,
  .empty-action:active { transform: scale(var(--press-scale)); }

  @media (max-width: 920px), (max-height: 700px) {
    .sae {
      display: block;
      overflow-y: auto;
    }
    .section.steer,
    .section.probe {
      max-height: none;
      overflow: visible;
    }
    .scroll,
    .steer-cards {
      flex: 0 0 auto;
      overflow: visible;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .add-btn,
    .empty-action { transition: none; }
  }

</style>
