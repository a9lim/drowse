<script lang="ts">
  import { flip } from "svelte/animate";
  import { motionDuration } from "../lib/motion";
  // J-LENS — the inspector column's Jacobian-lens tab: two sections,
  // card-based and symmetric with the CAA tab (every row wears RackCard;
  // the j-lens family accent is blue, marker ■/□).
  //
  //   STEER — one card per ``α jlens/<word>`` token atom in the ONE
  //           steering expression (the engine folds the lens direction
  //           over all fitted layers with whitened shares, exactly
  //           like a concept vector).  Per-card α slider + trigger
  //           pill (lens atoms run hotter than concept vectors;
  //           default 0.3).
  //   PROBE — the J-lens readout, pinned and unpinned in ONE section
  //           (pinning a token just makes its card persistent and
  //           gate-able — both card kinds are the same shape: strength
  //           bar + per-layer strength strip; the pinned card's ■ unpins,
  //           the unpinned card's □ pins).  Pinned ``jlens/<word>`` token
  //           probes first, then the exact whole-vocabulary aggregate cards
  //           not already pinned.  The card list owns
  //           the scroll (header + add form stay anchored, like the CAA
  //           racks).  The header's live toggle is the lens live switch:
  //           off ⇒ no per-step lens computation — pinned probes settle
  //           to the end-of-gen aggregate, discovery cards go quiet.  The
  //           full per-layer ranking lives in the transcript drilldown.
  //
  // SOURCE is the shared lifecycle shell: one selector uses or fetches an
  // artifact, followed by the same labelled custom row as SAE. Successful
  // preparation activates the source and live readout.

  import RuntimeJLensSourceSection from "@runtime-jlens-source";
  import RackSectionHeader from "./rack/RackSectionHeader.svelte";
  import JLensProbeCard from "./rack/JLensProbeCard.svelte";
  import { mergeInstrumentProbeRows } from "./rack/probeRows";
  import type { InstrumentProbeRow } from "./rack/probeRows";
  import AtomSteerCard from "./rack/AtomSteerCard.svelte";
  import { apiInstruments, describeError } from "../lib/runtime/services";
  import {
    addJLensToRack,
    activeProbeNames,
    attachProbe,
    lensState,
    lensAggregateForDisplay,
    lensReadoutForDisplay,
    probeRack,
    probeEntryForDisplay,
    sessionState,
    setLensWorkspaceSortMode,
    setLiveLens,
    steerRack,
    tokenHoverState,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type {
    JLensSteerEntry,
    ProbeRackEntry,
    ScalarReadingJSON,
  } from "../lib/types";
  import type { LensWorkspaceSortMode } from "../lib/stores.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const liveOn = $derived(lensState.layers !== null);
  const displayReadout = $derived(lensReadoutForDisplay());
  const displayAggregate = $derived(lensAggregateForDisplay());
  const displayLayers = $derived.by(() => {
    if (!tokenHoverState.active) return lensState.layers ?? [];
    return Object.keys(displayReadout ?? {}).map(Number).sort((a, b) => a - b);
  });

  // ---------- STEER: jlens-mode rack entries (alphabetical) ----------
  const steerCards = $derived.by(() => {
    const arr = [...steerRack.entries.entries()].filter(
      (kv): kv is [string, JLensSteerEntry] => kv[1].mode === "jlens",
    );
    arr.sort((a, b) => a[0].localeCompare(b[0]));
    return arr;
  });

  let steerInput = $state("");
  let steerBusy = $state(false);

  function bareWord(value: string): string {
    return value.trim().replace(/^jlens\//, "");
  }

  async function onAddSteer(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const submitted = steerInput;
    const word = bareWord(submitted);
    if (!word || steerBusy) return;
    steerBusy = true;
    try {
      const validated = await apiInstruments.validateLensToken(word);
      addJLensToRack(validated.word);
      if (steerInput === submitted) steerInput = "";
    } catch (e) {
      pushToast(`Couldn't steer toward jlens/${word}: ${describeError(e)}`, {
        kind: "error",
      });
    } finally {
      steerBusy = false;
    }
  }

  // ---------- PROBE: pinned probe cards + unpinned aggregate cards ----------
  // One merged section — pinning a workspace token just makes its card
  // persistent (and gate-able); both card families share the one sort
  // control (strength / name / depth).

  /** One card's props plus the shared sort keys — pinned and discovery
   *  rows produce the identical shape, so the roster is one list of one
   *  card. */
  interface WorkspaceCard extends InstrumentProbeRow {
    token: string;
    strength: number;
    com: number | null;
    spread: number | null;
    measured: boolean;
    series: (number | null)[];
    cells: { layer: number; p: number | null }[];
    pinned: boolean;
  }

  const SORT_OPTIONS: {
    value: LensWorkspaceSortMode;
    label: string;
  }[] = [
    { value: "strength", label: "strength" },
    { value: "name", label: "name" },
    { value: "depth", label: "depth" },
  ];

  /** Per-layer cells for a pinned probe — the store's axis-0 per-layer map. */
  function pinnedCells(
    entry: ProbeRackEntry,
  ): { layer: number; p: number | null }[] {
    const perLayer = entry.perLayer;
    if (!perLayer) return [];
    return Object.keys(perLayer)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => ({ layer: Number(layer), p: perLayer[layer] ?? null }));
  }

  /** Per-layer cells for a discovery token — its softmax probability in
   *  each streamed readout row; ``null`` = below that layer's top-k. */
  function readoutCells(token: string): { layer: number; p: number | null }[] {
    return displayLayers.map((layer) => {
      const pairs = displayReadout?.[String(layer)];
      if (!pairs || pairs.length === 0) return { layer, p: null };
      const hit = pairs.find(([text]) => text === token);
      return { layer, p: hit ? hit[1] : null };
    });
  }

  const pinnedCards = $derived.by((): WorkspaceCard[] => {
    const rows: WorkspaceCard[] = [];
    for (const name of activeProbeNames()) {
      const entry = probeEntryForDisplay(name);
      if (entry?.info.family !== "lens") continue;
      // A pinned lens probe reads the family's NATIVE one-channel
      // reading — value + depth summary, no coordinate vector to unwrap.
      const latest = (entry.aggregate ?? entry.reading) as
        | ScalarReadingJSON
        | null;
      const word = entry.info.word;
      rows.push({
        key: name,
        sortName: word,
        token: word,
        strength: latest?.value ?? entry.current ?? 0,
        measured: latest !== null || entry.sparkline.length > 0,
        com: latest?.depth?.center?.[0] ?? null,
        spread: latest?.depth?.spread?.[0] ?? null,
        series: entry.sparkline ?? [],
        cells: pinnedCells(entry),
        pinned: true,
      });
    }
    return rows;
  });

  const aggRows = $derived.by((): WorkspaceCard[] => {
    const rows = displayAggregate;
    if (!rows || rows.length === 0) return [];
    const hist = tokenHoverState.active ? [] : lensState.aggHistory;
    // Pinned tokens already have a persistent card — the aggregate group
    // carries only the unpinned remainder of the top-k.
    return rows
      .filter(([token]) => !pinnedCards.some((card) => card.token === token))
      .map(([token, strength, com, spread]) => ({
        key: `aggregate:${token}`,
        sortName: token.trim(),
        token,
        strength,
        measured: true,
        com,
        spread,
        series: tokenHoverState.active
          ? [strength]
          : hist.map((frame) => frame.find(([t]) => t === token)?.[1] ?? null),
        cells: readoutCells(token),
        pinned: false,
      }));
  });

  /** Pinned and discovered tokens are one visual roster. Persistence is an
   *  action/state difference, not a hidden first sort key. */
  const workspaceCards = $derived(
    mergeInstrumentProbeRows(
      pinnedCards,
      liveOn || tokenHoverState.active ? aggRows : [],
      lensState.workspaceSortMode,
    ),
  );

  let probeInput = $state("");
  let probeBusy = $state(false);

  async function pinWord(word: string): Promise<boolean> {
    const bare = bareWord(word);
    if (!bare || probeBusy) return false;
    const selector = `jlens/${bare}`;
    if (probeRack.active.includes(selector)) return true;
    probeBusy = true;
    try {
      const validated = await apiInstruments.validateLensToken(bare);
      const validatedSelector = `jlens/${validated.word}`;
      await attachProbe(validatedSelector);
      pushToast(`pinned ${validatedSelector}`, { kind: "info" });
      return true;
    } catch (e) {
      pushToast(`Couldn't pin ${selector}: ${describeError(e)}`, {
        kind: "error",
      });
      return false;
    } finally {
      probeBusy = false;
    }
  }

  async function onAddProbe(ev: SubmitEvent): Promise<void> {
    ev.preventDefault();
    const submitted = probeInput;
    if (await pinWord(submitted)) {
      if (probeInput === submitted) probeInput = "";
    }
  }

  function onToggleLive(): void {
    void setLiveLens(!liveOn);
  }

  function openConversation(): void {
    window.dispatchEvent(new CustomEvent("drowse:workspace", {
      detail: "conversation",
    }));
  }
</script>

<div class="jlens" aria-label="Layer prediction controls">
  <RuntimeJLensSourceSection />

  {#if fitted}
    <!-- STEER — token-atom cards in the shared steering expression. -->
    <section class="section steer">
      <RackSectionHeader
        title="J-lens steering"
        help="This steers the model toward predicting one tokenizer word. Lower strengths are usually easier to control."
        count={`${steerCards.length} term${steerCards.length === 1 ? "" : "s"}`}
      />

      {#if steerCards.length > 0}
        <div class="cards steer-cards" role="list">
          {#each steerCards as [name, entry] (name)}
            <div role="listitem" animate:flip={{ duration: motionDuration(160) }}>
              <AtomSteerCard mode="jlens" {name} {entry} />
            </div>
          {/each}
        </div>
      {:else}
        <p class="hint empty-copy">
          No word direction added. Enter a word below, then choose Add word.
        </p>
      {/if}

      <form class="add-form" onsubmit={onAddSteer}>
        <label class="add-field">
          <span class="add-label">Word</span>
          <input
            class="add-input"
            type="text"
            placeholder="e.g. calm"
            bind:value={steerInput}
            aria-label="Add a word prediction direction"
          />
        </label>
        <button
          type="submit"
          class="add-btn"
          disabled={steerBusy || !steerInput.trim()}
        >
          Add word
        </button>
      </form>
    </section>

    <!-- PROBE — the merged workspace readout: pinned token-probe cards
         (persistent, gate-able) + the unpinned live aggregate cards.
         The card list owns the scroll; header + add form stay anchored
         (the CAA racks' fixed-chrome / scrollable-middle shape). -->
    <section class="section probe">
      <RackSectionHeader
        title="J-lens readout"
        count={`${pinnedCards.length} pinned`}
        live={liveOn}
        liveBusy={lensState.busy}
        liveTitle={liveOn
          ? "turn live readout off"
          : "turn live readout on"}
        liveLabel="live predicted-word readings"
        liveHelp="Update predicted words while the model writes. Pinned words stay visible when it finishes."
        onLiveToggle={onToggleLive}
        sortValue={lensState.workspaceSortMode}
        sortOptions={SORT_OPTIONS}
        sortAriaLabel="Sort prediction words by"
        onSortChange={setLensWorkspaceSortMode}
      />

      <div class="scroll">
        {#if workspaceCards.length > 0}
          <div class="cards" role="list" aria-label="J-lens probe tokens">
            {#each workspaceCards as card (card.key)}
              <div role="listitem" animate:flip={{ duration: motionDuration(160) }}>
                <JLensProbeCard
                  token={card.token}
                  probeName={card.pinned ? card.key : undefined}
                  strength={card.measured ? card.strength : null}
                  com={card.com}
                  spread={card.spread}
                  series={card.series}
                  cells={card.cells}
                  pinned={card.pinned}
                  busy={probeBusy}
                  onpin={pinWord}
                />
              </div>
            {/each}
          </div>
        {/if}

        {#if tokenHoverState.active}
          {#if tokenHoverState.lensLoading}
            <p class="hint">reading hovered token…</p>
          {:else if tokenHoverState.lensError}
            <p class="hint read-error" role="alert">{tokenHoverState.lensError}</p>
          {:else if aggRows.length === 0}
            <p class="hint">no J-lens score for this token</p>
          {/if}
        {:else if liveOn}
          {#if aggRows.length > 0}
            <p class="hint drill-hint">select a token for layers</p>
          {:else}
            <div class="empty-state">
              <p class="hint">Pin a word, then send a message to track its prediction strength.</p>
              <button type="button" class="empty-action" onclick={openConversation}>
                Go to conversation
              </button>
            </div>
          {/if}
        {:else}
          <p class="hint">Live is off. Turn it on to see predicted words while the model writes, or pin a word below.</p>
        {/if}
      </div>

      <form class="add-form anchored" onsubmit={onAddProbe}>
        <label class="add-field">
          <span class="add-label">Word</span>
          <input
            class="add-input"
            type="text"
            placeholder="e.g. answer"
            bind:value={probeInput}
            aria-label="Watch a prediction word"
          />
        </label>
        <button
          type="submit"
          class="add-btn"
          disabled={probeBusy || !probeInput.trim()}
        >
          Watch word
        </button>
      </form>
    </section>
  {/if}
</div>

<style>
  /* Fixed-chrome column, matching the CAA rack-grid: STEER sizes to its
     content up to half the inspector, PROBE takes the rest and scrolls
     internally so the header + add form stay visible. */
  .jlens {
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
  }
  .section.steer {
    flex: 0 1 auto;
    min-height: 0;
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
    min-height: 0;
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
  /* Anchored footer — borderless, same padding treatment as the CAA
     racks' actions row. */
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
  .empty-action:hover { color: var(--pillar-lens); }
  .drill-hint {
    font-size: var(--text-xs);
    color: var(--fg-dim);
  }


  /* Card stack — same rhythm as the probe rack's strips. */
  .cards {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  /* ----- add forms ----- */
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
    background: color-mix(in srgb, var(--pillar-lens) 10%, transparent);
    color: var(--pillar-lens);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-size: var(--text-sm);
    padding: var(--space-xs) var(--space-3);
    cursor: pointer;
    flex: 0 0 auto;
  }
  .add-btn:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-lens) 18%, transparent);
  }
  .add-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .add-btn:active,
  .empty-action:active { transform: scale(var(--press-scale)); }

  @media (max-width: 920px), (max-height: 700px) {
    .jlens {
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
