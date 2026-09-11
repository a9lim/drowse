<script lang="ts">
  import MorphText from "../lib/ui/MorphText.svelte";
  import FluentIcon from "../lib/ui/FluentIcon.svelte";
  import SidebarIcon from "../lib/ui/SidebarIcon.svelte";
  import PopoutIcon from "../lib/ui/PopoutIcon.svelte";
  import { dockTokenDetails, hideTokenDetails, undockTokenDetails } from "../lib/stores/drawers.svelte";
  import { onDestroy, onMount, tick, untrack } from "svelte";
  import { fly } from "svelte/transition";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Per-token drilldown drawer — opens when a chat / raw-buffer token is
  // clicked.
  //
  // The drawer is a shell over four family tabs (drawers/token/):
  // geometry (Monitor readings), logits (top-K alts + fork), sae (gold),
  // j-lens (blue) — each speaking one shared grammar: the shell's
  // identity header (token text + id / raw / logprob chips), the context
  // ribbon, then a per-tab InstrumentHeader (provenance · source ·
  // steering · apply-recipe toggle) over the body.  The selected tab is
  // STICKY while available (drilldown.svelte.ts); otherwise opening the
  // drawer selects a captured or replayable view.
  //
  // Navigation is a conversation-walking cursor (drawers/token/cursor.ts):
  // ←/→ step tokens and ROLL ACROSS segment + turn boundaries (thinking →
  // response → next turn), so chat mode walks the same flat stream the
  // raw buffer shows; ↑/↓ jump turns, Home/End jump segment bounds, and
  // a fresh token click snaps the cursor back to its anchor.  Drawer
  // params still come in via openDrawer("token_drilldown", { turnIdx,
  // tokenIdx, isThinking? }) and index chatLog.turns.

  import {
    closeDrawer,
    chatLog,
    loomTree,
    loomUiState,
    samplingState,
    sessionState,
    effectiveRawMode,
    probeRack,
    lensSourceState,
    saeSourceState,
    instrumentFamily,
    saeLoaded,
    genStatus,
    sendFork,
    sendTextFork,
  } from "../lib/stores.svelte";
  import type {
    ChatTurn,
    LensTokenReadoutJSON,
    SaeTokenReadoutJSON,
    ScalarReadingJSON,
    TokenScore,
  } from "../lib/types";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import {
    clampCursor,
    jumpTurn,
    segmentTokens,
    segmentsOf,
    stepCursor,
    type SegmentKind,
    type TokenCursor,
  } from "./token/cursor";
  import {
    ReplayReadout,
    type GeometryTokenReadout,
  } from "./token/readout.svelte";
  import { availableDrilldownTab, drilldownUi, type DrilldownTab } from "./token/drilldown.svelte";
  import TokenRibbon from "./token/TokenRibbon.svelte";
  import GeometryTab from "./token/GeometryTab.svelte";
  import LogitsTab from "./token/LogitsTab.svelte";
  import SaeTab from "./token/SaeTab.svelte";
  import LensTab from "./token/LensTab.svelte";
  import { resolveReadoutTopK } from "../lib/readouts";
  import { selectionIn } from "../lib/motion";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import {
    currentCatalogInstrumentAvailability,
    type CatalogInstrumentAvailability,
  } from "../lib/runtime/instrumentAvailability";

  interface DrawerParams {
    turnIdx: number;
    tokenIdx: number;
    /** When true the click came from the thinking-collapsible body, so
     * the anchor segment is ``turn.thinkingTokens``. */
    isThinking?: boolean;
    /** Loom entry points prioritize replacement over the sticky tab. */
    initialTab?: DrilldownTab;
  }

  // ---- params → anchor cursor -------------------------------------------

  let { params: inputParams, docked = false, active = true, mobile = false, onclose }: { params?: unknown; docked?: boolean; active?: boolean; mobile?: boolean; onclose?: () => void } = $props();
  let inspectorEl: HTMLElement | null = $state(null);
  const params = $derived(inputParams as DrawerParams | null);

  function closeDetails(): void {
    if (onclose) onclose();
    else if (docked) hideTokenDetails();
    else closeDrawer();
  }

  function toggleDock(): void {
    const position = effCursor ? { turnIdx: effCursor.turnIdx, tokenIdx: effCursor.tokenIdx, isThinking: effCursor.seg === "thinking" } : params;
    if (docked) undockTokenDetails(position);
    else dockTokenDetails(position);
  }

  /** The position the user actually CLICKED — the drawer's anchor.  A
   *  fresh click (params object identity changes) snaps the cursor back
   *  here; the ↩ action does the same. */
  const anchor = $derived.by<TokenCursor | null>(() => {
    if (!params) return null;
    return {
      turnIdx: params.turnIdx,
      seg: params.isThinking ? "thinking" : "response",
      tokenIdx: params.tokenIdx,
    };
  });

  /** The walking cursor — every view reads the clamped ``effCursor``
   *  below; mutations write here. */
  let cursor = $state<TokenCursor | null>(null);

  /** Which branch we're inspecting — "primary" is the steered turn,
   * "shadow" the unsteered abPair when available.  Applies to the
   * cursor's current turn; walking into a different turn resets it. */
  type Branch = "primary" | "shadow";
  let branch: Branch = $state<Branch>("primary");

  const BRANCH_ITEMS: Array<{ value: Branch; label: string; title: string }> = [
    { value: "primary", label: "steered", title: "primary turn" },
    { value: "shadow", label: "unsteered", title: "A/B shadow turn" },
  ];

  $effect(() => {
    const a = anchor;
    cursor = a ? { ...a } : null;
    branch = "primary";
  });

  // ---- cursor resolution --------------------------------------------------

  /** The conversation with the shadow turn swapped in at the cursor's
   *  turn when the unsteered branch is selected. */
  const turnsView = $derived<ChatTurn[]>(
    chatLog.turns.map((t, i) =>
      i === (cursor?.turnIdx ?? -1) && branch === "shadow" && t.abPair
        ? t.abPair
        : t,
    ),
  );

  const segments = $derived(segmentsOf(turnsView));

  /** The effective inspected position — the cursor clamped onto the
   *  current segment list (token lists change while streaming). */
  const effCursor = $derived(cursor ? clampCursor(segments, cursor) : null);

  /** Reset the branch when the cursor crosses turns — the shadow pair is
   *  a per-turn artifact. */
  let lastBranchTurn = -1;
  $effect(() => {
    const t = effCursor?.turnIdx ?? -1;
    if (t !== lastBranchTurn) {
      lastBranchTurn = t;
      if (branch !== "primary") branch = "primary";
    }
  });

  /** The primary turn at the cursor (for abPair detection). */
  const turn = $derived<ChatTurn | null>(
    effCursor != null &&
      effCursor.turnIdx >= 0 &&
      effCursor.turnIdx < chatLog.turns.length
      ? chatLog.turns[effCursor.turnIdx]
      : null,
  );

  /** The turn actually inspected (shadow-swapped when selected). */
  const inspected = $derived<ChatTurn | null>(
    effCursor ? (turnsView[effCursor.turnIdx] ?? null) : null,
  );

  const tokenList = $derived<TokenScore[]>(
    effCursor ? segmentTokens(inspected, effCursor.seg) : [],
  );

  const token = $derived<TokenScore | null>(
    effCursor != null &&
      effCursor.tokenIdx >= 0 &&
      effCursor.tokenIdx < tokenList.length
      ? tokenList[effCursor.tokenIdx]
      : null,
  );

  const loomNodeId = $derived.by(() => {
    if (!effCursor) return null;
    if (branch === "shadow") {
      return inspected?.nodeId ?? null;
    }
    if (turn?.nodeId) return turn.nodeId;
    if (effCursor.turnIdx < 0 || loomTree.activePath.length === 0) return null;
    const visible = loomTree.activePath
      .map((id) => loomTree.nodes.get(id))
      .filter(Boolean)
      .filter((n) => !(n!.parent_id === null && n!.role === "system" && !n!.text));
    return visible[effCursor.turnIdx]?.id ?? null;
  });

  const hasReplayContext = $derived(
    loomNodeId != null && token?.rawIndex != null,
  );
  const canContinueFromToken = $derived(
    branch === "primary" &&
      loomNodeId != null &&
      token?.rawIndex != null &&
      token?.tokenId != null &&
      !genStatus.active,
  );
  const canReplaceToken = $derived(
    branch === "primary" &&
      loomNodeId != null &&
      token?.rawIndex != null &&
      !genStatus.active,
  );
  let continuingFromToken = $state(false);
  let continuationError = $state<string | null>(null);
  let replacementOpen = $state(false);
  let replacementText = $state("");
  let replacingToken = $state(false);
  let replacementInput = $state<HTMLInputElement | null>(null);

  $effect(() => {
    void loomNodeId;
    void token?.rawIndex;
    continuationError = null;
    replacementOpen = false;
    replacementText = "";
  });

  async function showReplacement(): Promise<void> {
    if (!canReplaceToken || !token) return;
    replacementText = token.text;
    replacementOpen = true;
    continuationError = null;
    await tick();
    replacementInput?.focus();
    const firstContent = replacementText.search(/\S/);
    const lastContent = replacementText.trimEnd().length;
    replacementInput?.setSelectionRange(
      firstContent >= 0 ? firstContent : 0,
      lastContent > 0 ? lastContent : replacementText.length,
    );
  }

  async function replaceToken(): Promise<void> {
    if (!canReplaceToken || !loomNodeId || token?.rawIndex == null) return;
    if (replacementText.length === 0) {
      continuationError = "Enter replacement text.";
      replacementInput?.focus();
      return;
    }
    continuationError = null;
    replacingToken = true;
    try {
      await sendTextFork(loomNodeId, token.rawIndex, replacementText);
      if (!docked || mobile) closeDetails();
      loomUiState.view = "map";
      window.dispatchEvent(new CustomEvent("drowse:workspace", { detail: "branches" }));
    } catch (error) {
      continuationError = userFacingError(
        error,
        "Unable to replace this token and start a new branch.",
      );
    } finally {
      replacingToken = false;
    }
  }

  async function continueFromToken(): Promise<void> {
    if (!canContinueFromToken || !loomNodeId || token?.rawIndex == null || token.tokenId == null) {
      return;
    }
    continuationError = null;
    continuingFromToken = true;
    try {
      await sendFork(loomNodeId, token.rawIndex, token.tokenId, true);
      if (!docked || mobile) closeDetails();
      loomUiState.view = "map";
      window.dispatchEvent(new CustomEvent("drowse:workspace", { detail: "branches" }));
    } catch (error) {
      continuationError = userFacingError(
        error,
        "Unable to continue this branch from the selected token.",
      );
    } finally {
      continuingFromToken = false;
    }
  }

  /** The durable generation record behind the inspected token. The
   *  detail view keeps this recipe beside the token rather than making
   *  users reconstruct it from the global controls. */
  const loomNode = $derived(
    loomNodeId ? (loomTree.nodes.get(loomNodeId) ?? null) : null,
  );
  const recipeSampling = $derived(loomNode?.recipe?.sampling ?? null);
  const recipeSteering = $derived(
    loomNode?.recipe?.steering ?? inspected?.appliedSteering ?? null,
  );
  function fmtSetting(value: number | null | undefined): string {
    if (value == null || !Number.isFinite(value)) return "-";
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }

  const recipeChips = $derived.by<string[]>(() => {
    const sampling = recipeSampling;
    const recipe = loomNode?.recipe;
    if (!sampling && !recipe) return [];
    const chips = [
      `Temperature ${fmtSetting(sampling?.temperature)}`,
      `Top P ${fmtSetting(sampling?.top_p)}`,
      `Top K ${fmtSetting(sampling?.top_k)}`,
      `Max tokens ${fmtSetting(sampling?.max_tokens)}`,
    ];
    const seed = recipe?.seed ?? sampling?.seed;
    if (seed != null) chips.push(`Seed ${seed}`);
    if (sampling?.presence_penalty) chips.push(`Presence penalty ${fmtSetting(sampling.presence_penalty)}`);
    if (sampling?.frequency_penalty) chips.push(`Frequency penalty ${fmtSetting(sampling.frequency_penalty)}`);
    if (sampling?.return_top_k != null) chips.push(`Return top K ${sampling.return_top_k}`);
    if (recipe?.thinking != null) chips.push(recipe.thinking ? "Thinking on" : "Thinking off");
    if ((recipe?.probes.length ?? 0) > 0) chips.push(`${recipe!.probes.length} recipe probes`);
    return chips;
  });

  // ---- navigation ---------------------------------------------------------

  function moveTo(next: TokenCursor | null): void {
    if (next) cursor = next;
  }

  const canStepBack = $derived(
    effCursor != null && stepCursor(segments, effCursor, -1) !== null,
  );
  const canStepFwd = $derived(
    effCursor != null && stepCursor(segments, effCursor, 1) !== null,
  );
  const canPrevTurn = $derived(
    effCursor != null && jumpTurn(segments, effCursor, -1) !== null,
  );
  const canNextTurn = $derived(
    effCursor != null && jumpTurn(segments, effCursor, 1) !== null,
  );

  function step(delta: 1 | -1): void {
    if (effCursor) moveTo(stepCursor(segments, effCursor, delta));
  }
  function turnHop(delta: 1 | -1): void {
    if (effCursor) moveTo(jumpTurn(segments, effCursor, delta));
  }
  function segEdge(where: "home" | "end"): void {
    if (!effCursor) return;
    cursor = {
      ...effCursor,
      tokenIdx: where === "home" ? 0 : Math.max(0, tokenList.length - 1),
    };
  }

  const atAnchor = $derived(
    effCursor != null &&
      anchor != null &&
      effCursor.turnIdx === anchor.turnIdx &&
      effCursor.seg === anchor.seg &&
      effCursor.tokenIdx === anchor.tokenIdx,
  );

  function resetToAnchor(): void {
    if (anchor) cursor = { ...anchor };
  }

  /** The counterpart segment of the current turn, when it exists —
   *  clicking the segment badge jumps to its start. */
  const otherSeg = $derived.by<SegmentKind | null>(() => {
    if (!effCursor) return null;
    const other: SegmentKind =
      effCursor.seg === "thinking" ? "response" : "thinking";
    return segments.some(
      (s) => s.turnIdx === effCursor.turnIdx && s.seg === other,
    )
      ? other
      : null;
  });

  function toggleSeg(): void {
    if (!effCursor || !otherSeg) return;
    cursor = { turnIdx: effCursor.turnIdx, seg: otherSeg, tokenIdx: 0 };
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.defaultPrevented || !active) return;
    if (mobile && !inspectorEl?.contains(ev.target as Node)) return;
    if (docked && !inspectorEl?.contains(ev.target as Node)) return;
    if (ev.key === "Escape") {
      ev.preventDefault();
      if (!docked || mobile) closeDetails();
      return;
    }
    // Never steal keys from a focusable field or the layer-strip
    // scrubbers (role="slider" owns its own arrow keys).
    const t = ev.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" ||
        t.tagName === "TEXTAREA" ||
        t.tagName === "SELECT" ||
        t.isContentEditable ||
        t.closest('[role="slider"], [role="listbox"]'))
    ) {
      return;
    }
    switch (ev.key) {
      case "ArrowLeft":
        step(-1);
        break;
      case "ArrowRight":
        step(1);
        break;
      case "ArrowUp":
        turnHop(-1);
        break;
      case "ArrowDown":
        turnHop(1);
        break;
      case "Home":
        segEdge("home");
        break;
      case "End":
        segEdge("end");
        break;
      default:
        return;
    }
    ev.preventDefault();
  }

  // ---- identity chips -------------------------------------------------

  const roleLabel = $derived(
    inspected ? (inspected.roleLabel ?? inspected.role) : "",
  );

  /** Rank of the chosen token within its captured alts, when present. */
  const chosenRank = $derived.by<number | null>(() => {
    const alts = token?.topAlts;
    if (!alts || alts.length === 0 || token?.tokenId == null) return null;
    const i = alts.findIndex((a) => a.id === token.tokenId);
    return i >= 0 ? i + 1 : null;
  });

  function fmtP(p: number): string {
    if (!Number.isFinite(p)) return "-";
    if (p >= 0.001) return p.toFixed(3);
    return p.toExponential(1);
  }

  // ---- tabs ----------------------------------------------------------

  const tabItems = $derived<Array<{
    value: DrilldownTab;
    label: string;
    meta: string;
    color?: string;
    title: string;
  }>>([
    {
      value: "geometry",
      label: "geometry",
      meta: String(Object.keys(token?.measurements?.instruments.geometry?.readings ?? {}).length),
      color: "var(--fg-dim)",
      title: "activation geometry",
    },
    {
      value: "logits",
      label: "logits",
      meta: String(token?.topAlts?.length ?? 0),
      title: "sampling alternatives",
    },
    {
      value: "sae",
      label: "sae",
      meta: String(token?.measurements?.instruments.sae?.readout?.features.length ?? 0),
      color: "var(--pillar-sae)",
      title: "sparse features",
    },
    {
      value: "lens",
      label: "j-lens",
      meta: String(token?.measurements?.instruments.lens?.readout?.layers.length ?? 0),
      color: "var(--pillar-lens)",
      title: "workspace readout",
    },
  ]);

  const hasAbPair = $derived(turn?.abPair != null);

  // ---- instrument readouts (captured-or-replay) ------------------------
  //
  // The token's loom-owned ``measurements`` envelope is authoritative;
  // replay covers old/missing channels and the explicit unsteered
  // counterfactual.  One ReplayReadout per family — the shell owns them
  // (and the steered flags) so tab switches don't lose state; the tabs
  // are presentational.

  const jlensFitted = $derived(sessionState.info?.jlens_fitted === true);
  const saeResident = $derived(saeLoaded());
  let saeAvailability = $state<CatalogInstrumentAvailability>("unknown");
  onMount(() => {
    let mounted = true;
    void currentCatalogInstrumentAvailability("sae").then((availability) => {
      if (mounted) saeAvailability = availability;
    });
    return () => {
      mounted = false;
    };
  });
  const lensTokenReplayAvailable = $derived(
    instrumentFamily("lens")?.capabilities.token_readout === true,
  );
  const saeTokenReplayAvailable = $derived(
    instrumentFamily("sae")?.capabilities.token_readout === true,
  );
  const geometryTokenReplayAvailable = $derived(
    instrumentFamily("geometry")?.capabilities.token_readout === true,
  );
  const lensReplayUnavailableReason =
    "J-lens token replay is unavailable in this runtime. J-lens data captured during generation can still be inspected.";
  const saeReplayUnavailableReason =
    "SAE token replay is unavailable in this runtime. Sparse-feature data captured during generation can still be inspected.";
  const geometryReplayUnavailableReason =
    "Geometry token replay is unavailable in this runtime. Geometry captured during generation can still be inspected.";

  // Share the logit-alternative width. Zero means the ordinary logit
  // capture is off, so retain the canonical eight-wide read-side view.
  const readoutTopK = $derived(resolveReadoutTopK(samplingState.return_top_k));

  const lensReadout = new ReplayReadout<LensTokenReadoutJSON>();
  const saeReadout = new ReplayReadout<SaeTokenReadoutJSON>();
  const geometryReadout = new ReplayReadout<GeometryTokenReadout>();

  const bodyStateKey = $derived.by(() => {
    const tab = drilldownUi.tab;
    if (!token || !effCursor) return params ? "missing:unavailable" : "missing:unselected";
    if (tab === "logits") return `${tab}:ready`;
    const readout = tab === "geometry"
      ? geometryReadout
      : tab === "sae"
        ? saeReadout
        : lensReadout;
    const state = readout.loading
      ? "loading"
      : readout.error
        ? "error"
        : readout.data
          ? "ready"
          : "empty";
    return `${tab}:${state}`;
  });

  onDestroy(() => {
    lensReadout.dispose();
    saeReadout.dispose();
    geometryReadout.dispose();
  });
  let lensSteered = $state(true);
  let saeSteered = $state(true);
  let geometrySteered = $state(true);

  /** ≥1 attached Monitor probe (anything in the rack that isn't a lens
   *  or SAE readout probe) — the geometry replay 400s on an empty roster. */
  const hasGeometryProbes = $derived(
    probeRack.active.some((name) => {
      return probeRack.entries.get(name)?.info.family === "geometry";
    }),
  );

  const lensPinned = $derived<Record<string, ScalarReadingJSON> | null>(
    token?.measurements?.instruments.lens?.readings ?? null,
  );
  const saePinned = $derived<Record<string, ScalarReadingJSON> | null>(
    token?.measurements?.instruments.sae?.readings ?? null,
  );

  const capturedLensData = $derived.by<LensTokenReadoutJSON | null>(() => {
    const lens = token?.measurements?.instruments.lens;
    if (!lens?.readout || !token) return null;
    return {
      node_id: loomNodeId ?? "",
      raw_index: token.rawIndex ?? -1,
      token_id: token.tokenId ?? -1,
      token_text: token.text,
      steering: lens.binding.steering,
      aggregate: lens.readout.aggregate,
      layers: lens.readout.layers,
    };
  });

  const capturedSaeData = $derived.by<SaeTokenReadoutJSON | null>(() => {
    const sae = token?.measurements?.instruments.sae;
    if (!sae?.readout || !token) return null;
    return {
      node_id: loomNodeId ?? "",
      raw_index: token.rawIndex ?? -1,
      token_id: token.tokenId ?? -1,
      token_text: token.text,
      steering: sae.binding.steering,
      layer: sae.binding.layer ?? -1,
      features: sae.readout.features,
    };
  });

  const capturedGeometryData = $derived.by<GeometryTokenReadout | null>(() => {
    const geometry = token?.measurements?.instruments.geometry;
    if (!geometry || Object.keys(geometry.readings ?? {}).length === 0) {
      return null;
    }
    return {
      steering: geometry.binding?.steering ?? null,
      readings: geometry.readings,
    };
  });

  let selectionAnchor = $state<DrawerParams | null>(null);
  $effect(() => {
    if (!active || !params || !token || !sessionState.info || selectionAnchor === params) return;
    drilldownUi.tab = availableDrilldownTab(
      params.initialTab ?? untrack(() => drilldownUi.tab),
      {
        lens: capturedLensData !== null || (hasReplayContext && jlensFitted && lensTokenReplayAvailable),
        sae: capturedSaeData !== null || (hasReplayContext && saeResident && saeTokenReplayAvailable),
        geometry: capturedGeometryData !== null || (hasReplayContext && hasGeometryProbes && geometryTokenReplayAvailable),
      },
    );
    selectionAnchor = params;
  });

  $effect(() => {
    if (!active || drilldownUi.tab !== "lens") return;
    const captured = capturedLensData;
    if (!lensTokenReplayAvailable && !lensSteered) lensSteered = true;
    if ((lensSteered || !lensTokenReplayAvailable) && captured) {
      lensReadout.adopt(
        captured,
        token?.measurements?.instruments.lens?.binding.source ?? null,
      );
      return;
    }
    lensReadout.clear();
    if (!lensTokenReplayAvailable) return;
    if (!jlensFitted) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    const replayTokenId = token?.tokenId ?? -1;
    const replayTokenText = token?.text ?? "";
    const fallbackSource =
      lensSourceState.sources.find((source) => source.active)?.source ?? null;
    lensReadout.replay(
      "lens",
      nodeId,
      rawIndex,
      { topK: readoutTopK, steered: lensSteered, raw: effectiveRawMode(), layers: "all" },
      (m) => {
        const lens = m.instruments.lens;
        if (!lens?.readout) {
          throw new Error("No J-lens reading was returned. Check the active lens source and try again.");
        }
        return {
          data: {
            node_id: nodeId,
            raw_index: rawIndex,
            token_id: replayTokenId,
            token_text: replayTokenText,
            steering: lens?.binding.steering ?? null,
            aggregate: lens.readout.aggregate,
            layers: lens.readout.layers,
          },
          source:
            lens?.binding.source ??
            fallbackSource ??
            null,
        };
      },
    );
  });

  $effect(() => {
    if (!active || drilldownUi.tab !== "sae") return;
    const captured = capturedSaeData;
    if (!saeTokenReplayAvailable && !saeSteered) saeSteered = true;
    if ((saeSteered || !saeTokenReplayAvailable) && captured) {
      saeReadout.adopt(
        captured,
        token?.measurements?.instruments.sae?.binding.source ?? null,
      );
      return;
    }
    saeReadout.clear();
    if (!saeTokenReplayAvailable) return;
    if (!saeResident) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    const replayTokenId = token?.tokenId ?? -1;
    const replayTokenText = token?.text ?? "";
    const fallbackSource =
      saeSourceState.sources.find((source) => source.active)?.source ??
      instrumentFamily("sae")?.source ??
      null;
    saeReadout.replay(
      "sae",
      nodeId,
      rawIndex,
      { topK: readoutTopK, steered: saeSteered, raw: effectiveRawMode() },
      (m) => {
        const sae = m.instruments.sae;
        if (!sae?.readout) {
          throw new Error("No SAE reading was returned. Check the active feature source and try again.");
        }
        return {
          data: {
            node_id: nodeId,
            raw_index: rawIndex,
            token_id: replayTokenId,
            token_text: replayTokenText,
            steering: sae?.binding.steering ?? null,
            layer: sae?.binding.layer ?? -1,
            features: sae.readout.features,
          },
          source:
            sae?.binding.source ??
            fallbackSource ??
            null,
        };
      },
    );
  });

  $effect(() => {
    if (!active || drilldownUi.tab !== "geometry") return;
    const captured = capturedGeometryData;
    if (!geometryTokenReplayAvailable && !geometrySteered) geometrySteered = true;
    if ((geometrySteered || !geometryTokenReplayAvailable) && captured) {
      geometryReadout.adopt(captured, null);
      return;
    }
    geometryReadout.clear();
    if (!geometryTokenReplayAvailable) return;
    if (!hasGeometryProbes) return;
    const nodeId = loomNodeId;
    const rawIndex = token?.rawIndex;
    if (!nodeId || rawIndex == null) return;
    geometryReadout.replay(
      "geometry",
      nodeId,
      rawIndex,
      { steered: geometrySteered, raw: effectiveRawMode() },
      (m) => {
        const geometry = m.instruments.geometry;
        if (!geometry) {
          throw new Error("No probe readings were returned. Check that the probes are attached and try again.");
        }
        return {
          data: {
            steering: geometry?.binding?.steering ?? null,
            readings: geometry?.readings ?? {},
          },
          source: null,
        };
      },
    );
  });
</script>

<svelte:window onkeydown={onKeydown} />

<aside data-morph-snapshot={effCursor ? `${effCursor.turnIdx}:${effCursor.seg}:${effCursor.tokenIdx}` : "none"}
  bind:this={inspectorEl}
  class="drawer"
  class:docked={docked || mobile}
  class:mobile
  data-token-details-scroll
  aria-label="Token drilldown"
  tabindex="-1"
>
  <header class="drawer-header">
    <div class="heading-row">
      {#if !mobile}
      <button type="button" class="dock-toggle" aria-pressed={docked}
        aria-label={docked ? "Undock token details" : "Dock token details"}
        {...{ "aria-description": (docked ? "Show token details in a window" : "Keep token details in a sidebar") }}
        onclick={toggleDock}>{#if docked}<PopoutIcon />{:else}<SidebarIcon side="right" />{/if}</button>
      {/if}
      <h2 class="eyebrow">Generated word details</h2>
      <DrawerCloseButton onclick={closeDetails} />
    </div>
    <div class="title">
      {#if token && effCursor}
        <div class="name-row">
          <code class="tok-text">
            <MorphText text={JSON.stringify(token.text)} numbers={false} identity={`${effCursor.turnIdx}:${effCursor.seg}:${effCursor.tokenIdx}`} />
          </code>
          <button
            type="button"
            class="kv-chip seg-chip"
            disabled={!otherSeg}
            onclick={toggleSeg}
            {...{ "aria-description": (otherSeg
              ? `Show the ${otherSeg} tokens from this same turn.`
              : undefined) }}
          >
            {#if sessionState.info?.is_base_model}Completion {effCursor.turnIdx}{:else}turn {effCursor.turnIdx} · {roleLabel} · {effCursor.seg}{/if}
          </button>
          {#if token.tokenId != null}
            <span class="kv-chip" {...{ "aria-description": "This token’s ID in the model’s vocabulary." }}>id {token.tokenId}</span>
          {/if}
          {#if token.rawIndex != null}
            <span class="kv-chip" {...{ "aria-description": "Position in the recorded generation, used to replay or branch from this token." }}>
              raw {token.rawIndex}
            </span>
          {:else}
            <span
              class="kv-chip warn"
              {...{ "aria-description": "The generation record is missing, so this token cannot be replayed or branched." }}
            >
              no replay
            </span>
          {/if}
          {#if token.logprob != null}
            <span
              class="kv-chip"
              {...{ "aria-description": "Probability after temperature and Top K / Top P filtering, not the model’s unmodified probability." }}
            >
              p <MorphText text={fmtP(Math.exp(token.logprob))} identity={`${effCursor.turnIdx}:${effCursor.seg}:${effCursor.tokenIdx}`} /> · logp <MorphText text={token.logprob.toFixed(3)} identity={`${effCursor.turnIdx}:${effCursor.seg}:${effCursor.tokenIdx}`} />{chosenRank !== null
                ? ` · rank ${chosenRank}/${token.topAlts?.length ?? 0}`
                : ""}
            </span>
          {/if}
        </div>
        <div class="nav-row">
          <span class="scrub">
            <button
              type="button"
              class="scrub-btn"
              disabled={!canStepBack}
              onclick={() => step(-1)}
              aria-label="Previous token"
              {...{ "aria-description": "Inspect the previous token" }}
            ><FluentIcon name="back" /></button>
            <span class="scrub-pos"><MorphText text={`${effCursor.tokenIdx + 1} / ${tokenList.length}`} /></span>
            <button
              type="button"
              class="scrub-btn"
              disabled={!canStepFwd}
              onclick={() => step(1)}
              aria-label="Next token"
              {...{ "aria-description": "Inspect the next token" }}
            ><FluentIcon name="next" /></button>
          </span>
          <span class="scrub">
            <button
              type="button"
              class="scrub-btn"
              disabled={!canPrevTurn}
              onclick={() => turnHop(-1)}
              aria-label="Previous turn"
              {...{ "aria-description": "Inspect the previous turn" }}
            ><FluentIcon name="up" /></button>
            <span class="scrub-pos"><MorphText text={`turn ${effCursor.turnIdx}`} /></span>
            <button
              type="button"
              class="scrub-btn"
              disabled={!canNextTurn}
              onclick={() => turnHop(1)}
              aria-label="Next turn"
              {...{ "aria-description": "Inspect the next turn" }}
            ><FluentIcon name="down" /></button>
          </span>
          {#if !atAnchor}
            <button
              type="button"
              class="scrub-btn scrub-home"
              onclick={resetToAnchor}
              aria-label="Return to the token you opened"

            ><FluentIcon name="return" /></button>
          {/if}
        </div>
        <div class="generation-context">
          <div class="recipe">
            <span class="recipe-label">generation recipe</span>
            <code class="recipe-steering" {...{ "aria-description": (recipeSteering ?? "no steering") }}>
              {recipeSteering ?? "unsteered"}
            </code>
            {#each recipeChips as chip (chip)}
              <span>{chip}</span>
            {/each}
          </div>
        </div>
      {:else}
        <div class="name-row">
          <span class="coord">
            {params ? "Selected token unavailable" : "No token selected"}
          </span>
        </div>
      {/if}
    </div>
  </header>

  {#if token && effCursor}
    <TokenRibbon
      tokens={tokenList}
      index={effCursor.tokenIdx}
      onjump={(i) => {
        if (effCursor) cursor = { ...effCursor, tokenIdx: i };
      }}
    />

    <section class="branch-point" aria-label="Selected token branch point">
      <div class="branch-point-copy">
        <strong>Branch from this token</strong>
        <span>Keep the text up to here and write a new ending. Your original {sessionState.info?.is_base_model ? "completion" : "response"} stays saved.</span>
      </div>
      <div class="branch-point-actions">
        <button
          type="button"
          class="secondary-action"
          disabled={!canReplaceToken || replacingToken}
          {...{ "aria-description": (canReplaceToken
            ? undefined
            : genStatus.active
              ? "Finish or stop the current generation first"
              : "This saved token does not have an exact replay boundary") }}
          onclick={() => {
            if (replacementOpen) {
              replacementOpen = false;
              continuationError = null;
            } else {
              void showReplacement();
            }
          }}
        >{replacementOpen ? "Cancel replacement" : "Replace token…"}</button>
        <button
          type="button"
          class="primary-action"
          disabled={!canContinueFromToken || continuingFromToken}
          {...{ "aria-description": (canContinueFromToken
            ? undefined
            : genStatus.active
              ? "Finish or stop the current generation first"
              : "This saved token does not have an exact replay boundary") }}
          onclick={() => void continueFromToken()}
        ><MorphText text={continuingFromToken ? "Starting…" : "Continue from here"} numbers={false} /></button>
      </div>
      {#if replacementOpen}
        <form class="replacement-form" onsubmit={(event) => {
          event.preventDefault();
          void replaceToken();
        }}>
          <label for="token-replacement">Replacement text</label>
          <div class="replacement-controls">
            <input
              id="token-replacement"
              bind:this={replacementInput}
              bind:value={replacementText}
              aria-describedby="token-replacement-help"
              autocomplete="off"
              spellcheck="true"
            />
            <button
              type="submit"
              class="primary-action"
              disabled={!canReplaceToken || replacingToken || replacementText.length === 0}
            ><MorphText text={replacingToken ? "Starting…" : "Start branch"} numbers={false} /></button>
          </div>
          <span id="token-replacement-help">
            Edit this token or replace it with a longer phrase. Spacing is preserved.
          </span>
        </form>
      {/if}
      {#if continuationError}
        <p class="branch-point-error" role="alert">{continuationError}</p>
      {/if}
    </section>
  {/if}

  <!-- View tabs + the steered/unsteered branch toggle when this turn has
       an A/B pair.  Tabs always render so users see the other views
       exist even when their capture is off. -->
  <div class="toolbar">
    <SegmentedTabs items={tabItems} bind:value={drilldownUi.tab} ariaLabel="Token detail view" />
    {#if hasAbPair}
      <SegmentedTabs items={BRANCH_ITEMS} bind:value={branch} ariaLabel="Token branch" />
    {/if}
  </div>

  <div class="body">
    {#key bodyStateKey}
      <div
        class="tab-content"
        in:selectionIn
      >
    {#if token && effCursor && drilldownUi.tab === "geometry" && !geometryTokenReplayAvailable && (hasGeometryProbes || capturedGeometryData)}
      <p class="replay-unavailable" role="note">{geometryReplayUnavailableReason}</p>
    {:else if token && effCursor && drilldownUi.tab === "lens" && !lensTokenReplayAvailable && (jlensFitted || capturedLensData)}
      <p class="replay-unavailable" role="note">{lensReplayUnavailableReason}</p>
    {:else if token && effCursor && drilldownUi.tab === "sae" && !saeTokenReplayAvailable && (saeResident || capturedSaeData)}
      <p class="replay-unavailable" role="note">{saeReplayUnavailableReason}</p>
    {/if}
    {#if !token || !effCursor}
      <div class="empty">{params ? "This token is no longer available. Select another token to inspect it." : "Select a token in your conversation or Loom to see its full details here."}</div>
    {:else if drilldownUi.tab === "geometry"}
      <GeometryTab
        readout={geometryReadout}
        returnToToken={{ turnIdx: effCursor.turnIdx, tokenIdx: effCursor.tokenIdx, isThinking: effCursor.seg === "thinking" }}
        bind:steered={geometrySteered}
        {hasGeometryProbes}
        {hasReplayContext}
        replayAvailable={geometryTokenReplayAvailable}
      />
    {:else if drilldownUi.tab === "logits"}
      <LogitsTab {token} nodeId={loomNodeId} />
    {:else if drilldownUi.tab === "sae"}
      <SaeTab
        readout={saeReadout}
        bind:steered={saeSteered}
        saeLoaded={saeResident}
        availability={saeAvailability}
        {hasReplayContext}
        pinned={saePinned}
        replayAvailable={saeTokenReplayAvailable}
      />
    {:else}
      <LensTab
        readout={lensReadout}
        bind:steered={lensSteered}
        {jlensFitted}
        {hasReplayContext}
        pinned={lensPinned}
        modelId={sessionState.info?.model_id ?? null}
        replayAvailable={lensTokenReplayAvailable}
      />
    {/if}
      </div>
    {/key}
  </div>

</aside>

<style>
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/token/expression sits in mono. */
  .drawer {
    display: block;
    height: 100%;
    min-height: 0;
    overflow-y: auto;
    overscroll-behavior-y: contain;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .drawer:focus {
    /* This scroll container is not a control; its children show keyboard focus. */
    outline: none !important;
  }

  .tab-content {
    display: grid;
    gap: var(--space-7);
    min-width: 0;
  }

  .drawer-header {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .heading-row { display: flex; align-items: center; gap: var(--space-3); width: 100%; min-width: 0; }
  .heading-row h2 { flex: 1; min-width: 0; margin: 0; }
  .dock-toggle {
    display: inline-grid;
    place-items: center;
    flex: none;
    width: var(--control-compact);
    height: var(--control-compact);
    padding: 0;
    border: 1px solid transparent;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-muted);
  }
  .dock-toggle:hover { background: var(--glass); color: var(--fg); }
  .dock-toggle[aria-pressed="true"] { background: var(--glass-strong); color: var(--accent); }
  .docked .eyebrow { font-size: var(--text); }
  .mobile .drawer-header { padding-top: 0; }
  .mobile :global(.drawer-close) { min-width: 44px; min-height: 44px; }
  .docked .branch-point { grid-template-columns: minmax(0, 1fr); }
  .docked .branch-point-actions { align-items: stretch; flex-direction: column; }
  .docked .replacement-controls { grid-template-columns: minmax(0, 1fr); }
  .docked .toolbar { align-items: flex-start; flex-direction: column; }
  .docked .toolbar :global(.sk-tabs) { flex-wrap: wrap; max-width: 100%; }
  .docked :global(.logit-grid),
  .docked :global(.geo-list),
  .docked :global(.aggregate-grid),
  .docked :global(.sae-list) { grid-template-columns: minmax(0, 1fr); }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-width: 0;
    flex: 1 1 auto;
  }
  .eyebrow {
    color: var(--fg);
    font-size: var(--text-lg);
    font-weight: var(--weight-medium);
    letter-spacing: 0;
  }
  .name-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
    flex-wrap: wrap;
  }
  .tok-text {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    background: var(--glass-strong);
    border-radius: var(--radius-sm);
    padding: var(--space-1) var(--space-3);
    word-break: break-all;
    max-width: 28ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .coord {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }

  /* Identity labels — the segment chip doubles as
   * the thinking/response jump when the turn has both. */
  .kv-chip {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    padding: var(--space-xs) 0;
    white-space: nowrap;
  }
  .kv-chip.warn {
    color: var(--fg-muted);
    font-style: italic;
  }
  button.seg-chip {
    min-height: var(--control-target);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  button.seg-chip:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  button.seg-chip:disabled {
    cursor: default;
  }

  .nav-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    flex-wrap: wrap;
  }
  .scrub {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .scrub-btn {
    min-width: var(--control-target);
    min-height: var(--control-target);
    background: var(--glass);
    color: var(--fg-muted);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font: inherit;
    font-size: var(--text-2xs);
    line-height: 1;
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .scrub-btn:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .scrub-btn:disabled {
    color: var(--fg-muted);
    opacity: var(--disabled-opacity);
    cursor: default;
  }
  .scrub-pos {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .scrub-home {
    color: var(--accent);
    font-size: var(--text-xs);
  }

  .generation-context {
    min-width: 0;
  }
  .recipe {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    flex-wrap: wrap;
    min-width: 0;
    margin: 0;
    padding: 0;
    border-radius: var(--radius);
    background: transparent;
  }
  .recipe-label {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    font-weight: var(--weight-medium);
    letter-spacing: 0;
    white-space: nowrap;
  }
  .recipe code {
    background: transparent;
    font-family: var(--font-mono);
  }
  .recipe {
    gap: var(--space-2) var(--space-3);
  }
  .recipe > span:not(.recipe-label),
  .recipe-steering {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .recipe-steering {
    color: var(--fg);
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 30ch;
  }

  /* Toolbar — view tabs left, branch toggle right (when A/B). */
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--space-5) var(--drawer-gutter-inline);
  }
  .branch-point {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-4) var(--space-6);
    margin: var(--space-4) var(--drawer-gutter-inline) 0;
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--input-well);
    box-shadow: var(--shadow-control);
    min-width: 0;
  }
  .branch-point-copy {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .branch-point-copy strong {
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .branch-point-copy span {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.45;
  }
  .branch-point-actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .branch-point-actions button {
    min-height: var(--control-target);
    padding: 0 var(--space-4);
    border: 1px solid transparent;
    border-radius: var(--radius);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .branch-point-actions .secondary-action {
    color: var(--fg-dim);
    background: var(--glass);
  }
  .branch-point-actions .primary-action {
    color: var(--action-ink);
    background: var(--action-bg);
  }
  .branch-point-actions button:hover:not(:disabled),
  .branch-point-actions button:focus-visible:not(:disabled) {
    filter: brightness(1.08);
    outline: none;
  }
  .branch-point-actions button:focus-visible {
    box-shadow: 0 0 0 2px var(--focus-ring);
  }
  .branch-point-actions button:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }
  .branch-point-actions button:disabled {
    opacity: var(--disabled-opacity);
    cursor: not-allowed;
  }
  .replacement-form {
    grid-column: 1 / -1;
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .replacement-form label {
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
  }
  .replacement-controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: var(--space-2);
  }
  .replacement-controls input {
    min-width: 0;
    min-height: var(--control-target);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
    padding: 0 var(--space-3);
    background: var(--input-well);
    color: var(--fg);
    font: inherit;
    font-family: var(--font-mono);
  }
  .replacement-controls input:focus-visible {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--focus-ring);
    outline: none;
  }
  .replacement-controls .primary-action {
    min-height: var(--control-target);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: 0 var(--space-4);
    color: var(--action-ink);
    background: var(--action-bg);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
  }
  .replacement-form > span {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .branch-point-error {
    grid-column: 1 / -1;
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-xs);
  }
  .replay-unavailable {
    margin: 0 0 var(--space-3);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--input-well);
    color: var(--fg-muted);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    line-height: 1.5;
  }

  .body {
    display: flex;
    flex-direction: column;
    gap: var(--space-6);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
    line-height: 1.5;
    max-width: 62ch;
  }

  @media (max-width: 820px) {
    .branch-point {
      grid-template-columns: minmax(0, 1fr);
    }
    .branch-point-actions {
      align-items: stretch;
      flex-direction: column;
    }
    .replacement-controls {
      grid-template-columns: 1fr;
    }
    .toolbar {
      align-items: flex-start;
      flex-direction: column;
    }
  }
</style>
