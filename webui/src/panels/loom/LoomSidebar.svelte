<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import { slidingSelection } from "../../lib/slidingSelection";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // Conversation loom. Turns advance from left to right and sibling
  // continuations separate vertically around their shared parent.
  //
  // Supports navigation, editing and regeneration, search, starring and notes,
  // compare pinning, fan-out, filtering, and cross-branch diff.

  import { onMount, tick, untrack } from "svelte";
  import { loomLiveNode } from "./loomLiveNode";
  import { MediaQuery, SvelteMap, SvelteSet } from "svelte/reactivity";
  import { fade, fly, slide } from "svelte/transition";
  import {
    applyTreeFilter,
    autoRegenState,
    chatLog,
    clearChat,
    clearNodeSelection,
    clearTreeFilter,
    currentRecipeOverride,
    edgeLabelCache,
    fetchEdgeLabel,
    filterState,
    highlightState,
    genStatus,
    loomContinueFromCommitted,
    loomTree,
    loomNodeIntersectsGeneration,
    LOOM_DELETE_DURING_GENERATION_MESSAGE,
    loomNavigate,
    loomEdit,
    loomBranch,
    loomSwapSeat,
    sessionState,
    loomDelete,
    loomStar,
    loomNote,
    loomRegenerateActive,
    loomUiState,
    nodeSelection,
    openDrawer,
    pinNodeForComparison,
    pinnedComparison,
    refreshLoomTree,
    sendFork,
    pushToast,
    toggleNodeSelection,
  } from "../../lib/stores.svelte";
  import LoomNode from "./LoomNode.svelte";
  import { segmentLoom } from "./loomSegments";
  import LoomWeave from "./LoomWeave.svelte";
  import {
    layoutLoom,
    loomBranchJunctions,
    loomMessageWidth,
    type LoomLayoutEdge,
  } from "./loomLayout";
  import { splitTokenSentences } from "./loomSentences";
  import {
    centerLoomRect,
    fitLoomCamera,
    LOOM_MAX_ZOOM,
    LOOM_MIN_ZOOM,
    resizeLoomCamera,
    zoomLoomAt,
    type LoomCamera,
  } from "./loomViewport";
  import Select from "../../lib/Select.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import type { LoomNodeJSON } from "../../lib/types";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { loomSearchExcerpt, loomTextMatches } from "../../lib/loomSearch";
  import { apiTree } from "../../lib/runtime/services";
  import { flushConversationAutosave, savedConversationState } from "../../lib/stores/savedConversations.svelte";
  import {
    pinchMetrics,
    scaleFromPinch,
    type GesturePoint,
  } from "../../lib/pointerGesture";
  import {
    modalIn,
    collapseIn,
    collapseOut,
    modalOut,
    scrimIn,
    scrimOut,
  } from "../../lib/motion";

  let { active = true, headersVisible = true }: { active?: boolean; headersVisible?: boolean } = $props();

  // ----------------------------------------- flat tree walk + depth --

  /** Depth-first walk of the tree starting at root.  Yields one row
   *  per node with its depth, so the renderer indents linearly. */
  interface Row {
    node: LoomNodeJSON;
    depth: number;
    isActivePath: boolean;
    isDead: boolean;
    /** Phase 5 filter dim: true iff the filter is on and this node
     *  doesn't match.  Rendered at 50% opacity — distinct visual
     *  channel from the 30% dead-branch dim. */
    filteredOut: boolean;
  }

  const activePathSet = $derived(new Set(loomTree.activePath));
  const selectionSet = $derived(new Set(nodeSelection.ids));
  const collapsedIds = new SvelteSet<string>();
  const sentenceBranchAvailable = true;

  // --------------------------------------- compare bar (active node) --
  //
  // A single capability-aware "compare": a generated node compares its
  // siblings; a committed node compares its generated children.  Seat does
  // not decide which branch is a model result.

  /** The turn whose generated continuations the compare acts on. */
  const compareUserParentId = $derived.by<string | null>(() => {
    const id = loomTree.active_node_id;
    const n = id ? (loomTree.nodes.get(id) ?? null) : null;
    if (!n) return null;
    if (n.recipe !== null) return n.parent_id;
    if (n.role !== "system") return n.id;
    return null;
  });

  const comparableNodes = $derived.by<LoomNodeJSON[]>(() => {
    if (!compareUserParentId) return [];
    return (loomTree.children_of.get(compareUserParentId) ?? [])
      .map((id) => loomTree.nodes.get(id))
      .filter((n): n is LoomNodeJSON => n != null && n.recipe !== null);
  });

  function compareBranch(): void {
    if (!compareUserParentId || comparableNodes.length < 2) return;
    openDrawer("node_compare", {
      node_ids: comparableNodes.map((n) => n.id),
      parent_id: compareUserParentId,
    });
  }

  /** Logit-pass: order children by ``mean_logprob`` when the sibling-sort
   *  filter directive is active.  Returns the children list in the order
   *  to *visit* (DFS pushes in reverse, so this is "first-out" order).
   *  Nodes without ``mean_logprob`` sink to the end and preserve their
   *  insertion-order tiebreak. */
  function _orderedChildren(parentId: string): string[] {
    const children = loomTree.children_of.get(parentId) ?? [];
    const mode = loomUiState.siblingSort;
    if (mode === "default" || children.length < 2) return children;
    // ``surprise`` first ⇒ most-surprising (lowest logprob) first.
    // ``confidence`` first ⇒ highest logprob first.
    const sign = mode === "surprise" ? 1 : -1;
    const indexed = children.map((id, idx) => ({
      id,
      idx,
      lp: loomTree.nodes.get(id)?.mean_logprob ?? null,
    }));
    indexed.sort((a, b) => {
      // Stable: null sinks; equal logprobs preserve insertion order.
      if (a.lp === null && b.lp === null) return a.idx - b.idx;
      if (a.lp === null) return 1;
      if (b.lp === null) return -1;
      const delta = sign * (a.lp - b.lp);
      return delta !== 0 ? delta : a.idx - b.idx;
    });
    return indexed.map((e) => e.id);
  }

  const rows = $derived.by<Row[]>(() => {
    const out: Row[] = [];
    if (!loomTree.root_id) return out;
    const matching = filterState.matchingIds;
    const liveTurn = genStatus.active && chatLog.pendingIndex !== null
      ? chatLog.turns[chatLog.pendingIndex] : null;
    // Touch the sort key so $derived re-runs when it flips.  Reading
    // ``loomUiState.siblingSort`` happens inside ``_orderedChildren``
    // already; this is the Svelte 5 idiom for keeping the dependency
    // explicit when the helper is called inside a tight loop.
    void loomUiState.siblingSort;
    const stack: { id: string; depth: number; deadAncestor: boolean }[] = [
      { id: loomTree.root_id, depth: 0, deadAncestor: false },
    ];
    while (stack.length) {
      const { id, depth, deadAncestor } = stack.pop()!;
      const savedNode = loomTree.nodes.get(id);
      if (!savedNode) continue;
      const node = liveTurn && loomTree.pendingNodeId === id
        ? loomLiveNode(savedNode, liveTurn) : savedNode;
      const onActive = activePathSet.has(id);
      const isDead = deadAncestor || !onActive;
      // Skip the synthetic root system node from the visible list —
      // it has no text and is just an anchor.
      if (!(node.parent_id === null && node.role === "system" && !node.text)) {
        const filteredOut = matching !== null && !matching.has(id);
        out.push({
          node,
          depth,
          isActivePath: onActive,
          isDead: isDead && !onActive,
          filteredOut,
        });
      }
      // Push children in reverse so DFS visits them in order, then
      // apply the optional sibling-sort.
      const children = matching === null && collapsedIds.has(id) ? [] : _orderedChildren(id);
      for (let i = children.length - 1; i >= 0; i--) {
        stack.push({
          id: children[i],
          depth: depth + (node.parent_id === null && node.role === "system" && !node.text ? 0 : 1),
          deadAncestor: deadAncestor || (!onActive && node.parent_id !== null),
        });
      }
    }
    return out;
  });

  const rowById = $derived(new Map(rows.map((row) => [row.node.id, row])));
  const coarsePointer = new MediaQuery("(pointer: coarse)");
  function nodeCardHeight(node: LoomNodeJSON, width: number): number {
    if (node.tokens && node.tokens.length > 0) {
      const availableWidth = width - 60;
      const sentences = splitTokenSentences(node.tokens);
      let rows = 0;
      for (const sentence of sentences) {
        rows += 1;
        let usedWidth = 0;
        for (const token of node.tokens.slice(sentence.start, sentence.end + 1)) {
          const label = (token.text || "∅").replace(/\s/g, " ");
          const tokenWidth = Math.min(availableWidth, Math.max(coarsePointer.current ? 24 : 18, 16 + label.length * 7));
          if (usedWidth > 0 && usedWidth + 3 + tokenWidth > availableWidth) {
            rows += 1;
            usedWidth = tokenWidth;
          } else {
            usedWidth += (usedWidth > 0 ? 3 : 0) + tokenWidth;
          }
        }
      }
      return coarsePointer.current
        ? 120 + rows * 47 + sentences.length * 56
        : 105 + rows * 27 + sentences.length * 38;
    }
    const textLength = (node.text ?? "").trim().length;
    const lines = Math.min(6, Math.max(1, Math.ceil(textLength / Math.max(1, (width - 40) / 7))));
    return 98 + lines * 18;
  }

  const segments = $derived(segmentLoom(rows.map(row => row.node), activePathSet));
  const measuredHeights = new SvelteMap<string, number>();
  $effect(() => {
    const ids = new Set(segments.map(segment => segment.id));
    for (const id of measuredHeights.keys()) {
      if (!ids.has(id)) measuredHeights.delete(id);
    }
  });
  const segmentById = $derived(new Map(segments.map(segment => [segment.id, segment])));
  const segmentWidths = $derived(new Map(segments.map(segment => {
    const tokens = segment.node.tokens?.slice(segment.start, segment.end);
    return [segment.id, loomMessageWidth(tokens?.length ? tokens.map(token => token.text).join("") : segment.node.text ?? "", tokens?.length ?? 0)];
  })));
  const mapRowById = $derived(new Map(segments.map(segment => [segment.id, {
    ...rowById.get(segment.node.id)!,
    depth: segment.depth,
    isActivePath: segment.memberIds.some(id => activePathSet.has(id)),
    isDead: segment.memberIds.every(id => rowById.get(id)!.isDead),
    filteredOut: segment.memberIds.every(id => rowById.get(id)!.filteredOut),
  }])));
  const graph = $derived.by(() => layoutLoom(segments.map((segment) => ({
    id: segment.id,
    parentId: segment.parentId,
    depth: segment.depth,
    width: segmentWidths.get(segment.id),
    height: measuredHeights.has(segment.id)
      ? measuredHeights.get(segment.id)!
      : nodeCardHeight({ ...segment.node, tokens: segment.node.tokens?.slice(segment.start, segment.end) ?? null }, segmentWidths.get(segment.id)!),
  }))));
  const branchPointCount = $derived(loomBranchJunctions(graph).length);
  const visibleTokenCount = $derived(segments.reduce(
    (total, segment) => total + segment.end - segment.start,
    0,
  ));
  const branchJunctions = $derived(loomBranchJunctions(graph));

  $effect(() => {
    if (!active || loomUiState.view !== "map" || genStatus.active) return;
    void loomTree.rev;
    const currentRows = rows;
    untrack(() => {
      for (const row of currentRows) {
        if (row.node.parent_id && row.node.recipe !== null) {
          fetchEdgeLabel(row.node.parent_id, row.node.id);
        }
      }
    });
  });

  function edgeIsActive(edge: LoomLayoutEdge): boolean {
    return Boolean(mapRowById.get(edge.parentId)?.isActivePath && mapRowById.get(edge.childId)?.isActivePath);
  }

  function edgeIsFiltered(edge: LoomLayoutEdge): boolean {
    return mapRowById.get(edge.childId)?.filteredOut ?? false;
  }

  function edgeStyle(edge: LoomLayoutEdge): string {
    const value = loomTree.nodes.get(edge.childId)?.mean_logprob;
    if (typeof value !== "number" || !Number.isFinite(value) || value > 0) return "";
    const intensity = 1 - Math.exp(value);
    return `--edge-width:${(1.25 + 1.75 * intensity).toFixed(2)}px;--edge-opacity:${(0.42 + 0.5 * intensity).toFixed(3)}`;
  }

  function divergenceLabel(node: LoomNodeJSON): string | null {
    if (!node.parent_id || !node.raw_token_ids?.length) return null;
    const siblings = (loomTree.children_of.get(node.parent_id) ?? [])
      .map((id) => loomTree.nodes.get(id))
      .filter((sibling): sibling is LoomNodeJSON =>
        sibling != null && sibling.id !== node.id && Boolean(sibling.raw_token_ids?.length)
      );
    if (siblings.length === 0) return null;

    let deepest = 0;
    for (const sibling of siblings) {
      const other = sibling.raw_token_ids!;
      const limit = Math.min(node.raw_token_ids.length, other.length);
      let index = 0;
      while (index < limit && node.raw_token_ids[index] === other[index]) index += 1;
      if (index < limit || node.raw_token_ids.length !== other.length) {
        deepest = Math.max(deepest, index);
      }
    }
    if (deepest === 0) return "alternate reply";
    const token = node.tokens?.find((row) => row.raw_index === deepest)?.text.trim();
    if (token) {
      const clean = token.replace(/\s+/g, " ").slice(0, 16);
      return `fork · “${clean}”`;
    }
    return `fork · token ${deepest + 1}`;
  }

  // ----------------------------------------- ring decoration --------

  /** Per-node ring fill keyed off the currently-selected highlight
   *  probe. Returns the node's
   *  aggregate reading for ``highlightState.target`` in [-1, 1], or
   *  ``null`` when no probe is selected, the node has no aggregate
   *  readings yet, or the selected probe is missing from this node's
   *  reading map.  ``LoomNode`` renders the ring only when this value
   *  is non-null. */
  function ringFor(node: LoomNodeJSON): number | null {
    const target = highlightState.target;
    if (!target) return null;
    const readings = node.aggregate_readings;
    if (!readings) return null;
    const v = readings[target];
    return typeof v === "number" ? v : null;
  }

  /** Logit-pass: the badge value to render on a node — the node's own
   *  ``mean_logprob``, or null when capture wasn't live (which suppresses
   *  the badge entirely in ``LoomNode``). */
  function weightBadgeFor(node: LoomNodeJSON): number | null {
    const v = node.mean_logprob;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }

  /** Steering-delta label for the edge into ``node``. Generated-only:
   *  committed / system nodes don't carry steering. */
  function steerLabelFor(node: LoomNodeJSON): string | null {
    if (!node.parent_id) return null;
    if (node.recipe === null) return null;
    return edgeLabelCache.get(`${node.parent_id}|${node.id}`) ?? null;
  }

  // ----------------------------------------- filter input ------------

  let searchInput: HTMLInputElement | null = $state(null);
  let searchNodeId = $state<string | null>(null);
  let searchSegmentId = $state<string | null>(null);
  let searchRootId = $state(loomTree.root_id);
  const searchResults = $derived(filterState.matchingIds === null ? [] : rows.filter(row => !row.filteredOut && (loomUiState.view !== "saved" || row.node.starred)));
  const searchIndex = $derived(searchResults.findIndex(row => row.node.id === searchNodeId));
  const searchResult = $derived(searchResults[Math.max(0, searchIndex)]?.node ?? null);
  const searchExcerpt = $derived(searchResult ? loomSearchExcerpt(searchResult.text ?? "", filterState.mode === "text" ? filterState.expr : "") : null);

  $effect(() => {
    void filterState.expr;
    void filterState.mode;
    searchNodeId = null;
    searchSegmentId = null;
  });
  $effect(() => {
    const rootId = loomTree.root_id;
    void loomTree.rev;
    untrack(() => {
      if (rootId !== searchRootId) {
        searchRootId = rootId;
        clearTreeFilter();
      } else if (filterState.expr.trim()) {
        void applyTreeFilter(filterState.expr);
      }
    });
  });

  function editFilter(value: string): void {
    clearTreeFilter();
    filterState.expr = value;
    if (filterState.mode === "text") void applyTreeFilter(value);
  }

  function changeSearchMode(mode: string): void {
    const expr = filterState.expr;
    clearTreeFilter();
    filterState.mode = mode === "advanced" ? "advanced" : "text";
    filterState.expr = expr;
    if (filterState.mode === "text") void applyTreeFilter(expr);
  }

  async function showSearchMatch(direction = 0): Promise<void> {
    if (!searchResults.length) return;
    const index = searchIndex < 0 ? (direction < 0 ? searchResults.length - 1 : 0)
      : direction === 0 ? searchIndex
      : (searchIndex + direction + searchResults.length) % searchResults.length;
    const node = searchResults[index].node;
    searchNodeId = node.id;
    loomUiState.view = "map";
    await tick();
    const candidates = segments.filter(segment => segment.memberIds.includes(node.id));
    const segment = candidates.find(segment => filterState.mode === "text" && loomTextMatches(
      segment.node.tokens?.slice(segment.start, segment.end).map(token => token.text).join("") || segment.node.text || "", filterState.expr,
    )) ?? candidates.find(segment => segment.id === node.id) ?? candidates[0];
    if (!segment) return;
    searchSegmentId = segment.id;
    centerNode(segment.id);
    await tick();
    const card = viewportEl?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(segment.id)}"]`);
    const sentence = [...(card?.querySelectorAll<HTMLElement>(".sentence-node") ?? [])].find(element =>
      filterState.mode === "text" && loomTextMatches(element.querySelector(".sentence-tokens")?.textContent ?? "", filterState.expr));
    const placed = graph.nodes.find(node => node.id === segment.id);
    if (sentence && card && placed && viewportEl) {
      const rect = sentence.getBoundingClientRect();
      const cardRect = card.getBoundingClientRect();
      const scale = cardRect.width / placed.width;
      camera = centerLoomRect(viewportSize(), {
        x: placed.x + (rect.left - cardRect.left) / scale,
        y: placed.y + (rect.top - cardRect.top) / scale,
        width: rect.width / scale,
        height: rect.height / scale,
      }, camera.zoom);
    }
  }

  function onFilterKey(ev: KeyboardEvent): void {
    if (ev.isComposing) return;
    if (ev.key === "Enter") {
      ev.preventDefault();
      ev.stopPropagation();
      if (filterState.mode === "advanced" && filterState.matchingIds === null) void applyTreeFilter(filterState.expr);
      else void showSearchMatch(ev.shiftKey ? -1 : 1);
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      clearTreeFilter();
    }
  }

  // ------------------------------------------- focus / keyboard nav --

  let focusedId: string | null = $state(null);
  type LoomView = "weave" | "map" | "path" | "options" | "saved";
  interface PathSnippet {
    key: string;
    node: LoomNodeJSON;
    sentenceIndex: number | null;
    start: number;
    end: number;
    text: string;
  }

  const cursorNode = $derived.by<LoomNodeJSON | null>(() => {
    const id = focusedId ?? loomTree.active_node_id;
    return id ? (loomTree.nodes.get(id) ?? null) : null;
  });
  const activePathRows = $derived.by<Row[]>(() => loomTree.activePath
    .map((id) => rowById.get(id))
    .filter((row): row is Row => row != null));
  const pathSnippets = $derived.by<PathSnippet[]>(() => {
    const snippets: PathSnippet[] = [];
    for (const row of activePathRows) {
      const node = row.node;
      const sentences = splitTokenSentences(node.tokens ?? []);
      if (sentences.length === 0) {
        snippets.push({
          key: node.id,
          node,
          sentenceIndex: null,
          start: 0,
          end: -1,
          text: node.text,
        });
        continue;
      }
      for (let index = 0; index < sentences.length; index += 1) {
        const sentence = sentences[index];
        snippets.push({
          key: `${node.id}:${sentence.start}:${sentence.end}`,
          node,
          sentenceIndex: index,
          start: sentence.start,
          end: sentence.end,
          text: sentence.text,
        });
      }
    }
    return snippets;
  });
  const optionNodes = $derived.by<LoomNodeJSON[]>(() => {
    const node = cursorNode;
    if (!node) return [];
    const parentId = node.recipe !== null && node.parent_id ? node.parent_id : node.id;
    return _orderedChildren(parentId)
      .map((id) => loomTree.nodes.get(id))
      .filter((child): child is LoomNodeJSON => child != null);
  });
  const savedRows = $derived(rows.filter((row) => row.node.starred && !row.filteredOut));
  let viewportEl: HTMLDivElement | null = $state(null);
  let lastViewportSize = { width: 0, height: 0 };
  let camera: LoomCamera = $state({ x: 0, y: 0, zoom: 1 });
  let cameraInitialized = $state(false);
  let cameraAnimating = $state(false);
  let cameraAnimationTimer: ReturnType<typeof setTimeout> | null = null;
  let dragging = $state(false);
  let dragPointerId: number | null = null;
  let dragOrigin = { x: 0, y: 0, cameraX: 0, cameraY: 0 };
  const touchPointers = new Map<number, GesturePoint>();
  let pinchGesture: {
    distance: number;
    zoom: number;
    worldX: number;
    worldY: number;
  } | null = null;
  let suppressNodeClickUntil = 0;

  const loomDepthStyle = $derived.by(() => {
    const nearScale = 1 + (camera.zoom - 1) * 0.32;
    const farScale = 1 + (camera.zoom - 1) * 0.12;
    const nearSpacing = 40 * nearScale;
    const farSpacing = 76 * farScale;
    return [
      `--loom-grid-size:${nearSpacing.toFixed(3)}px`,
      `--loom-grid-x:${((camera.x * 0.32) % nearSpacing).toFixed(3)}px`,
      `--loom-grid-y:${((camera.y * 0.32) % nearSpacing).toFixed(3)}px`,
      `--loom-far-size:${farSpacing.toFixed(3)}px`,
      `--loom-far-x:${((camera.x * 0.12) % farSpacing).toFixed(3)}px`,
      `--loom-far-y:${((camera.y * 0.12) % farSpacing).toFixed(3)}px`,
    ].join(";");
  });

  function setCameraMotion(enabled: boolean): void {
    if (cameraAnimationTimer !== null) {
      clearTimeout(cameraAnimationTimer);
      cameraAnimationTimer = null;
    }
    cameraAnimating = enabled;
    if (enabled) {
      cameraAnimationTimer = setTimeout(() => {
        cameraAnimating = false;
        cameraAnimationTimer = null;
      }, 220);
    }
  }

  function viewportSize(): { width: number; height: number } {
    return {
      width: viewportEl?.clientWidth ?? 0,
      height: viewportEl?.clientHeight ?? 0,
    };
  }

  function fitView(): void {
    if (!viewportEl || graph.nodes.length === 0) return;
    setCameraMotion(cameraInitialized);
    camera = fitLoomCamera(viewportSize(), graph);
    cameraInitialized = true;
    lastViewportSize = viewportSize();
  }

  function focusInitialView(): void {
    if (!active || loomUiState.view !== "map" || !viewportEl) return;
    camera.zoom = 0.9;
    if (loomTree.active_node_id) centerNode(loomTree.active_node_id);
    else fitView();
  }

  function zoomView(
    nextZoom: number,
    anchor?: { x: number; y: number },
    animate = true,
  ): void {
    const size = viewportSize();
    setCameraMotion(animate);
    camera = zoomLoomAt(camera, nextZoom, anchor ?? {
      x: size.width / 2,
      y: size.height / 2,
    });
    cameraInitialized = true;
  }

  function onViewportWheel(ev: WheelEvent): void {
    ev.preventDefault();
    if (ev.ctrlKey || ev.metaKey) {
      const rect = viewportEl?.getBoundingClientRect();
      if (!rect) return;
      const factor = Math.exp(-ev.deltaY * 0.002);
      zoomView(camera.zoom * factor, {
        x: ev.clientX - rect.left,
        y: ev.clientY - rect.top,
      }, false);
      return;
    }
    setCameraMotion(false);
    camera.x -= ev.shiftKey && ev.deltaX === 0 ? ev.deltaY : ev.deltaX;
    camera.y -= ev.shiftKey ? 0 : ev.deltaY;
    cameraInitialized = true;
  }

  function pointInViewport(ev: PointerEvent): GesturePoint | null {
    const rect = viewportEl?.getBoundingClientRect();
    if (!rect) return null;
    return {
      x: ev.clientX - rect.left,
      y: ev.clientY - rect.top,
    };
  }

  function captureViewportPointer(pointerId: number): void {
    if (!viewportEl) return;
    try {
      viewportEl.setPointerCapture(pointerId);
    } catch {
      // The pointer may already have ended between events.
    }
  }

  function releaseViewportPointer(pointerId: number): void {
    if (!viewportEl?.hasPointerCapture(pointerId)) return;
    try {
      viewportEl.releasePointerCapture(pointerId);
    } catch {
      // A cancelled pointer releases capture automatically.
    }
  }

  function touchPair(): [GesturePoint, GesturePoint] | null {
    const points = [...touchPointers.values()];
    return points.length >= 2 ? [points[0], points[1]] : null;
  }

  function beginPinch(): void {
    const pair = touchPair();
    if (!pair) return;
    const metrics = pinchMetrics(pair[0], pair[1]);
    if (metrics.distance <= 0) return;
    setCameraMotion(false);
    dragging = false;
    dragPointerId = null;
    pinchGesture = {
      distance: metrics.distance,
      zoom: camera.zoom,
      worldX: (metrics.center.x - camera.x) / camera.zoom,
      worldY: (metrics.center.y - camera.y) / camera.zoom,
    };
    suppressNodeClickUntil = performance.now() + 350;
    for (const pointerId of touchPointers.keys()) captureViewportPointer(pointerId);
  }

  function updatePinch(): void {
    const pair = touchPair();
    if (!pair || !pinchGesture) return;
    const metrics = pinchMetrics(pair[0], pair[1]);
    const zoom = scaleFromPinch(
      pinchGesture.zoom,
      pinchGesture.distance,
      metrics.distance,
      LOOM_MIN_ZOOM,
      LOOM_MAX_ZOOM,
    );
    camera = {
      x: metrics.center.x - pinchGesture.worldX * zoom,
      y: metrics.center.y - pinchGesture.worldY * zoom,
      zoom,
    };
    cameraInitialized = true;
  }

  function onViewportPointerDown(ev: PointerEvent): void {
    if (ev.button !== 0) return;
    const target = ev.target as HTMLElement | null;
    if (ev.pointerType === "touch") {
      const point = pointInViewport(ev);
      if (!point) return;
      touchPointers.set(ev.pointerId, point);
      if (touchPointers.size >= 2) {
        beginPinch();
        ev.preventDefault();
        return;
      }
    }
    if (target?.closest(".tree-node-wrap, .loom-view-controls")) return;
    setCameraMotion(false);
    dragging = true;
    dragPointerId = ev.pointerId;
    dragOrigin = {
      x: ev.clientX,
      y: ev.clientY,
      cameraX: camera.x,
      cameraY: camera.y,
    };
    captureViewportPointer(ev.pointerId);
    ev.preventDefault();
  }

  function onViewportPointerMove(ev: PointerEvent): void {
    if (ev.pointerType === "touch" && touchPointers.has(ev.pointerId)) {
      const point = pointInViewport(ev);
      if (point) touchPointers.set(ev.pointerId, point);
      if (pinchGesture && touchPointers.size >= 2) {
        updatePinch();
        suppressNodeClickUntil = performance.now() + 350;
        ev.preventDefault();
        return;
      }
    }
    if (!dragging || dragPointerId !== ev.pointerId) return;
    camera.x = dragOrigin.cameraX + ev.clientX - dragOrigin.x;
    camera.y = dragOrigin.cameraY + ev.clientY - dragOrigin.y;
    cameraInitialized = true;
    ev.preventDefault();
  }

  function endViewportGesture(ev: PointerEvent): void {
    releaseViewportPointer(ev.pointerId);
    if (ev.pointerType === "touch") {
      const wasPinching = pinchGesture !== null;
      touchPointers.delete(ev.pointerId);
      if (wasPinching) {
        if (touchPointers.size >= 2) beginPinch();
        else pinchGesture = null;
      }
    }
    if (dragPointerId === ev.pointerId) {
      dragging = false;
      dragPointerId = null;
    }
  }

  function toggleChildren(nodeId: string): void {
    if (collapsedIds.has(nodeId)) collapsedIds.delete(nodeId);
    else collapsedIds.add(nodeId);
    void tick().then(fitView);
  }

  let lastGraphNodeCount = 0;
  $effect(() => {
    const viewport = viewportEl;
    if (!active || loomUiState.view !== "map" || !viewport) return;
    const observer = new ResizeObserver(() => {
      const size = viewportSize();
      if (size.width === 0 || size.height === 0) return;
      if (!cameraInitialized) focusInitialView();
      else if (lastViewportSize.width > 0 && lastViewportSize.height > 0) {
        setCameraMotion(false);
        camera = resizeLoomCamera(camera, lastViewportSize, size);
      }
      lastViewportSize = size;
    });
    observer.observe(viewport);
    return () => observer.disconnect();
  });

  $effect(() => {
    if (!active || loomUiState.view !== "map") return;
    const nodeCount = graph.nodes.length;
    void graph.width;
    void graph.height;
    const previousCount = lastGraphNodeCount;
    lastGraphNodeCount = nodeCount;
    if (nodeCount === 0) {
      cameraInitialized = false;
      return;
    }
    void tick().then(() => {
      if (!cameraInitialized || previousCount === 0) focusInitialView();
      else if (nodeCount !== previousCount) centerCurrent();
    });
  });

  // When the active node changes, focus moves to it (only when there's
  // no manual focus selected yet).
  $effect(() => {
    const activeId = loomTree.active_node_id;
    const activeIsVisible = activeId !== null && rows.some((row) => row.node.id === activeId);
    const focusedIsVisible = focusedId !== null && rows.some((row) => row.node.id === focusedId);
    if (!focusedIsVisible) {
      focusedId = activeIsVisible ? activeId : (rows[0]?.node.id ?? null);
    }
  });

  function focusedIndex(): number {
    if (!focusedId) return -1;
    return rows.findIndex((r) => r.node.id === focusedId);
  }

  async function focusNode(nodeId: string, scroll = true): Promise<void> {
    focusedId = nodeId;
    await tick();
    const el = viewportEl?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(nodeId)}"]`);
    if (!el) return;
    el.focus({ preventScroll: true });
    if (scroll) centerNode(nodeId);
  }

  function centerNode(nodeId: string): void {
    const placed = graph.nodes.find((node) => node.id === nodeId);
    if (!placed || !viewportEl) return;
    setCameraMotion(cameraInitialized);
    camera = centerLoomRect(viewportSize(), {
      x: placed.x,
      y: placed.y,
      width: placed.width,
      height: placed.height,
    }, camera.zoom);
    cameraInitialized = true;
    lastViewportSize = viewportSize();
  }

  function centerCurrent(): void {
    if (loomTree.active_node_id) centerNode(loomTree.active_node_id);
  }

  function setLoomView(view: LoomView): void {
    loomUiState.view = view;
    closeMenu(false);
    if (view === "map") {
      void tick().then(() => {
        if (loomTree.active_node_id) void focusNode(loomTree.active_node_id, false);
        focusInitialView();
      });
    }
  }

  function readableNodeText(node: LoomNodeJSON): string {
    return (node.text ?? "").replace(/\s+/g, " ").trim() || "Empty turn";
  }

  function nodeRole(node: LoomNodeJSON): string {
    return node.role_label?.trim() || node.role;
  }

  async function usePath(node: LoomNodeJSON): Promise<void> {
    focusedId = node.id;
    if (loomTree.active_node_id !== node.id) await loomNavigate(node.id);
  }

  function moveVisible(dir: -1 | 1): void {
    const index = focusedIndex();
    const next = rows[index + dir];
    if (next) void focusNode(next.node.id);
  }

  function moveParent(): void {
    if (!focusedId) return;
    const parentId = loomTree.nodes.get(focusedId)?.parent_id;
    if (parentId && rows.some((row) => row.node.id === parentId)) {
      void focusNode(parentId);
    }
  }

  function moveFirstChild(): void {
    if (!focusedId) return;
    const firstChild = _orderedChildren(focusedId)[0];
    if (firstChild && rows.some((row) => row.node.id === firstChild)) {
      void focusNode(firstChild);
    }
  }

  // ----------------------------------------- context menu state --

  interface MenuState {
    open: boolean;
    nodeId: string | null;
  }
  let menu: MenuState = $state({ open: false, nodeId: null });
  let menuEl: HTMLElement | null = $state(null);
  let menuPosition = $state({ x: 12, y: 12 });
  let menuPositioned = $state(false);
  let menuOpener: HTMLElement | null = null;
  let menuTrackingFrame: number | null = null;

  function portal(node: HTMLElement): { destroy: () => void } {
    document.body.appendChild(node);
    return {
      destroy: () => node.remove(),
    };
  }

  function openMenu(ev: MouseEvent, nodeId: string): void {
    ev.preventDefault();
    ev.stopPropagation();
    const opener = (ev.currentTarget as HTMLElement | null) ??
      viewportEl?.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(nodeId)}"]`) ?? null;
    void showMenu(nodeId, opener);
  }

  async function showMenu(
    nodeId: string,
    opener: HTMLElement | null,
  ): Promise<void> {
    menuOpener = opener;
    menuPositioned = false;
    focusedId = nodeId;
    setCameraMotion(false);
    await tick();
    menu = {
      open: true,
      nodeId,
    };
    await tick();
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    positionOpenMenu();
    await tick();
    menuPositioned = true;
    trackOpenMenu();
    await tick();
    menuItems()[0]?.focus();
  }

  function positionOpenMenu(): void {
    if (!menuEl || !menu.nodeId) return;
    // Resolve the visible tree item on every frame. The canvas is transformed
    // for pan/zoom, so a stale trigger or its untransformed wrapper can place
    // the fixed, portalled menu hundreds of pixels away from the card.
    const card = viewportEl?.querySelector<HTMLElement>(
      `[data-node-id="${CSS.escape(menu.nodeId)}"]`,
    ) ?? (menuOpener?.isConnected ? menuOpener : null);
    const anchor = card?.getBoundingClientRect();
    const margin = 12;
    const gap = 12;
    const menuRect = { width: menuEl.offsetWidth, height: menuEl.offsetHeight };
    const anchorRect = anchor ?? {
      left: margin,
      right: margin,
      top: margin,
      bottom: margin,
    };
    let x = anchorRect.right + gap;
    let y = anchorRect.top;
    if (x + menuRect.width > window.innerWidth - margin) {
      x = anchorRect.left - menuRect.width - gap;
    }
    if (x < margin) {
      x = anchorRect.left;
      y = anchorRect.bottom + gap;
    }
    if (y + menuRect.height > window.innerHeight - margin) {
      y = Math.max(margin, window.innerHeight - menuRect.height - margin);
    }
    const nextX = Math.max(margin, Math.min(x, window.innerWidth - menuRect.width - margin));
    const nextY = Math.max(margin, y);
    if (
      Math.abs(menuPosition.x - nextX) > 0.5 ||
      Math.abs(menuPosition.y - nextY) > 0.5
    ) {
      menuPosition = { x: nextX, y: nextY };
    }
  }

  function trackOpenMenu(): void {
    if (menuTrackingFrame !== null) cancelAnimationFrame(menuTrackingFrame);
    const update = () => {
      if (!menu.open) {
        menuTrackingFrame = null;
        return;
      }
      positionOpenMenu();
      menuTrackingFrame = requestAnimationFrame(update);
    };
    menuTrackingFrame = requestAnimationFrame(update);
  }

  function openMenuFromKeyboard(nodeId: string): void {
    const opener = viewportEl?.querySelector<HTMLElement>(
      `[data-node-id="${CSS.escape(nodeId)}"]`,
    ) ?? null;
    void showMenu(nodeId, opener);
  }

  function menuItems(): HTMLButtonElement[] {
    if (!menuEl) return [];
    return Array.from(
      menuEl.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)'),
    );
  }

  function onMenuKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closeMenu();
      return;
    }
    if (ev.key === "Tab") {
      closeMenu(false);
      return;
    }
    const items = menuItems();
    if (items.length === 0) return;
    const index = Math.max(0, items.indexOf(document.activeElement as HTMLButtonElement));
    let next = -1;
    if (ev.key === "ArrowDown") next = (index + 1) % items.length;
    else if (ev.key === "ArrowUp") next = (index - 1 + items.length) % items.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = items.length - 1;
    if (next >= 0) {
      ev.preventDefault();
      ev.stopPropagation();
      items[next].focus();
    }
  }

  function closeMenu(restoreFocus = true): void {
    const opener = menuOpener;
    const nodeId = menu.nodeId;
    menu = { open: false, nodeId: null };
    menuPositioned = false;
    if (menuTrackingFrame !== null) cancelAnimationFrame(menuTrackingFrame);
    menuTrackingFrame = null;
    menuEl = null;
    if (restoreFocus) {
      void tick().then(() => {
        if (opener?.isConnected) opener.focus({ preventScroll: true });
        else if (nodeId) void focusNode(nodeId, false);
      });
    }
  }

  // ----------------------------------------- external modal requests --

  /** When App.svelte fires Ctrl+R/E/B/N/D it bumps
   *  ``loomUiState.modalRequest.seq``.  Mirror that into the local
   *  modal state. */
  let _lastSeenSeq = $state(0);
  $effect(() => {
    const req = loomUiState.modalRequest;
    if (req.seq !== _lastSeenSeq && req.kind) {
      _lastSeenSeq = req.seq;
      void openModal(req.kind, req.nodeId, req.text, req.n);
    }
  });

  // ----------------------------------------- modals --

  interface ModalState {
    kind:
      | null
      | "regenerate"
      | "edit"
      | "branch"
      | "delete"
      | "clear"
      | "note"
      | "navpicker"
      | "search"
      | "fanout"
      | "regen_mode";
    nodeId: string | null;
    text: string;
    n: number;
    /** Steering selector for the fan-out modal. */
    vector?: string;
    /** Mode for the regen-with-modifier modal. */
    mode?: string;
    /** Inline validation error.  Non-empty means commit failed; the
     *  modal stays open with this message rendered below the input. */
    error?: string;
  }
  let modal: ModalState = $state({
    kind: null,
    nodeId: null,
    text: "",
    n: 1,
    vector: "",
    mode: "unsteered",
    error: "",
  });
  /** Focusable ref shared by every modal input variant — the regenerate
   *  / regen_mode number-of-siblings field swaps in a themed
   *  ``NumberInput`` which exposes a structurally-matching
   *  ``focus()`` / ``select()`` pair, so the modal-open auto-focus
   *  works regardless of which branch is rendered. */
  type FocusableRef = {
    focus: () => void;
    select?: () => void;
  };
  let modalInput: FocusableRef | null = $state(null);
  let modalEl: HTMLElement | null = $state(null);
  let modalCancelButton: HTMLButtonElement | null = $state(null);
  let modalTrigger: HTMLElement | null = null;
  let asideEl: HTMLElement | null = $state(null);
  let backgroundStates: Array<{
    element: HTMLElement;
    inert: boolean;
    ariaHidden: string | null;
  }> = [];

  function hideModalBackground(): void {
    if (backgroundStates.length > 0) return;
    const zone = asideEl?.closest<HTMLElement>(".loom-zone");
    const layout = zone?.parentElement;
    if (!zone || !layout) return;
    backgroundStates = Array.from(layout.children)
      .filter((element): element is HTMLElement => element instanceof HTMLElement && element !== zone)
      .map((element) => ({
        element,
        inert: element.hasAttribute("inert"),
        ariaHidden: element.getAttribute("aria-hidden"),
      }));
    for (const state of backgroundStates) {
      state.element.setAttribute("inert", "");
      state.element.setAttribute("aria-hidden", "true");
    }
  }

  function restoreModalBackground(): void {
    for (const state of backgroundStates) {
      if (!state.element.isConnected) continue;
      if (!state.inert) state.element.removeAttribute("inert");
      if (state.ariaHidden === null) state.element.removeAttribute("aria-hidden");
      else state.element.setAttribute("aria-hidden", state.ariaHidden);
    }
    backgroundStates = [];
  }

  function focusInitialModalControl(): void {
    if (modalInput) {
      modalInput.focus();
      modalInput.select?.();
    } else {
      modalCancelButton?.focus();
    }
  }

  async function openModal(
    kind: ModalState["kind"],
    nodeId: string | null,
    initialText: string = "",
    initialN: number = 1,
    trigger?: HTMLElement | null,
  ): Promise<void> {
    if (kind === "delete" && nodeId && loomNodeIntersectsGeneration(nodeId)) {
      pushToast(LOOM_DELETE_DURING_GENERATION_MESSAGE, { kind: "warning" });
      return;
    }
    modalTrigger = trigger ??
      (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    modalInput = null;
    modal = {
      kind,
      nodeId,
      text: initialText,
      n: initialN,
      vector: "",
      mode: "unsteered",
      error: "",
    };
    await tick();
    focusInitialModalControl();
    asideEl?.setAttribute("inert", "");
    asideEl?.setAttribute("aria-hidden", "true");
    hideModalBackground();
  }

  function closeModal(): void {
    const trigger = modalTrigger;
    asideEl?.removeAttribute("inert");
    asideEl?.removeAttribute("aria-hidden");
    restoreModalBackground();
    modal = {
      kind: null,
      nodeId: null,
      text: "",
      n: 1,
      vector: "",
      mode: "unsteered",
      error: "",
    };
    modalInput = null;
    modalEl = null;
    modalCancelButton = null;
    modalTrigger = null;
    void tick().then(() => {
      if (trigger?.isConnected) trigger.focus({ preventScroll: true });
      else {
        const fallback = focusedId && rows.some((row) => row.node.id === focusedId)
          ? focusedId
          : (loomTree.active_node_id ?? rows[0]?.node.id ?? null);
        if (fallback) void focusNode(fallback, false);
      }
    });
  }

  function modalFocusable(): HTMLElement[] {
    if (!modalEl) return [];
    return Array.from(
      modalEl.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), [tabindex]:not([tabindex="-1"])',
      ),
    );
  }

  function onModalKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closeModal();
      return;
    }
    if (ev.key !== "Tab") return;
    const focusable = modalFocusable();
    if (focusable.length === 0) {
      ev.preventDefault();
      modalEl?.focus();
      return;
    }
    const active = document.activeElement as HTMLElement | null;
    const index = active ? focusable.indexOf(active) : -1;
    if (ev.shiftKey && index <= 0) {
      ev.preventDefault();
      focusable[focusable.length - 1].focus();
    } else if (!ev.shiftKey && (index < 0 || index === focusable.length - 1)) {
      ev.preventDefault();
      focusable[0].focus();
    }
  }

  function setModalError(message: string): void {
    // Clear input focus so the error reads; modal stays open.
    modal = { ...modal, error: message };
  }

  let modalBusy = $state(false);
  let createdBranchId = $state<string | null>(null);
  async function commitModal(): Promise<void> {
    if (modalBusy) return;
    modalBusy = true;
    try { await applyModal(); }
    finally { modalBusy = false; }
  }

  async function applyModal(): Promise<void> {
    const m = modal;
    if (!m.kind || !m.nodeId) return closeModal();
    switch (m.kind) {
      case "clear":
        if (genStatus.active) { setModalError("Stop the current reply before clearing the loom."); return; }
        try {
          await flushConversationAutosave();
          await apiTree.reset();
          savedConversationState.activeId = null;
          savedConversationState.avatarSeed = null;
          savedConversationState.accent = "purple";
          await refreshLoomTree();
          clearNodeSelection();
          clearTreeFilter();
          pushToast("Loom cleared. The previous conversation is still in Saved chats.", { kind: "info" });
        } catch (error) {
          setModalError(userFacingError(error, "The loom could not be cleared. Your saved chats were not deleted."));
          return;
        }
        break;
      case "regenerate":
        await loomRegenerateActive(Math.max(1, Math.floor(m.n)));
        break;
      case "edit":
        await loomEdit(m.nodeId, m.text);
        break;
      case "branch": {
        const newId = await loomBranch(m.nodeId, m.text);
        if (newId) { await loomNavigate(newId); createdBranchId = newId; }
        break;
      }
      case "delete":
        if (loomNodeIntersectsGeneration(m.nodeId)) {
          setModalError(LOOM_DELETE_DURING_GENERATION_MESSAGE);
          return;
        }
        if (!await loomDelete(m.nodeId)) {
          setModalError(loomTree.error ?? "The branch could not be removed. Try again.");
          return;
        }
        break;
      case "note":
        await loomNote(m.nodeId, m.text);
        break;
      case "navpicker": {
        // Resolve a node by id-prefix or full id.  Ambiguity keeps the
        // modal open with a list of matches so the user can disambiguate
        // rather than navigating to a random first match (Map insertion
        // order leaks the bug otherwise).
        const r = resolveByPrefix(m.text);
        if (r.id) {
          await loomNavigate(r.id);
          break;
        }
        if (r.matches.length === 0) {
          setModalError(`no node matches '${m.text}'`);
          return;
        }
        const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
        setModalError(
          `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")",
        );
        return;
      }
      case "search": {
        await applyTreeFilter(m.text, "text");
        if (!filterState.matchingIds?.size) {
          setModalError(`No text match for '${m.text}'.`);
          return;
        }
        closeModal();
        loomUiState.view = "map";
        await tick();
        searchInput?.focus();
        await showSearchMatch();
        return;
      }
      case "fanout": {
        // Phase 5 fan-out: anchor on the user node, send one regen
        // per alpha as a sibling.  We use a sequential dispatch
        // rather than the engine's ``generate_sweep`` because the
        // active rack already carries the rest of the steering
        // context — we only need to overlay the swept vector's α
        // per call.
        const vector = (m.vector ?? "").trim();
        if (!vector) {
          setModalError("vector name required");
          return;
        }
        const alphas = parseAlphaList(m.text);
        if (alphas.length === 0) {
          setModalError(
            "couldn't parse alphas, try a comma list (0.0, 0.3, 0.7) " +
            "or linspace(-1, 1, 5)",
          );
          return;
        }
        const userId = m.nodeId;
        for (const alpha of alphas) {
          // Recipe override carries the per-row alpha for the swept
          // vector.  The engine's modifier resolver accepts a partial
          // recipe expression on the ``steering`` axis.
          await loomContinueFromCommitted(userId, {
            n: 1,
            recipe_override: `steering=${alpha} ${vector}`,
          });
        }
        break;
      }
      case "regen_mode": {
        // Manual regen-with-modifier: regenerate a generated node, or
        // author fresh continuations under a committed node, and dispatch N
        // siblings under the
        // chosen mode.  Modes are resolved engine-side.
        const node = loomTree.nodes.get(m.nodeId);
        const mode = (m.mode ?? "unsteered").trim();
        const N = Math.max(1, Math.floor(m.n));
        if (node?.recipe === null) {
          await loomContinueFromCommitted(m.nodeId, {
            n: N,
            recipe_override: mode,
          });
        } else {
          // Generated node — set it active then regenerate in its own seat.
          if (loomTree.active_node_id !== m.nodeId) {
            await loomNavigate(m.nodeId);
          }
          await loomRegenerateActive(N, { recipe_override: mode });
        }
        break;
      }
    }
    closeModal();
  }

  // ----------------------------------------- alpha-list parser ------

  /** Parse a fan-out alpha string (comma list / ``linspace(a, b, n)`` /
   *  ``start:stop:step``) into ``number[]``. */
  function parseAlphaList(raw: string): number[] {
    const trimmed = raw.trim();
    if (!trimmed) return [];

    const linspaceMatch = trimmed.match(
      /^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$/i,
    );
    if (linspaceMatch) {
      const start = Number(linspaceMatch[1]);
      const stop = Number(linspaceMatch[2]);
      const count = Number(linspaceMatch[3]);
      if (!Number.isFinite(start) || !Number.isFinite(stop)) return [];
      if (!Number.isInteger(count) || count < 1) return [];
      if (count === 1) return [start];
      const step = (stop - start) / (count - 1);
      const out: number[] = [];
      for (let i = 0; i < count; i++) out.push(start + step * i);
      return out;
    }

    if (trimmed.includes(":")) {
      const parts = trimmed.split(":").map((p) => p.trim());
      if (parts.length !== 3) return [];
      const start = Number(parts[0]);
      const stop = Number(parts[1]);
      const step = Number(parts[2]);
      if (![start, stop, step].every(Number.isFinite)) return [];
      if (step === 0) return [];
      if ((stop - start) * step < 0) return [];
      const out: number[] = [];
      const eps = Math.abs(step) * 1e-9;
      const ascending = step > 0;
      let v = start;
      let guard = 0;
      while (
        (ascending ? v <= stop + eps : v >= stop - eps) &&
        guard++ < 10000
      ) {
        out.push(Number.parseFloat(v.toPrecision(12)));
        v += step;
      }
      if (guard >= 10000) return [];
      return out;
    }

    const out: number[] = [];
    for (const part of trimmed.split(",")) {
      const t = part.trim();
      if (!t) continue;
      const v = Number(t);
      if (!Number.isFinite(v)) return [];
      out.push(v);
    }
    return out;
  }

  interface PrefixResolution {
    /** Single unambiguous match — caller can navigate immediately. */
    id: string | null;
    /** Every node id whose ulid starts with the prefix.  ``length > 1``
     *  signals ambiguity; ``length == 0`` signals no match. */
    matches: string[];
  }

  function resolveByPrefix(prefix: string): PrefixResolution {
    const p = prefix.trim();
    if (!p) return { id: null, matches: [] };
    if (p === "root") {
      const root = loomTree.root_id;
      return root ? { id: root, matches: [root] } : { id: null, matches: [] };
    }
    const matches: string[] = [];
    for (const id of loomTree.nodes.keys()) {
      if (id === p) return { id, matches: [id] };  // exact match wins
      if (id.startsWith(p)) matches.push(id);
    }
    if (matches.length === 1) return { id: matches[0], matches };
    return { id: null, matches };
  }

  // --------------------------------------- click + global keys --

  function onNodeClick(node: LoomNodeJSON, ev: MouseEvent): void {
    if (performance.now() < suppressNodeClickUntil) {
      ev.preventDefault();
      ev.stopPropagation();
      return;
    }
    // Ctrl/Cmd-click on a generated node toggles its multi-select
    // membership for the cross-branch diff drawer.  Plain click
    // still navigates.
    if ((ev.ctrlKey || ev.metaKey) && node.recipe !== null) {
      ev.preventDefault();
      ev.stopPropagation();
      toggleNodeSelection(node.id);
      void focusNode(node.id, false);
      return;
    }
    void focusNode(node.id, false);
    void loomNavigate(node.id);
  }

  async function growFromNode(node: LoomNodeJSON): Promise<void> {
    if (node.recipe === null) {
      await loomContinueFromCommitted(node.id, { n: 1 });
      return;
    }
    if (loomTree.active_node_id !== node.id) await loomNavigate(node.id);
    await loomRegenerateActive(1);
  }

  function branchFromNode(node: LoomNodeJSON, trigger: HTMLElement | null): void {
    void openModal("branch", node.id, node.text ?? "", 1, trigger);
  }

  /** Keep the generated prefix through this token and resample everything
   * after it. Passing the boundary's original token id through the existing
   * fork protocol preserves every token up to the branch point exactly. */
  async function branchAfterToken(node: LoomNodeJSON, tokenIndex: number): Promise<void> {
    if (genStatus.active) {
      pushToast("Finish or stop the current reply before starting another branch.");
      return;
    }
    const token = node.tokens?.[tokenIndex];
    if (token?.raw_index == null || token.token_id == null) {
      pushToast("This saved reply does not contain an exact token boundary.", { kind: "error" });
      return;
    }
    try {
      await sendFork(node.id, token.raw_index, token.token_id, true);
    } catch (error) {
      pushToast(userFacingError(
        error,
        "A new continuation could not be started from that word.",
      ), { kind: "error" });
    }
  }

  /** A Loom token is a branch point, not an immediate generation command.
   * Move onto its saved path, then open the shared token workbench at the
   * exact token so replacement, readouts, and continuation remain separate,
   * reversible choices. */
  async function inspectLoomToken(node: LoomNodeJSON, tokenIndex: number): Promise<void> {
    if (!node.tokens?.[tokenIndex]) {
      pushToast("That token is no longer available in this branch.", { kind: "error" });
      return;
    }
    if (loomTree.active_node_id !== node.id) {
      await loomNavigate(node.id);
    }
    await tick();
    const turnIdx = chatLog.turns.findIndex((turn) => turn.nodeId === node.id);
    if (turnIdx < 0) {
      pushToast("That branch could not be opened for token editing.", { kind: "error" });
      return;
    }
    openDrawer("token_drilldown", {
      turnIdx,
      tokenIdx: tokenIndex,
      initialTab: "logits",
    });
  }

  /** True when a keydown originated in a text-entry element (an ``<input>``,
   *  ``<textarea>``, ``<select>``, or a ``contenteditable`` host) — used to
   *  keep the sidebar's bare-key nav shortcuts from stealing keystrokes meant
   *  for a nested field. */
  function _isEditableTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      el.isContentEditable
    );
  }

  function onSidebarKey(ev: KeyboardEvent): void {
    if (modal.kind !== null || menu.open) return;
    const target = ev.target as HTMLElement | null;
    const treeItem = target?.closest<HTMLElement>('[role="treeitem"]') ?? null;
    if (!treeItem || !viewportEl?.contains(treeItem)) return;
    if (target !== treeItem && target?.closest("button, a, [role='button'], [role='menuitem']")) {
      return;
    }
    const k = ev.key;
    if (k !== "Escape" && _isEditableTarget(ev.target)) return;
    if (k === "ArrowDown" || k === "j") {
      ev.preventDefault();
      moveVisible(+1);
      return;
    }
    if (k === "ArrowUp" || k === "k") {
      ev.preventDefault();
      moveVisible(-1);
      return;
    }
    if (k === "ArrowLeft" || k === "h") {
      ev.preventDefault();
      moveParent();
      return;
    }
    if (k === "ArrowRight" || k === "l") {
      ev.preventDefault();
      moveFirstChild();
      return;
    }
    if (k === "Home" && rows.length > 0) {
      ev.preventDefault();
      void focusNode(rows[0].node.id);
      return;
    }
    if (k === "End" && rows.length > 0) {
      ev.preventDefault();
      void focusNode(rows[rows.length - 1].node.id);
      return;
    }
    if (k === "ContextMenu" || (k === "F10" && ev.shiftKey)) {
      if (focusedId) {
        ev.preventDefault();
        openMenuFromKeyboard(focusedId);
      }
      return;
    }
    if (k === "Enter" || k === " ") {
      if (focusedId) {
        ev.preventDefault();
        void loomNavigate(focusedId);
      }
      return;
    }
    if (k === "s" && focusedId) {
      ev.preventDefault();
      const node = loomTree.nodes.get(focusedId);
      void loomStar(focusedId, !node?.starred);
      return;
    }
    if (k === "n" && focusedId) {
      ev.preventDefault();
      const node = loomTree.nodes.get(focusedId);
      void openModal("note", focusedId, node?.notes ?? "");
      return;
    }
    if (k === "/") {
      ev.preventDefault();
      void openModal("search", focusedId ?? loomTree.active_node_id);
      return;
    }
    if (k === "Escape") {
      if (loomUiState.filterHelpOpen) {
        ev.preventDefault();
        loomUiState.filterHelpOpen = false;
      }
      return;
    }
  }

  // Close context menu on outside click or Escape (window-level).
  function onWindowClick(ev: MouseEvent): void {
    if (!menu.open) return;
    const t = ev.target as HTMLElement | null;
    if (t && t.closest(".loom-menu")) return;
    closeMenu(false);
  }
  function onWindowKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      if (menu.open) { closeMenu(); ev.preventDefault(); return; }
      if (modal.kind) { closeModal(); ev.preventDefault(); return; }
    }
  }

  // ---------------------------------------- menu actions --

  async function menuRegenerate(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    if (loomTree.active_node_id !== nid) await loomNavigate(nid);
    await openModal("regenerate", nid, "", 1, trigger);
  }
  async function menuEdit(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await openModal("edit", nid, node?.text ?? "", 1, trigger);
  }
  async function menuBranch(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await openModal("branch", nid, node?.text ?? "", 1, trigger);
  }
  async function menuSwapSeat(): Promise<void> {
    // Seat-swap branch (the cast model): a sibling with the same text
    // and the seat flipped — identical bytes, one seat bit different,
    // the controlled experiment on the seat prior.  No modal: the text
    // is by definition unchanged.
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const newId = await loomSwapSeat(nid);
    if (newId) await loomNavigate(newId);
  }
  async function menuNavigate(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    await loomNavigate(nid);
  }
  async function menuDelete(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    await openModal("delete", nid, "", 1, trigger);
  }
  async function menuStar(): Promise<void> {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await loomStar(nid, !node?.starred);
  }
  async function menuNote(): Promise<void> {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    await openModal("note", nid, node?.notes ?? "", 1, trigger);
  }

  // ---------------------------------- phase 5 context-menu actions --

  function menuPin(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    pinNodeForComparison(nid);
  }

  function menuFanOut(): void {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    // Anchor the fan-out on the committed turn.  If the user clicked a
    // generated node, walk up to its parent.
    let anchorId = nid;
    if (node && node.recipe !== null && node.parent_id) {
      anchorId = node.parent_id;
    }
    void openModal("fanout", anchorId, "0.0, 0.3, 0.6", 1, trigger);
  }

  function menuCompareBranch(): void {
    const nid = menu.nodeId;
    closeMenu(false);
    if (!nid) return;
    const node = loomTree.nodes.get(nid);
    // A committed node compares generated children; a generated node
    // compares its generated sibling set.
    let parentId: string | null = null;
    if (node && node.recipe === null && node.role !== "system") parentId = nid;
    else if (node && node.recipe !== null) parentId = node.parent_id;
    if (!parentId) return;
    const generatedChildren = (loomTree.children_of.get(parentId) ?? []).filter(
      (id) => loomTree.nodes.get(id)?.recipe !== null,
    );
    if (generatedChildren.length < 2) return;
    openDrawer("node_compare", {
      node_ids: generatedChildren,
      parent_id: parentId,
    });
  }

  function menuToggleSelection(): void {
    const nid = menu.nodeId;
    closeMenu();
    if (!nid) return;
    toggleNodeSelection(nid);
  }

  function menuCompareSelected(): void {
    if (menu.open) closeMenu(false);
    if (nodeSelection.ids.length < 2) return;
    openDrawer("node_compare", { node_ids: [...nodeSelection.ids] });
  }

  function menuRegenWithMode(): void {
    const nid = menu.nodeId;
    const trigger = menuOpener;
    closeMenu(false);
    if (!nid) return;
    void openModal("regen_mode", nid, "", 1, trigger);
  }

  // ---------------------------------------- refresh / error UI --

  function fullRefresh(): void {
    // Re-fetch — useful if the user suspects drift.
    void refreshLoomTree();
  }

  onMount(() => {
    const initialId = focusedId ?? loomTree.active_node_id ?? rows[0]?.node.id;
    if (initialId && rows.some((row) => row.node.id === initialId)) {
      void focusNode(initialId, false);
    }
    void tick().then(focusInitialView);
    return () => {
      if (cameraAnimationTimer !== null) clearTimeout(cameraAnimationTimer);
      if (menuTrackingFrame !== null) cancelAnimationFrame(menuTrackingFrame);
      touchPointers.clear();
      restoreModalBackground();
    };
  });
</script>

<svelte:window onclick={onWindowClick} onkeydown={onWindowKey} onresize={positionOpenMenu} />

<aside
  class="loom-sidebar"
  aria-label="Conversation loom panel"
  bind:this={asideEl}
>
  {#if headersVisible}
  <header class="loom-header" id="loom-tools-header" inert={!headersVisible} in:slide={collapseIn()} out:slide={collapseOut()}>
    <div class="loom-heading">
      <span class="title">Conversation loom</span>
      <span class="loom-subtitle">Explore, branch, and return to any point.</span>
    </div>
    <!-- Tree-scope actions — clear / save / load act on the whole
         conversation tree, not on the active chat path, so they live in
         the threads-column header rather than buried in the chat. -->
    <button
      type="button"
      class="action-btn"
      onclick={clearChat}
      {...{ "aria-description": "Start a new path from the root; keep all existing branches" }}
    >
      Start over
    </button>
    <button type="button" class="action-btn" disabled={genStatus.active || !loomTree.loaded || loomTree.nodes.size <= 1}
      {...{ "aria-description": "Save this conversation and start with an empty loom" }}
      onclick={() => void openModal("clear", loomTree.root_id)}>Clear loom…</button>
    <button type="button" class="action-btn"
      disabled={!loomTree.active_node_id || loomTree.active_node_id === loomTree.root_id || loomNodeIntersectsGeneration(loomTree.active_node_id)}
      {...{ "aria-description": "Remove the current turn and all branches that follow it" }}
      onclick={() => void openModal("delete", loomTree.active_node_id)}>Cut branch…</button>
    <button
      type="button"
      class="action-btn"
      onclick={() => openDrawer("save_conversation")}
    >
      Save
    </button>
    <button
      type="button"
      class="action-btn"
      onclick={() => openDrawer("load_conversation")}
    >
      Open
    </button>
    <button
      type="button"
      class="action-btn"
      onclick={compareBranch}
      disabled={comparableNodes.length < 2}
      {...{ "aria-description": "Compare alternate replies to the same message" }}
    >
      {comparableNodes.length >= 2 ? `Compare ${comparableNodes.length}` : "Compare"}
    </button>
    <button
      type="button"
      class="icon-btn"
      onclick={fullRefresh}
      {...{ "aria-description": "refresh" }}
      aria-label="Refresh"
    ><FluentIcon name="refresh" /></button>
  </header>

  {/if}
  <nav class="loom-views" aria-label="Loom views" use:slidingSelection>
    <button
      type="button"
      class:active={loomUiState.view === "weave"}
      aria-current={loomUiState.view === "weave" ? "page" : undefined}
      onclick={() => setLoomView("weave")}
    ><strong>Weave</strong><span>Text & alternatives</span></button>
    <button
      type="button"
      class:active={loomUiState.view === "map"}
      aria-current={loomUiState.view === "map" ? "page" : undefined}
      onclick={() => setLoomView("map")}
    ><strong>Map</strong><span>All branches</span></button>
    <button
      type="button"
      class:active={loomUiState.view === "path"}
      aria-current={loomUiState.view === "path" ? "page" : undefined}
      onclick={() => setLoomView("path")}
    ><strong>Current path</strong><span>{pathSnippets.length} {pathSnippets.length === 1 ? "snippet" : "snippets"}</span></button>
    <button
      type="button"
      class:active={loomUiState.view === "options"}
      aria-current={loomUiState.view === "options" ? "page" : undefined}
      onclick={() => setLoomView("options")}
    ><strong>Next options</strong><span>{optionNodes.length} available</span></button>
    <button
      type="button"
      class:active={loomUiState.view === "saved"}
      aria-current={loomUiState.view === "saved" ? "page" : undefined}
      onclick={() => setLoomView("saved")}
    ><strong>Starred</strong><span>{savedRows.length} {savedRows.length === 1 ? "point" : "points"}</span></button>
  </nav>

  <div class="compact-loom-view">
    <span>View</span>
    <Select ariaLabel="Loom view" value={loomUiState.view} options={[
      { value: "weave", label: "Weave · Text and alternatives" },
      { value: "map", label: "Map · All branches" },
      { value: "path", label: "Current path" },
      { value: "options", label: "Next options" },
      { value: "saved", label: "Starred" },
    ]} onchange={value => setLoomView(value as LoomView)} />
  </div>

  {#if loomUiState.view === "map" || loomUiState.view === "saved"}
    <div class="filter-bar">
    <label class="filter-field">
    <span>{filterState.mode === "text" ? "Search messages" : "Advanced filters"}</span>
    <input
      type="search"
      class="filter-input"
      bind:this={searchInput}
      value={filterState.expr}
      oninput={(event) => editFilter(event.currentTarget.value)}
      onkeydown={onFilterKey}
      placeholder={filterState.mode === "text" ? "Words or a phrase…" : "starred, text:fox"}
      aria-invalid={filterState.error ? true : undefined}
      aria-describedby="loom-search-status"
    />
    </label>
    <Select value={filterState.mode} options={[{ value: "text", label: "Text" }, { value: "advanced", label: "Advanced" }]}
      ariaLabel="Search mode" onchange={changeSearchMode} />
    {#if filterState.mode === "advanced"}
      <button type="button" class="quiet-action" onclick={() => void applyTreeFilter(filterState.expr)}>Apply</button>
    <!-- Logit-pass (Decision 8): help popover for the filter grammar.
         Clicked-toggle keeps the popover anchored without stealing
         keyboard focus from the filter input.  The grammar text mirrors
         tree_filter.py's accepted forms plus the client-side sort
         directive added in Phase 4. -->
    <button
      type="button"
      class="icon-btn help-btn"
      class:on={loomUiState.filterHelpOpen}
      onclick={() => (loomUiState.filterHelpOpen = !loomUiState.filterHelpOpen)}

      aria-label="How advanced filters work"
      aria-expanded={loomUiState.filterHelpOpen}
    ><FluentIcon name="help" /></button>
    {/if}
    {#if filterState.expr}
      <button
        type="button"
        class="icon-btn"
        onclick={() => { clearTreeFilter(); searchInput?.focus(); }}

        aria-label="Clear search"
      ><FluentIcon name="dismiss" /></button>
    {/if}
    </div>
    <div class="search-feedback">
      <p id="loom-search-status" class:err={filterState.error !== null} role="status">
        <MorphText text={filterState.loading ? "Searching…" : filterState.error ?? (filterState.matchingIds !== null
          ? searchResults.length === 0 ? "No messages match. Try different words or clear the search."
            : `${searchIndex >= 0 ? `${searchIndex + 1} of ` : ""}${searchResults.length} matching ${searchResults.length === 1 ? "message" : "messages"}`
          : filterState.mode === "advanced" ? "Press Enter or Apply to run the filters." : "Search all branches. Enter moves to the next match; Shift+Enter moves back.")} />
      </p>
      {#if searchResult && searchExcerpt}
        <div class="search-result">
          <button type="button" class="search-excerpt" aria-label="Show match in map" onclick={() => void showSearchMatch()}>
            <span class="snippet-role">{nodeRole(searchResult)}</span>
            <span dir="auto">{searchExcerpt.before}<mark>{searchExcerpt.match}</mark>{searchExcerpt.after}</span>
          </button>
          <button type="button" class="icon-btn" aria-label="Previous match" onclick={() => void showSearchMatch(-1)}><FluentIcon name="up" /></button>
          <button type="button" class="icon-btn" aria-label="Next match" onclick={() => void showSearchMatch(1)}><FluentIcon name="down" /></button>
        </div>
      {/if}
    </div>
  {/if}

  {#if filterState.mode === "advanced" && loomUiState.filterHelpOpen && (loomUiState.view === "map" || loomUiState.view === "saved")}
    <!-- Inline grammar reference + worked examples.  Dismissible by
         clicking the ? again or pressing Esc on the sidebar. -->
    <div class="filter-help" role="region" aria-label="Filter grammar help" in:slide={collapseIn()} out:slide={collapseOut()}>
      <p>Combine filters with commas. A branch must match every filter.</p>
      <ul>
        <li><code>text:&lt;words&gt;</code> searches message text</li>
        <li><code>starred</code> shows saved branches</li>
        <li><code>&lt;measurement&gt; &gt; &lt;number&gt;</code> filters by a reading</li>
        <li><code>agg:</code>, <code>any:</code>, or <code>last:</code> chooses where to measure</li>
        <li><code>sort:surprise</code> or <code>sort:confidence</code> changes reply order</li>
      </ul>
      <p>Examples</p>
      <ul class="examples">
        <li><code>agg:angry.calm &gt; 0.4</code></li>
        <li><code>starred, text:fox</code></li>
        <li><code>sort:surprise, agg:honest &lt; 0</code></li>
      </ul>
    </div>
  {/if}

  <!-- Logit-pass: edges are always surprise-weighted — stroke width and
       opacity thicken toward surprising children (low mean_logprob), and
       every node carries its mean_logprob badge.  Only surfaces a bar
       when a ``sort:`` filter directive is active. -->
  {#if loomUiState.siblingSort !== "default"}
    <div class="weight-bar">
      <span class="weight-label" {...{ "aria-description": "sibling sort" }}>
        sort:{loomUiState.siblingSort}
      </span>
    </div>
  {/if}

  {#if nodeSelection.ids.length > 0}
    <div class="selection-bar">
      <span><MorphText text={`${nodeSelection.ids.length} selected`} /></span>
      <button
        type="button"
        class="action-btn"
        onclick={menuCompareSelected}
        disabled={nodeSelection.ids.length < 2}
      >compare</button>
      <button
        type="button"
        class="action-btn"
        onclick={clearNodeSelection}
      >clear</button>
    </div>
  {/if}

  {#if cursorNode}
    <div class="focused-branch" aria-label="Focused branch summary">
      {#if cursorNode.id === createdBranchId}<span role="status">Branch created</span>{/if}
      <span><MorphText text={cursorNode.role_label || cursorNode.role} numbers={false} /></span>
      <span><MorphText text={`Depth ${rowById.get(cursorNode.id)?.depth ?? 0}`} /></span>
      <span><MorphText text={`${cursorNode.tokens?.length ?? 0} recorded tokens`} /></span>
    </div>
  {/if}
  <div class="weave-host" hidden={loomUiState.view !== "weave" || !!loomTree.error}>
    {#key loomTree.root_id}
      <LoomWeave
        active={active && loomUiState.view === "weave"}
        oninspect={(node, index) => void inspectLoomToken(node, index)}
        onwrite={branchFromNode}
      />
    {/key}
  </div>

  {#if loomTree.error}
    <div class="empty err">
      <p>tree unavailable</p>
      <button type="button" onclick={fullRefresh}>retry</button>
    </div>
  {:else if loomUiState.view !== "weave"}
  {#if rows.length === 0}
    <div class="empty">
      <p>No branches yet</p>
      <p>Send a message in Conversation to create the first path.</p>
    </div>
  {:else if loomUiState.view === "map"}
    <div class="loom-map-bar">
      <div class="loom-map-summary">
        <strong>Conversation map</strong>
        <span><MorphText text={rows.length} /> {rows.length === 1 ? "turn" : "turns"}</span>
        <span class="map-separator" aria-hidden="true">·</span>
        <span><MorphText text={branchPointCount} /> {branchPointCount === 1 ? "fork" : "forks"}</span>
        {#if visibleTokenCount > 0}
          <span class="map-separator" aria-hidden="true">·</span>
          <span><MorphText text={visibleTokenCount} /> {visibleTokenCount === 1 ? "token" : "tokens"}</span>
        {/if}
        <span class="path-key"><i aria-hidden="true"></i> current path</span>
      </div>
      <div class="loom-view-controls" role="group" aria-label="Loom view controls">
        <button
          type="button"
          onclick={() => zoomView(camera.zoom - 0.12)}
          data-cursor="zoom-out"
          disabled={camera.zoom <= LOOM_MIN_ZOOM}
          aria-label="Zoom out"

        ><FluentIcon name="subtract" /></button>
        <output aria-label="Loom zoom"><RollingNumber value={Math.round(camera.zoom * 100)} />%</output>
        <button
          type="button"
          onclick={() => zoomView(camera.zoom + 0.12)}
          data-cursor="zoom-in"
          disabled={camera.zoom >= LOOM_MAX_ZOOM}
          aria-label="Zoom in"

        ><FluentIcon name="add" /></button>
        <button type="button" onclick={fitView} aria-label="Fit whole loom" >
          Fit
        </button>
        <button type="button" onclick={centerCurrent} aria-label="Center current path" >
          Current
        </button>
      </div>
    </div>
    <div
      class="tree-scroll loom-viewport"
      class:dragging
      class:camera-animating={cameraAnimating}
      role="tree"
      aria-label="Conversation loom"
      aria-describedby="loom-touch-help"
      aria-multiselectable="true"
      tabindex="-1"
      bind:this={viewportEl}
      onwheel={onViewportWheel}
      onpointerdown={onViewportPointerDown}
      onpointermove={onViewportPointerMove}
      onpointerup={endViewportGesture}
      onpointercancel={endViewportGesture}
      onlostpointercapture={endViewportGesture}
    >
      <p id="loom-touch-help" class="touch-gesture-hint">Drag to move · pinch to zoom</p>
      <div class="loom-depth-field" aria-hidden="true" style={loomDepthStyle}></div>
      <div
        class="loom-canvas"
        class:camera-animating={cameraAnimating}
        style={`width:${graph.width}px;height:${graph.height}px;transform:translate3d(${camera.x}px,${camera.y}px,0) scale(${camera.zoom})`}
        data-loom-nodes={graph.nodes.length}
        data-loom-edges={graph.edges.length}
        data-loom-zoom={camera.zoom.toFixed(2)}
        data-loom-camera-x={camera.x.toFixed(2)}
        data-loom-camera-y={camera.y.toFixed(2)}
      >
        <svg
          class="loom-edges"
          width={graph.width}
          height={graph.height}
          viewBox={`0 0 ${graph.width} ${graph.height}`}
          aria-hidden="true"
        >
          {#each graph.edges as edge (`${edge.parentId}|${edge.childId}`)}
            <path
              d={edge.path}
              class="edge-depth"
              class:active-edge={edgeIsActive(edge)}
              class:quiet-edge={!edgeIsActive(edge)}
              class:filtered-edge={edgeIsFiltered(edge)}
              style={edgeStyle(edge)}
            />
            <path
              d={edge.path}
              class="edge-line"
              class:active-edge={edgeIsActive(edge)}
              class:quiet-edge={!edgeIsActive(edge)}
              class:filtered-edge={edgeIsFiltered(edge)}
              style={edgeStyle(edge)}
              data-loom-edge={`${edge.parentId}|${edge.childId}`}
            />
          {/each}
          {#each branchJunctions as junction (junction.id)}
            <circle
              class="junction-halo"
              class:active-junction={mapRowById.get(junction.id)?.isActivePath}
              cx={junction.x}
              cy={junction.y}
              r="9"
            />
            <circle
              class="junction-core"
              class:active-junction={mapRowById.get(junction.id)?.isActivePath}
              cx={junction.x}
              cy={junction.y}
              r="4"
              data-loom-junction={junction.id}
            />
          {/each}
        </svg>

        {#each graph.nodes as placed (placed.id)}
          {@const row = mapRowById.get(placed.id)!}
          {@const segment = segmentById.get(placed.id)!}
          <div
            class="tree-node-wrap"
            role="none"
            class:filtered-out={row.filteredOut}
            class:search-match={filterState.matchingIds !== null && !row.filteredOut}
            class:search-current={searchSegmentId === placed.id}
            class:selected={!segment.shared && selectionSet.has(row.node.id)}
            class:pinned={!segment.shared && pinnedComparison.nodeId === row.node.id}
            class:active-path-node={row.isActivePath}
            class:current-node={!segment.shared && loomTree.active_node_id === row.node.id}
            class:dead-node={row.isDead}
            data-loom-node-id={segment.id}
            data-loom-shared={segment.shared ? segment.memberIds.length : undefined}
            data-loom-depth={placed.depth}
            style={`left:${placed.x}px;top:${placed.y}px;width:${placed.width}px;height:${placed.height}px;--branch-depth:${placed.depth}`}
          >
            <LoomNode
              onmeasure={(height) => { if (measuredHeights.get(segment.id) !== height) measuredHeights.set(segment.id, height); }}
              node={row.node}
              displayId={segment.id}
              tokenStart={segment.start}
              tokenEnd={segment.end}
              sharedCount={segment.shared ? segment.memberIds.length : 0}
              sharedUsesActivePath={segment.shared && activePathSet.has(row.node.id)}
              onActivePath={row.isActivePath}
              focused={!segment.shared && focusedId === row.node.id}
              current={!segment.shared && loomTree.active_node_id === row.node.id}
              selected={selectionSet.has(row.node.id)}
              level={row.depth + 1}
              hasChildren={(loomTree.children_of.get(row.node.id)?.length ?? 0) > 0}
              collapsed={collapsedIds.has(row.node.id)}
              dead={row.isDead}
              streaming={!segment.shared && loomTree.pendingNodeId === row.node.id}
              ring={segment.shared ? null : ringFor(row.node)}
              weightBadge={segment.shared ? null : weightBadgeFor(row.node)}
              steerLabel={segment.shared ? null : steerLabelFor(row.node)}
              forkLabel={segment.shared ? null : segment.start > 0 ? "continuation" : divergenceLabel(row.node)}
              onclick={segment.shared ? undefined : (ev) => onNodeClick(row.node, ev)}
              onfocus={segment.shared ? undefined : () => { focusedId = row.node.id; }}
              onkeydown={segment.shared ? undefined : onSidebarKey}
              oncontextmenu={segment.shared ? undefined : (ev) => openMenu(ev, row.node.id)}
              onactions={(ev) => openMenu(ev, row.node.id)}
              ongrow={() => void growFromNode(row.node)}
              onbranch={(ev) => branchFromNode(row.node, ev.currentTarget as HTMLElement)}
              onselecttoken={(tokenIndex) => void inspectLoomToken(row.node, tokenIndex)}
              onbranchsentence={(tokenIndex) => void branchAfterToken(row.node, tokenIndex)}
              sentenceBranchAvailable={sentenceBranchAvailable && !genStatus.active}
              ontogglechildren={() => toggleChildren(row.node.id)}
            />
          </div>
        {/each}
      </div>
    </div>
  {:else if loomUiState.view === "path"}
    <section class="loom-projection path-projection" aria-labelledby="current-path-title">
      <header class="projection-heading">
        <div>
          <h2 id="current-path-title">Current path</h2>
          <p>Each generated sentence is a branch point. Choose one to keep it and explore a different continuation.</p>
        </div>
        {#if loomTree.active_node_id}
          <button type="button" class="secondary-action" onclick={() => setLoomView("options")}>See alternatives</button>
        {/if}
      </header>
      <div class="path-stream" role="list" aria-label="Current conversation path">
        {#each pathSnippets as snippet, index (snippet.key)}
          {@const boundary = snippet.end >= 0 ? snippet.node.tokens?.[snippet.end] : null}
          {@const canBranch = snippet.sentenceIndex !== null && sentenceBranchAvailable &&
            !genStatus.active && boundary?.raw_index != null && boundary?.token_id != null}
          <article
            class="path-snippet"
            class:current-snippet={loomTree.active_node_id === snippet.node.id && index === pathSnippets.length - 1}
            data-path-snippet={snippet.key}
            role="listitem"
          >
            <div class="snippet-rail" aria-hidden="true">
              <span></span>
            </div>
            <div class="snippet-card">
              <header>
                <span class="snippet-role">{nodeRole(snippet.node)}</span>
                {#if snippet.sentenceIndex !== null}
                  <span class="snippet-index">sentence {snippet.sentenceIndex + 1}</span>
                {/if}
                {#if loomTree.active_node_id === snippet.node.id}
                  <span class="current-label">current</span>
                {/if}
                <button
                  type="button"
                  class="snippet-actions"
                  aria-label={`Actions for ${nodeRole(snippet.node)}`}
                  onclick={(ev) => openMenu(ev, snippet.node.id)}
                >•••</button>
              </header>
              {#if snippet.sentenceIndex !== null && snippet.node.tokens}
                <div class="snippet-tokens">
                  {#each snippet.node.tokens.slice(snippet.start, snippet.end + 1) as token, offset (`${token.raw_index ?? snippet.start + offset}:${token.token_id ?? token.text}`)}
                    {@const tokenIndex = snippet.start + offset}
                    <button
                      type="button"
                      class="path-token"
                      data-cursor="inspect"
                      data-loom-token-node={snippet.node.id}
                      data-token-index={tokenIndex}
                      {...{ "aria-description": "Open branch-point tools" }}
                      aria-label={`Open token ${tokenIndex + 1}: ${token.text.trim() || "whitespace"}`}
                      aria-haspopup="dialog"
                      onclick={() => void inspectLoomToken(snippet.node, tokenIndex)}
                    >{token.text || "∅"}</button>
                  {/each}
                </div>
              {:else}
                <button type="button" class="snippet-text" onclick={() => void usePath(snippet.node)}>
                  {readableNodeText(snippet.node)}
                </button>
              {/if}
              <footer>
                <button type="button" class="quiet-action" onclick={() => void usePath(snippet.node)}>
                  Use this point
                </button>
                {#if snippet.sentenceIndex !== null}
                  <button
                    type="button"
                    class="branch-action"
                    disabled={!canBranch}
                    {...{ "aria-description": (canBranch
                      ? "Keep everything through this sentence and generate a new continuation"
                      : "Exact sentence replay is unavailable while the model is busy") }}
                    onclick={() => void branchAfterToken(snippet.node, snippet.end)}
                  >Branch after sentence</button>
                {/if}
              </footer>
            </div>
          </article>
        {/each}
      </div>
    </section>
  {:else if loomUiState.view === "options"}
    <section class="loom-projection options-projection" aria-labelledby="next-options-title">
      <header class="projection-heading">
        <div>
          <h2 id="next-options-title">Next options</h2>
          <p>{cursorNode ? readableNodeText(cursorNode) : "Choose a point in the map to see its alternatives."}</p>
        </div>
        {#if cursorNode}
          <div class="projection-actions">
            <button type="button" class="primary-action" disabled={genStatus.active} onclick={() => void growFromNode(cursorNode)}>
              <MorphText text={genStatus.active ? "Generating…" : "Generate another"} />
            </button>
            {#if cursorNode.parent_id}
              <button type="button" class="secondary-action" onclick={(ev) => branchFromNode(cursorNode, ev.currentTarget as HTMLElement)}>
                Write an alternative
              </button>
            {/if}
          </div>
        {/if}
      </header>
      {#if optionNodes.length === 0}
        <div class="projection-empty">
          <strong>No alternatives here yet</strong>
          <p>Generate another reply to make this point branch.</p>
        </div>
      {:else}
        <div class="option-grid" role="list" aria-label="Alternative paths">
          {#each optionNodes as node, index (node.id)}
            <article
              class="option-card"
              class:active-option={loomTree.active_node_id === node.id}
              role="listitem"
              data-option-node={node.id}
            >
              <header>
                <span class="option-number">Option {index + 1}</span>
                <span class="snippet-role">{nodeRole(node)}</span>
                {#if loomTree.active_node_id === node.id}<span class="current-label">current</span>{/if}
                {#if node.starred}<span class="saved-mark">saved</span>{/if}
                <button type="button" class="snippet-actions" aria-label={`Actions for option ${index + 1}`} onclick={(ev) => openMenu(ev, node.id)}>•••</button>
              </header>
              <button type="button" class="option-preview" onclick={() => void usePath(node)}>
                {readableNodeText(node)}
              </button>
              <footer>
                <button type="button" class="primary-action" onclick={() => void usePath(node)}>Use this path</button>
                <button type="button" class="quiet-action" onclick={() => void loomStar(node.id, !node.starred)}>
                  <MorphText text={node.starred ? "Remove saved" : "Save for later"} />
                </button>
              </footer>
            </article>
          {/each}
        </div>
      {/if}
    </section>
  {:else}
    <section class="loom-projection saved-projection" aria-labelledby="saved-points-title">
      <header class="projection-heading">
        <div>
          <h2 id="saved-points-title">Starred points</h2>
          <p>Keep useful branches close without changing the current conversation.</p>
        </div>
      </header>
      {#if savedRows.length === 0}
        <div class="projection-empty">
          <strong><MorphText text={filterState.matchingIds !== null ? "No starred messages match" : "Nothing saved yet"} /></strong>
          <p>{filterState.matchingIds !== null ? "Try different words or clear the search." : "Use a node’s menu and choose “Save this branch.”"}</p>
        </div>
      {:else}
        <div class="saved-list" role="list" aria-label="Starred conversation points">
          {#each savedRows as row (row.node.id)}
            <article class="saved-card" role="listitem" data-saved-node={row.node.id}>
              <div class="saved-copy">
                <span class="snippet-role">{nodeRole(row.node)}</span>
                <strong>{readableNodeText(row.node)}</strong>
                {#if row.node.notes}<p>{row.node.notes}</p>{/if}
              </div>
              <div class="saved-actions">
                <button type="button" class="primary-action" onclick={() => void usePath(row.node)}>Use this path</button>
                <button type="button" class="quiet-action" onclick={() => void loomStar(row.node.id, false)}>Remove</button>
                <button type="button" class="snippet-actions" aria-label={`Actions for ${nodeRole(row.node)}`} onclick={(ev) => openMenu(ev, row.node.id)}>•••</button>
              </div>
            </article>
          {/each}
        </div>
      {/if}
    </section>
  {/if}
  {/if}

</aside>

{#if menu.open && menu.nodeId}
  {@const menuNode = loomTree.nodes.get(menu.nodeId)}
  {@const menuPreview = (menuNode?.text ?? "").replace(/\s+/g, " ").trim() || "Empty turn"}
  {@const menuRole = menuNode?.role_label?.trim() || menuNode?.role || "turn"}
  {@const cmpParentId =
    menuNode?.recipe === null && menuNode?.role !== "system"
      ? menu.nodeId
      : menuNode?.recipe !== null
        ? (menuNode?.parent_id ?? null)
        : null}
  {@const cmpCount = cmpParentId
    ? (loomTree.children_of.get(cmpParentId) ?? []).filter(
        (id) => loomTree.nodes.get(id)?.recipe !== null,
      ).length
    : 0}
  <div
    class="loom-menu"
    class:positioned={menuPositioned}
    style={`left:${menuPosition.x}px;top:${menuPosition.y}px`}
    role="menu"
    aria-label="Node actions"
    tabindex="-1"
    bind:this={menuEl}
    use:portal
    onkeydown={onMenuKey}
    out:fade={scrimOut()}
  >
    <div class="loom-menu-context" role="presentation">
      <span>{menuRole}</span>
      <strong>{menuPreview}</strong>
    </div>
    <div class="loom-menu-section" role="presentation">
      <button type="button" role="menuitem" onclick={menuNavigate}>Use this path</button>
      <button type="button" role="menuitem" onclick={menuRegenerate}>Generate another reply…</button>
      <button type="button" role="menuitem" onclick={menuRegenWithMode}>Generate several replies…</button>
      <button type="button" role="menuitem" onclick={menuEdit}>Edit this turn…</button>
      <button type="button" role="menuitem" onclick={menuBranch}>Start a branch…</button>
      {#if (sessionState.info?.scene_mode ?? false) && (menuNode?.role === "user" || menuNode?.role === "assistant")}
        <button
          type="button"
          role="menuitem"
          {...{ "aria-description": "Swap the speaker and create a branch" }}
          onclick={menuSwapSeat}
        >Swap speaker on a new branch</button>
      {/if}
    </div>
    <div class="loom-menu-section" role="presentation">
      <button type="button" role="menuitem" onclick={menuStar}>
        <MorphText text={menuNode?.starred ? "Remove saved marker" : "Save this branch"} />
      </button>
      <button type="button" role="menuitem" onclick={menuNote}>Add a note…</button>
    </div>
    <div class="loom-menu-section" role="presentation">
      {#if menuNode?.recipe !== null}
        <button type="button" role="menuitem" onclick={menuPin}>
          {pinnedComparison.nodeId === menu.nodeId
            ? "Remove comparison pin"
            : "Pin for comparison"}
        </button>
        <button type="button" role="menuitem" onclick={menuToggleSelection}>
          {selectionSet.has(menu.nodeId)
            ? "Remove from comparison"
            : "Add to comparison"}
        </button>
      {/if}
      <button
        type="button"
        role="menuitem"
        onclick={menuCompareBranch}
        disabled={cmpCount < 2}
        {...{ "aria-description": (cmpCount < 2 ? "Create at least two replies first" : "") }}
      >Compare replies…</button>
      <button type="button" role="menuitem" onclick={menuFanOut}>Create guided variations…</button>
    </div>
    <div class="loom-menu-section danger-section" role="presentation">
      <button
        type="button"
        role="menuitem"
        onclick={menuDelete}
        class="danger"
        disabled={menu.nodeId !== null && loomNodeIntersectsGeneration(menu.nodeId)}
        {...{ "aria-description": (menu.nodeId !== null && loomNodeIntersectsGeneration(menu.nodeId)
          ? LOOM_DELETE_DURING_GENERATION_MESSAGE
          : "") }}
      >Delete this branch…</button>
    </div>
  </div>
{/if}

{#if modal.kind}
  <div
    class="loom-modal-backdrop"
    aria-hidden="true"
    onclick={closeModal}
    in:fade={scrimIn()}
    out:fade={scrimOut()}
  ></div>
  <div
    class="loom-modal"
    role="dialog"
    aria-modal="true"
    aria-labelledby="loom-modal-title"
    tabindex="-1"
    bind:this={modalEl}
    onkeydown={onModalKey}
    in:fly={modalIn(8)}
    out:fly={modalOut(4)}
  >
    <header class="modal-header">
      <h2 id="loom-modal-title">
        {#if modal.kind === "regenerate"}Generate alternatives
        {:else if modal.kind === "edit"}Edit turn
        {:else if modal.kind === "branch"}Start a branch
        {:else if modal.kind === "delete"}Delete branch
        {:else if modal.kind === "clear"}Clear loom
        {:else if modal.kind === "note"}Add note
        {:else if modal.kind === "navpicker"}Go to turn
        {:else if modal.kind === "search"}Search conversation
        {:else if modal.kind === "fanout"}Create guided variations
        {:else if modal.kind === "regen_mode"}Generate several replies
        {/if}
      </h2>
      <button type="button" class="icon-btn" onclick={closeModal} aria-label="Close"><FluentIcon name="dismiss" /></button>
    </header>
    <div class="modal-body">
      {#if modal.kind === "regenerate"}
        <label>
          <span>Number of replies</span>
          <NumberInput
            bind:this={modalInput}
            value={modal.n}
            min={1}
            max={16}
            step={1}
            oninput={(v) => { if (v !== null) modal.n = v; }}
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Uses the current model settings.</p>
      {:else if modal.kind === "delete"}
        <p>Remove this turn and every continuation below it?</p>
        <blockquote>{loomTree.nodes.get(modal.nodeId ?? "")?.text.slice(0, 240) || "Empty turn"}</blockquote>
        <p class="hint danger">Earlier turns and sibling branches stay. This also updates the saved chat and cannot be undone. Download a copy first if you want a backup.</p>
      {:else if modal.kind === "clear"}
        <p>Save this conversation and start with an empty loom?</p>
        <p class="hint">The full conversation, including all branches, stays in Saved chats. Model files and response settings are kept.</p>
      {:else if modal.kind === "navpicker" || modal.kind === "search"}
        <input
          bind:this={modalInput as HTMLInputElement}
          bind:value={modal.text}
          type="text"
          aria-label={modal.kind === "navpicker" ? "Node ID prefix" : "Search node text"}
          placeholder={modal.kind === "navpicker" ? "node id prefix (or 'root')" : "search node text"}
          onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
        />
      {:else if modal.kind === "fanout"}
        <label>
          <span>Steering direction</span>
          <input
            bind:this={modalInput as HTMLInputElement}
            bind:value={modal.vector}
            type="text"
            placeholder="e.g. honest, calm, deer.wolf"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <label>
          <span>Strengths</span>
          <input
            bind:value={modal.text}
            type="text"
            placeholder="0.0, 0.3, 0.6 · linspace(-1, 1, 5) · 0:1:0.25"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Enter a comma-separated list, linspace(), or start:stop:step.</p>
      {:else if modal.kind === "regen_mode"}
        <label>
          <span>Variation style</span>
          <Select
            value={modal.mode ?? "unsteered"}
            options={[
              { value: "unsteered", label: "unsteered" },
              { value: "inverted", label: "inverted" },
              { value: "reseed", label: "reseed" },
              { value: "cool", label: "cool" },
              { value: "hot", label: "hot" },
            ]}
            onchange={(v) => { modal.mode = v; }}
            ariaLabel="regen mode"
          />
        </label>
        <label>
          <span>Number of replies</span>
          <NumberInput
            bind:this={modalInput}
            value={modal.n}
            min={1}
            max={16}
            step={1}
            oninput={(v) => { if (v !== null) modal.n = v; }}
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Keeps this branch's settings and applies the selected variation.</p>
      {:else}
        <textarea
          bind:this={modalInput as HTMLTextAreaElement}
          bind:value={modal.text}
          rows="6"
          aria-label={modal.kind === "edit" ? "Node text" : modal.kind === "branch" ? "Branch text" : "Node note"}
          placeholder={modal.kind === "branch" ? "(empty = branch from blank)" : ""}
          onkeydown={(ev) => {
            if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
              void commitModal();
            }
            // Prevent browser bold-formatting on Ctrl+B inside the
            // textarea for branch buffers.
            if (ev.key === "b" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
            }
          }}
        ></textarea>
        <p class="hint">⌃⏎ / ⌘⏎</p>
      {/if}
      {#if modal.error}
        <p class="modal-error" role="alert">{modal.error}</p>
      {/if}
    </div>
    <footer class="modal-footer">
      <button
        type="button"
        class="cancel"
        bind:this={modalCancelButton}
        onclick={closeModal}
      >cancel</button>
      <button
        type="button"
        class={modal.kind === "delete" || modal.kind === "clear" ? "danger" : "primary"}
        onclick={() => void commitModal()}
        disabled={modalBusy || (modal.kind === "clear" && genStatus.active) || (modal.kind === "delete" && modal.nodeId !== null &&
          loomNodeIntersectsGeneration(modal.nodeId))}
        {...{ "aria-description": (modal.kind === "delete" && modal.nodeId !== null &&
          loomNodeIntersectsGeneration(modal.nodeId)
          ? LOOM_DELETE_DURING_GENERATION_MESSAGE
          : "") }}
      >
        <MorphText text={modalBusy ? (modal.kind === "branch" ? "Creating branch…" : "Applying…")
          : modal.kind === "delete" ? "Delete branch"
          : modal.kind === "clear" ? "Save and clear loom"
          : modal.kind === "edit" || modal.kind === "note" ? "Save"
          : modal.kind === "branch" ? "Create branch"
          : modal.kind === "navpicker" ? "Go"
          : modal.kind === "search" ? "Search" : "Generate"} numbers={false} />
      </button>
    </footer>
  </div>
{/if}

<style>
  .focused-branch { display: flex; flex-wrap: wrap; gap: 8px 16px; padding: var(--space-1) var(--surface-padding); color: var(--fg-dim); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .weave-host { display: flex; flex: 1; min-height: 0; min-width: 0; }
  .weave-host[hidden] { display: none; }
  .loom-sidebar {
    display: flex;
    flex-direction: column;
    height: 100%;
    color: var(--fg);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    min-width: 0;
    min-height: 0;
    overflow: clip;
  }

  /* Section bars are borderless — typography + their own padding carry
   * the divide between them. */
  .filter-bar {
    display: flex;
    flex: 0 0 auto;
    gap: var(--space-2);
    align-items: center;
    flex-wrap: wrap;
    padding: var(--space-4) var(--surface-gutter);
  }
  .filter-field { display: grid; flex: 1 1 16rem; gap: var(--space-2); min-width: 0; color: var(--fg-dim); }
  .filter-field > span { font-size: var(--text-xs); }
  .filter-bar :global(.sk-select) { align-self: end; width: 7.5rem; flex: 0 0 7.5rem; }
  .filter-bar > button { align-self: end; }
  .search-feedback { flex: none; padding: 0 var(--surface-gutter) var(--space-4); }
  .search-feedback p { margin: 0; font-size: var(--text-xs); line-height: 1.5; color: var(--fg-dim); }
  .search-feedback p.err { color: var(--accent-red); }
  .search-result { display: flex; align-items: center; gap: var(--space-2); margin-top: var(--space-3); }
  .search-excerpt { display: grid; flex: 1; min-width: 0; gap: var(--space-2); padding: var(--space-3); background: var(--input-well); text-align: start; border: 0; border-radius: var(--radius); color: var(--fg); line-height: 1.5; overflow-wrap: anywhere; }
  .search-excerpt mark { background: var(--accent-subtle); color: inherit; text-decoration: underline; text-underline-offset: 2px; }
  .search-excerpt > span:last-child { display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .tree-node-wrap.search-match :global(.node) { outline: 1px solid var(--accent); outline-offset: 2px; }
  .tree-node-wrap.search-current :global(.node) { outline-width: 3px; outline-offset: 4px; }
  .filter-input {
    flex: 1 1 auto;
    min-height: var(--control-field);
    /* Recessed input well — the accent focus ring is the only border. */
    background: var(--input-well);
    color: var(--fg-strong);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    min-width: 0;
  }
  .filter-input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .filter-input::-webkit-search-cancel-button { -webkit-appearance: none; }

  /* Logit-pass: help affordance, help popover, weight-mode picker. */
  .help-btn.on {
    color: var(--accent);
  }
  .filter-help {
    padding: var(--space-4) var(--surface-gutter);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    line-height: 1.45;
    max-height: 14em;
    overflow: auto;
  }
  .filter-help code {
    color: var(--accent);
    background: transparent;
    font-family: var(--font-mono);
  }
  .filter-help ul {
    margin: var(--space-1) 0 var(--space-3) var(--space-sm);
    padding: 0;
    list-style: disc;
  }
  .filter-help li {
    margin: var(--space-1) 0;
  }
  .filter-help .examples code {
    color: var(--fg-strong);
  }
  .weight-bar {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    padding: var(--space-4) var(--surface-gutter);
    font-size: var(--text-xs);
    color: var(--fg-dim);
  }
  .weight-label {
    text-transform: uppercase;
    letter-spacing: 0;
  }

  .selection-bar {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-4) var(--surface-gutter);
    color: var(--accent);
    font-size: var(--text-xs);
  }
  .action-btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    min-height: var(--control-target);
  }
  .action-btn:hover:not(:disabled) {
    background: var(--glass-strong);
    color: var(--accent);
  }
  .action-btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }

  .loom-header {
    display: flex;
    flex: 0 0 auto;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    padding: var(--surface-padding);
  }
  .loom-heading {
    display: flex;
    flex: 1 1 18rem;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .title {
    color: var(--fg);
    font-family: var(--font-structure);
    font-weight: var(--weight-display);
    letter-spacing: -0.02em;
    font-size: var(--text-md);
  }
  .loom-subtitle {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .loom-views {
    display: grid;
    flex: 0 0 auto;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: var(--space-2);
    padding: 0 var(--surface-gutter) var(--space-4);
  }
  .compact-loom-view { display: none; }
  .loom-views button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-xs);
    min-width: 0;
    min-height: var(--control-target);
    padding: var(--space-3) var(--space-4);
    border: 1px solid transparent;
    border-radius: var(--radius);
    background: var(--input-well);
    color: var(--fg-muted);
    cursor: pointer;
    text-align: start;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .loom-views button:hover,
  .loom-views button:focus-visible {
    color: var(--fg);
    background: var(--glass);
    outline: none;
  }
  .loom-views button:focus-visible {
    border-color: var(--focus-ring);
  }
  .loom-views button:active {
    transform: scale(var(--press-scale));
  }
  .loom-views button.active {
    color: var(--fg);
    border-color: color-mix(in srgb, var(--accent) 35%, var(--glass-line));
    background: color-mix(in srgb, var(--accent) 9%, var(--bg-elev));
    box-shadow: inset 0 1px 0 color-mix(in srgb, var(--fg) 5%, transparent);
  }
  .loom-views strong {
    overflow: hidden;
    max-width: 100%;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .loom-views span {
    overflow: hidden;
    max-width: 100%;
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .icon-btn {
    background: transparent;
    border: 0;
    color: var(--fg-dim);
    cursor: pointer;
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-sm);
    font: inherit;
    font-family: var(--font-mono);
    min-width: var(--control-target);
    min-height: var(--control-target);
  }
  .icon-btn:hover {
    color: var(--accent);
  }

  .loom-projection {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    min-height: 0;
    overflow: auto;
    overscroll-behavior: contain;
    padding: var(--surface-padding);
    background:
      radial-gradient(circle at 50% 0, color-mix(in srgb, var(--accent) 4%, transparent), transparent 40%),
      var(--bg);
  }
  .projection-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    margin: 0 auto var(--space-6);
    width: min(100%, 960px);
  }
  .projection-heading > div:first-child {
    min-width: 0;
  }
  .snippet-role,
  .snippet-index,
  .option-number,
  .saved-mark,
  .current-label {
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }
  .projection-heading h2 {
    margin: 0 0 var(--space-2);
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text-lg);
    font-weight: var(--weight-display);
    letter-spacing: -0.025em;
  }
  .projection-heading p,
  .projection-empty p {
    margin: 0;
    color: var(--fg-muted);
    line-height: 1.5;
  }
  .projection-actions,
  .saved-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .primary-action,
  .secondary-action,
  .quiet-action,
  .branch-action {
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-4);
    border: 1px solid transparent;
    border-radius: var(--radius);
    cursor: pointer;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-medium);
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .primary-action {
    color: var(--text-on-accent);
    background: var(--accent);
  }
  .primary-action:hover:not(:disabled),
  .primary-action:focus-visible:not(:disabled) {
    background: var(--accent-light);
    outline: none;
  }
  .secondary-action {
    color: var(--fg);
    background: var(--glass);
  }
  .quiet-action {
    color: var(--fg-dim);
    background: transparent;
  }
  .secondary-action:hover,
  .quiet-action:hover,
  .secondary-action:focus-visible,
  .quiet-action:focus-visible {
    color: var(--fg);
    background: var(--glass-bright);
    outline: none;
  }
  .branch-action {
    color: var(--accent);
    border-color: color-mix(in srgb, var(--accent) 28%, transparent);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
  }
  .branch-action:hover:not(:disabled),
  .branch-action:focus-visible:not(:disabled) {
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    outline: none;
  }
  .primary-action:active:not(:disabled),
  .secondary-action:active:not(:disabled),
  .quiet-action:active:not(:disabled),
  .branch-action:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }
  .primary-action:disabled,
  .branch-action:disabled {
    opacity: 0.42;
    cursor: not-allowed;
  }
  .path-stream {
    display: flex;
    flex-direction: column;
    width: min(100%, 820px);
    margin: 0 auto;
  }
  .path-snippet {
    display: grid;
    grid-template-columns: 28px minmax(0, 1fr);
    min-width: 0;
  }
  .snippet-rail {
    position: relative;
    display: flex;
    justify-content: center;
  }
  .snippet-rail::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1px;
    background: color-mix(in srgb, var(--fg-muted) 38%, transparent);
  }
  .path-snippet:first-child .snippet-rail::before {
    top: 24px;
  }
  .path-snippet:last-child .snippet-rail::before {
    bottom: calc(100% - 24px);
  }
  .snippet-rail span {
    position: relative;
    z-index: 1;
    width: 9px;
    height: 9px;
    margin-top: var(--space-md);
    border: 2px solid var(--bg);
    border-radius: 50%;
    background: var(--fg-muted);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--fg-muted) 48%, transparent);
  }
  .current-snippet .snippet-rail span {
    background: var(--accent);
    box-shadow: 0 0 0 5px color-mix(in srgb, var(--accent) 10%, transparent);
  }
  .snippet-card {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-width: 0;
    margin: 0 0 var(--space-3) var(--space-2);
    padding: var(--surface-padding);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--bg-elev);
    box-shadow: inset 0 1px 0 color-mix(in srgb, var(--fg) 4%, transparent);
  }
  .current-snippet .snippet-card {
    border-color: color-mix(in srgb, var(--accent) 42%, var(--glass-line));
    background: var(--surface-sheen), color-mix(in srgb, var(--accent) 5%, var(--bg-elev));
  }
  .snippet-card > header,
  .option-card > header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
  }
  .snippet-role {
    color: var(--fg-dim);
  }
  .snippet-index,
  .option-number {
    color: var(--fg-muted);
  }
  .current-label {
    padding: var(--space-xs) var(--space-xs);
    border-radius: var(--radius-pill);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }
  .snippet-actions {
    display: inline-grid;
    place-items: center;
    width: var(--control-target);
    height: var(--control-target);
    margin-inline-start: auto;
    padding: 0;
    border: 0;
    border-radius: var(--radius-pill);
    color: var(--fg-muted);
    background: transparent;
    cursor: pointer;
    font-family: var(--font-data);
  }
  .snippet-actions:hover,
  .snippet-actions:focus-visible {
    color: var(--fg);
    background: var(--glass-bright);
    outline: none;
  }
  .snippet-text,
  .option-preview {
    width: 100%;
    min-width: 0;
    padding: 0;
    border: 0;
    color: var(--fg);
    background: transparent;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-reading);
    font-size: var(--text-md);
    line-height: 1.5;
    text-align: start;
  }
  .snippet-text:focus-visible,
  .option-preview:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 4px;
  }
  .snippet-tokens {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0;
    min-width: 0;
  }
  .path-token {
    min-width: 24px;
    min-height: 24px;
    font-family: var(--font-reading) !important;
    font-weight: var(--weight-reading) !important;
    padding: 0;
    border: 0;
    border-radius: var(--radius-sm);
    color: var(--fg);
    background: transparent;
    cursor: pointer;
    font: inherit;
    font-family: var(--font-reading);
    font-size: var(--text-md);
    line-height: 1.45;
    text-align: start;
  }
  .path-token:hover,
  .path-token:focus-visible {
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    outline: none;
  }
  @media (pointer: coarse) {
    .path-token { min-width: var(--control-target); min-height: var(--control-target); }
  }
  .snippet-card > footer,
  .option-card > footer {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: var(--space-2);
  }
  .option-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
    gap: var(--space-4);
    width: min(100%, 1080px);
    margin: 0 auto;
  }
  .option-card {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 210px;
    padding: var(--surface-padding);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
    background: var(--bg-elev);
    box-shadow: inset 0 1px 0 color-mix(in srgb, var(--fg) 4%, transparent);
  }
  .option-card.active-option {
    border-color: color-mix(in srgb, var(--accent) 48%, var(--glass-line));
    background: color-mix(in srgb, var(--accent) 5%, var(--bg-elev));
  }
  .option-preview {
    display: -webkit-box;
    overflow: hidden;
    flex: 1 1 auto;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 6;
    line-clamp: 6;
  }
  .saved-mark {
    color: var(--accent-yellow);
  }
  .projection-empty {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    min-height: 240px;
    text-align: center;
  }
  .projection-empty strong {
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text-lg);
    font-weight: var(--weight-display);
  }
  .saved-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    width: min(100%, 900px);
    margin: 0 auto;
  }
  .saved-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--surface-padding);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--bg-elev);
  }
  .saved-copy {
    display: flex;
    flex: 1 1 auto;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .saved-copy strong {
    overflow: hidden;
    color: var(--fg);
    font-family: var(--font-reading);
    font-weight: var(--weight-reading-medium);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .saved-copy p {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }

  .loom-map-bar {
    display: flex;
    flex: 0 0 auto;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-3) var(--surface-gutter);
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }
  .loom-map-summary,
  .loom-view-controls {
    display: flex;
    align-items: center;
  }
  .loom-map-summary {
    min-width: 0;
    gap: var(--space-2);
    white-space: nowrap;
  }
  .loom-map-summary strong {
    color: var(--fg-strong);
    font-family: var(--font-structure);
    font-weight: var(--weight-medium);
  }
  .map-separator {
    color: var(--fg-subtle);
  }
  .path-key {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    margin-inline-start: var(--space-3);
  }
  .path-key i {
    display: block;
    width: 1.4rem;
    height: 2px;
    border-radius: var(--radius-pill);
    background: var(--accent);
  }
  .loom-view-controls {
    flex: none;
    font-size: var(--text-sm);
    gap: var(--space-xs);
    padding: var(--space-xs);
    border-radius: var(--radius-group);
    background: var(--surface-sheen), var(--input-well);
  }
  .loom-view-controls button,
  .loom-view-controls output {
    display: inline-grid;
    min-width: var(--control-target);
    min-height: var(--control-target);
    padding: 0 var(--space-2);
    place-items: center;
    border: 0;
    border-radius: var(--radius-inset);
    color: var(--fg-dim);
    background: transparent;
    font: inherit;
  }
  .loom-view-controls button {
    cursor: pointer;
  }
  .loom-view-controls button:hover:not(:disabled),
  .loom-view-controls button:focus-visible {
    color: var(--fg);
    background: var(--glass-bright);
    outline: none;
  }
  .loom-view-controls button:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }
  .loom-view-controls button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .loom-view-controls output {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    white-space: nowrap;
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
  }

  .tree-scroll {
    position: relative;
    isolation: isolate;
    overflow: clip;
    overscroll-behavior: contain;
    padding: 0;
    min-height: 0;
    flex: 1 1 auto;
    touch-action: none;
    cursor: grab;
    background: color-mix(in srgb, var(--bg) 90%, var(--bg-alt));
  }
  .tree-scroll::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 4;
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 5%, transparent),
      var(--loom-canvas-shade);
  }
  .tree-scroll.dragging,
  .tree-scroll.dragging :global(*) {
    cursor: grabbing !important;
  }
  .touch-gesture-hint {
    display: none;
  }

  .loom-depth-field {
    position: absolute;
    z-index: 0;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
  }
  .loom-depth-field::before,
  .loom-depth-field::after {
    content: "";
    position: absolute;
    inset: calc(-12% - 48px);
    pointer-events: none;
    background-repeat: repeat;
  }
  .loom-depth-field::before {
    background-image: radial-gradient(
      circle,
      color-mix(in srgb, var(--fg-subtle) 42%, transparent) 1px,
      transparent 1.4px
    );
    background-size: var(--loom-grid-size) var(--loom-grid-size);
    background-position: var(--loom-grid-x) var(--loom-grid-y);
    opacity: 0.52;
  }
  .loom-depth-field::after {
    background-image: radial-gradient(
      circle,
      color-mix(in srgb, var(--fg-subtle) 30%, transparent) 0.7px,
      transparent 1.1px
    );
    background-size: var(--loom-far-size) var(--loom-far-size);
    background-position: var(--loom-far-x) var(--loom-far-y);
    opacity: 0.35;
  }

  .loom-canvas {
    position: relative;
    z-index: 1;
    min-width: 100%;
    min-height: 100%;
    transform-origin: 0 0;
    will-change: transform;
  }
  .loom-canvas.camera-animating {
    transition: transform var(--dur) var(--ease-enter);
  }
  .loom-edges {
    position: absolute;
    inset: 0;
    overflow: visible;
    pointer-events: none;
  }
  .loom-edges path {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    vector-effect: non-scaling-stroke;
    transition:
      stroke var(--dur-fast) var(--ease-out),
      opacity var(--dur-fast) var(--ease-out);
  }
  .loom-edges path.edge-depth {
    stroke: var(--loom-edge-depth);
    stroke-width: calc(var(--edge-width, 1.5px) + 7px);
    opacity: 0.5;
    transform: translateY(3px);
    filter: blur(1.5px);
  }
  .loom-edges path.edge-line {
    stroke: color-mix(in srgb, var(--fg-muted) 58%, var(--bg));
    stroke-width: var(--edge-width, 1.5px);
    opacity: var(--edge-opacity, 0.55);
  }
  .loom-edges path.edge-depth.active-edge {
    stroke: color-mix(in srgb, var(--accent) 32%, var(--loom-edge-depth));
    opacity: 0.7;
  }
  .loom-edges path.edge-line.active-edge {
    stroke: var(--accent);
    opacity: 0.92;
  }
  .loom-edges path.filtered-edge {
    opacity: 0.18;
  }
  .loom-edges circle.junction-halo {
    fill: color-mix(in srgb, var(--bg) 72%, transparent);
    stroke: transparent;
  }
  .loom-edges circle.junction-halo.active-junction {
    fill: color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .loom-edges circle.junction-core {
    fill: var(--bg-elev);
    stroke: var(--fg-muted);
    stroke-width: 1.5px;
  }
  .loom-edges circle.junction-core.active-junction {
    fill: var(--accent);
    stroke: color-mix(in srgb, var(--accent) 30%, var(--bg));
  }
  .tree-node-wrap {
    position: absolute;
    box-sizing: border-box;
    z-index: calc(10 + var(--branch-depth, 0));
    min-width: 0;
    min-height: 0;
    transition:
      left var(--dur) var(--ease-out),
      top var(--dur) var(--ease-out),
      opacity var(--dur-fast) var(--ease-out);
  }
  .tree-node-wrap.active-path-node {
    z-index: calc(30 + var(--branch-depth, 0));
  }
  .tree-node-wrap.current-node {
    z-index: 60;
  }
  .tree-node-wrap.dead-node {
    z-index: calc(5 + var(--branch-depth, 0));
  }

  .tree-node-wrap.filtered-out :global(.node) {
    color: var(--fg-muted);
    opacity: 0.55;
  }
  .tree-node-wrap.filtered-out:hover :global(.node) {
    color: var(--fg);
    opacity: 1;
  }
  .tree-node-wrap.selected {
    --loom-node-outline: var(--accent);
  }
  .tree-node-wrap.selected :global(.node) {
    background: var(--accent-subtle);
  }
  .tree-node-wrap.pinned {
    --loom-node-outline: var(--fg-muted);
  }

  @media (prefers-reduced-motion: reduce) {
    .tree-node-wrap,
    .loom-edges path {
      transition: none;
    }
  }

  @media (prefers-contrast: more), (forced-colors: active) {
    .loom-depth-field {
      display: none;
    }
  }

  @media (max-width: 560px) {
    .loom-header {
      padding-inline: var(--surface-padding);
    }
    .loom-heading {
      flex-basis: 100%;
    }
    .loom-subtitle {
      display: none;
    }
    .loom-views {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      padding-inline: var(--space-4);
    }
    .loom-views button {
      padding-inline: var(--space-3);
    }
    .loom-projection {
      padding-inline: var(--surface-padding);
    }
    .projection-heading,
    .saved-card {
      align-items: stretch;
      flex-direction: column;
    }
    .projection-actions,
    .saved-actions {
      width: 100%;
    }
    .projection-actions button,
    .saved-actions .primary-action,
    .saved-actions .quiet-action {
      flex: 1 1 auto;
    }
    .path-snippet {
      grid-template-columns: 18px minmax(0, 1fr);
    }
    .snippet-card,
    .option-card,
    .saved-card {
      padding: var(--surface-padding);
    }
    .snippet-card > footer,
    .option-card > footer {
      align-items: stretch;
      flex-direction: column;
    }
    .snippet-card > footer button,
    .option-card > footer button {
      width: 100%;
    }
    .loom-map-bar {
      align-items: flex-start;
      flex-direction: column;
    }
    .loom-view-controls {
      width: 100%;
    }
    .loom-view-controls button,
    .loom-view-controls output {
      flex: 1 1 auto;
    }
    .path-key {
      display: none;
    }
  }

  /* Short screens keep the view chooser on one touch-sized row. */
  @media (max-width: 560px), (max-height: 600px) {
    .loom-header {
      flex: 0 0 auto;
      flex-wrap: nowrap;
      gap: var(--space-1);
      padding: var(--surface-padding);
      overflow-x: auto;
      overscroll-behavior-inline: contain;
      scrollbar-width: none;
    }

    .loom-header::-webkit-scrollbar,
    .loom-views::-webkit-scrollbar,
    .loom-map-bar::-webkit-scrollbar {
      display: none;
    }

    .loom-heading {
      flex: 0 0 auto;
    }

    .title {
      font-size: var(--text-sm);
      white-space: nowrap;
    }

    .loom-subtitle {
      display: none;
    }

    .loom-header .action-btn,
    .loom-header .icon-btn {
      flex: 0 0 auto;
      min-height: var(--control-target);
      padding-inline: var(--space-3);
    }

    .loom-views {
      display: none;
    }

    .compact-loom-view { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: var(--space-sm); flex: 0 0 auto; padding: var(--space-xs) var(--surface-gutter); font-size: var(--text-sm); color: var(--fg-muted); }

    .loom-views button {
      flex: 0 0 auto;
      flex-direction: row;
      align-items: center;
      min-height: var(--control-target);
      padding: var(--space-2) var(--space-3);
    }

    .loom-views span {
      display: none;
    }

    .filter-bar {
      flex: 0 0 auto;
      padding: var(--space-1) var(--surface-gutter);
    }

    .filter-input {
      min-height: var(--control-target);
    }
    .filter-field { flex-basis: 8rem; }

    .loom-map-bar {
      flex: 0 0 auto;
      flex-wrap: nowrap;
      flex-direction: row;
      align-items: center;
      gap: var(--space-1);
      padding: var(--space-1) var(--surface-gutter);
      overflow-x: auto;
      overscroll-behavior-inline: contain;
      scrollbar-width: none;
    }

    .loom-map-summary {
      display: none;
    }

    .loom-view-controls {
      width: max-content;
      min-width: 0;
      margin-inline: auto;
      padding: 0;
      background: transparent;
    }

    .loom-view-controls button,
    .loom-view-controls output {
      flex: 0 0 auto;
      min-width: var(--control-target);
      min-height: var(--control-target);
      background: var(--input-well);
    }

    .tree-scroll {
      min-height: min(6rem, 25dvh);
    }

    .loom-projection {
      padding: var(--surface-padding);
    }

    .projection-heading {
      margin-bottom: var(--space-4);
    }
  }

  @media (max-width: 760px), (max-height: 600px) {
    .filter-input,
    .modal-body input,
    .modal-body textarea {
      font-size: var(--text-input-touch);
    }
  }

  @media (pointer: coarse) {
    .touch-gesture-hint {
      position: absolute;
      inset-inline-start: var(--space-3);
      bottom: var(--space-3);
      z-index: 5;
      display: block;
      margin: 0;
      padding: var(--space-1) var(--space-3);
      border-radius: var(--radius-pill);
      color: var(--fg-muted);
      background: color-mix(in srgb, var(--bg) 78%, transparent);
      font-size: var(--text-2xs);
      pointer-events: none;
      backdrop-filter: blur(8px);
    }
  }

  .empty {
    padding: var(--space-5);
    color: var(--fg-muted);
    font-size: var(--text-sm);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .empty:not(.err) {
    flex: 1 1 auto;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  .empty:not(.err) p:first-child {
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text-lg);
    font-weight: var(--weight-display);
  }
  .empty p {
    margin: 0;
  }
  .empty.err {
    color: var(--accent-red);
  }
  .empty button {
    align-self: flex-start;
    background: var(--danger-bg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    color: var(--accent-red);
    padding: var(--space-1) var(--space-4);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
    min-height: var(--control-target);
  }
  .empty button:hover {
    background: var(--danger-hover);
  }

  .hint {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .modal-error {
    font-size: var(--text-sm);
    margin-top: var(--space-3);
    color: var(--error-text);
    background: var(--surface-sheen), var(--popup-bg);
    border-radius: var(--radius-sm);
    padding: var(--surface-padding);
    font-family: var(--font-mono);
  }

  @media (prefers-contrast: more) {
    .modal-error { background: var(--bg); }
  }
  @media (forced-colors: active) {
    .modal-error { background: Canvas; color: CanvasText; }
  }

  .loom-menu {
    position: fixed;
    box-sizing: border-box;
    margin: 0;
    z-index: var(--z-modal);
    background: var(--surface-sheen), var(--popup-bg);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    padding: var(--surface-padding);
    box-shadow: var(--popup-shadow);
    min-width: 248px;
    width: min(280px, calc(100vw - 24px));
    max-width: calc(100vw - 16px);
    max-height: min(680px, calc(100dvh - 24px));
    overflow-y: auto;
    overscroll-behavior: contain;
    display: flex;
    flex-direction: column;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    opacity: 0;
    pointer-events: none;
  }
  .loom-menu.positioned {
    opacity: 1;
    pointer-events: auto;
  }
  .loom-menu-context {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
    padding: var(--space-4) var(--space-4) var(--space-5);
  }
  .loom-menu-context span {
    color: var(--accent);
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .loom-menu-context strong {
    overflow: hidden;
    color: var(--fg);
    font-family: var(--font-reading);
    font-weight: var(--weight-reading-medium);
    line-height: 1.35;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .loom-menu-section {
    display: flex;
    flex-direction: column;
    padding: var(--space-2) 0;
    border-top: 1px solid color-mix(in srgb, var(--glass-line) 72%, transparent);
  }
  .loom-menu button {
    background: transparent;
    border: 0;
    text-align: start;
    padding: var(--space-3) var(--space-4);
    color: var(--fg-strong);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-structure);
    font-weight: var(--weight-reading-medium);
    min-height: var(--control-target);
    border-radius: var(--radius);
  }
  .loom-menu button:hover:not(:disabled),
  .loom-menu button:focus-visible:not(:disabled) {
    background: var(--bg-elev);
    color: var(--accent);
    outline: none;
  }
  .loom-menu button.danger {
    color: var(--accent-red);
  }
  .loom-menu button.danger:hover {
    background: color-mix(in srgb, var(--accent-red) 12%, transparent);
  }
  .loom-menu button:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .loom-menu .danger-section {
    padding-bottom: 0;
  }

  .loom-modal-backdrop {
    position: fixed;
    inset: 0;
    background: var(--scrim-strong);
    z-index: var(--z-modal);
    border: 0;
    cursor: pointer;
  }
  .loom-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: calc(var(--z-modal) + 1);
    background: var(--surface-sheen), var(--popup-bg);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    box-shadow: var(--popup-shadow);
    width: min(640px, calc(100vw - 2rem));
    min-width: 0;
    max-height: calc(100dvh - 2rem);
    display: flex;
    flex-direction: column;
    font-family: var(--font-reading);
    font-size: var(--text);
    color: var(--fg);
  }
  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    color: var(--accent);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .modal-header h2 {
    margin: 0;
    font: inherit;
    font-weight: var(--weight-medium);
  }
  .modal-body {
    padding: var(--panel-padding) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
    overflow-y: auto;
  }
  .modal-body label {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .modal-body input,
  .modal-body textarea {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-reading);
    resize: vertical;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
  }
  .modal-body input:focus,
  .modal-body textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .modal-body blockquote {
    margin: 0;
    overflow-wrap: anywhere;
    color: var(--accent-yellow);
    background: var(--surface-sheen), var(--bg-elev);
    padding: var(--surface-padding);
    border-radius: var(--radius);
  }
  .modal-body .danger {
    color: var(--accent-red);
  }
  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .modal-footer button {
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius);
    color: var(--fg-strong);
    padding: var(--space-3) var(--space-5);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
  }
  .modal-footer .primary {
    background: var(--accent);
    color: var(--text-on-accent);
  }
  .modal-footer .primary:hover {
    background: var(--accent-light);
  }
  .modal-footer .danger {
    background: var(--danger-bg);
    color: var(--accent-red);
  }
  .modal-footer .danger:hover {
    background: var(--danger-hover);
  }
  .modal-footer .cancel:hover {
    background: var(--glass-strong);
  }
</style>
