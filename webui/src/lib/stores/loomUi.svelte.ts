// Loom sidebar UI state — everything the threads column renders that is
// NOT the tree itself.
//
// Four independent knobs, all in-memory (the server recomputes each from
// primitives, so none of it is persisted): the modal-request signal the
// global Ctrl+R/E/B/N/D accelerators poke, the lazy steering-delta edge
// label cache, the filter-grammar slice (server-resolved node set plus the
// client-side ``sort:`` term), and the two selection surfaces — the pinned
// comparison node and the multi-select feeding the cross-branch diff
// drawer.

import { SvelteMap } from "svelte/reactivity";
import { apiTree, ApiError } from "../runtime/services";
import { userFacingError } from "../runtime/userFacingError";
import { loomTree } from "./loom.svelte";
import { loomTextMatches } from "../loomSearch";

/** Sidebar-modal kind, also pokeable from App.svelte via the global
 *  Ctrl+R/E/B/N/D shortcuts.  ``null`` = no modal. */
export type LoomModalKind =
  | null
  | "regenerate"
  | "edit"
  | "branch"
  | "delete"
  | "note"
  | "navpicker"
  | "search";

export interface LoomUiState {
  /** Synchronized Loom subview. The authoritative tree is shared; this only
   * changes which projection of it is visible. */
  view: "weave" | "map" | "path" | "options" | "saved";
  /** Request flag: when the App's Ctrl+R/etc handlers want to open a
   *  modal inside the sidebar, they bump this counter and the sidebar
   *  reacts.  Counter lets the same modal be re-requested back-to-back
   *  (e.g. user closes regen modal then hits Ctrl+R again). */
  modalRequest: {
    seq: number;
    kind: LoomModalKind;
    nodeId: string | null;
    text: string;
    n: number;
  };
  /** Logit-pass: sibling sort key derived from filter grammar
   *  ``sort:surprise`` / ``sort:confidence``.  ``"default"`` preserves
   *  server insertion order.  Parsed client-side out of the filter
   *  input before the rest of the expression is sent to the server. */
  siblingSort: "default" | "surprise" | "confidence";
  /** Filter help popover visibility (Decision 8).  Toggled by the
   *  ``?`` button next to the filter input. */
  filterHelpOpen: boolean;
}

/** Loom (threads) UI state.  The threads column is a permanent part of
 *  the layout, so this carries no open/closed flag — only the modal
 *  request signal and the sort / filter-help knobs. */
export const loomUiState: LoomUiState = $state({
  view: "map",
  modalRequest: { seq: 0, kind: null, nodeId: null, text: "", n: 1 },
  siblingSort: "default",
  filterHelpOpen: false,
});

// ================================ steering-delta edge labels ==========
//
// Lazy per-edge label cache.  In-memory only; not persisted across
// reloads (the engine recomputes it from primitives).

/** Lazy cache of steering-delta labels for `parent_id|child_id` edges.
 *  The sidebar fetches on first render; SvelteMap so individual entries
 *  trigger reactivity in the edge components. */
export const edgeLabelCache: Map<string, string> = $state(new SvelteMap());

/** In-flight fetch dedupe — keys we've already kicked off a request
 *  for.  Cleared after the response lands so retries are possible
 *  when the rev changes. */
const _edgeLabelInFlight = new Map<string, object>();

function _edgeKey(parentId: string, childId: string): string {
  return `${parentId}|${childId}`;
}

/** Fetch (and cache) the steering-delta label for an edge.  Returns
 *  immediately when the entry is already cached.  Bumps reactivity
 *  when the label arrives so all consumers re-render. */
export function fetchEdgeLabel(parentId: string, childId: string): void {
  const key = _edgeKey(parentId, childId);
  if (edgeLabelCache.has(key)) return;
  if (_edgeLabelInFlight.has(key)) return;
  const request = {};
  _edgeLabelInFlight.set(key, request);
  apiTree
    .edgeLabel(parentId, childId)
    .then((r) => {
      if (_edgeLabelInFlight.get(key) === request) edgeLabelCache.set(key, r.label);
    })
    .catch(() => {
      // Labels are optional. Leave failures uncached so the next tree or
      // view update can retry; the fetch effect does not observe this cache.
    })
    .finally(() => {
      if (_edgeLabelInFlight.get(key) === request) _edgeLabelInFlight.delete(key);
    });
}

/** Invalidate changed edges, or every edge when replacing the whole tree. */
export function invalidateEdgeLabels(keys?: ReadonlySet<string>): void {
  if (keys === undefined) {
    edgeLabelCache.clear();
    _edgeLabelInFlight.clear();
    return;
  }
  for (const key of keys) {
    edgeLabelCache.delete(key);
    _edgeLabelInFlight.delete(key);
  }
}

// ----------------------------------------------------- filter --------

export interface FilterState {
  mode: "text" | "advanced";
  /** User-entered expression string.  Empty = filter off. */
  expr: string;
  /** Text or advanced-filter matching ids. When ``expr`` is empty this is
   *  ``null`` — the UI then renders every node at full opacity. */
  matchingIds: Set<string> | null;
  /** Last parse / fetch error to surface in the input. */
  error: string | null;
  /** Pending state for the spinner. */
  loading: boolean;
}

export const filterState: FilterState = $state({
  mode: "text",
  expr: "",
  matchingIds: null,
  error: null,
  loading: false,
});

let filterRequest = 0;

/** Strip ``sort:surprise`` / ``sort:confidence`` terms out of the filter
 *  expression before it reaches the server.  Sort is a client-side
 *  rendering concern (the DFS walk in LoomSidebar reorders siblings),
 *  so the server filter grammar doesn't need to know about it.  Stashes
 *  the resolved mode on ``loomUiState.siblingSort`` and returns the
 *  cleaned expression for the server.  Unknown ``sort:`` values fall
 *  through to the server, which will surface a parse error — that's
 *  the right UX (typo discovery), better than silently dropping. */
function _consumeSortPrefix(expr: string): string {
  // Match a comma-separated ``sort:<value>`` term anywhere in the
  // expression.  Comma is the filter grammar's AND separator so this
  // composes cleanly with other terms.
  const sortRe = /(?:^|,)\s*sort:(surprise|confidence)\s*(?=,|$)/gi;
  let mode: "default" | "surprise" | "confidence" = "default";
  const cleaned = expr.replace(sortRe, (_match, value: string) => {
    mode = value.toLowerCase() as "surprise" | "confidence";
    return "";
  });
  loomUiState.siblingSort = mode;
  // Drop leading / trailing commas and collapse double commas left by
  // the replace.
  return cleaned.replace(/,,+/g, ",").replace(/^\s*,|,\s*$/g, "").trim();
}

export async function applyTreeFilter(expr: string, mode = filterState.mode): Promise<void> {
  const request = ++filterRequest;
  const rootId = loomTree.root_id;
  const current = () => request === filterRequest && rootId === loomTree.root_id;
  filterState.expr = expr;
  filterState.mode = mode;
  const trimmed = expr.trim();
  if (!trimmed) {
    filterState.matchingIds = null;
    filterState.error = null;
    filterState.loading = false;
    loomUiState.siblingSort = "default";
    return;
  }
  const nodes = [...loomTree.nodes.values()];
  if (mode === "text") {
    loomUiState.siblingSort = "default";
    filterState.matchingIds = new Set(nodes.filter(node => loomTextMatches(node.text ?? "", trimmed)).map(node => node.id));
    filterState.error = null;
    filterState.loading = false;
    return;
  }
  // Logit-pass: peel the client-side sort term off before sending to
  // the server.  Server filter grammar stays unchanged.
  const advancedExpr = _consumeSortPrefix(trimmed);
  if (!advancedExpr) {
    // Only ``sort:...`` was provided — no node-set filter, just a sort
    // directive.  Clear the matching-set so every node renders; the
    // sidebar's DFS picks up ``siblingSort`` independently.
    filterState.matchingIds = null;
    filterState.error = null;
    filterState.loading = false;
    return;
  }
  const clauses = advancedExpr.split(",").map(clause => clause.trim()).filter(Boolean);
  const textClauses = clauses.filter(clause => /^text:/i.test(clause)).map(clause => clause.slice(5).trim());
  const starred = clauses.some(clause => clause.toLowerCase() === "starred");
  const serverExpr = clauses.filter(clause => !/^text:/i.test(clause) && clause.toLowerCase() !== "starred").join(",");
  const localIds = new Set(nodes.filter(node => (!starred || node.starred) && textClauses.every(text => loomTextMatches(node.text ?? "", text))).map(node => node.id));
  filterState.loading = true;
  filterState.error = null;
  filterState.matchingIds = null;
  try {
    if (textClauses.some(text => !text)) throw new Error("Enter words after text:.");
    const ids = serverExpr ? (await apiTree.filter(serverExpr)).matching_node_ids : [...localIds];
    if (!current()) return;
    filterState.matchingIds = new Set(ids.filter(id => localIds.has(id)));
  } catch (e) {
    if (!current()) return;
    if (e instanceof ApiError) {
      const detail =
        e.body && typeof e.body === "object" && "detail" in (e.body as object)
          ? String((e.body as { detail: unknown }).detail)
          : e.message;
      filterState.error = userFacingError(detail, "The conversation could not be searched.");
    } else {
      filterState.error = userFacingError(e, "The conversation could not be searched.");
    }
    filterState.matchingIds = null;
  } finally {
    if (current()) filterState.loading = false;
  }
}

export function clearTreeFilter(): void {
  filterRequest += 1;
  filterState.expr = "";
  filterState.matchingIds = null;
  filterState.error = null;
  filterState.loading = false;
  // Logit-pass: clear the sibling-sort directive too — Esc / ✕ on the
  // filter input is the canonical "go back to default rendering" gesture.
  loomUiState.siblingSort = "default";
}

// ------------------------------------------- branch pinning ----------

/** Pinned-sibling state for the right-column comparison pane.  A node
 *  id (or ``null`` to default to A/B-style shadow).  Set via the
 *  context menu's "Pin to comparison" action. */
export const pinnedComparison: { nodeId: string | null } = $state({
  nodeId: null,
});

export function pinNodeForComparison(nodeId: string | null): void {
  pinnedComparison.nodeId = nodeId;
}

export function unpinComparison(): void {
  pinnedComparison.nodeId = null;
}

// ------------------------------- node multi-select for diff ---------

/** Multi-select for the cross-branch diff drawer.  Right-click on a
 *  generated node toggles its membership; "Compare selected" opens the
 *  drawer with these ids.  Clears on drawer close or successful diff. */
export const nodeSelection: { ids: string[] } = $state({ ids: [] });

export function toggleNodeSelection(nodeId: string): void {
  const idx = nodeSelection.ids.indexOf(nodeId);
  if (idx === -1) nodeSelection.ids = [...nodeSelection.ids, nodeId];
  else nodeSelection.ids = nodeSelection.ids.filter((id) => id !== nodeId);
}

export function clearNodeSelection(): void {
  nodeSelection.ids = [];
}

// ----------------------------------------- modal request signal ------

/** Bump the modalRequest signal so the LoomSidebar opens the named
 *  modal with the given seed values. */
export function requestLoomModal(
  kind: LoomModalKind,
  opts: { nodeId?: string | null; text?: string; n?: number } = {},
): void {
  loomUiState.modalRequest = {
    seq: loomUiState.modalRequest.seq + 1,
    kind,
    nodeId: opts.nodeId ?? loomTree.active_node_id,
    text: opts.text ?? "",
    n: opts.n ?? 1,
  };
}
