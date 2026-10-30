// THE drawer registry — one row per ``DrawerName``.
//
// Typed as ``Record<DrawerName, DrawerEntry>``, so adding a name to the
// union without wiring it here is a compile error, and every fact about a
// drawer lives in one row: the component, the params the host folds in,
// whether it takes the narrow panel, and how it is reached.
//
// "How it is reached" is the load-bearing part.  A non-null ``launcher``
// means the drawer appears in All tools, and ``RAIL_CATEGORIES``
// below is *derived* from those rows rather than maintained beside them —
// the previous hand-written list was the one place a drawer could be
// declared, exported, and rendered while still being unreachable.  A
// drawer that is genuinely opened from a surface rather than the palette
// sets ``launcher: null`` and must then name that surface in ``via``; the
// union type makes "no palette entry" a claim someone wrote down instead
// of an omission nobody noticed.
//
// ``RackDrawer`` is the shared rack browser — one component reskinned by
// geometry family.  ``family: "subspace"`` (flat pca / baked fits, white
// accent) and ``family: "manifold"`` (curved spectral / authored fits,
// purple accent) are mirror images, differing only by accent, label, and
// catalog filter, so they are two registry rows over one component with
// different ``params``.  The "+ build manifold" launcher inside it routes
// to ``manifold_builder`` for both families (a flat fit is just a pca
// manifold).

import type { Component } from "svelte";

import type { DrawerName } from "../lib/types";
import { runtimeClient } from "../lib/runtime/client";
import { drawerAvailability } from "../lib/runtime/ui-capabilities";


/** Each drawer owns its params; responsive drawers can consume the host's
 *  presentation and dismissal props. */
export type DrawerComponent = Component<{ params: unknown; mobile?: boolean; onclose?: () => void }>;

/** Palette group keys.  Typing ``launcher.group`` as this union means a
 *  typo can't strand a tool in a group that never renders. */
export type DrawerGroupKey = "manifolds" | "analysis" | "session";

export interface DrawerLauncher {
  group: DrawerGroupKey;
  /** Palette entry label, trailing ellipsis included. */
  label: string;
  /** Extra match text for palette filtering (synonyms, old names). */
  keywords?: string;
}

/** Drawers load on first open: the registry is imported by the app shell,
 *  so a static import here would put every drawer in the first paint. */
type DrawerLoader = () => Promise<{ default: DrawerComponent }>;

interface DrawerBase {
  load: DrawerLoader;
  /** Folded into ``drawerState.params`` before render — how one component
   *  serves two drawer names. */
  params?: Record<string, unknown>;
  /** Content-driven sizing: forms and pickers get the narrow panel, while
   *  analysis views keep the wide one. */
  narrow?: boolean;
}

/** A drawer is either palette-launched or reached from a named surface —
 *  never neither. */
export type DrawerEntry =
  | (DrawerBase & { launcher: DrawerLauncher; via?: never })
  | (DrawerBase & { launcher: null; via: string });

/** Palette groups, in display order.  Every ``DrawerGroupKey`` appears
 *  exactly once. */
export const DRAWER_GROUPS: readonly { key: DrawerGroupKey; label: string }[] = [
  // The single steering-authoring surface.  Concepts are manifolds now —
  // a flat (2-node / personas) fit is just a pca manifold — so there's no
  // separate "subspaces" group; flat authoring folds into the manifold
  // builder's pca path.  The catalog is the shared RackDrawer
  // (family-split), reached from the rack "+" buttons.
  { key: "manifolds", label: "Shape responses" },
  { key: "analysis", label: "Understand responses" },
  { key: "session", label: "Settings and help" },
];

/** Declaration order is the palette order within a group. */
export const DRAWERS: Record<DrawerName, DrawerEntry> = {
  appearance: {
    load: () => import("./AppearanceDrawer.svelte"),
    narrow: true,
    launcher: { group: "session", label: "Appearance…", keywords: "theme background wallpaper image pixel dither" },
  },
  // ----------------------------------------------- palette-launched ----
  manifold_builder: {
    load: () => import("./ManifoldBuilderDrawer.svelte"),
    narrow: true,
    launcher: {
      group: "manifolds",
      label: "Create a concept or scale…",
      keywords: "extract author create concept vector fit",
    },
  },
  manifold_merge: {
    load: () => import("./ManifoldMergeDrawer.svelte"),
    launcher: {
      group: "manifolds",
      label: "Combine response controls…",
      keywords: "union corpora",
    },
  },
  manifold_pack: {
    load: () => import("./ManifoldPacksDrawer.svelte"),
    launcher: {
      group: "manifolds",
      label: "Manage downloaded controls…",
      keywords: "pack packs install search huggingface hub catalog",
    },
  },
  template_lab: {
    load: () => import("./TemplateLabDrawer.svelte"),
    launcher: {
      group: "manifolds",
      label: "Test prompt templates…",
      keywords: "score completion slot restricted choice",
    },
  },
  cast: {
    load: () => import("./CastDrawer.svelte"),
    launcher: {
      group: "manifolds",
      label: "Role settings…",
      keywords: "role behavior guidance speaker",
    },
  },
  correlation: {
    load: () => import("./CorrelationDrawer.svelte"),
    launcher: {
      group: "analysis",
      label: "Compare saved readings…",
      keywords: "cosine similarity vectors",
    },
  },
  compare: {
    load: () => import("./CompareDrawer.svelte"),
    launcher: {
      group: "analysis",
      label: "Compare controls by layer…",
      keywords: "cross-layer cosine",
    },
  },
  health: {
    load: () => import("./HealthDrawer.svelte"),
    launcher: { group: "session", label: "Model health…", keywords: "device dtype" },
  },
  session_admin: {
    load: () => import("@runtime-session-drawer"),
    launcher: { group: "session", label: "API access…", keywords: "auth api key bearer" },
  },
  local_runtime: {
    load: () => import("@runtime-local-drawer"),
    launcher: {
      group: "session",
      label: "Model settings…",
      keywords: "local device model storage gpu webgpu offline delete unload switch diagnostics",
    },
  },
  help: {
    load: () => import("./HelpDrawer.svelte"),
    launcher: {
      group: "session",
      label: "Help and shortcuts…",
      keywords: "keyboard grammar cheatsheet",
    },
  },

  feedback: {
    load: () => import("./FeedbackDrawer.svelte"),
    narrow: true,
    launcher: { group: "session", label: "Submit Feedback…", keywords: "contact support bug suggestion feedback" },
  },

  // -------------------------------------------- opened in context -----
  subspace: {
    load: () => import("./RackDrawer.svelte"),
    params: { family: "subspace" },
    narrow: true,
    launcher: null,
    via: 'the steering and probe racks\' "+ add" buttons',
  },
  manifolds: {
    load: () => import("./RackDrawer.svelte"),
    params: { family: "manifold" },
    narrow: true,
    launcher: null,
    via: 'the steering and probe racks\' "+ add" buttons',
  },
  save_conversation: {
    load: () => import("./SaveConversationDrawer.svelte"),
    narrow: true,
    launcher: {
      group: "session",
      label: "Save current chat…",
      keywords: "conversation name local storage backup",
    },
  },
  download_chat: {
    load: () => import("./DownloadChatDrawer.svelte"),
    narrow: true,
    launcher: null,
    via: "the workbench header download button",
  },
  load_conversation: {
    load: () => import("./LoadConversationDrawer.svelte"),
    narrow: true,
    launcher: {
      group: "session",
      label: "Saved chats…",
      keywords: "conversation library open rename avatar delete import backup",
    },
  },
  system_prompt: {
    load: () => import("./SystemPromptDrawer.svelte"),
    narrow: true,
    launcher: null,
    via: "the sampling strip's system-prompt button",
  },
  advanced_sampling: {
    load: () => import("./AdvancedSamplingDrawer.svelte"),
    launcher: null,
    via: "the sampling strip's advanced button",
  },
  token_drilldown: {
    load: () => import("./TokenDrilldownDrawer.svelte"),
    launcher: null,
    via: "selecting a transcript or raw-buffer token",
  },
  probe_inspector: {
    load: () => import("./ProbeInspectorDrawer.svelte"),
    launcher: null,
    via: "a probe card's ⓘ button",
  },
  node_compare: {
    load: () => import("./NodeCompareDrawer.svelte"),
    launcher: null,
    via: "the loom sidebar's compare actions",
  },
  transcript: {
    load: () => import("./TranscriptDrawer.svelte"),
    launcher: {
      group: "analysis",
      label: "Conversation transcript…",
      keywords: "transcript export import yaml conversation",
    },
  },
};

const loadedDrawers = new Map<DrawerName, DrawerComponent>();

/** The drawer's component when it has already loaded, so reopening renders
 *  synchronously. */
export function loadedDrawer(name: DrawerName): DrawerComponent | undefined {
  return loadedDrawers.get(name);
}

/** Load (once) and return the drawer's component. */
export async function loadDrawer(name: DrawerName): Promise<DrawerComponent> {
  const cached = loadedDrawers.get(name);
  if (cached) return cached;
  const component = (await DRAWERS[name].load()).default;
  loadedDrawers.set(name, component);
  return component;
}

/** The props the host passes to the drawer component: whatever
 *  ``openDrawer`` carried, with the registry's fixed params folded on
 *  top. */
export function drawerParams(name: DrawerName, params: unknown): unknown {
  const fixed = DRAWERS[name].params;
  if (!fixed) return params;
  return { ...(params as Record<string, unknown>), ...fixed };
}

export interface RailTool {
  label: string;
  drawer: DrawerName;
  keywords?: string;
}

export interface RailCategory {
  key: DrawerGroupKey;
  label: string;
  tools: RailTool[];
}

/** The All tools launcher list, DERIVED from the registry — a tool cannot go
 *  missing here without its row also losing its ``launcher``. */
export const RAIL_CATEGORIES: RailCategory[] = DRAWER_GROUPS.map((group) => {
  const tools: RailTool[] = [];
  for (const [drawer, entry] of Object.entries(DRAWERS) as [
    DrawerName,
    DrawerEntry,
  ][]) {
    if (entry.launcher === null || entry.launcher.group !== group.key) continue;
    if (runtimeClient.mode === "http" && drawer === "local_runtime") continue;
    if (runtimeClient.mode !== "http" && drawer === "health") continue;
    if (runtimeClient.mode !== "http" && drawer === "session_admin") continue;
    if (!drawerAvailability(drawer).available) continue;
    tools.push({
      label: entry.launcher.label,
      drawer,
      keywords: entry.launcher.keywords,
    });
  }
  return { key: group.key, label: group.label, tools };
});
