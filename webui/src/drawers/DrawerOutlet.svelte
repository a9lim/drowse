<script lang="ts">
  // Renders a registry drawer, loading its chunk on first use.  Once loaded
  // the component renders synchronously on every later open.  Props other
  // than ``name`` and ``onready`` pass through (the docked token inspector
  // takes extras).  ``onready`` fires once the drawer is in the DOM, which on
  // a first open is after its chunk loads, not on the next tick.
  import type { Component } from "svelte";
  import type { DrawerName } from "../lib/types";
  import { loadDrawer, loadedDrawer } from "./index";

  let { name, onready, ...props }: {
    name: DrawerName;
    params: unknown;
    mobile?: boolean;
    onclose?: () => void;
    onready?: (name: DrawerName) => void;
    [extra: string]: unknown;
  } = $props();

  type AnyDrawer = Component<Record<string, unknown>>;
  let loaded = $state.raw<{ name: DrawerName; drawer: AnyDrawer } | null>(null);
  let failed = $state.raw<{ name: DrawerName; error: unknown } | null>(null);
  const Drawer = $derived(
    (loadedDrawer(name) as AnyDrawer | undefined)
      ?? (loaded?.name === name ? loaded.drawer : undefined),
  );

  $effect(() => {
    const target = name;
    if (loadedDrawer(target)) return;
    let live = true;
    loadDrawer(target).then(
      (drawer) => { if (live) loaded = { name: target, drawer: drawer as AnyDrawer }; },
      (error: unknown) => { if (live) failed = { name: target, error }; },
    );
    return () => { live = false; };
  });

  $effect(() => {
    if (Drawer) onready?.(name);
  });
</script>

{#if Drawer}
  <Drawer {...props} />
{:else if failed?.name === name}
  <p class="drawer-loading" role="alert">This tool could not load: {failed.error instanceof Error ? failed.error.message : String(failed.error)}</p>
{:else}
  <p class="drawer-loading" role="status">Loading…</p>
{/if}

<style>
  .drawer-loading {
    margin: 0;
    padding: var(--space-4);
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
</style>
