<script lang="ts">
  // Renders a registry drawer, loading its chunk on first use.  Once loaded
  // the component renders synchronously on every later open.  Props other
  // than ``name`` pass through (the docked token inspector takes extras).
  import type { Component } from "svelte";
  import type { DrawerName } from "../lib/types";
  import { loadDrawer, loadedDrawer } from "./index";

  let { name, ...props }: {
    name: DrawerName;
    params: unknown;
    mobile?: boolean;
    onclose?: () => void;
    [extra: string]: unknown;
  } = $props();

  type AnyDrawer = Component<Record<string, unknown>>;
  const ready = $derived(loadedDrawer(name) as AnyDrawer | undefined);
  const pending = $derived(ready ? null : loadDrawer(name) as Promise<AnyDrawer>);
</script>

{#if ready}
  {@const Drawer = ready}
  <Drawer {...props} />
{:else if pending}
  {#await pending}
    <p class="drawer-loading" role="status">Loading…</p>
  {:then Drawer}
    <Drawer {...props} />
  {:catch error}
    <p class="drawer-loading" role="alert">This tool could not load: {error instanceof Error ? error.message : String(error)}</p>
  {/await}
{/if}

<style>
  .drawer-loading {
    margin: 0;
    padding: var(--space-4);
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
</style>
