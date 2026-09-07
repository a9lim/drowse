<script lang="ts">
  import { onMount } from "svelte";
  import { createTabIcon, tabTitle, type TabState } from "../tabIdentity";
  let { state: tabState = "home", title }: { state?: TabState; title?: string } = $props();
  let icon = $state<ReturnType<typeof createTabIcon> | null>(null);
  onMount(() => {
    const controller = createTabIcon();
    icon = controller;
    return () => controller.dispose();
  });
  $effect(() => { icon?.update(tabState); });
</script>

<svelte:head><title>{title ?? tabTitle(tabState)}</title></svelte:head>
