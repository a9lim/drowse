<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import { closeDrawer } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => void _drawerProps.params);
</script>

<section class="drawer-shell" aria-label="Unavailable runtime drawer">
  <header>
    <div>
      <h2>This tool is not available here</h2>
      <p>Close this panel and choose an available action from All tools.</p>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>
</section>

<style>
  .drawer-shell {
    min-height: 100%;
    color: var(--fg);
  }

  header {
    display: flex;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }

  h2 {
    margin: 0;
    color: var(--fg);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }

  p {
    margin: var(--space-2) 0 0;
    color: var(--fg-muted);
  }
</style>
