<script lang="ts">
  import ContactForm from "../lib/ui/ContactForm.svelte";
  import { CONTACT_ADDRESS } from "../lib/contact";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import { closeDrawer } from "../lib/stores/drawers.svelte";
  import { runtimeClient } from "../lib/runtime/client";
  let { params: _params }: { params: unknown } = $props();
</script>

<section class="feedback-drawer" aria-label="Submit Feedback">
  <header><h2>Share feedback</h2><DrawerCloseButton onclick={closeDrawer} label="Close feedback" /></header>
  <div class="body"><ContactForm source="feedback" ondone={closeDrawer} /><p class="contact-link">Need help? <a href={runtimeClient.mode === "http" ? "https://drowse.ai/contact" : "/contact"} target="_blank" rel="noreferrer">Contact us<span class="sr-only"> (opens a new tab)</span></a> or email <a href={`mailto:${CONTACT_ADDRESS}`}>{CONTACT_ADDRESS}</a>.</p></div>
</section>

<style>
  .feedback-drawer { display: flex; flex-direction: column; height: 100%; min-height: 0; }
  header { display: flex; flex: none; align-items: center; justify-content: space-between; gap: var(--space-md); padding: var(--surface-padding); }
  h2 { margin: 0; font-family: var(--font-structure); font-size: var(--text-lg); font-weight: var(--weight-structure); }
  .body { overflow-y: auto; overscroll-behavior: contain; padding: 0 var(--surface-padding) var(--surface-padding); scrollbar-gutter: stable; }
  .contact-link { margin: var(--space-lg) 0 0; color: var(--fg-dim); font-size: var(--text-xs); line-height: 1.5; }
  a { color: var(--accent); text-underline-offset: 3px; }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; border-radius: var(--radius-sm); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
</style>
