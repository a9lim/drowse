<script lang="ts">
  import type { Snippet } from "svelte";
  import TabIdentity from "../../../lib/ui/TabIdentity.svelte";
  import PageHeader from "../PageHeader.svelte";
  import PageFooter from "../PageFooter.svelte";
  import PwaUpdatePrompt from "../PwaUpdatePrompt.svelte";

  let { title, heading, lede, children, markdown }: {
    title: string;
    heading: string;
    lede: string;
    children: Snippet;
    /** Path of the page's Markdown twin, when it has one. */
    markdown?: string;
  } = $props();
</script>

<TabIdentity {title} />
<PwaUpdatePrompt />

<div class="site-page">
  <a class="skip-link" href="#site-page-main">Skip to content</a>
  <PageHeader />
  <main id="site-page-main" tabindex="-1">
    <header class="page-intro">
      <h1>{heading}</h1>
      <p class="lede">{lede}</p>
    </header>
    {@render children()}
    {#if markdown}<p class="alternate"><a href={markdown} type="text/markdown">Read this page as Markdown</a></p>{/if}
  </main>
  <PageFooter />
</div>

<style>
  .site-page { display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; background: var(--bg); color: var(--fg); }
  .skip-link { position: fixed; inset: 8px auto auto 8px; z-index: 10; padding: var(--space-sm); background: var(--accent); color: var(--text-on-accent); border-radius: var(--radius); text-decoration: none; transform: translateY(-160%); }
  .skip-link:focus { transform: none; }
  main { flex: 1; width: min(100%, var(--page-max)); margin-inline: auto; padding-inline: var(--page-gutter); padding-block-end: var(--section-space); }
  main:focus { outline: none; }
  .page-intro { display: grid; gap: var(--space-6); max-width: 52rem; padding-block: var(--section-space) var(--surface-padding); }
  h1 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-section-display-compact); line-height: 1.05; letter-spacing: -0.05em; text-wrap: balance; }
  .lede { margin: 0; max-width: 60ch; color: var(--fg-dim); font-size: var(--text-lg); line-height: 1.55; text-wrap: pretty; }
  .alternate { margin: var(--space-8) 0 0; font-size: var(--text-sm); }
  .alternate a { display: inline-flex; align-items: center; min-height: var(--control-target); color: var(--fg-muted); text-underline-offset: 4px; }
  .alternate a:hover { color: var(--fg); }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; border-radius: var(--radius-sm); }
</style>
