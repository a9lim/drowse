<script lang="ts">
  import SitePage from "./SitePage.svelte";
  import GuideContent from "./GuideContent.svelte";
  import { htmlGuides } from "../../../../scripts/site-content.mjs";

  let { path }: { path: string } = $props();
  const page = $derived(htmlGuides.find(guide => guide.path === path && path.startsWith("/learn/"))!);
</script>

<SitePage title={`${page.heading} · Drowse`} heading={page.heading} lede={page.intro} markdown={`${page.path}.md`}>
  <nav class="breadcrumb" aria-label="Breadcrumb"><a href="/learn">Learn</a></nav>
  <GuideContent {page} />
</SitePage>

<style>
  .breadcrumb { padding-block-end: var(--space-6); font-size: var(--text-sm); }
  .breadcrumb a { display: inline-flex; align-items: center; min-height: var(--control-target); color: var(--accent); font-family: var(--font-structure); text-underline-offset: 4px; }
  .breadcrumb a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; border-radius: var(--radius-sm); }
</style>
