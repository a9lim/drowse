<script lang="ts">
  import SitePage from "./SitePage.svelte";
  import GuideContent from "./GuideContent.svelte";
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import { htmlGuides } from "../../../../scripts/site-content.mjs";

  const page = htmlGuides.find(guide => guide.path === "/learn")!;
  const guides = htmlGuides.filter(guide => guide.path.startsWith("/learn/"));
</script>

<SitePage title="Learn · Drowse" heading={page.heading} lede={page.intro} markdown="/learn.md">
  <section class="guides" aria-labelledby="guides-title">
    <h2 id="guides-title">Guides</h2>
    <ul>
      {#each guides as guide}
        <li>
          <a href={guide.path}>
            <span class="guide-title">{guide.heading}</span>
            <span class="guide-summary">{guide.description}</span>
            <FluentIcon name="next" size={20} />
          </a>
        </li>
      {/each}
    </ul>
  </section>
  <GuideContent {page} />
</SitePage>

<style>
  .guides { display: grid; gap: var(--space-6); padding-block: var(--surface-padding) var(--section-space); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-model-title); line-height: 1.15; letter-spacing: -0.035em; }
  ul { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-6); margin: 0; padding: 0; list-style: none; }
  a { position: relative; display: grid; gap: var(--space-3); height: 100%; padding: var(--surface-padding); padding-inline-end: var(--space-8); border-radius: var(--radius-lg); background: var(--surface-card); box-shadow: var(--shadow-card); color: var(--fg); text-decoration: none; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  a:hover { background: var(--surface-card-hover); box-shadow: var(--shadow-card-active); }
  a:active { transform: scale(var(--press-scale)); }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; }
  a > :global(svg) { position: absolute; inset-block-start: var(--surface-padding); inset-inline-end: var(--space-5); color: var(--fg-muted); }
  .guide-title { font-family: var(--font-structure); font-weight: var(--weight-structure); font-size: var(--text-lg); line-height: 1.25; text-wrap: balance; }
  .guide-summary { color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.55; text-wrap: pretty; }
  @media (max-width: 760px) { ul { grid-template-columns: minmax(0, 1fr); gap: var(--space-4); } }
  @media (prefers-reduced-motion: reduce) { a { transition: none; } }
</style>
