<script lang="ts">
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import { htmlGuidePaths, type GuidePage, type GuideLink } from "../../../../scripts/site-content.mjs";

  let { page }: { page: GuidePage } = $props();

  // Internal guides without an HTML page (such as /developers) link to their Markdown resource.
  const htmlRoutes = new Set(["/", "/app", "/credits", "/contact", "/contribute", "/third-party-licenses", ...htmlGuidePaths]);
  const internalHref = (href: string) => {
    const path = href.split(/[?#]/, 1)[0];
    return htmlRoutes.has(path) ? href : `${path}.md`;
  };
  const isExternal = (link: GuideLink) => !link.href.startsWith("/");
  const slug = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
</script>

{#each page.sections as section}
  <section class="guide-section" aria-labelledby={`section-${slug(section.heading)}`}>
    <h2 id={`section-${slug(section.heading)}`}>{section.heading}</h2>
    <div class="section-body">
      {#each section.paragraphs ?? [] as paragraph}<p>{paragraph}</p>{/each}
      {#if section.bullets}
        <ul class="bullets">{#each section.bullets as bullet}<li>{bullet}</li>{/each}</ul>
      {/if}
      {#if section.code}<pre class="code"><code>{section.code}</code></pre>{/if}
      {#if section.links?.length}
        <ul class="links">
          {#each section.links as link}
            <li>
              {#if isExternal(link)}
                <a href={link.href} rel="noreferrer"><span>{link.label}</span><FluentIcon name="external" size={18} /></a>
              {:else}
                <a href={internalHref(link.href)}><span>{link.label}</span><FluentIcon name="next" size={18} /></a>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </section>
{/each}

<style>
  .guide-section { display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr); gap: var(--space-6) var(--space-8); padding-block: var(--surface-padding); border-top: 1px solid var(--grid-line); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-model-title); line-height: 1.15; letter-spacing: -0.035em; text-wrap: balance; }
  .section-body { display: grid; gap: var(--space-5); min-width: 0; }
  p, li { color: var(--fg-dim); font-size: var(--text-md); line-height: 1.65; text-wrap: pretty; }
  p { margin: 0; max-width: 68ch; }
  .bullets { display: grid; gap: var(--space-2); margin: 0; padding-inline-start: var(--space-6); max-width: 68ch; }
  .code { margin: 0; overflow-x: auto; padding: var(--space-4) var(--space-5); border-radius: var(--radius); background: var(--surface-card); box-shadow: var(--shadow-rack); color: var(--fg); font-family: var(--font-mono); font-size: var(--text-sm); line-height: 1.6; }
  .links { display: flex; flex-wrap: wrap; gap: var(--space-4); margin: 0; padding: 0; list-style: none; }
  .links li { min-width: 0; }
  .links a { display: flex; align-items: center; gap: var(--space-4); min-height: 48px; padding: var(--space-3) var(--space-5); border-radius: var(--radius); background: var(--surface-card); box-shadow: var(--shadow-rack); color: var(--fg); font-size: var(--text-sm); text-decoration: none; overflow-wrap: anywhere; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  .links a:hover { background: var(--surface-card-hover); box-shadow: var(--shadow-rack-hover); }
  .links a:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }
  .links a > :global(svg) { flex: none; color: var(--fg-muted); }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; }
  @media (max-width: 760px) {
    .guide-section { grid-template-columns: minmax(0, 1fr); }
  }
  @media (prefers-reduced-motion: reduce) { .links a { transition: none; } }
</style>
