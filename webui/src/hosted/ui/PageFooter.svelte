<script lang="ts">
  import { CONTACT_ADDRESS } from "../../lib/contact";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";

  type Current = "home" | "credits" | "contact" | "learn" | "privacy" | "about" | "contribute" | "neurons";
  let { current, hosted = true, compact = false }: { current?: Current; hosted?: boolean; compact?: boolean } = $props();

  const releaseBuild = typeof __DROWSE_HOSTED_RELEASE__ !== "undefined" && __DROWSE_HOSTED_RELEASE__;
  const sourceRevision = typeof __DROWSE_SOURCE_REVISION__ === "string" ? __DROWSE_SOURCE_REVISION__ : "preview";
  // AGPL section 13 source offer: release builds link the exact revision they were built from.
  const sourceUrl = typeof __DROWSE_SOURCE_URL__ === "string" ? __DROWSE_SOURCE_URL__ : "https://github.com/a9lim/drowse";
  const version = typeof __DROWSE_VERSION__ === "string" ? __DROWSE_VERSION__ : "";
  const repositoryUrl = "https://github.com/a9lim/drowse";

  const site = $derived(hosted ? "" : "https://drowse.ai");
  type FooterLink = { label: string; href: string; page?: Current; detail?: string; external?: boolean };
  const groups: { title: string; links: FooterLink[] }[] = $derived([
    { title: "Drowse", links: [
      { label: "Home", href: `${site}/`, page: "home" },
      { label: "Open Drowse", href: `${site}/app` },
      { label: "Models", href: `${site}/app?choose=1` },
      { label: "Learn", href: `${site}/learn`, page: "learn" },
      { label: "Neurons", href: `${site}/neurons`, page: "neurons" },
    ] },
    { title: "Project", links: [
      { label: "Credits", href: `${site}/credits`, page: "credits" },
      { label: "Contribute", href: `${site}/contribute`, page: "contribute" },
      { label: "Changelog", href: `${repositoryUrl}/releases`, external: true },
    ] },
    { title: "Trust", links: [
      { label: "Privacy", href: `${site}/privacy`, page: "privacy" },
      { label: "License", href: `${site}/LICENSE`, detail: "GNU AGPL v3 or later" },
      { label: "Third-party licenses", href: `${site}/third-party-licenses` },
      { label: "Security", href: `${repositoryUrl}/security/policy`, external: true },
    ] },
    { title: "Contact", links: [
      { label: "Contact us", href: `${site}/contact`, page: "contact" },
      { label: "Email us", href: `mailto:${CONTACT_ADDRESS}` },
    ] },
  ]);
</script>

<footer class="page-footer" class:compact>
  <div class="footer-content">
    <nav class="footer-groups" aria-label="Footer">
      {#each groups as group}
        <div class="footer-group">
          <h2>{group.title}</h2>
          <ul>
            {#each group.links as link}
              <li>
                <a href={link.href} aria-current={link.page && current === link.page ? "page" : undefined} rel={link.external ? "noreferrer" : undefined}>
                  <span>{link.label}{#if link.detail}<span class="link-detail">{link.detail}</span>{/if}</span>
                  {#if link.external}<FluentIcon name="external" size={16} /><span class="sr-only">(opens GitHub)</span>{/if}
                </a>
              </li>
            {/each}
          </ul>
        </div>
      {/each}
    </nav>
    <div class="project-info">
      <span translate="no">Drowse{version ? ` ${version}` : ""}</span>
      {#if releaseBuild}<span class="build" {...{ "aria-description": (sourceRevision) }}>Build {sourceRevision.slice(0, 12)}</span>{/if}
      <a href={sourceUrl} rel="noreferrer">
        <span>{releaseBuild ? "Source for this build" : "Source code"}</span>
        <FluentIcon name="external" size={16} /><span class="sr-only">(opens GitHub)</span>
      </a>
    </div>
  </div>
</footer>

<style>
  .page-footer {
    position: relative;
    z-index: 1;
    flex: 0 0 auto;
    width: min(100%, var(--page-max));
    margin-inline: auto;
    padding-inline: max(var(--page-gutter), env(safe-area-inset-left)) max(var(--page-gutter), env(safe-area-inset-right));
  }
  .footer-content {
    display: grid;
    gap: var(--space-md);
    padding-block: var(--surface-padding) max(var(--surface-padding), env(safe-area-inset-bottom));
    border-top: 1px solid var(--grid-line);
  }
  .footer-groups { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--space-md) var(--space-8); }
  .footer-group { display: grid; align-content: start; gap: var(--space-xs); min-width: 0; }
  h2 {
    margin: 0;
    padding-inline: var(--space-2);
    color: var(--fg-muted);
    font-family: var(--font-structure);
    font-size: var(--text-xs);
    font-weight: var(--weight-structure);
  }
  ul { display: grid; margin: 0; padding: 0; list-style: none; }
  .footer-group a, .project-info a {
    display: inline-flex;
    align-items: center;
    gap: var(--space-xs);
    min-height: var(--control-target);
    padding-inline: var(--space-2);
    border-radius: var(--radius-sm);
    color: var(--fg-dim);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    text-decoration: none;
    transition: color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out);
  }
  .footer-group a > span { display: grid; }
  .link-detail { color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-reading); }
  .footer-group a :global(svg), .project-info a :global(svg) { flex: none; color: var(--fg-muted); }
  .footer-group a:hover, .project-info a:hover { color: var(--fg); background: var(--bg-hover); }
  .footer-group a[aria-current="page"] { color: var(--accent); }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .project-info { display: flex; flex-wrap: wrap; align-items: center; gap: 0 var(--space-6); color: var(--fg-muted); font-size: var(--text-xs); }
  .project-info > span { padding-inline: var(--space-2); }
  .project-info a { font-size: var(--text-xs); }
  .build { font-family: var(--font-mono); }
  @media (max-width: 760px) {
    .footer-groups { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .project-info { gap: 0 var(--space-2); }
  }
  @media (prefers-reduced-motion: reduce) { a { transition: none; } }
  .page-footer.compact { width: 100%; padding: 0; }
  .compact .footer-content { gap: var(--space-xs); padding: var(--space-2) 0 0; }
  .compact .footer-groups { gap: var(--space-xs) var(--space-md); }
  .compact .project-info { font-size: var(--text-2xs); }
  .compact .footer-group a { font-size: var(--text-xs); }
</style>
