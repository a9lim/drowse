<script lang="ts">
  let { current, hosted = true, compact = false }: { current?: "credits"; hosted?: boolean; compact?: boolean } = $props();

  const releaseBuild = typeof __DROWSE_HOSTED_RELEASE__ !== "undefined" && __DROWSE_HOSTED_RELEASE__;
  const sourceRevision = typeof __DROWSE_SOURCE_REVISION__ === "string" ? __DROWSE_SOURCE_REVISION__ : "preview";
  const sourceUrl = typeof __DROWSE_SOURCE_URL__ === "string" ? __DROWSE_SOURCE_URL__ : "https://github.com/a9lim/drowse";
</script>

<footer class="page-footer" class:compact>
  <div class="footer-content">
    <div class="project-info">
      <span>Open source · AGPL-3.0-or-later</span>
      {#if releaseBuild}<span class="build" title={sourceRevision}>Build {sourceRevision.slice(0, 12)}</span>{/if}
    </div>
    <nav aria-label="Footer">
      {#if hosted}<a href="/credits" aria-current={current === "credits" ? "page" : undefined}>Credits</a>{/if}
      <a href={sourceUrl} rel="noreferrer">Contribute</a>
      <a href={hosted ? "/LICENSE" : `${sourceUrl}/blob/main/LICENSE`}>License</a>
    </nav>
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
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4) var(--space-8);
    min-height: 80px;
    padding-block: var(--surface-padding) max(var(--surface-padding), env(safe-area-inset-bottom));
    border-top: 1px solid var(--grid-line);
  }
  .project-info { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-6); color: var(--fg-muted); font-size: var(--text-xs); }
  .build { font-family: var(--font-mono); }
  nav { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-5); }
  nav a {
    display: inline-flex;
    align-items: center;
    min-height: var(--control-target);
    min-width: 40px;
    padding-inline: var(--space-2);
    border-radius: var(--radius-sm);
    color: var(--fg-dim);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    text-decoration: none;
    transition: color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out);
  }
  nav a:hover { color: var(--fg); background: var(--bg-hover); }
  nav a[aria-current="page"] { color: var(--accent); }
  @media (max-width: 760px) {
    .footer-content { align-items: start; flex-direction: column; }
  }
  @media (prefers-reduced-motion: reduce) { nav a { transition: none; } }
  .page-footer.compact { width: 100%; padding: 0; }
  .compact .footer-content {
    min-height: 0;
    align-items: start;
    gap: var(--space-2);
    padding: var(--space-2) 0 0;
  }
  .compact .project-info { padding-inline: var(--space-2); font-size: var(--text-2xs); }
  .compact nav { gap: 0; }
  .compact nav a { font-size: var(--text-xs); }
</style>
