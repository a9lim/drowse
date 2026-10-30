<script lang="ts">
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import SitePage from "./SitePage.svelte";

  const groups = [
    {
      id: "drowse", title: "Drowse",
      body: "Drowse is free software under the GNU Affero General Public License, version 3 or later.",
      links: [{ label: "GNU AGPL v3 or later", href: "/LICENSE" }],
    },
    {
      id: "software", title: "Open-source software",
      body: "The npm packages whose code ships in this build, with each package's license and notice files.",
      links: [{ label: "Third-party software notices", href: "/third-party-licenses.txt" }],
    },
    {
      id: "fonts", title: "Fonts and icons",
      body: "Wix Madefor and Martian Mono are under the SIL Open Font License 1.1. Fluent icons and Torph are under the MIT License.",
      links: [
        { label: "Wix Madefor", href: "/LICENSE-Wix-Madefor.txt" },
        { label: "Martian Mono", href: "/LICENSE-Martian-Mono.txt" },
        { label: "Fluent UI icons", href: "/LICENSE-Fluent-Icons.txt" },
        { label: "Fluent System Icons", href: "/LICENSE-Fluent-System-Icons.txt" },
        { label: "Torph", href: "/LICENSE-Torph.txt" },
      ],
    },
    {
      id: "models", title: "Models and packs",
      body: "Models and instrument packs are downloaded separately and are covered by their own licenses.",
      links: [],
    },
  ];
</script>

<SitePage title="Third-party licenses · Drowse" heading="Licenses." lede="The open-source work that Drowse is built on, and the terms it ships under.">
  {#each groups as group}
    <section class="license-group" aria-labelledby={`licenses-${group.id}`}>
      <div class="copy">
        <h2 id={`licenses-${group.id}`}>{group.title}</h2>
        <p>{group.body}</p>
      </div>
      {#if group.links.length}
        <ul>
          {#each group.links as link}
            <li><a href={link.href} type="text/plain"><span>{link.label}</span><FluentIcon name="next" size={18} /></a></li>
          {/each}
        </ul>
      {/if}
    </section>
  {/each}
</SitePage>

<style>
  .license-group { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); gap: var(--space-6) var(--space-8); padding-block: var(--surface-padding); border-top: 1px solid var(--grid-line); }
  .copy { display: grid; gap: var(--space-4); align-content: start; }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-model-title); line-height: 1.15; letter-spacing: -0.035em; }
  p { margin: 0; max-width: 60ch; color: var(--fg-dim); font-size: var(--text-md); line-height: 1.6; text-wrap: pretty; }
  ul { display: flex; flex-wrap: wrap; align-content: start; gap: var(--space-4); margin: 0; padding: 0; list-style: none; }
  a { display: flex; align-items: center; gap: var(--space-4); min-height: 48px; padding: var(--space-3) var(--space-5); border-radius: var(--radius); background: var(--surface-card); box-shadow: var(--shadow-rack); color: var(--fg); font-size: var(--text-sm); text-decoration: none; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  a:hover { background: var(--surface-card-hover); box-shadow: var(--shadow-rack-hover); }
  a:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }
  a > :global(svg) { flex: none; color: var(--fg-muted); }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; }
  @media (max-width: 760px) { .license-group { grid-template-columns: minmax(0, 1fr); } }
  @media (prefers-reduced-motion: reduce) { a { transition: none; } }
</style>
