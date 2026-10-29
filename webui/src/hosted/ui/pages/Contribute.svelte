<script lang="ts">
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import SitePage from "./SitePage.svelte";
  import { contributeLinks } from "../../../../scripts/site-content.mjs";

  const version = typeof __DROWSE_VERSION__ === "string" ? __DROWSE_VERSION__ : "unknown";
  const revision = typeof __DROWSE_ARTIFACT_SOURCE_REVISION__ === "string" ? __DROWSE_ARTIFACT_SOURCE_REVISION__ : "";
  const channel = typeof __DROWSE_HOSTED_CHANNEL__ === "string" ? __DROWSE_HOSTED_CHANNEL__ : "preview";
  const sourceUrl = revision ? `${contributeLinks.repository}/tree/${revision}` : contributeLinks.repository;
  // Only build identity is prefilled. Prompts, chats, and model output stay out of the report.
  const issueBody = [
    "**What happened?**", "", "", "**Steps to reproduce**", "1. ", "", "**What did you expect?**", "", "",
    "**Environment**",
    `- Drowse ${version} (${channel}${revision ? `, build ${revision.slice(0, 12)}` : ""})`,
    "- Browser and device: ",
    "- Model: ",
    "", "_Please leave out private prompts, chats, and personal data._",
  ].join("\n");
  const issueUrl = `${contributeLinks.newIssue}?body=${encodeURIComponent(issueBody)}`;

  const ways = [
    { id: "bug", title: "Report a bug", body: "Tell us what happened and how to reproduce it. The report fills in this build's version for you; leave out private prompts and chats.", links: [{ label: "Open an issue", href: issueUrl }] },
    { id: "code", title: "Fix or improve code", body: "Read the contributing guide, then pick an issue that fits. Small, focused changes are easiest to review.", links: [{ label: "Contributing guide", href: contributeLinks.contributing }, { label: "Good first issues", href: contributeLinks.goodFirstIssues }] },
    { id: "model", title: "Add a model or pack", body: "Hosted models ship only with a verified artifact closure and tests on real devices. Start with the browser runtime guide.", links: [{ label: "Browser runtime guide", href: contributeLinks.hostedModels }] },
    { id: "teach", title: "Study or teach with it", body: "Using Drowse in a course, workshop, or study? Tell us what you need.", links: [{ label: "Contact us about teaching", href: "/contact?reason=teaching" }] },
    { id: "support", title: "Support the project", body: "Ask about funding and sponsorship through the contact form.", links: [{ label: "Contact us about funding", href: "/contact?reason=funding" }] },
    { id: "security", title: "Report a security issue privately", body: "Please don't open a public issue for a vulnerability. Follow the security policy instead.", links: [{ label: "Security policy", href: contributeLinks.securityPolicy }] },
  ];
</script>

<SitePage title="Contribute · Drowse" heading="Help build Drowse." lede="Drowse is open source. Here are the ways you can help.">
  <ul class="ways">
    {#each ways as way}
      <li class="way">
        <h2 id={`way-${way.id}`}>{way.title}</h2>
        <p>{way.body}</p>
        <div class="actions">
          {#each way.links as link}
            {@const external = !link.href.startsWith("/")}
            <a href={link.href} rel={external ? "noreferrer" : undefined}>
              <span>{link.label}</span><FluentIcon name={external ? "external" : "next"} size={18} />
            </a>
          {/each}
        </div>
      </li>
    {/each}
  </ul>

  <section class="source" aria-labelledby="source-title">
    <h2 id="source-title">Source for this build</h2>
    <p><code>{revision ? revision.slice(0, 12) : "unknown"}</code> · Drowse {version}</p>
    <a class="source-action" href={sourceUrl} rel="noreferrer">View on GitHub</a>
  </section>
</SitePage>

<style>
  .ways { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-6); margin: 0; padding: 0 0 var(--surface-padding); list-style: none; }
  .way { display: grid; grid-template-rows: auto 1fr auto; gap: var(--space-4); padding: var(--surface-padding); border-radius: var(--radius-lg); background: var(--surface-card); box-shadow: var(--shadow-card); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-lg); line-height: 1.2; letter-spacing: -0.02em; text-wrap: balance; }
  p { margin: 0; max-width: 60ch; color: var(--fg-dim); font-size: var(--text-md); line-height: 1.6; text-wrap: pretty; }
  .actions { display: flex; flex-wrap: wrap; gap: var(--space-4); }
  .actions a { display: inline-flex; align-items: center; gap: var(--space-3); min-height: 48px; padding: var(--space-3) var(--space-5); border-radius: var(--radius); background: var(--control-sheen), var(--glass); box-shadow: var(--shadow-control); color: var(--fg); font-size: var(--text-sm); font-weight: var(--weight-structure); text-decoration: none; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  .actions a:hover { background: var(--control-sheen), var(--glass-strong); box-shadow: var(--shadow-control-hover); }
  .actions a:active, .source-action:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }
  .actions a > :global(svg) { color: var(--fg-muted); }
  .source { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-4) var(--space-6); padding-block: var(--surface-padding); border-top: 1px solid var(--grid-line); }
  .source h2 { flex-basis: 100%; }
  code { font-family: var(--font-mono); color: var(--fg); }
  .source-action { display: inline-flex; justify-content: center; align-items: center; min-height: 48px; padding: var(--space-4) var(--space-6); border-radius: var(--radius); background: var(--control-sheen), var(--action-bg); box-shadow: var(--shadow-control); color: var(--action-ink); font-weight: var(--weight-structure); text-decoration: none; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  .source-action:hover { background: var(--control-sheen), var(--action-hover); box-shadow: var(--shadow-control-hover); }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; }
  @media (max-width: 760px) { .ways { grid-template-columns: minmax(0, 1fr); gap: var(--space-4); } }
  @media (prefers-reduced-motion: reduce) { .actions a, .source-action { transition: none; } }
</style>
