<script lang="ts">
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import a9limPortrait from "../../assets/credits/a9lim.jpg";
  import treetownPortrait from "../../assets/credits/treetown.jpg";
  import katPortrait from "../../assets/credits/transkatgirl.jpg";
  import vogelPortrait from "../../assets/credits/voooooogel.jpg";
  import motionPortrait from "../../assets/credits/motion_so.jpg";
  import claudeLogo from "../../assets/credits/claude.svg";
  import openaiLogo from "../../assets/model-providers/openai.png";
  import VerificationBadge from "./VerificationBadge.svelte";
  import HeroShader from "./HeroShader.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import PwaUpdatePrompt from "./PwaUpdatePrompt.svelte";

  const sourceUrl = typeof __DROWSE_SOURCE_URL__ === "string"
    ? __DROWSE_SOURCE_URL__ : "https://github.com/a9lim/drowse";
  // X profile verification snapshot, checked 2026-09-06 (voooooogel: 2026-09-07).
  const team = [
    { name: "a9lim", handle: "@_a9lim", profile: "https://x.com/_a9lim", website: "a9l.im", portrait: a9limPortrait, verification: "blue" },
    { name: "treetown", handle: "@treetowntree", profile: "https://x.com/treetowntree", website: "logits.ml", portrait: treetownPortrait, verification: "blue" },
  ] as const;
  const thanks = [
    { handle: "@transkatgirl", profile: "https://x.com/transkatgirl", portrait: katPortrait, verification: "blue" },
    { handle: "@voooooogel", profile: "https://x.com/voooooogel", portrait: vogelPortrait, verification: "blue" },
    { handle: "@motion_so", profile: "https://x.com/motion_so", portrait: motionPortrait, verification: "gold" },
  ] as const;
  const models = [
    { name: "Opus 4.6", provider: "claude" },
    { name: "Opus 4.7", provider: "claude" },
    { name: "Opus 4.8", provider: "claude" },
    { name: "Opus 5", provider: "claude" },
    { name: "Fable 5", provider: "claude" },
    { name: "GPT-6 Sol", provider: "openai" },
    { name: "GPT-6 Astra", provider: "openai" },
  ] as const;
</script>

<TabIdentity state="credits" title="Credits · Drowse" />
<PwaUpdatePrompt />
<svelte:head>
  <meta name="description" content="The people behind Drowse, a local workbench for exploring language models." />
</svelte:head>

<div class="credits-shell">
  <a class="skip-link" href="#credits-main">Skip to content</a>
  <PageHeader current="credits" />

  <main id="credits-main" tabindex="-1">
    <section class="credits-intro" aria-labelledby="credits-title">
      <div class="intro-copy">
        <h1 id="credits-title">Credits.</h1>
        <p class="lede">The people behind Drowse.</p>
      </div>
      <div class="credits-art-frame" aria-hidden="true">
        <div class="credits-art"><HeroShader contained /></div>
      </div>
    </section>

    <section class="team-section" aria-labelledby="team-title">
      <div class="section-heading">
        <h2 id="team-title">The team</h2>
        <p>The people who built &amp; maintain Drowse</p>
      </div>
      <div class="team-list">
        {#each team as member}
          <article class="team-member">
            <div class="member-header">
              <img class="portrait" src={member.portrait} width="88" height="88" alt="" />
              <div class="member-copy">
                <h3><span>{member.name}</span><VerificationBadge status={member.verification} /></h3>
                <a class="member-handle" href={member.profile} rel="noreferrer">{member.handle}</a>
              </div>
            </div>
            <a class="member-website" href={`https://${member.website}`} rel="noreferrer">
              <span><span class="link-label">Website</span><span class="site-name">{member.website}</span></span>
              <FluentIcon name="external" size={20} />
            </a>
          </article>
        {/each}
      </div>
    </section>

    <section class="thanks-section" aria-labelledby="thanks-title">
      <div class="section-heading">
        <h2 id="thanks-title">Special thanks</h2>
        <p>People &amp; Groups whose work inspired Drowse or have supported/contributed in some way</p>
      </div>
      <ul class="thanks-list">
        {#each thanks as contributor}
          <li>
            <a href={contributor.profile} aria-label={contributor.handle} aria-describedby={`${contributor.handle}-verification`} rel="noreferrer">
              <img class="thanks-portrait" src={contributor.portrait} width="48" height="48" alt="" />
              <span class="thanks-name"><span>{contributor.handle}</span><VerificationBadge status={contributor.verification} id={`${contributor.handle}-verification`} /></span>
              <FluentIcon name="external" size={18} />
            </a>
          </li>
        {/each}
      </ul>
    </section>

    <section class="models-section" aria-labelledby="models-title">
      <div class="section-heading">
        <h2 id="models-title">Models of Technical Staff</h2>
        <p>The models that worked hard to bring Drowse to life.</p>
      </div>
      <ul class="models-list">
        {#each models as model}
          <li class="model-credit">
            <span class="model-logo" data-provider={model.provider} aria-hidden="true">
              <span class="model-mark" style:mask-image={`url("${model.provider === "claude" ? claudeLogo : openaiLogo}")`}></span>
            </span>
            <span>{model.name}</span>
          </li>
        {/each}
      </ul>
    </section>

    <section class="contribute-section" aria-labelledby="contribute-title">
      <div>
        <h2 id="contribute-title">Want to help?</h2>
        <p>Report a bug or suggest a change on GitHub. If something broke, include the steps to reproduce it.</p>
      </div>
      <a class="contribute-action" href={sourceUrl} rel="noreferrer">Contribute on GitHub</a>
    </section>
  </main>

  <PageFooter current="credits" />
</div>

<style>
  .credits-shell { display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; isolation: isolate; background: var(--bg); color: var(--fg); }
  a { text-decoration: none; }
  a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; border-radius: var(--radius-sm); }
  .skip-link { position: fixed; inset: 8px auto auto 8px; z-index: 10; padding: var(--space-sm); background: var(--accent); color: var(--text-on-accent); border-radius: var(--radius); transform: translateY(-160%); }
  .skip-link:focus { transform: none; }
  main { flex: 1; width: min(100%, var(--page-max)); margin-inline: auto; padding-inline: var(--page-gutter); }
  .credits-intro { position: relative; isolation: isolate; display: flex; align-items: center; min-height: clamp(24rem, 42vw, 36rem); padding-block: var(--surface-padding); }
  .intro-copy { position: relative; z-index: 1; min-width: 0; }
  h1, h2, h3, p { margin: 0; }
  h1, h2, h3, .lede { color: var(--fg); }
  h1, h2, h3 { font-family: var(--font-structure); font-weight: var(--weight-display); text-wrap: balance; }
  h1 { font-size: var(--text-section-display); line-height: 0.95; letter-spacing: -0.065em; }
  .lede { margin-block: var(--space-6); max-width: 16ch; font-size: var(--text-feature-title); line-height: 1.15; letter-spacing: -0.025em; text-wrap: pretty; }
  .credits-art-frame { position: absolute; z-index: -1; inset-block: -3rem -2rem; left: calc(50% - 50vw); width: 100vw; overflow: hidden; pointer-events: none; mask-image: linear-gradient(transparent, black 12%, black 76%, transparent); }
  .credits-art { position: absolute; width: clamp(44rem, 92vw, 96rem); aspect-ratio: 1; top: 50%; right: -24%; transform: translateY(-50%); mask-image: radial-gradient(ellipse, black 42%, transparent 72%); }
  .team-section { padding-block: var(--surface-padding) var(--surface-padding); }
  .section-heading { display: grid; gap: var(--space-4); margin-block-end: var(--space-7); }
  .section-heading p { max-width: 65ch; color: var(--fg-muted); font-size: var(--text-md); line-height: 1.6; text-wrap: pretty; }
  h2 { font-size: var(--text-model-title); line-height: 1.1; letter-spacing: -0.035em; }
  .team-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-6); }
  .team-member { overflow: hidden; padding: var(--surface-padding); border-radius: var(--radius-lg); background: var(--surface-card); box-shadow: var(--shadow-card); }
  .member-header { display: flex; align-items: center; gap: var(--space-6); }
  .portrait { display: block; flex: none; width: 88px; height: 88px; object-fit: cover; border-radius: 50%; box-shadow: 0 0 0 4px var(--glass), var(--shadow-control); }
  .member-copy { min-width: 0; overflow-wrap: anywhere; }
  .member-handle { display: inline-flex; align-items: flex-start; min-height: 44px; padding-block-start: var(--space-1); color: var(--accent); text-underline-offset: 4px; }
  .member-handle:hover { text-decoration: underline; }
  h3 { font-size: var(--text-model-title); line-height: 1.15; letter-spacing: -0.035em; }
  h3, .thanks-name { display: flex; align-items: center; gap: var(--space-1); }
  .member-website { display: flex; align-items: center; justify-content: space-between; gap: var(--space-4); margin-block-start: var(--space-7); padding: var(--space-4) var(--space-5); border-radius: var(--radius); background: var(--control-sheen), var(--glass); box-shadow: var(--shadow-control); color: var(--fg); line-height: 1.5; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  .member-website > span { display: grid; gap: var(--space-1); }
  .link-label { color: var(--fg-muted); font-size: var(--text-xs); }
  .site-name { font-size: var(--text-sm); font-weight: var(--weight-structure); }
  .member-website :global(svg) { flex: none; color: var(--fg-muted); transition: transform var(--dur) var(--ease-out); }
  .member-website:hover { background: var(--control-sheen), var(--glass-strong); box-shadow: var(--shadow-control-hover); }
  .member-website:hover :global(svg) { transform: translate(2px, -2px); }
  .member-website:active, .thanks-list a:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }
  .thanks-section, .models-section { padding-block: var(--surface-padding) var(--surface-padding); }
  .thanks-list, .models-list { display: flex; flex-wrap: wrap; gap: var(--space-5); margin: 0; padding: 0; list-style: none; }
  .thanks-list li { min-width: 0; }
  .thanks-list a { display: flex; align-items: center; gap: var(--space-4); padding: var(--space-4) var(--space-5); border-radius: var(--radius); background: var(--surface-card); box-shadow: var(--shadow-rack); color: var(--fg); font-size: var(--text-sm); overflow-wrap: anywhere; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  .thanks-list a:hover { background: var(--surface-card-hover); box-shadow: var(--shadow-rack-hover); }
  .thanks-list a > :global(svg) { flex: none; color: var(--fg-muted); }
  .thanks-portrait { display: block; flex: none; width: 48px; height: 48px; object-fit: cover; border-radius: 50%; }
  .model-credit { display: flex; align-items: center; gap: var(--space-4); min-width: 0; padding: var(--space-4) var(--space-5); border-radius: var(--radius); background: var(--surface-card); box-shadow: var(--shadow-rack); font-size: var(--text-sm); }
  .model-logo { display: grid; place-items: center; flex: none; width: 48px; height: 48px; border-radius: 50%; background: var(--brand-claude); }
  .model-logo[data-provider="openai"] { background: var(--brand-openai); }
  .model-mark { width: 30px; height: 30px; background: var(--brand-logo-ink); mask-size: contain; mask-position: center; mask-repeat: no-repeat; mask-mode: alpha; }
  .contribute-section { display: flex; justify-content: space-between; align-items: center; gap: var(--space-6); padding-block: var(--surface-padding) var(--surface-padding); }
  .contribute-section p { max-width: 49ch; margin-block-start: var(--space-4); color: var(--fg-muted); font-size: var(--text-md); line-height: 1.6; text-wrap: pretty; }
  .contribute-action { flex: none; display: inline-flex; justify-content: center; align-items: center; min-height: 48px; padding: var(--space-4) var(--space-6); border-radius: var(--radius); background: var(--control-sheen), var(--action-bg); box-shadow: var(--shadow-control); color: var(--action-ink); font-weight: var(--weight-structure); transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  .contribute-action:hover { background: var(--control-sheen), var(--action-hover); box-shadow: var(--shadow-control-hover); }
  .contribute-action:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }
  @media (max-width: 760px) {
    .credits-intro { min-height: 25rem; }
    .credits-art { width: 48rem; right: calc(24% - 31rem); }
    .team-list { grid-template-columns: 1fr; gap: var(--space-4); }
    .contribute-section { align-items: start; flex-direction: column; }
  }
  @media (max-width: 560px) {
    h1 { font-size: var(--text-section-display); }
    .lede { font-size: var(--text-feature-title); }
    .team-member { padding: var(--surface-padding); }
    .member-header { gap: var(--space-4); }
    .portrait { width: 64px; height: 64px; }
    h3 { font-size: var(--text-lg); }
  }
  @media (prefers-contrast: more), (forced-colors: active) {
    .credits-art { display: none; }
  }
</style>
