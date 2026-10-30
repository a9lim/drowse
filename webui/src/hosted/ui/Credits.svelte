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
  import Button from "../../lib/ui/Button.svelte";
  import MorphText from "../../lib/ui/MorphText.svelte";
  import { onMount } from "svelte";

  const pageDescription = "The people, research, models and software behind Drowse, a local workbench for exploring language models.";
  const releaseBuild = typeof __DROWSE_HOSTED_RELEASE__ !== "undefined" && __DROWSE_HOSTED_RELEASE__;
  const sourceRevision = typeof __DROWSE_SOURCE_REVISION__ === "string" ? __DROWSE_SOURCE_REVISION__ : "preview";
  const version = typeof __DROWSE_VERSION__ === "string" ? __DROWSE_VERSION__ : "5.3.0";
  const repositoryUrl = "https://github.com/a9lim/drowse";
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
  const providerNames = { claude: "Anthropic", openai: "OpenAI" } as const;
  const models = [
    { name: "Opus 4.6", provider: "claude" },
    { name: "Opus 4.7", provider: "claude" },
    { name: "Opus 4.8", provider: "claude" },
    { name: "Opus 5", provider: "claude" },
    { name: "Fable 5", provider: "claude" },
    { name: "GPT-5.6 Sol", provider: "openai" },
    { name: "GPT-6 Astra", provider: "openai" },
  ] as const;

  type License = { label: string; href: string };
  type Credit = { name: string; by?: string; note: string; href?: string; license?: License };
  const thirdParty = "/third-party-licenses";
  // Sources follow browser-runtime/runtime-lock.json. Licenses follow the hosted model closure checks.
  const catalogModels: Credit[] = [
    { name: "Gemma 3", by: "Google", href: "https://ai.google.dev/gemma",
      note: "270M, 1B and 4B instruction-tuned, and 1B base. Instruction-tuned weights come from Unsloth's copies. Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.",
      license: { label: "Gemma Terms of Use", href: "https://ai.google.dev/gemma/terms" } },
    { name: "Qwen3 and Qwen3.5", by: "Qwen team, Alibaba Cloud", href: "https://huggingface.co/Qwen",
      note: "Qwen3 1.7B and 4B, and Qwen3.5 2B base.", license: { label: "Apache 2.0", href: thirdParty } },
    { name: "GPT-2", by: "OpenAI", href: "https://huggingface.co/openai-community/gpt2",
      note: "The original 124M model.", license: { label: "Modified MIT", href: "https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE" } },
    { name: "Pythia", by: "EleutherAI", href: "https://huggingface.co/EleutherAI/pythia-70m-deduped",
      note: "70M, trained on deduplicated data.", license: { label: "Apache 2.0", href: thirdParty } },
  ];
  // From the research lineage in the project README.
  const research: Credit[] = [
    { name: "Representation Engineering", by: "Zou et al., 2023", href: "https://arxiv.org/abs/2310.01405", note: "The approach Drowse builds on." },
    { name: "repeng", by: "Theia Vogel", href: "https://github.com/vgel/repeng", note: "The best-known compact implementation of representation engineering." },
    { name: "Difference-of-means extraction", by: "Im and Li, 2025", href: "https://arxiv.org/abs/2502.02716", note: "Two-pole extraction follows this method." },
    { name: "Manifold steering", by: "Goodfire", href: "https://arxiv.org/abs/2605.05115", note: "Manifold steering follows this work." },
    { name: "The Assistant Axis", by: "Anthropic", href: "https://arxiv.org/abs/2601.10387", note: "The personas source is derived from its framing." },
    { name: "Verbalizable workspace", by: "Gurnee et al., 2026", href: "https://transformer-circuits.pub/2026/workspace/index.html", note: "J-LENS support implements this method." },
  ];
  const interpretability: Credit[] = [
    { name: "SAELens", by: "SAELens contributors", href: "https://github.com/jbloomAus/SAELens", note: "Loads published sparse autoencoders in the Python edition." },
    { name: "Gemma Scope 2", by: "Google DeepMind", note: "Sparse autoencoders for Gemma 3, used for SAE readings." },
    { name: "Neuronpedia", href: "https://www.neuronpedia.org", note: "Feature descriptions for Gemma Scope 2 features." },
  ];
  const software: Credit[] = [
    { name: "WebLLM", by: "MLC", href: "https://github.com/mlc-ai/web-llm", note: "Runs models in the browser. Drowse uses its own fork, @drowse/web-llm.", license: { label: "Apache 2.0", href: thirdParty } },
    { name: "Apache TVM", by: "Apache Software Foundation", href: "https://github.com/apache/tvm", note: "Compiles models for WebGPU.", license: { label: "Apache 2.0", href: thirdParty } },
    { name: "Svelte", by: "Svelte contributors", href: "https://svelte.dev", note: "The interface framework.", license: { label: "MIT", href: thirdParty } },
    { name: "Vite", by: "Vite contributors", href: "https://vite.dev", note: "Builds the site.", license: { label: "MIT", href: thirdParty } },
    { name: "fflate", by: "Arjun Barrett", href: "https://github.com/101arrowz/fflate", note: "Compression for archives and downloads.", license: { label: "MIT", href: thirdParty } },
    { name: "noble-hashes", by: "Paul Miller", href: "https://github.com/paulmillr/noble-hashes", note: "Checks downloaded files.", license: { label: "MIT", href: thirdParty } },
    { name: "YAML", by: "Eemeli Aro", href: "https://eemeli.org/yaml/", note: "Reads and writes transcripts.", license: { label: "ISC", href: thirdParty } },
    { name: "Blobatar", by: "Alain00", href: "https://github.com/Alain00/blobatar", note: "Chat avatars.", license: { label: "MIT", href: thirdParty } },
    { name: "Torph", by: "Lochie Axon", href: "https://torph.lochie.me", note: "Text morphing.", license: { label: "MIT", href: "/LICENSE-Torph.txt" } },
    { name: "Wix Madefor", by: "The Wix Madefor Project Authors", href: "https://github.com/wix/wixmadefor", note: "Display and text font.", license: { label: "SIL Open Font License 1.1", href: "/LICENSE-Wix-Madefor.txt" } },
    { name: "Martian Mono", by: "The Martian Mono Project Authors", href: "https://github.com/evilmartians/mono", note: "Data and code font.", license: { label: "SIL Open Font License 1.1", href: "/LICENSE-Martian-Mono.txt" } },
    { name: "Fluent UI System Icons", by: "Microsoft", href: "https://github.com/microsoft/fluentui-system-icons", note: "Interface icons.", license: { label: "MIT", href: "/LICENSE-Fluent-System-Icons.txt" } },
  ];
  const creditGroups = [
    { id: "research", title: "Research Drowse builds on", lede: "If you use Drowse in published research, cite these methods too.", items: research },
    { id: "interpretability", title: "Interpretability tools and data", lede: "Tools and data behind SAE readings and feature labels.", items: interpretability },
    { id: "catalog-models", title: "Models", lede: "Open models you can run in Drowse. Each keeps its own license.", items: catalogModels },
    { id: "software", title: "Software, fonts and icons", lede: "Open-source projects Drowse ships with or is built with.", items: software },
  ];

  const citation = [
    "@software{drowse,",
    "  author = {{a9lim} and {treetown}},",
    "  title = {Drowse},",
    "  year = {2026},",
    `  version = {${version}},`,
    ...(releaseBuild && /^[0-9a-f]{40}$/.test(sourceRevision) ? [`  commit = {${sourceRevision}},`] : []),
    `  url = {${repositoryUrl}},`,
    "}",
  ].join("\n");
  let citationBlock = $state<HTMLElement>();
  let citationStatus = $state("");
  let citationTimer: ReturnType<typeof setTimeout> | undefined;

  function copyWithSelection(text: string): boolean {
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.className = "sr-only";
    document.body.append(field);
    field.select();
    try { return document.execCommand("copy"); }
    catch { return false; }
    finally { field.remove(); }
  }

  async function copyCitation() {
    let copied = false;
    try { await navigator.clipboard.writeText(citation); copied = true; }
    catch { copied = copyWithSelection(citation); }
    if (!copied && citationBlock) {
      const range = document.createRange();
      range.selectNodeContents(citationBlock);
      const selection = window.getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
    citationStatus = copied ? "Citation copied." : "Couldn't copy. The citation is selected, so you can copy it yourself.";
    clearTimeout(citationTimer);
    citationTimer = setTimeout(() => { citationStatus = ""; }, 5000);
  }

  // Update the page's one description tag instead of adding a second one.
  let mounted = $state(false);
  onMount(() => {
    mounted = true;
    let meta = document.head.querySelector<HTMLMetaElement>('meta[name="description"]');
    const created = !meta;
    const previous = meta?.content ?? "";
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = "description";
      document.head.append(meta);
    }
    const tag = meta;
    tag.content = pageDescription;
    return () => {
      clearTimeout(citationTimer);
      if (created) tag.remove();
      else tag.content = previous;
    };
  });
</script>

<TabIdentity state="credits" title="Credits · Drowse" />
<!-- The update prompt touches window on teardown, so it only renders in the browser. -->
{#if mounted}<PwaUpdatePrompt />{/if}

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
        <p>The people who build and maintain Drowse.</p>
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
        <p>People and groups who inspired, supported or contributed to Drowse.</p>
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
            <span>{model.name}<span class="sr-only"> by {providerNames[model.provider]}</span></span>
          </li>
        {/each}
      </ul>
    </section>

    {#each creditGroups as group}
      <section class="credit-section" aria-labelledby={`${group.id}-title`}>
        <div class="section-heading">
          <h2 id={`${group.id}-title`}>{group.title}</h2>
          <p>{group.lede}</p>
        </div>
        <ul class="credit-list">
          {#each group.items as item}
            <li class="credit-card">
              <h3 class="credit-name">
                {#if item.href}
                  <a href={item.href} rel="noreferrer">
                    <span>{item.name}</span><FluentIcon name="external" size={16} />{#if item.href.startsWith("https://github.com/")}<span class="sr-only">(opens GitHub)</span>{/if}
                  </a>
                {:else}{item.name}{/if}
              </h3>
              {#if item.by}<p class="credit-by">{item.by}</p>{/if}
              <p class="credit-note">{item.note}</p>
              {#if item.license}<a class="credit-license" href={item.license.href} rel={item.license.href.startsWith("/") ? undefined : "noreferrer"}>License: {item.license.label}</a>{/if}
            </li>
          {/each}
        </ul>
        {#if group.id === "software"}
          <p class="credit-footnote">Full license texts for every bundled package are on the <a href={thirdParty}>third-party licenses</a> page.</p>
        {/if}
      </section>
    {/each}

    <section class="cite-section" aria-labelledby="cite-title">
      <div class="section-heading">
        <h2 id="cite-title">Cite Drowse</h2>
        <p>If Drowse helped your work, cite the version you used{releaseBuild ? " and this build's commit" : ""}, along with the exact model checkpoint.</p>
      </div>
      <div class="citation">
        <!-- svelte-ignore a11y_no_noninteractive_tabindex (The scrollable citation must be keyboard-accessible.) -->
        <pre bind:this={citationBlock} tabindex="0" role="region" aria-label="BibTeX citation"><code>{citation}</code></pre>
        <div class="citation-actions">
          <Button onclick={copyCitation}><FluentIcon name="copy" /><MorphText text={citationStatus === "Citation copied." ? "Copied" : "Copy BibTeX"} /></Button>
          <span class="citation-status" role="status">{citationStatus}</span>
        </div>
      </div>
    </section>

    <section class="contribute-section" aria-labelledby="contribute-title">
      <div>
        <h2 id="contribute-title">Want to help?</h2>
        <p>Report a bug, suggest a change, or send a fix. If something broke, include the steps to reproduce it.</p>
      </div>
      <a class="contribute-action" href="/contribute">Ways to contribute</a>
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
  .credit-section, .cite-section { padding-block: var(--surface-padding) var(--surface-padding); }
  .credit-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 18rem), 1fr)); gap: var(--space-5); margin: 0; padding: 0; list-style: none; }
  .credit-card { display: flex; flex-direction: column; gap: var(--space-2); min-width: 0; padding: var(--space-5); border-radius: var(--radius); background: var(--surface-card); box-shadow: var(--shadow-rack); }
  .credit-name { font-size: var(--text-lg); letter-spacing: -0.02em; }
  .credit-name a { display: inline-flex; align-items: center; gap: var(--space-2); min-height: var(--control-target); color: var(--fg); overflow-wrap: anywhere; }
  .credit-name a:hover { color: var(--accent); }
  .credit-name a :global(svg) { flex: none; color: var(--fg-muted); }
  .credit-by { color: var(--fg-dim); font-size: var(--text-sm); }
  .credit-note { color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.5; text-wrap: pretty; }
  .credit-license { display: inline-flex; align-items: center; align-self: start; min-height: var(--control-target); margin-block-start: auto; color: var(--accent); font-size: var(--text-sm); text-underline-offset: 4px; }
  .credit-license:hover, .credit-footnote a:hover { text-decoration: underline; }
  .credit-footnote { margin-block-start: var(--space-5); color: var(--fg-muted); font-size: var(--text-sm); }
  .credit-footnote a { color: var(--accent); text-decoration: underline; text-underline-offset: 4px; }
  .citation { display: grid; gap: var(--space-4); max-width: 48rem; padding: var(--space-5); border-radius: var(--radius-lg); background: var(--surface-card); box-shadow: var(--shadow-card); }
  .citation pre { margin: 0; padding: var(--space-4); overflow-x: auto; border-radius: var(--radius); background: var(--input-well); box-shadow: var(--shadow-well); color: var(--fg); font-family: var(--font-mono); font-size: var(--text-sm); line-height: 1.6; }
  .citation pre:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .citation-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-4); }
  .citation-status { color: var(--fg-dim); font-size: var(--text-sm); }
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
    /* The intro only needs room for its copy. The artwork sits behind it. */
    .credits-intro { min-height: 0; padding-block: var(--space-lg) var(--space-md); }
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
