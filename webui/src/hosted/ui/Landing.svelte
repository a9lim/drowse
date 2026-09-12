<script lang="ts">
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import { onMount } from "svelte";
  import CapabilityDemo from "./CapabilityDemo.svelte";
  import HeroShader from "./HeroShader.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import ModelProviderLogo from "./ModelProviderLogo.svelte";
  import { siteTitle } from "../../../scripts/site-metadata.mjs";

  let plainBackground = $state(false);

  onMount(() => {
    const contrast = matchMedia("(prefers-contrast: more), (forced-colors: active)");
    const updateContrast = () => { plainBackground = contrast.matches; };
    updateContrast();
    contrast.addEventListener("change", updateContrast);
    return () => {
      contrast.removeEventListener("change", updateContrast);
    };
  });

  const capabilities = [
    {
      title: "Inspect the prediction",
      body: "See which tokens the model could have chosen. Add compatible J-lens or sparse autoencoder (SAE) packs to read predictions across layers or inspect learned features.",
    },
    {
      title: "Change an influence",
      body: "Steer a reply toward a concept, word, or available feature. Adjust the strength without changing the model's weights.",
    },
    {
      title: "Compare what changes",
      body: "Try another continuation from a token in the Loom and compare it with the original. Your work autosaves in this browser.",
    },
  ];

  const modelGroups = [
    {
      id: "chat-models",
      title: "Chat models",
      description: ["Follow instructions.", "Explore a conversation."],
      models: [
        { name: "Gemma 3 1B", id: "gemma3-1b", provider: "Google" },
        { name: "Gemma 3 4B", id: "gemma3-4b", provider: "Google" },
        { name: "Qwen3 1.7B", id: "qwen3-1.7b", provider: "Qwen" },
        { name: "Qwen3 4B", id: "qwen3-4b", provider: "Qwen" },
      ],
    },
    {
      id: "base-models",
      title: "Base models",
      description: ["Continue raw text.", "Study next-token predictions."],
      models: [
        { name: "Gemma 3 1B PT", id: "gemma3-1b-pt", provider: "Google" },
        { name: "GPT-2 · 124M", id: "gpt2-base", provider: "OpenAI" },
        { name: "Pythia 70M", id: "pythia-70m-base", provider: "EleutherAI · deduped" },
        { name: "Qwen 3.5 2B Base", id: "qwen35-2b-base", provider: "Qwen" },
      ],
    },
  ];
</script>

<TabIdentity title={siteTitle} />

<a class="skip-link" href="#main">Skip to content</a>

<div class="landing-shell">
  <svg class="glass-definitions" width="0" height="0" aria-hidden="true" focusable="false">
    <defs>
      <filter id="landing-glass" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">
        <feImage href="/images/landing-glass-map.svg" width="100%" height="100%" preserveAspectRatio="none" result="rim" />
        <feDisplacementMap in="SourceGraphic" in2="rim" scale="32" xChannelSelector="R" yChannelSelector="G" />
      </filter>
    </defs>
  </svg>
  {#if !plainBackground}<HeroShader />{/if}

  <div class="page-layer">
    <PageHeader current="home" />

    <main id="main">
      <section class="hero" aria-labelledby="hero-title">
        <div class="hero-copy">
          <h1 id="hero-title">See inside<br />your model.</h1>
          <p class="lede">
            Run a language model on your device. Inspect its predictions and test
            how changes to its internal activity affect its replies.
          </p>
          <div class="hero-action-row">
            <a class="primary-action" href="/app">Open Drowse</a>
            <span>Chrome or Safari · compatible device required</span>
          </div>
          <p class="trust-line">Open source. Inference and saved work stay on your device.</p>
        </div>
      </section>

      <div class="orb-passage" aria-hidden="true"></div>

      <section class="capabilities" id="what-you-can-do" aria-labelledby="capabilities-title">
        <div class="section-intro">
          <h2 id="capabilities-title">Test what shapes a reply.</h2>
          <p class="section-description">Ask a question, then try it with different response controls. Compare the predictions and replies in your browser, with no notebook or server to set up.</p>
        </div>

        <ol>
          {#each capabilities as capability}
            <li>
              <h3>{capability.title}</h3>
              <p>{capability.body}</p>
            </li>
          {/each}
        </ol>
        <div class="demo-panel"><CapabilityDemo /></div>
      </section>

      <section class="models" id="models" aria-labelledby="models-title">
        <div class="models-heading">
          <h2 id="models-title">Choose a model.</h2>
          <p>Use a chat model for conversation or a base model for text completion. Drowse checks what your device can run when you open the app.</p>
        </div>

        <div class="model-groups">
          {#each modelGroups as group}
            <section class="model-group" aria-labelledby={`${group.id}-title`}>
              <h3 id={`${group.id}-title`}>{group.title}</h3>
              <p>{group.description[0]}<br />{group.description[1]}</p>
              <ul class="model-list">
                {#each group.models as model}
                  <li class="model-row">
                    <ModelProviderLogo modelId={model.id} size={32} />
                    <span class="model-copy"><strong>{model.name}</strong><span>{model.provider}</span></span>
                  </li>
                {/each}
              </ul>
            </section>
          {/each}
        </div>

        <div class="closing-action">
          <a class="primary-action" href="/app">Open Drowse</a>
          <p>Choose your model and tools inside the app.</p>
        </div>
      </section>
    </main>

    <PageFooter />
  </div>
</div>

<style>
  :global(html:has(.landing-shell)) {
    scrollbar-gutter: auto;
    scrollbar-width: none;
  }

  :global(html:has(.landing-shell)::-webkit-scrollbar) {
    display: none;
  }

  @media (prefers-reduced-motion: no-preference) {
    :global(html:has(.landing-shell)) { scroll-behavior: smooth; }
  }

  .skip-link {
    position: fixed;
    z-index: 1000;
    top: 8px;
    inset-inline-start: 8px;
    padding: var(--space-xs) var(--space-sm);
    border-radius: var(--radius);
    background: var(--accent);
    color: var(--text-on-accent);
    transform: translateY(-160%);
  }

  .skip-link:focus { transform: translateY(0); }

  .landing-shell {
    min-height: 100%;
    overflow-x: clip;
    color: var(--fg);
    isolation: isolate;
  }

  .page-layer { position: relative; }

  .hero-copy,
  .capabilities,
  .models {
    width: min(100%, var(--page-max));
    margin-inline: auto;
    padding-inline: max(var(--surface-padding), env(safe-area-inset-left)) max(var(--surface-padding), env(safe-area-inset-right));
  }

  .hero {
    -webkit-user-select: none;
    user-select: none;
    min-height: max(42rem, calc(100svh - 88px));
    position: relative;
    display: flex;
    align-items: center;
    padding-block: var(--surface-padding) var(--surface-padding);
  }

  .hero-copy {
    margin-inline: auto;
  }
  .hero-copy > * { max-width: min(36rem, 52%); }
  .orb-passage { height: 65svh; }
  .closing-action { display: flex; align-items: center; flex-wrap: wrap; gap: var(--space-md); margin-top: var(--space-lg); }
  .closing-action p { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }

  h1,
  h2,
  h3 { font-family: var(--font-structure); text-wrap: balance; }

  h1 {
    margin: var(--space-lg) 0 0;
    font-size: var(--text-hero-compact);
    font-weight: var(--weight-display);
    letter-spacing: -0.076em;
    line-height: 0.98;
  }

  .lede {
    margin: calc(var(--space-unit) * 5) 0 0;
    color: var(--fg-strong);
    font-family: var(--font-reading);
    font-size: var(--text-hero-lede);
    line-height: 1.55;
    text-wrap: pretty;
  }

  .hero-action-row {
    display: flex;
    align-items: flex-start;
    flex-direction: column;
    gap: var(--space-7);
    margin-top: calc(var(--space-unit) * 5);
  }

  .primary-action {
    display: inline-flex;
    flex-shrink: 0;
    white-space: nowrap;
    min-height: 56px;
    align-items: center;
    justify-content: center;
    padding: var(--space-6) var(--space-7);
    border-radius: var(--radius);
    background: var(--control-sheen), var(--action-bg);
    box-shadow: var(--shadow-control);
    color: var(--action-ink);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure-bold);
    text-decoration: none;
    transition: transform var(--dur-fast) var(--ease-out), background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out);
  }

  .primary-action:hover { background: var(--control-sheen), var(--action-hover); box-shadow: var(--shadow-control-hover); transform: translateY(-1px); }

  .primary-action:active { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }

  .hero-action-row span {
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }

  .trust-line {
    margin: var(--space-7) 0 0;
    color: var(--fg-muted);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    line-height: 1.6;
    text-wrap: pretty;
  }

  .capabilities {
    position: relative;
    display: grid;
    gap: var(--space-xl);
    padding-block: var(--surface-padding);
  }

  .section-intro h2,
  .models h2 {
    margin: 0;
    font-size: var(--text-section-display);
    font-weight: var(--weight-display);
    letter-spacing: -0.06em;
    line-height: 0.94;
  }

  .models-heading > p {
    max-width: 42ch;
    margin: var(--space-md) 0 0;
    color: var(--fg-dim);
    font-size: var(--text-section-lede);
    line-height: 1.55;
  }

  .section-description { max-width: 64ch; margin: var(--space-8) 0 0; font-size: var(--text-section-lede); line-height: 1.6; }

  .capabilities ol {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-lg);
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .capabilities li {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-sm);
    align-items: baseline;
    align-content: start;
  }

  .capabilities li,
  .model-group,
  .demo-panel {
    --fg: var(--landing-panel-ink);
    --fg-dim: var(--landing-panel-muted);
    --fg-muted: var(--landing-panel-muted);
    position: relative;
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--landing-panel-bg);
    color: var(--fg);
    font: var(--weight-reading) var(--text-md)/1.5 var(--font-reading);
    text-shadow: none;
    -webkit-backdrop-filter: blur(1px);
    backdrop-filter: blur(1px);
  }

  .glass-definitions { position: absolute; pointer-events: none; }
  .demo-panel { min-width: 0; margin-top: 32px; }

  .capabilities li > *,
  .model-group > *,
  .demo-panel > :global(*) { position: relative; z-index: 1; }

  @supports (backdrop-filter: url("#landing-glass")) and (not (-webkit-touch-callout: none)) {
    .capabilities li,
    .model-group,
    .demo-panel {
      backdrop-filter: url("#landing-glass") blur(1px);
    }
  }

  .capabilities h3 {
    margin: 0;
    font-size: var(--text-lg);
    font-weight: var(--weight-structure-bold);
    letter-spacing: -0.035em;
  }

  .capabilities li p {
    margin: 0;
    color: var(--fg-dim);
    line-height: 1.6;
    text-wrap: pretty;
  }

  .models {
    position: relative;
    padding-block: var(--surface-padding);
  }

  .models-heading {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(240px, 0.6fr);
    gap: var(--space-xl);
    align-items: end;
  }

  .model-groups { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: var(--space-xl); margin-top: var(--space-lg); }
  .model-group { display: grid; grid-template-rows: subgrid; grid-row: span 3; row-gap: 0; min-width: 0; }
  .model-group h3 {
    margin: 0;
    font-size: var(--text-feature-title);
    font-weight: var(--weight-structure-bold);
    line-height: 1.2;
    text-box-edge: cap alphabetic;
    text-box-trim: trim-start;
  }
  .model-group > p { margin: var(--space-xs) 0 var(--space-md); color: var(--fg-muted); line-height: 1.6; }
  .model-list { display: grid; grid-auto-rows: 1fr; gap: var(--space-md); list-style: none; margin: 0; padding: 0; }

  .model-row {
    display: grid;
    grid-template-columns: 32px minmax(0, 1fr);
    gap: var(--space-sm);
    align-items: center;
  }

  .model-row strong {
    font: inherit;
    font-weight: var(--weight-reading-medium);
  }

  .model-copy { display: grid; gap: var(--space-xs); min-width: 0; }
  .model-copy > span { color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.6; }
  .model-row:last-child .model-copy { align-self: end; }
  .model-row:last-child .model-copy > span {
    text-box-edge: cap alphabetic;
    text-box-trim: trim-end;
  }
  .model-row strong { overflow-wrap: anywhere; }

  h1, h2, h3, .lede,
  .hero-action-row > span, .trust-line,
  .capabilities li p, .models-heading > p,
  .model-row strong {
    color: var(--fg);
  }


  @media (prefers-contrast: more), (forced-colors: active), (prefers-reduced-transparency: reduce) {
    .capabilities li,
    .model-group,
    .demo-panel {
      --fg: CanvasText;
      --fg-dim: CanvasText;
      --fg-muted: CanvasText;
      background: Canvas;
      -webkit-backdrop-filter: none;
      backdrop-filter: none;
      box-shadow: none;
      text-shadow: none;
    }
  }

  @media (prefers-contrast: more), (forced-colors: active) {
    .landing-shell :is(a, h1, h2, h3, p, span, strong):not(.primary-action) {
      color: var(--fg);
    }
    .orb-passage { display: none; }
  }

  @media (max-width: 760px) {
    .hero { min-height: calc(100svh - 70px); align-items: flex-start; padding-block: var(--surface-padding) var(--surface-padding); }
    .hero-copy { margin-inline: auto; }
    .hero-copy > * { max-width: 100%; }
    h1 { font-size: var(--text-hero-compact); line-height: 1; }
    .lede { margin-top: var(--space-5); }
    .orb-passage { height: 35svh; }
    .capabilities { grid-template-columns: 1fr; gap: calc(var(--space-unit) * 7); }
    .capabilities ol { grid-template-columns: 1fr; gap: var(--space-lg); }
    .models-heading { grid-template-columns: 1fr; gap: var(--space-sm); }
    .model-groups { grid-template-columns: 1fr; row-gap: var(--space-xl); }
    .model-group { grid-template-rows: auto auto auto; grid-row: auto; }
    .model-list { grid-auto-rows: auto; }
  }

  @media (prefers-reduced-motion: reduce) {
    .orb-passage { display: none; }
  }
</style>
