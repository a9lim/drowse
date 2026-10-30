<script lang="ts">
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import { onMount, tick } from "svelte";
  import CapabilityDemo from "./CapabilityDemo.svelte";
  import HeroShader from "./HeroShader.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import ModelProviderLogo from "./ModelProviderLogo.svelte";
  import { siteTitle, sourceRepository } from "../../../scripts/site-metadata.mjs";

  let plainBackground = $state(false);
  // The prerendered page states the requirement; the probe only replaces it with a result.
  let webGpu = $state<"unknown" | "available" | "unavailable">("unknown");

  async function probeWebGpu(): Promise<typeof webGpu> {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (typeof gpu?.requestAdapter !== "function") return "unavailable";
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      // Asks for an adapter only: no device, model, or runtime code is loaded here.
      const adapter = await Promise.race([
        gpu.requestAdapter(),
        new Promise<"timeout">((resolve) => { timer = setTimeout(() => resolve("timeout"), 3000); }),
      ]);
      return adapter === "timeout" ? "unknown" : adapter ? "available" : "unavailable";
    } catch {
      return "unavailable";
    } finally {
      clearTimeout(timer);
    }
  }

  // Mounting replaces the prerendered DOM, which drops the browser's jump to `#section`.
  let hashScrollY: number | null = null;
  function restoreHash() {
    let id = "";
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = id ? document.getElementById(id) : null;
    if (!target) return;
    target.scrollIntoView({ block: "start", behavior: "instant" });
    if (target.hasAttribute("tabindex")) target.focus({ preventScroll: true });
    hashScrollY = window.scrollY;
  }

  async function demoReady() {
    // The recorded example changes the page height; jump again once it is in place,
    // unless the reader has already scrolled away from the target.
    if (hashScrollY === null) return;
    await tick();
    requestAnimationFrame(() => { if (hashScrollY !== null && Math.abs(window.scrollY - hashScrollY) < 4) restoreHash(); });
  }

  onMount(() => {
    const contrast = matchMedia("(prefers-contrast: more), (forced-colors: active)");
    const updateContrast = () => { plainBackground = contrast.matches; };
    updateContrast();
    contrast.addEventListener("change", updateContrast);
    let active = true;
    void probeWebGpu().then((result) => { if (active) webGpu = result; });
    requestAnimationFrame(restoreHash);
    return () => {
      active = false;
      contrast.removeEventListener("change", updateContrast);
    };
  });

  const requirement = $derived(
    webGpu === "available"
      ? "WebGPU is available in this browser. Drowse checks your device's memory next."
      : webGpu === "unavailable"
        ? "This browser can't run Drowse yet. Try a current version of Chrome, Edge or Safari 26 or later on a computer with a graphics processor."
        : "Needs a browser with WebGPU, such as current Chrome, Edge or Safari 26 or later.",
  );

  const capabilities = [
    {
      title: "Inspect the prediction",
      body: "See which words the model could have chosen at each step. Add word insights (J-lens) to read predictions across layers, or feature insights (SAE) to inspect learned features.",
    },
    {
      title: "Change an influence",
      body: "An influence nudges the model's internal activity toward a concept, word, or feature while it writes. Strength sets how hard it pushes. The model's weights never change.",
    },
    {
      title: "Compare what changes",
      body: "Branch from any token in the Loom, Drowse's tree of continuations, and compare the new reply with the original. Your work autosaves in this browser.",
    },
  ];

  // Mirrors the verified runtime lock and the display names in the signed catalog.
  // `id` is the catalog variant the app's `?model=` parameter selects;
  // scripts/landing-models.test.mjs checks this list against browser-runtime/runtime-lock.json.
  // Download range (model plus required core pack) from signed catalog sequence 9.
  const storageRange = "0.3 to 2.6 GB";
  const modelGroups = [
    {
      id: "chat-models",
      title: "Chat models",
      description: ["Follow instructions.", "Explore a conversation."],
      models: [
        { name: "Gemma 3 270M", id: "gemma3-270m-instruct-q0f32", provider: "Google" },
        { name: "Gemma 3 1B", id: "gemma3-1b-instruct-q4f32_1", provider: "Google" },
        { name: "Gemma 3 4B", id: "gemma3-4b-instruct-q4f32_1", provider: "Google" },
        { name: "Qwen3 1.7B", id: "qwen3-1.7b-q4f32_1", provider: "Qwen" },
        { name: "Qwen3 4B", id: "qwen3-4b-q4f32_1", provider: "Qwen" },
      ],
    },
    {
      id: "base-models",
      title: "Base models",
      description: ["Continue raw text.", "Study next-token predictions."],
      models: [
        { name: "Gemma 3 1B PT", id: "gemma3-1b-pt-q4f32_1", provider: "Google" },
        { name: "GPT-2 Base (124M)", id: "gpt2-base-q0f32", provider: "OpenAI" },
        { name: "Pythia 70M Deduped Base", id: "pythia-70m-base-q0f32", provider: "EleutherAI" },
        { name: "Qwen 3.5 2B Base", id: "qwen35-2b-base-q4f32_1", provider: "Qwen" },
      ],
    },
  ];

  const requirements = [
    {
      title: "A browser with WebGPU",
      body: "Current Chrome, Edge, Brave, or Arc, or Safari 26 or later. Firefox support is a preview.",
    },
    {
      title: "Storage",
      body: `${storageRange} per model, plus optional insight packs. Your browser keeps them, so each model downloads once.`,
    },
    {
      title: "Memory",
      body: "About 4 GB for the smallest chat model, up to 16 GB or more for the 4B models.",
    },
    {
      title: "Phones",
      body: "Phones run compact models as a preview. Use a computer for the larger models.",
    },
  ];

  const glossary = [
    { term: "Influence", definition: "A change Drowse adds to the model's internal activity while it writes, pulling replies toward or away from a concept. Also called steering." },
    { term: "Strength (α)", definition: "How hard an influence pushes. 0 leaves the model unsteered; negative values push the other way." },
    { term: "Word insights (J-lens)", definition: "A lens that shows which words each layer is leaning toward, before the model makes its final prediction." },
    { term: "Feature insights (SAE)", definition: "A sparse autoencoder splits the model's activity into features: learned patterns that often match a recognizable idea." },
    { term: "Loom", definition: "A tree of continuations. Branch from any token, try another path, and compare the results." },
    { term: "Base model", definition: "A model trained only to continue text, without chat tuning. It predicts what comes next instead of following requests." },
  ];

  const pythonInstall = `${sourceRepository}#installation`;
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
  {#if !plainBackground}<HeroShader motionControl />{/if}

  <div class="page-layer">
    <PageHeader current="home" />

    <main id="main" tabindex="-1">
      <section class="hero" aria-labelledby="hero-title">
        <div class="hero-copy">
          <h1 id="hero-title">See inside<br />your model.</h1>
          <p class="lede">
            Run a language model on your device. Inspect its predictions and test
            how changes to its internal activity affect its replies.
          </p>
          <p class="audience">For researchers, students, educators, and anyone curious about how language models work.</p>
          <div class="hero-action-row">
            <a class="primary-action" href="/app">Open Drowse</a>
            <span class="requirement" data-webgpu={webGpu}>{requirement}</span>
          </div>
          <p class="trust-line">Open source. Inference and saved work stay on your device. <a href="/privacy">What leaves your device <span aria-hidden="true">→</span></a></p>
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
        <div class="demo-panel"><CapabilityDemo onready={demoReady} /></div>
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
                  <li>
                    <a class="model-row" href={`/app?choose=1&model=${encodeURIComponent(model.id)}`} data-model={model.id}>
                      <ModelProviderLogo modelId={model.id} size={32} />
                      <span class="model-copy"><strong>{model.name}</strong><span>{model.provider}</span></span>
                    </a>
                  </li>
                {/each}
              </ul>
            </section>
          {/each}
        </div>
        <p class="model-note">Gemma 3 270M runs at full precision, so its download is larger than Gemma 3 1B's.</p>
      </section>

      <section class="guide" id="requirements" aria-labelledby="requirements-title">
        <h2 id="requirements-title">What you need.</h2>
        <ul class="guide-grid">
          {#each requirements as item}
            <li class="guide-panel">
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </li>
          {/each}
        </ul>
      </section>

      <section class="guide" id="glossary" aria-labelledby="glossary-title">
        <h2 id="glossary-title">Words you'll see.</h2>
        <dl class="guide-grid">
          {#each glossary as entry}
            <div class="guide-panel">
              <dt>{entry.term}</dt>
              <dd>{entry.definition}</dd>
            </div>
          {/each}
        </dl>
      </section>

      <section class="guide" id="faq" aria-labelledby="faq-title">
        <h2 id="faq-title">Common questions.</h2>
        <div class="faq guide-panel">
          <details>
            <summary>Does anything leave my device?</summary>
            <p>Models and optional packs download from Hugging Face. Some feature descriptions are fetched from Neuronpedia when you look them up; that request names the feature, not your text. Your prompts, chats, and saved work stay in this browser. The contact form sends only what you enter in it. <a href="/privacy">Read the privacy details</a>.</p>
          </details>
          <details>
            <summary>Why is the download large?</summary>
            <p>A model is mostly its weights: hundreds of millions to billions of numbers. Each model is {storageRange}, plus any insight packs you choose. Your browser keeps the files, so you only download a model once.</p>
          </details>
          <details>
            <summary>Will it work on my phone?</summary>
            <p>Phones with WebGPU can run compact models as a preview, within the memory the browser allows. For the larger models and insight packs, use a computer.</p>
          </details>
          <details>
            <summary>Is steering the same as fine-tuning?</summary>
            <p>No. Fine-tuning trains the model further and changes its weights. Steering adds an influence to the model's activity while it writes. Set the strength to 0 and you get the unsteered model back.</p>
          </details>
          <details>
            <summary>What's a base model?</summary>
            <p>A model trained to continue text, before any chat or instruction tuning. It doesn't answer requests; it predicts what comes next. That makes it useful for studying raw next-token predictions.</p>
          </details>
          <details>
            <summary>Can I use it in a class?</summary>
            <p>Yes. Drowse is free and open source, and it needs only a browser. Each student downloads models to their own device, so check the storage and memory needs above first. Each model's license still applies.</p>
          </details>
          <details>
            <summary>How do I cite it?</summary>
            <p>Cite the Drowse version and the exact model checkpoint you used, alongside the methods Drowse builds on. The <a href="/credits">Credits</a> page lists them.</p>
          </details>
          <details>
            <summary>How is this different from TransformerLens or Neuronpedia?</summary>
            <p>TransformerLens is a Python library for writing interpretability code against a model's internals. Neuronpedia is a hosted platform for exploring and sharing published features. Drowse is an interactive workbench that runs the model on your device, so you can inspect, steer, and branch without code or a server. They work well together.</p>
          </details>
        </div>
      </section>

      <section class="guide" id="python" aria-labelledby="python-title">
        <div class="python-band guide-panel">
          <h2 id="python-title">Need more models?</h2>
          <p>Drowse Local runs supported Hugging Face models with Python, fits your own lenses, trains SAEs, and serves OpenAI- and Ollama-compatible APIs.</p>
          <a class="text-link" href={pythonInstall}>Install from source <span aria-hidden="true">↗</span></a>
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
    position: relative;
    min-height: 100%;
    overflow-x: clip;
    color: var(--fg);
    isolation: isolate;
  }

  .page-layer { position: relative; }

  .hero-copy,
  .capabilities,
  .models,
  .guide {
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

  .audience {
    margin: var(--space-md) 0 0;
    color: var(--fg-muted);
    font-family: var(--font-reading);
    font-size: var(--text-md);
    line-height: 1.6;
    text-wrap: pretty;
  }

  .hero-action-row .requirement {
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

  .trust-line a,
  .guide-panel a,
  .text-link {
    color: inherit;
    text-decoration: underline;
    text-decoration-thickness: 1px;
    text-underline-offset: 0.2em;
  }

  .trust-line a:hover,
  .guide-panel a:hover,
  .text-link:hover { color: var(--accent); }

  .landing-shell a:focus-visible,
  .faq summary:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 3px;
  }

  main:focus { outline: none; }

  .capabilities {
    position: relative;
    display: grid;
    gap: var(--space-xl);
    padding-block: var(--surface-padding);
  }

  .section-intro h2,
  .models h2,
  .guide h2 {
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
  .demo-panel,
  .guide-panel {
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
  .demo-panel > :global(*),
  .guide-panel > * { position: relative; z-index: 1; }

  @supports (backdrop-filter: url("#landing-glass")) and (not (-webkit-touch-callout: none)) {
    .capabilities li,
    .model-group,
    .demo-panel,
    .guide-panel {
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
    align-items: start;
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
  /* The lists differ in length, so rows take their natural height instead of sharing 1fr tracks. */
  .model-list { display: grid; grid-auto-rows: auto; align-content: start; gap: var(--space-md); list-style: none; margin: 0; padding: 0; }

  .model-row {
    display: grid;
    grid-template-columns: 32px minmax(0, 1fr);
    gap: var(--space-sm);
    align-items: center;
    min-height: 44px;
    color: inherit;
    text-decoration: none;
  }

  .model-row:hover strong { text-decoration: underline; text-underline-offset: 0.2em; }

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
  .model-note { max-width: 64ch; margin: var(--space-md) 0 0; font-size: var(--text-sm); line-height: 1.6; text-wrap: pretty; }

  .guide {
    position: relative;
    display: grid;
    gap: var(--space-lg);
    padding-block: var(--surface-padding);
  }

  .guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
    gap: var(--space-lg);
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .guide-panel h3,
  .guide-panel dt,
  .faq summary {
    margin: 0;
    font-family: var(--font-structure);
    font-size: var(--text-lg);
    font-weight: var(--weight-structure-bold);
    letter-spacing: -0.035em;
    text-wrap: balance;
  }

  .guide-panel p,
  .guide-panel dd {
    margin: var(--space-sm) 0 0;
    color: var(--fg-dim);
    line-height: 1.6;
    text-wrap: pretty;
  }

  .faq { display: grid; gap: var(--space-sm); }
  .faq summary {
    display: flex;
    align-items: center;
    min-height: 44px;
    cursor: pointer;
    font-size: var(--text-md);
  }
  .faq summary::before {
    content: "+";
    flex: none;
    width: var(--space-lg);
    font-family: var(--font-data);
    color: var(--fg-muted);
  }
  .faq details[open] > summary::before { content: "−"; }
  .faq summary::-webkit-details-marker { display: none; }
  .faq summary { list-style: none; }
  .faq details > p { max-width: 70ch; margin: 0 0 var(--space-sm) var(--space-lg); }

  .python-band { display: grid; gap: var(--space-sm); justify-items: start; }
  .python-band h2 { font-size: var(--text-feature-title); letter-spacing: -0.035em; line-height: 1.2; }
  .python-band p { max-width: 64ch; margin: 0; }
  .text-link { display: inline-flex; align-items: center; gap: var(--space-xs); min-height: 44px; font-weight: var(--weight-reading-medium); }

  h1, h2, h3, .lede, .audience,
  .hero-action-row > span, .trust-line,
  .capabilities li p, .models-heading > p, .model-note,
  .model-row strong {
    color: var(--fg);
  }


  @media (prefers-contrast: more), (forced-colors: active), (prefers-reduced-transparency: reduce) {
    .capabilities li,
    .model-group,
    .demo-panel,
    .guide-panel {
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
    .landing-shell :is(a, h1, h2, h3, p, span, strong, dt, dd, summary):not(.primary-action) {
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
    .guide-grid { grid-template-columns: 1fr; }
  }

  @media (prefers-reduced-motion: reduce) {
    .orb-passage { display: none; }
  }
</style>
