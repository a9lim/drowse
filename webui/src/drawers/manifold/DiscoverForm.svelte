<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  // Auto-generated authoring: hand the model a flat concept list, the
  // K-tuple generator produces per-concept corpora against the shared
  // baseline prompts, then the fitter derives coords per-model via PCA or
  // spectral embedding.  No coords to author; no domain to pick.
  //
  // Generate and fit are deliberately two server calls — a flaky
  // generation leaves inspectable corpora — but "fit now" chains them so
  // the common case is one gesture.

  import { onMount, tick } from "svelte";

  import {
    apiManifoldFitStream,
    apiManifoldGenerateStream,
    describeError,
  } from "../../lib/runtime/services";
  import { openDrawer, refreshManifoldList, attachProbe } from "../../lib/stores.svelte";
  import { pushToast } from "../../lib/stores/toasts.svelte";
  import { manifoldJobs } from "../../lib/stores/manifoldJobs.svelte";
  import { EXAMPLES_PER_ROUND, EXAMPLE_BUDGETS } from "../../lib/exampleBudget";
  import { isFittingCancellation } from "../../lib/runtime/fittingCancellation";
  import type { FitManifoldRequest, GenerateManifoldRequest } from "../../lib/types";
  import Button from "../../lib/ui/Button.svelte";
  import Checkbox from "../../lib/Checkbox.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Radio from "../../lib/Radio.svelte";
  import Select from "../../lib/Select.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import DiscoverTuningFields from "./DiscoverTuningFields.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    parseTokens,
    slug,
    tuningHyperparams,
    tuningMessages,
    type ManifoldIdentity,
  } from "./shared";
  import { getRuntimeManifoldFitMaxIntrinsicDim } from "../../lib/runtime/registry";
  import { runtimeClient } from "../../lib/runtime/client";
  import { PER_NODE_ROLE_HELP } from "../../lib/manifolds/selectors";
  import {
    refreshSaeSources,
    saeSourceState,
  } from "../../lib/stores/instruments.svelte";

  let { identity, oncomplete }: { identity: ManifoldIdentity; oncomplete?: (result: { namespace: string; name: string }) => void } = $props();

  type DiscoverKind = "abstract" | "concrete" | "custom";

  const maxDimLimit = getRuntimeManifoldFitMaxIntrinsicDim();
  const browserMode = runtimeClient.mode !== "http";
  const tuning = $state(defaultTuning(maxDimLimit, browserMode));
  let conceptsText = $state("");
  // Conversational corpus knobs: ``kind`` frames each concept's system
  // prompt (abstract → "someone {c}", concrete → "{article} {c}");
  // ``samplesPerPrompt`` is the in-character responses generated per
  // shared baseline prompt.
  let kind: DiscoverKind = $state("concrete");
  let customSystem = $state("You are {c}.");
  let samplesPerPrompt = $state(1);
  let budget = $state("1");
  // Persona-manifold opt-in: when set, each concept slug doubles as the
  // matching node's assistant-role substitution at fit time, producing a
  // role-paired manifold (steering through it implies the nearest node's
  // role at decode time).  The slug regex matches the engine's role
  // validation — concepts that pass ``slug()`` are already in
  // ``[a-z0-9._-]+``.
  let rolePerNode = $state(false);
  let force = $state(false);
  let saeRelease = $state("");
  let alsoFit = $state(true);
  let advancedOpen = $state(false);
  let submitting = $state(false);
  let result = $state<{ namespace: string; name: string; fitted: boolean } | null>(null);
  let lastFit: FitManifoldRequest = {};
  let error = $state<string | null>(null);
  let attaching = $state(false);
  let attached = $state(false);
  const anotherJob = $derived(manifoldJobs.current !== null && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status));
  let validationAttempted = $state(false);
  let formRegion: HTMLDivElement | null = $state(null);
  let conceptsInput: HTMLTextAreaElement | null = $state(null);
  let samplesField: HTMLLabelElement | null = $state(null);
  let customSystemInput: HTMLTextAreaElement | null = $state(null);

  const concepts = $derived(parseTokens(conceptsText));
  const examplesPerConcept = $derived(samplesPerPrompt * EXAMPLES_PER_ROUND);
  const totalExamples = $derived(examplesPerConcept * concepts.length);
  const browserSaeOptions = $derived([
    { value: "", label: "Standard fit (no feature pack)" },
    ...saeSourceState.sources.map((source) => ({
      value: source.source,
      label: source.name?.trim() || source.source,
    })),
  ]);

  onMount(() => {
    if (browserMode) void refreshSaeSources();
  });

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(identity.name)) {
      messages.push("name required");
    }
    if (concepts.length < 2) {
      messages.push(`concepts: ${concepts.length} / 2`);
    }
    const seen = new Set<string>();
    for (const c of concepts) {
      const s = slug(c);
      if (!s) {
        messages.push(`invalid concept "${c}"`);
      } else if (seen.has(s)) {
        messages.push(`duplicate concept "${s}"`);
      } else {
        seen.add(s);
      }
    }
    if (!Number.isSafeInteger(samplesPerPrompt) || samplesPerPrompt < 1) {
      messages.push("Choose a whole number of responses per prompt, at least 1.");
    }
    if (kind === "custom") {
      const template = customSystem.trim();
      if (!template) {
        messages.push("system template required");
      } else if (!template.includes("{c}")) {
        messages.push('system template needs "{c}"');
      }
    }
    messages.push(...tuningMessages(tuning, maxDimLimit));
    return { ok: messages.length === 0, messages };
  });

  const conceptsError = $derived.by<string | null>(() => {
    if (!validationAttempted) return null;
    if (concepts.length < 2) return "Enter at least two concepts.";
    const seen = new Set<string>();
    for (const concept of concepts) {
      const formatted = slug(concept);
      if (!formatted) return `Concept “${concept}” needs a letter or number.`;
      if (seen.has(formatted)) return "Each concept must be unique after formatting.";
      seen.add(formatted);
    }
    return null;
  });

  const samplesError = $derived(
    validationAttempted && (!Number.isSafeInteger(samplesPerPrompt) || samplesPerPrompt < 1)
      ? "Use a whole number, at least 1."
      : null,
  );

  const customSystemError = $derived.by<string | null>(() => {
    if (!validationAttempted || kind !== "custom") return null;
    if (!customSystem.trim()) return "Enter a system template.";
    if (!customSystem.includes("{c}")) return 'Include the "{c}" placeholder.';
    return null;
  });

  $effect(() => {
    const input = samplesField?.querySelector("input");
    if (!input) return;
    if (samplesError) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", "discover-samples-error");
    } else {
      input.removeAttribute("aria-invalid");
      input.removeAttribute("aria-describedby");
    }
  });

  function sharedNameInput(): HTMLInputElement | null {
    const body = formRegion?.closest(".mb-form");
    return body?.querySelector<HTMLInputElement>(
      ":scope > .grid2 > .field:nth-child(2) input",
    ) ?? null;
  }

  function tuningInput(label: string): HTMLInputElement | null {
    const fields = formRegion?.querySelectorAll<HTMLElement>(".field") ?? [];
    for (const field of fields) {
      const text = field.querySelector<HTMLElement>(".label")?.textContent?.trim();
      if (text === label) return field.querySelector<HTMLInputElement>("input");
    }
    return null;
  }

  async function focusFirstInvalid(): Promise<void> {
    await tick();
    if (!slug(identity.name)) {
      sharedNameInput()?.focus();
      return;
    }
    if (conceptsError) {
      conceptsInput?.focus();
      return;
    }
    if (samplesError) {
      samplesField?.querySelector<HTMLInputElement>("input")?.focus();
      return;
    }
    if (customSystemError) {
      customSystemInput?.focus();
      return;
    }
    const tuningErrors = tuningMessages(tuning, maxDimLimit);
    if (tuningErrors.length === 0) return;
    advancedOpen = true;
    await tick();
    const label = tuning.maxDim < 1 || maxDimLimit !== null && tuning.maxDim > maxDimLimit
      ? "max dim"
      : "variance";
    tuningInput(label)?.focus();
  }

  async function save(): Promise<void> {
    if (submitting || anotherJob) return;
    validationAttempted = true;
    if (!validation.ok) {
      await focusFirstInvalid();
      return;
    }
    submitting = true;
    error = null;
    result = null;
    attached = false;
    const { namespace, name, description } = identitySlugs(identity);
    const hyperparams = tuningHyperparams(tuning);
    const req: GenerateManifoldRequest = {
      namespace,
      name,
      description,
      concepts: concepts.map((c) => slug(c)),
      kind,
      custom_system: kind === "custom" ? customSystem.trim() : undefined,
      samples_per_prompt: samplesPerPrompt,
      fit_mode: tuning.fitMode,
      hyperparams,
      force,
      role_per_node: rolePerNode,
    };
    lastFit = { sae: saeRelease.trim() || null, fit_mode: tuning.fitMode, hyperparams };
    const fitRequested = alsoFit;
    try {
      await apiManifoldGenerateStream(req, () => {}, fitRequested);
      result = { namespace, name, fitted: false };
      if (fitRequested) {
        await apiManifoldFitStream(namespace, name, lastFit, () => {});
        result = { namespace, name, fitted: true };
      }
      await refreshManifoldList();
    } catch (e) {
      error = isFittingCancellation(e)
        ? result ? "Fitting cancelled. Your examples are saved; you can fit them below." : "Generation cancelled. Completed concept groups are kept; run again to fill any missing groups."
        : `${result ? "Examples were saved, but fitting failed" : "Could not generate the examples"}: ${describeError(e)}`;
      await refreshManifoldList();
    } finally {
      submitting = false;
    }
  }

  async function fitSaved(): Promise<void> {
    if (!result || submitting || anotherJob) return;
    const saved = result;
    submitting = true;
    error = null;
    try {
      await apiManifoldFitStream(saved.namespace, saved.name, lastFit, () => {});
      result = { ...saved, fitted: true };
      await refreshManifoldList();
    } catch (e) {
      error = isFittingCancellation(e) ? "Fitting cancelled. Your saved examples are ready to try again." : `Could not fit the saved examples: ${describeError(e)}`;
    } finally { submitting = false; }
  }

  async function addProbe(): Promise<void> {
    if (!result?.fitted || attaching) return;
    attaching = true;
    try {
      await attachProbe(`${result.namespace}/${result.name}`);
      attached = true;
      pushToast("Probe added. Generate a reply to see its readings.");
    } catch (e) { error = `Could not add the probe: ${describeError(e)}`; }
    finally { attaching = false; }
  }
</script>

<div
  bind:this={formRegion}
  class="form-stack"
  role="form"
  aria-label="Generate manifold"
  aria-busy={submitting}
>
  <fieldset class="creation-fields" disabled={submitting || anotherJob}>
  <section class="step">
    <h2 class="step-title">Concepts to compare</h2>
    <label class="field">
      <span class="label">Concepts (at least 2)</span>
      <textarea
        bind:this={conceptsInput}
        class="input"
        rows="4"
        placeholder="pirate, assistant"
        bind:value={conceptsText}
        spellcheck="false"
        aria-invalid={!!conceptsError}
        aria-describedby={conceptsError ? "discover-concepts-error" : undefined}
      ></textarea>
      {#if conceptsError}
        <span id="discover-concepts-error" class="field-error">{conceptsError}</span>
      {/if}
      <span class="dim-note">
        <strong><MorphText text={concepts.length} /></strong> concepts · separate with commas or new lines; join words with underscores
      </span>
    </label>
    <div>
      <div class="field">
        <span class="label">What do the concepts describe?</span>
        <div class="radio-row" role="radiogroup" aria-label="Concept kind">
          <Radio bind:group={kind} value="concrete" label="Roles or characters" />
          <Radio bind:group={kind} value="abstract" label="Traits or feelings" />
          <Radio bind:group={kind} value="custom" label="Custom instructions" />
        </div>
      </div>
    </div>
    {#if kind === "custom"}
      <label class="field">
        <span class="label">system template</span>
        <textarea
          bind:this={customSystemInput}
          class="input"
          rows="3"
          bind:value={customSystem}
          spellcheck="false"
          aria-invalid={!!customSystemError}
          aria-describedby={customSystemError
            ? "discover-system-error"
            : undefined}
        ></textarea>
        {#if customSystemError}
          <span id="discover-system-error" class="field-error">
            {customSystemError}
          </span>
        {/if}
      </label>
    {/if}
  </section>

  <section class="step" aria-labelledby="example-budget-title">
    <h2 id="example-budget-title" class="step-title">Example budget</h2>
    <div class="budget-options" role="radiogroup" aria-label="Example budget">
      {#each EXAMPLE_BUDGETS as option}
        <label class="budget-option" class:selected={budget === String(option.rounds)}>
          <input type="radio" name="example-budget" value={String(option.rounds)} bind:group={budget} onchange={() => (samplesPerPrompt = option.rounds)} />
          <span><strong>{option.label}</strong><span>{option.rounds * EXAMPLES_PER_ROUND} examples per concept</span><small>{option.description}</small></span>
        </label>
      {/each}
      <label class="budget-option" class:selected={budget === "custom"}>
        <input type="radio" name="example-budget" value="custom" bind:group={budget} />
        <span><strong>Custom</strong><span>Choose the number of responses per prompt.</span></span>
      </label>
    </div>
    {#if budget === "custom"}
      <label bind:this={samplesField} class="field">
        <span class="label">Responses per prompt</span>
        <NumberInput value={samplesPerPrompt} min={1} step={1} oninput={(v) => { if (v !== null) samplesPerPrompt = v; }} />
        {#if samplesError}<span id="discover-samples-error" class="field-error">{samplesError}</span>{/if}
      </label>
    {/if}
    <p class="budget-total"><strong>{Number.isSafeInteger(totalExamples) && totalExamples > 0 ? `${totalExamples.toLocaleString()} examples total` : "Choose concepts and an example budget"}</strong><span>{concepts.length} concepts × {examplesPerConcept} examples</span></p>
    <p class="budget-help">Each round covers all {EXAMPLES_PER_ROUND} prompts. More examples can reduce sampling noise, but do not guarantee a better probe. Start with Standard, then inspect the result.</p>
  </section>

  <AdvancedSection bind:expanded={advancedOpen}>
    <FitMethodPicker {tuning} linearOnly={browserMode} spectralNote="curved · best with ≥50 nodes" onchange={(fitMode) => (tuning.fitMode = fitMode)} />
    {#if browserMode}
      <div class="field">
        <span class="label">feature space for fit</span>
        <Select
          bind:value={saeRelease}
          options={browserSaeOptions}
          disabled={saeSourceState.loading || saeSourceState.busy}
          ariaLabel="Feature space used for manifold fitting"
        />
        {#if saeSourceState.error}
          <span class="field-error">{saeSourceState.error}</span>
        {:else if !saeSourceState.loading && saeSourceState.sources.length === 0}
          <span class="dim-note">No compatible learned-feature pack is installed.</span>
        {/if}
      </div>
    {:else}
      <label class="field">
        <span class="label">SAE release</span>
        <input
          class="input"
          bind:value={saeRelease}
          placeholder="e.g. gemma-scope-2-4b-it-res"
          spellcheck="false"
        />
      </label>
    {/if}
    <DiscoverTuningFields {tuning} {maxDimLimit} />
    <div class="check-stack">
      <Checkbox bind:checked={alsoFit} label="Fit after generating examples" />
      <Checkbox
        bind:checked={rolePerNode}
        label="Use a separate role for each concept"
        title={PER_NODE_ROLE_HELP}
      />
      {#if rolePerNode}
        <p class="role-hint">
          each concept becomes its assistant voice; unsupported chat templates fail before fitting
        </p>
      {/if}
      <Checkbox bind:checked={force} label="Replace existing examples with this name" />
      <p class="budget-help">Without replacement, existing concept groups are reused. To change their example budget, use a new name or enable replacement.</p>
    </div>
  </AdvancedSection>
  </fieldset>
  {#if error}<p class="creation-error" role="alert">{error}</p>{/if}
  {#if result && !submitting}
    <section class="creation-result" aria-label="Creation result">
      <strong>{result.fitted ? "Ready to use" : "Examples saved"}: {result.namespace}/{result.name}</strong>
      <p>{result.fitted ? "Add a probe to measure the next reply, or open the library to steer with it." : "Fit these saved examples when you are ready. You do not need to generate them again."}</p>
      <div class="result-actions">
        {#if result.fitted}<Button variant="solid" disabled={attached || attaching} onclick={() => void addProbe()}>{attached ? "Probe added" : "Add probe"}</Button>
        {:else}<Button variant="solid" disabled={anotherJob} onclick={() => void fitSaved()}>Fit saved examples</Button>{/if}
        <Button onclick={() => oncomplete ? oncomplete(result!) : openDrawer("subspace")}>Open library</Button>
      </div>
    </section>
  {/if}

  <ValidationBlock
    verb="generate examples"
    messages={validationAttempted ? validation.messages : []}
  />

  <button
    type="button"
    class="save-btn"
    disabled={submitting || anotherJob}
    onclick={save}
  >
    <MorphText text={alsoFit ? "Generate and fit" : "Generate examples"} />
  </button>
</div>

<style>
  .field-error {
    color: var(--accent-red);
    font-size: var(--text-xs);
    line-height: 1.4;
  }

  textarea[aria-invalid="true"] {
    border-color: var(--accent-red);
  }
  .creation-fields { display: flex; flex-direction: column; gap: var(--space-lg); border: 0; padding: 0; margin: 0; min-width: 0; }
  .budget-options { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-sm); }
  .budget-option { display: flex; align-items: start; gap: var(--space-sm); min-height: var(--control-target); background: var(--workspace-field-bg); border: 1px solid transparent; border-radius: var(--radius-group); padding: var(--space-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), border-color var(--dur-fast) var(--ease-out); }
  .budget-option.selected { background: var(--workspace-neutral-bg); border-color: var(--fg-dim); }
  .budget-option:has(input:focus-visible) { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .creation-fields:disabled .budget-option { cursor: not-allowed; opacity: var(--disabled-opacity); }
  .budget-option input { margin-block-start: var(--space-xs); accent-color: var(--accent); flex: none; }
  .budget-option > span { display: grid; gap: var(--space-xs); min-width: 0; }
  .budget-option span span, .budget-option small { font-size: var(--text-sm); color: var(--fg-dim); line-height: 1.5; font-variant-numeric: tabular-nums; text-wrap: pretty; }
  .step-title, .budget-option strong { text-wrap: balance; }
  .budget-total { display: flex; flex-wrap: wrap; gap: var(--space-sm); margin: var(--space-sm) 0 0; font-variant-numeric: tabular-nums; }
  .budget-total span { color: var(--fg-dim); }
  .budget-help { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.5; text-wrap: pretty; }
  .creation-error { color: var(--accent-red); line-height: 1.5; overflow-wrap: anywhere; }
  .creation-result { display: grid; gap: var(--space-sm); padding: var(--surface-padding); background: var(--workspace-field-bg); border-radius: var(--popup-radius); overflow-wrap: anywhere; }
  .creation-result p { margin: 0; line-height: 1.5; color: var(--fg-dim); text-wrap: pretty; }
  .result-actions { display: flex; flex-wrap: wrap; gap: var(--space-sm); }
  .save-btn { transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .save-btn:active:not(:disabled) { scale: var(--press-scale); }
  @media (hover: hover) { .creation-fields:not(:disabled) .budget-option:hover { background: var(--workspace-neutral-hover); } }
  @media (forced-colors: active) { .budget-option { border-color: CanvasText; } .budget-option.selected { border-color: Highlight; outline: 1px solid Highlight; } }
  @container (max-width: 360px) { .budget-options { grid-template-columns: 1fr; } }
  @media (max-width: 380px) { .budget-options { grid-template-columns: 1fr; } }
</style>
