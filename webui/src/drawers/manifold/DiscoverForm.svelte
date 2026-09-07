<script lang="ts">
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
  import { closeDrawer, openDrawer, refreshManifoldList } from "../../lib/stores.svelte";
  import { dismissToast, pushToast, updateToast } from "../../lib/stores/toasts.svelte";
  import type { GenerateManifoldRequest } from "../../lib/types";
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
    progressMessage,
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

  let { identity, oncomplete }: { identity: ManifoldIdentity; oncomplete?: () => void } = $props();

  type DiscoverKind = "abstract" | "concrete" | "custom";

  const maxDimLimit = getRuntimeManifoldFitMaxIntrinsicDim();
  const browserMode = runtimeClient.mode !== "http";
  const tuning = $state(defaultTuning(maxDimLimit));
  let conceptsText = $state("");
  // Conversational corpus knobs: ``kind`` frames each concept's system
  // prompt (abstract → "someone {c}", concrete → "{article} {c}");
  // ``samplesPerPrompt`` is the in-character responses generated per
  // shared baseline prompt.
  let kind: DiscoverKind = $state("abstract");
  let customSystem = $state("You are {c}.");
  let samplesPerPrompt = $state(1);
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
  let progress = $state("");
  let submitting = $state(false);
  let validationAttempted = $state(false);
  let formRegion: HTMLDivElement | null = $state(null);
  let conceptsInput: HTMLTextAreaElement | null = $state(null);
  let samplesField: HTMLLabelElement | null = $state(null);
  let customSystemInput: HTMLTextAreaElement | null = $state(null);

  const concepts = $derived(parseTokens(conceptsText));
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
    if (samplesPerPrompt <= 0) {
      messages.push("samples / prompt > 0");
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
    validationAttempted && samplesPerPrompt <= 0
      ? "Use at least one sample per prompt."
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
    if (submitting) return;
    validationAttempted = true;
    if (!validation.ok) {
      await focusFirstInvalid();
      return;
    }
    submitting = true;
    progress = "Starting generation…";
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
    const toastId = pushToast(`generating ${namespace}/${name} corpora…`, {
      kind: "info",
      ttlMs: null,
    });
    try {
      await apiManifoldGenerateStream(req, (ev) => {
        if (ev.event !== "progress") return;
        const msg = progressMessage(ev);
        if (msg) {
          progress = msg;
          updateToast(toastId, { detail: msg });
        }
      });
      dismissToast(toastId);
      if (alsoFit) {
        // Chain the fit immediately — the user opted into the two-step.
        // The fit endpoint accepts the discover-mode hyperparams as an
        // override; passing them here keeps the sidecar metadata in sync
        // even if the folder already had matching values from generate.
        const fitToastId = pushToast(`fitting ${namespace}/${name}…`, {
          kind: "info",
          ttlMs: null,
        });
        progress = "Starting fit…";
        try {
          await apiManifoldFitStream(
            namespace,
            name,
            {
              sae: saeRelease.trim() || null,
              fit_mode: tuning.fitMode,
              hyperparams,
            },
            (ev) => {
              if (ev.event !== "progress") return;
              const msg = progressMessage(ev);
              if (msg) {
                progress = msg;
                updateToast(fitToastId, { detail: msg });
              }
            },
          );
          dismissToast(fitToastId);
          pushToast(`fit ${namespace}/${name} (${tuning.fitMode})`, {
            kind: "info",
          });
        } catch (e) {
          dismissToast(fitToastId);
          pushToast(`Couldn't fit the manifold: ${describeError(e)}`, {
            kind: "error",
            ttlMs: null,
          });
        }
      } else {
        pushToast(
          `Generated ${namespace}/${name}. Open Manifolds to fit it.`,
          { kind: "info" },
        );
      }
      await refreshManifoldList();
      if (oncomplete) oncomplete();
      else { closeDrawer(); openDrawer("manifolds"); }
    } catch (e) {
      dismissToast(toastId);
      pushToast(`Couldn't generate the manifold: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
      progress = "";
    }
  }
</script>

<div
  bind:this={formRegion}
  class="form-stack"
  role="form"
  aria-label="Generate manifold"
  aria-busy={submitting}
>
  <section class="step">
    <h2 class="step-title">concepts</h2>
    <label class="field">
      <span class="label">concepts * · ≥2</span>
      <textarea
        bind:this={conceptsInput}
        class="input"
        rows="4"
        placeholder="pirate caveman assistant scholar robot"
        bind:value={conceptsText}
        spellcheck="false"
        aria-invalid={!!conceptsError}
        aria-describedby={conceptsError ? "discover-concepts-error" : undefined}
      ></textarea>
      {#if conceptsError}
        <span id="discover-concepts-error" class="field-error">{conceptsError}</span>
      {/if}
      <span class="dim-note">
        <strong>{concepts.length}</strong> parsed
      </span>
    </label>
    <div class="grid2">
      <div class="field">
        <span class="label">kind</span>
        <div class="radio-row" role="radiogroup" aria-label="Concept kind">
          <Radio bind:group={kind} value="abstract" label="abstract" />
          <Radio bind:group={kind} value="concrete" label="concrete" />
          <Radio bind:group={kind} value="custom" label="custom" />
        </div>
      </div>
      <label bind:this={samplesField} class="field">
        <span class="label">samples / prompt</span>
        <NumberInput
          value={samplesPerPrompt}
          min={1}
          step={1}
          oninput={(v) => {
            if (v !== null) samplesPerPrompt = v;
          }}
        />
        {#if samplesError}
          <span id="discover-samples-error" class="field-error">{samplesError}</span>
        {/if}
      </label>
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

  <FitMethodPicker
    {tuning}
    linearOnly={browserMode}
    spectralNote="curved · best with ≥50 nodes"
    onchange={(fitMode) => (tuning.fitMode = fitMode)}
  />

  <AdvancedSection bind:expanded={advancedOpen}>
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
      <Checkbox bind:checked={alsoFit} label="fit now" />
      <Checkbox
        bind:checked={rolePerNode}
        label="node roles"
        title={PER_NODE_ROLE_HELP}
      />
      {#if rolePerNode}
        <p class="role-hint">
          each concept becomes its assistant voice; unsupported chat templates fail before fitting
        </p>
      {/if}
      <Checkbox bind:checked={force} label="overwrite" />
    </div>
  </AdvancedSection>

  <p class="progress form-status" role="status" aria-live="polite" aria-atomic="true">
    {progress}
  </p>

  <ValidationBlock
    verb="generating"
    messages={validationAttempted ? validation.messages : []}
  />

  <button
    type="button"
    class="save-btn"
    disabled={submitting}
    onclick={save}
  >
    {submitting ? "generating…" : alsoFit ? "generate + fit" : "generate"}
  </button>
</div>

<style>
  .field-error {
    color: var(--accent-red);
    font-size: var(--text-xs);
    line-height: 1.4;
  }

  .form-status:empty {
    display: none;
  }

  textarea[aria-invalid="true"] {
    border-color: var(--accent-red);
  }
</style>
