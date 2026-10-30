<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../../lib/workspaceController";
  import { templatedDraftSchema } from "../../lib/interfaceSchemas";
  import { ToolError } from "../../lib/webmcp/types";

  import MorphText from "../../lib/ui/MorphText.svelte";
  // Derive a manifold from a standalone template.
  //
  // Some categories you *reference* rather than embody — days, months,
  // durations — so the persona-framed generation doesn't apply.  A
  // template (slot + candidate values + multi-turn contexts) materializes
  // those deterministically: no model call, the node corpus is just the
  // slot-filled assistant turns.
  //
  // This tab does NOT author templates.  It used to, with its own
  // single-user-turn editor, which was a strict subset of the template
  // lab's multi-turn editor and had already drifted away from the server's
  // validation once.  Authoring now lives in exactly one place — the
  // template lab's build tab, linked below — and this tab picks one of
  // those templates and runs the derivation the lab has no verb for.

  import { onMount, tick } from "svelte";
  import {
    apiManifoldFitStream,
    apiManifolds,
    apiTemplates,
    describeError,
  } from "../../lib/runtime/services";
  import { isFittingCancellation } from "../../lib/runtime/fittingCancellation";
  import { runtimeClient } from "../../lib/runtime/client";
  import {
    getHostedController,
    getRuntimeManifoldFitMaxIntrinsicDim,
  } from "../../lib/runtime/registry";
  import { closeDrawer, openDrawer, refreshManifoldList } from "../../lib/stores.svelte";
  import { dismissToast, pushToast } from "../../lib/stores/toasts.svelte";
  import type {
    CreateManifoldFromTemplateRequest,
    TemplateSummary,
  } from "../../lib/types";
  import Checkbox from "../../lib/Checkbox.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Select from "../../lib/Select.svelte";
  import AdvancedSection from "../../lib/builder/AdvancedSection.svelte";
  import ValidationBlock from "../../lib/builder/ValidationBlock.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import FitMethodPicker from "./FitMethodPicker.svelte";
  import {
    defaultTuning,
    identitySlugs,
    maxDimensionValidationMessage,
    slug,
    type ManifoldIdentity,
  } from "./shared";

  let { identity, oncomplete }: { identity: ManifoldIdentity; oncomplete?: () => void } = $props();

  const maxDimLimit = getRuntimeManifoldFitMaxIntrinsicDim();
  const browserMode = runtimeClient.mode !== "http";
  const tuning = $state(defaultTuning(maxDimLimit, browserMode));
  let templates: TemplateSummary[] = $state([]);
  let loadingTemplates = $state(true);
  let selectedKey = $state("");
  let maxDim: number | null = $state(null);
  let alsoFit = $state(true);
  let advancedOpen = $state(false);
  let submitting = $state(false);
  let validationAttempted = $state(false);
  let progress = $state("");
  let formRegion: HTMLDivElement | null = $state(null);
  let templateField: HTMLLabelElement | null = $state(null);
  let maxDimField: HTMLLabelElement | null = $state(null);
  let authorTemplateButton: HTMLButtonElement | null = $state(null);
  let fittingActive = $state(false);
  let cancelling = $state(false);
  const hostedController = getHostedController();

  onMount(async () => {
    try {
      templates = (await apiTemplates.list()).templates;
    } catch (e) {
      pushToast(`couldn't load templates: ${describeError(e)}`, {
        kind: "error",
      });
    } finally {
      loadingTemplates = false;
    }
  });

  const options = $derived(
    templates.map((t) => ({
      value: `${t.namespace}/${t.name}`,
      label: `${t.namespace}/${t.name} · ${t.slot} · ${t.n_values}×${t.n_contexts}`,
    })),
  );

  const selected = $derived(
    templates.find((t) => `${t.namespace}/${t.name}` === selectedKey) ?? null,
  );

  const validation = $derived.by<{ ok: boolean; messages: string[] }>(() => {
    const messages: string[] = [];
    if (!slug(identity.name)) messages.push("name required");
    if (!selected) messages.push("template required");
    const maxDimensionMessage = maxDimensionValidationMessage(maxDim, maxDimLimit);
    if (maxDimensionMessage) messages.push(maxDimensionMessage);
    return { ok: messages.length === 0, messages };
  });

  const templateError = $derived(
    validationAttempted && !selected ? "Choose a template." : null,
  );
  const maxDimError = $derived(
    validationAttempted
      ? maxDimensionValidationMessage(maxDim, maxDimLimit)
      : null,
  );

  $effect(() => {
    const trigger = templateField?.querySelector("button");
    if (!trigger) return;
    if (templateError) {
      trigger.setAttribute("aria-invalid", "true");
      trigger.setAttribute("aria-describedby", "templated-source-error");
    } else {
      trigger.removeAttribute("aria-invalid");
      trigger.removeAttribute("aria-describedby");
    }
  });

  $effect(() => {
    const input = maxDimField?.querySelector("input");
    if (!input) return;
    if (maxDimError) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", "templated-max-dim-error");
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

  async function focusFirstInvalid(): Promise<void> {
    await tick();
    if (!slug(identity.name)) {
      sharedNameInput()?.focus();
      return;
    }
    if (templateError) {
      const trigger = templateField?.querySelector<HTMLButtonElement>("button");
      (trigger ?? authorTemplateButton)?.focus();
      return;
    }
    if (!maxDimError) return;
    advancedOpen = true;
    await tick();
    maxDimField?.querySelector<HTMLInputElement>("input")?.focus();
  }

  /** Hand authoring to the one editor that speaks the full multi-turn
   *  context shape, then come back here to derive. */
  function openTemplateLab(): void {
    closeDrawer();
    openDrawer("template_lab", { tab: "build" });
  }

  async function cancelFit(): Promise<void> {
    if (!hostedController || !fittingActive || cancelling) return;
    cancelling = true;
    progress = "Cancelling fit…";
    try {
      await hostedController.cancelFitting();
    } catch (e) {
      pushToast(`Couldn't cancel the fit: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      cancelling = false;
    }
  }

  async function save(): Promise<void> {
    if (submitting) return;
    validationAttempted = true;
    if (!validation.ok || !selected) {
      await focusFirstInvalid();
      return;
    }
    submitting = true;
    progress = "Starting authoring…";
    const { namespace, name, description } = identitySlugs(identity);
    const hyperparams: Record<string, number> = {};
    if (maxDim !== null && maxDim >= 1) hyperparams.max_dim = maxDim;
    const req: CreateManifoldFromTemplateRequest = {
      namespace,
      name,
      description,
      fit_mode: tuning.fitMode,
      template_ref: `${selected.namespace}/${selected.name}`,
      hyperparams,
    };
    const toastId = pushToast(`authoring ${namespace}/${name}…`, {
      kind: "info",
      ttlMs: null,
    });
    try {
      await apiManifolds.createFromTemplate(req);
      dismissToast(toastId);
      if (alsoFit) {
        progress = "Starting fit…";
        fittingActive = true;
        try {
          await apiManifoldFitStream(
            namespace,
            name,
            { fit_mode: tuning.fitMode, hyperparams },
            (ev) => {
              if (ev.event !== "progress") return;
              const msg =
                ev.data && typeof ev.data === "object"
                  ? (ev.data as { message?: string }).message
                  : null;
              if (msg) {
                progress = msg;
              }
            },
          );
          pushToast(`fit ${namespace}/${name} (${tuning.fitMode})`, {
            kind: "info",
          });
        } catch (e) {
          if (isFittingCancellation(e)) {
            pushToast(
              `Fit cancelled. ${namespace}/${name} was kept and can be fitted later.`,
              { kind: "info" },
            );
          } else {
            pushToast(`Couldn't fit the manifold: ${describeError(e)}`, {
              kind: "error",
              ttlMs: null,
            });
          }
        } finally {
          fittingActive = false;
          cancelling = false;
        }
      } else {
        pushToast(
          `Created ${namespace}/${name}. Open Manifolds to fit it.`,
          { kind: "info" },
        );
      }
      await refreshManifoldList();
      if (oncomplete) oncomplete();
      else { closeDrawer(); openDrawer("manifolds"); }
    } catch (e) {
      dismissToast(toastId);
      pushToast(`Couldn't create the manifold: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      submitting = false;
      progress = "";
    }
  }

  onInterfaceMount(() => registerInterfaceController("manifold_templated", {
    schema: templatedDraftSchema,
    read: () => ({ busy: submitting || fittingActive || cancelling, values: { selected_key: selectedKey, max_dim: maxDim, also_fit: alsoFit, advanced_open: advancedOpen, tuning }, validation, available_templates: options, loading_templates: loadingTemplates }),
    update: async (change) => {
      if (submitting || fittingActive || cancelling) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      if (change.selected_key && !options.some(option => option.value === change.selected_key)) throw new ToolError("NOT_FOUND", "Choose a template from available_templates.");
      if (change.selected_key !== undefined) selectedKey = change.selected_key as string;
      if (change.max_dim !== undefined) maxDim = change.max_dim as number | null;
      if (change.also_fit !== undefined) alsoFit = change.also_fit as boolean;
      if (change.advanced_open !== undefined) advancedOpen = change.advanced_open as boolean;
      if (change.tuning) Object.assign(tuning, change.tuning);
    },
  }));
</script>

<div
  bind:this={formRegion}
  class="form-stack"
  role="form"
  aria-label="Build manifold from template"
  aria-busy={submitting}
>
  <section class="step">
    <h2 class="step-title">template</h2>
    {#if loadingTemplates}
      <p class="muted">loading…</p>
    {:else if templates.length === 0}
      <p class="muted">no templates yet</p>
    {:else}
      <label bind:this={templateField} class="field template-source">
        <span class="label">source *</span>
        <Select
          value={selectedKey}
          options={[{ value: "", label: "Choose a template" }, ...options]}
          ariaLabel="Template"
          onchange={(v) => {
            selectedKey = String(v);
          }}
        />
        {#if templateError}
          <span id="templated-source-error" class="field-error">{templateError}</span>
        {/if}
      </label>
      {#if selected}
        <p class="dim-note">
          slot <strong>{selected.slot}</strong> ·
          <strong>{selected.n_values}</strong> values ×
          <strong>{selected.n_contexts}</strong> contexts
        </p>
      {/if}
    {/if}
    <button
      bind:this={authorTemplateButton}
      type="button"
      class="add-node"
      onclick={openTemplateLab}
    >
      author a template…
    </button>
  </section>

  <FitMethodPicker
    {tuning}
    linearOnly={browserMode}
    onchange={(fitMode) => (tuning.fitMode = fitMode)}
  />

  <AdvancedSection bind:expanded={advancedOpen}>
    <label bind:this={maxDimField} class="field">
      <span class="label">max dim</span>
      <NumberInput
        value={maxDim}
        min={1}
        max={maxDimLimit ?? undefined}
        step={1}
        allowEmpty
        placeholder={maxDimLimit === null ? "auto" : `auto · max ${maxDimLimit}`}
        oninput={(v) => {
          maxDim = v;
        }}
      />
      {#if maxDimError}
        <span id="templated-max-dim-error" class="field-error">{maxDimError}</span>
      {:else if maxDimLimit !== null}
        <span class="dim-note">
          hosted browser limit · <strong>{maxDimLimit}</strong> dimensions
        </span>
      {/if}
    </label>
    <div class="check-stack">
      <Checkbox bind:checked={alsoFit} label="fit now" />
    </div>
  </AdvancedSection>

  <p class="progress form-status" role="status" aria-live="polite" aria-atomic="true">
    <MorphText text={progress} />
  </p>

  <ValidationBlock
    verb="building"
    messages={validationAttempted ? validation.messages : []}
  />

  <div class="form-actions">
    {#if hostedController && fittingActive}
      <Button variant="ghost" disabled={cancelling} onclick={cancelFit}>
        <MorphText text={cancelling ? "cancelling…" : "cancel"} numbers={false} />
      </Button>
    {/if}
    <button
      type="button"
      class="save-btn"
      disabled={submitting}
      onclick={save}
    >
      <MorphText text={submitting ? "building…" : alsoFit ? "build + fit" : "build"} numbers={false} />
    </button>
  </div>
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

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
  }

  .form-actions .save-btn {
    flex: 1;
  }

  .template-source :global(.sk-select-trigger[aria-invalid="true"]) {
    border-color: var(--accent-red);
  }
</style>
