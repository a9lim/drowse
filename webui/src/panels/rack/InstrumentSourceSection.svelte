<script lang="ts">
  // Canonical SOURCE section for the SAE and J-LENS pillars. Prepared and
  // provider-backed artifacts share one selector. Local authoring is an
  // explicit selector mode: its fields stay out of the way until chosen, and
  // the source-row action becomes the pillar's fit/train action.

  import type { Snippet } from "svelte";
  import type { InstrumentSourceJSON } from "../../lib/types";
  import Select from "../../lib/Select.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import RackSectionHeader from "./RackSectionHeader.svelte";

  interface ProviderOption {
    value: string;
    label: string;
    disabled?: boolean;
  }

  let {
    ready,
    sources,
    value = $bindable(),
    busy = false,
    sourceError = null,
    working = false,
    allowLocal = true,
    selectionCurrent = true,
    onuse,
    providerOptions,
    providerPlaceholder = "provider source",
    onfetch,
    localControls,
    localSectionLabel = null,
    localActionLabel = "",
    localActionDisabled = false,
    onlocal = () => undefined,
    sourceControls,
    progress,
    warning,
    messages,
    unavailableMessage = null,
  }: {
    ready: boolean;
    sources: InstrumentSourceJSON[];
    value: string;
    busy?: boolean;
    sourceError?: string | null;
    working?: boolean;
    allowLocal?: boolean;
    /** Whether secondary source settings match the resident runtime. */
    selectionCurrent?: boolean;
    onuse: (source: string) => void;
    providerOptions: ProviderOption[];
    providerPlaceholder?: string;
    onfetch: (source: string) => void;
    localControls?: Snippet;
    localSectionLabel?: string | null;
    localActionLabel?: string;
    localActionDisabled?: boolean;
    onlocal?: () => void;
    /** Optional controls that sit directly below the source picker. */
    sourceControls?: Snippet;
    progress?: Snippet;
    warning?: Snippet;
    messages?: Snippet;
    unavailableMessage?: string | null;
  } = $props();
  let selectionTouched = $state(false);

  const options = $derived.by(() => {
    const prepared = new Map(sources.map((source) => [source.source, source]));
    const providers = new Set(providerOptions.map((option) => option.value));
    // Provider options define the product order whether each source is
    // already prepared or still needs fetching. Provider-specific labels and
    // disabled state therefore stay stable across lifecycle transitions.
    const result: ProviderOption[] = providerOptions.map((option) => ({
      ...option,
      value: prepared.get(option.value)?.source ?? option.value,
    }));
    // Named local fits and any future external bindings follow the supported
    // provider tier in the order supplied by the server.
    for (const source of sources) {
      if (providers.has(source.source)) continue;
      if (
        !allowLocal &&
        (source.kind === "local" ||
          source.source === "local" ||
          source.source.startsWith("local:"))
      ) continue;
      result.push({ value: source.source, label: source.source });
    }
    if (allowLocal && !prepared.has("local")) {
      result.push({ value: "local", label: "local" });
    }
    return result;
  });
  // The picker owns the *validity* of the bound selection; the panels own
  // what a selection means.  An empty or no-longer-offered value falls
  // back to the active prepared source, then the first prepared source,
  // then the first provider option — which is also what makes a fresh
  // mount land on the active source without the panel arranging it.
  $effect(() => {
    const offered = new Set(options.map((option) => option.value));
    const active = sources.find(
      (source) => source.active && offered.has(source.source),
    )?.source;
    if (!selectionTouched && active) {
      value = active;
      return;
    }
    if (value && offered.has(value)) return;
    value =
      active ??
      sources.find((source) => offered.has(source.source))?.source ??
      options[0]?.value ??
      "";
  });

  const localSelected = $derived(value === "local");
  const selectedSource = $derived(sources.find((source) => source.source === value));
  const selectedProviderOption = $derived(
    providerOptions.find((option) => option.value === value),
  );
  const selectedOption = $derived(options.find((option) => option.value === value));

  function applySource(): void {
    if (!value) return;
    if (localSelected) {
      onlocal();
      return;
    }
    if (selectedSource) {
      // A prepared external source and its provider intentionally share one
      // identifier (for example ``neuronpedia``).  Treat it as prepared first;
      // only re-enter the provider fetch when the active binding is not usable.
      if (selectedSource.active && !ready && selectedProviderOption) onfetch(value);
      else onuse(value);
    }
    else onfetch(value);
  }
</script>

<section class="source-section">
  <RackSectionHeader
    title="Compatible data"
    help="Features and layer predictions use a pack built for this model. Packs stay on this device."
    count={ready ? "installed" : "needed"}
  />

  {#if sourceError}
    <p class="source-error" role="alert">{sourceError}</p>
  {/if}

  {#if options.length === 0 && !allowLocal && unavailableMessage}
    <p class="source-unavailable">{unavailableMessage}</p>
  {/if}

  {#if working && progress}
    <div class="progress">{@render progress()}</div>
  {:else}
    <div class="setup-stack">
      <div class="setup-group">
        <div class="setup-row source-row">
          <div class="setup-controls">
            <Select
              bind:value
              {options}
              placeholder={providerPlaceholder}
              disabled={busy || options.length === 0}
              ariaLabel="Compatible data pack"
              onchange={() => (selectionTouched = true)}
            />
          </div>
          <div class="setup-action">
            <Button
              size="sm"
              variant="solid"
              {busy}
              disabled={busy || !value ||
                (localSelected
                  ? localActionDisabled
                  : selectedOption?.disabled === true ||
                    (selectedSource?.active === true &&
                      selectionCurrent &&
                      (ready || selectedProviderOption === undefined)))}
              onclick={applySource}
            >
              {busy
                ? "Preparing…"
                : localSelected
                  ? localActionLabel
                  : selectedSource?.active
                  ? ready && selectionCurrent
                    ? "In use"
                    : ready
                    ? "Use pack"
                    : selectedProviderOption
                    ? "Download again"
                      : "Unavailable"
                  : selectedSource
                    ? "Use pack"
                    : "Download"}
            </Button>
          </div>
        </div>
        {#if sourceControls}
          <div class="setup-row supplemental-row">
            <div class="setup-controls">{@render sourceControls()}</div>
          </div>
        {/if}
      </div>
      {#if localSelected}
        <div class="setup-group local-group">
          {#if localSectionLabel}
            <span class="setup-label">{localSectionLabel}</span>
          {/if}
          {#if localControls}
            <div class="setup-row local-row">
              <div class="setup-controls">{@render localControls()}</div>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  {#if warning}
    <div class="warning">{@render warning()}</div>
  {/if}

  {#if messages}
    <div class="messages">{@render messages()}</div>
  {/if}
</section>

<style>
  .source-section {
    display: flex;
    flex: 0 0 auto;
    flex-direction: column;
    gap: var(--space-3);
    min-width: 0;
    padding: var(--surface-padding);
  }
  .setup-stack,
  .setup-group,
  .progress,
  .warning,
  .messages {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .source-unavailable {
    margin: 0;
    color: var(--fg-muted);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    line-height: 1.45;
  }
  .setup-group {
    gap: var(--space-1);
  }
  .setup-label {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .setup-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: stretch;
    gap: var(--space-2);
    min-width: 0;
  }
  .local-row {
    grid-template-columns: minmax(0, 1fr);
  }
  .supplemental-row {
    grid-template-columns: minmax(0, 1fr);
  }
  .setup-controls {
    display: flex;
    align-items: stretch;
    gap: var(--space-2);
    min-width: 0;
  }
  .setup-action {
    display: flex;
    align-items: flex-end;
  }
  .setup-action :global(button) {
    height: var(--control-compact);
  }
  .setup-controls :global(input),
  .setup-controls :global(.source-field) {
    min-width: 0;
  }
  .setup-controls :global(.setup-field) {
    display: flex;
    flex: 1 1 0;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .setup-controls :global(.setup-field-narrow) {
    flex: 0.65 1 4.5rem;
  }
  .setup-controls :global(.setup-field-medium) {
    flex: 0.85 1 6.5rem;
  }
  .setup-controls :global(.setup-field-wide) {
    flex: 1.3 1 8rem;
  }
  .setup-controls :global(.setup-field-label) {
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    letter-spacing: 0.06em;
    line-height: 1;
    text-transform: uppercase;
  }
  .setup-controls :global(.setup-field input) {
    box-sizing: border-box;
    min-height: var(--control-compact);
    width: 100%;
  }
  .source-error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
  }
  /* Optional snippets often render no DOM at all. Empty flex children still
     consumed a section gap, producing the large SOURCE→STEER void. */
  .warning:empty,
  .messages:empty {
    display: none;
  }
</style>
