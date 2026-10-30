<script lang="ts">
  import RollingNumber from "../lib/ui/RollingNumber.svelte";
  import ResetSettings from "../lib/ui/ResetSettings.svelte";
  import { settingsResetState } from "../lib/stores/settingsReset.svelte";
  // SamplingStrip keeps the four everyday controls visible. Secondary
  // generation and inspection controls live in AdvancedSamplingDrawer.
  //
  // Every edit applies immediately.  temperature / top-p / top-k /
  // max-tokens / thinking PATCH the session defaults as the user moves
  // them; seed and the advanced extras (penalties, stop strings, logit
  // bias, return_top_k) have no PATCH path, so ``sendGenerate`` packs
  // them onto each call's ``SamplingConfig``.  Either way the value the
  // strip shows is the value the next generation uses.
  //
  // Empty seed = null = no per-call seed pin (model RNG).  The 🎲 button
  // fills with a fresh ``Math.floor(Math.random() * 2**31)`` integer.

  import {
    samplingState,
    sessionState,
    setSampling,
    patchSessionDefaults,
    openDrawer,
  } from "../lib/stores.svelte";
  import Slider from "../lib/Slider.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import Checkbox from "../lib/Checkbox.svelte";
  import InfoTip from "../lib/ui/InfoTip.svelte";
  import { getRuntimeCapabilities, getHostedController } from "../lib/runtime/registry";
  import { outputTokenLimitForSignals } from "../lib/runtime/outputTokenPolicy";
  import { SAMPLING_HELP } from "../lib/parameterHelp";

  // ------------------------------------------------------------------- consts

  // Placeholder defaults shown while the session info hasn't landed yet.
  // These never reach the server — the strip is disabled in this state.
  const PLACEHOLDER = {
    temperature: 1.0,
    top_p: 1.0,
    max_tokens: 512,
  };

  const TEMP_MIN = 0;
  const TEMP_MAX = 2;
  const TEMP_STEP = 0.05;
  const TOP_P_MIN = 0;
  const TOP_P_MAX = 1;
  const TOP_P_STEP = 0.01;
  const MAX_TOK_MIN = 1;

  const HELP = SAMPLING_HELP;

  function describedBy(node: HTMLElement, id: string) {
    let descriptionId = id;
    let input: HTMLInputElement | null = null;
    const connect = () => {
      input = node.querySelector("input");
      input?.setAttribute("aria-describedby", descriptionId);
    };
    queueMicrotask(connect);
    return {
      update(next: string) {
        if (input?.getAttribute("aria-describedby") === descriptionId) {
          input.removeAttribute("aria-describedby");
        }
        descriptionId = next;
        connect();
      },
      destroy() {
        if (input?.getAttribute("aria-describedby") === descriptionId) {
          input.removeAttribute("aria-describedby");
        }
      },
    };
  }

  // ------------------------------------------------------------------- ready

  /** True once session info has loaded — gates control enable state. */
  const ready = $derived(sessionState.info !== null && !settingsResetState.busy);

  /** True iff thinking is supported for this model.  ``supports_thinking``
   * comes off the session info and may flip once the model loads. */
  const thinkingSupported = $derived(
    sessionState.info?.supports_thinking ?? false,
  );

  /** True iff the user can actually turn thinking off (the chat template
   *  has an ``enable_thinking`` switch).  Forced-thinking models leave
   *  the toggle locked and read-only so the user knows clicking it
   *  is a no-op.  The field is required by the current session contract
   *  so we don't lock controls against backends that pre-date the
   *  field. */
  const thinkingOptional = $derived(
    sessionState.info?.thinking_is_optional === true,
  );

  // (The per-message role boxes moved to the composer's cast row —
  // Chat.svelte ``speaking as`` / ``reply as`` chips, same client state.)

  /** Tri-state for the title attribute and disabled gate. */
  const thinkingForced = $derived(
    thinkingSupported && !thinkingOptional,
  );

  // ------------------------------------------------------------------- views
  //
  // Each control's *display* value reads ``samplingState`` first (which the
  // store's bootstrap populates from session config) and falls back to a
  // placeholder when it's still null.

  const tempView = $derived(samplingState.temperature ?? PLACEHOLDER.temperature);
  const topPView = $derived(samplingState.top_p ?? PLACEHOLDER.top_p);
  const maxTokenLimit = $derived(
    sessionState.info ? outputTokenLimitForSignals(
      getRuntimeCapabilities()?.signals,
      getHostedController()?.snapshot.contextTokens ?? undefined,
    ) : MAX_TOK_MIN,
  );
  const maxView = $derived(
    Math.min(samplingState.max_tokens || PLACEHOLDER.max_tokens, maxTokenLimit),
  );
  const thinkingView = $derived(samplingState.thinking ?? false);

  // ------------------------------------------------------------------- writes

  /** PATCH the server with a single field.  Errors surface as a
   * console.warn — the strip itself stays usable (local state already
   * updated; user can retry). */
  async function persistDefault(
    body: Partial<{
      temperature: number;
      top_p: number;
      top_k: number | null;
      max_tokens: number;
      thinking: boolean;
    }>,
  ): Promise<void> {
    try {
      await patchSessionDefaults(body);
    } catch (e) {
      console.warn("[sampling] patch failed", e);
    }
  }

  function onTemp(v: number): void {
    setSampling("temperature", v);
    void persistDefault({ temperature: v });
  }

  function onTopP(v: number): void {
    setSampling("top_p", v);
    void persistDefault({ top_p: v });
  }

  function onMax(raw: number | null): void {
    if (raw === null) return;
    const v = Math.max(MAX_TOK_MIN, Math.min(maxTokenLimit, Math.floor(raw)));
    setSampling("max_tokens", v);
    void persistDefault({ max_tokens: v });
  }

  function onThinking(v: boolean): void {
    setSampling("thinking", v);
    void persistDefault({ thinking: v });
  }

  function openSystemPrompt(): void {
    openDrawer("system_prompt");
  }

  function openAdvanced(): void {
    openDrawer("advanced_sampling");
  }
</script>

<section class="sampling-strip" aria-label="sampling controls">
  <!-- Row 1: temperature + top-p sliders -->
  <div class="row sliders">
    <div class="control" use:describedBy={"sampling-temperature-help"}>
      <span class="control-label">
        <span class="label">Temperature</span>
        <InfoTip text={HELP.temperature} label="About Temperature" />
      </span>
      <span class="slider-cell">
        <Slider
          value={tempView}
          min={TEMP_MIN}
          max={Math.max(TEMP_MAX, tempView)}
          step={TEMP_STEP}
          disabled={!ready}
          oninput={onTemp}
          ariaLabel="Temperature"
        />
      </span>
      <span class="value"><RollingNumber value={tempView} digits={2} /></span>
      <span id="sampling-temperature-help" class="sr-only">{HELP.temperature}</span>
    </div>

    <div class="control" use:describedBy={"sampling-top-p-help"}>
      <span class="control-label">
        <span class="label">Top P</span>
        <InfoTip text={HELP.topP} label="About Top P" />
      </span>
      <span class="slider-cell">
        <Slider
          value={topPView}
          min={TOP_P_MIN}
          max={TOP_P_MAX}
          step={TOP_P_STEP}
          disabled={!ready}
          oninput={onTopP}
          ariaLabel="Top P"
        />
      </span>
      <span class="value"><RollingNumber value={topPView} digits={2} /></span>
      <span id="sampling-top-p-help" class="sr-only">{HELP.topP}</span>
    </div>
  </div>
  <div class="row details">
    <div class="control" use:describedBy={"sampling-max-tokens-help"}>
      <span class="control-label">
        <span class="label">Max tokens</span>
        <InfoTip text={HELP.maxTokens} label="About Max tokens" />
      </span>
      <span class="num-cell">
        <NumberInput
          value={maxView}
          min={MAX_TOK_MIN}
          max={maxTokenLimit}
          step={1}
          disabled={!ready}
          onchange={onMax}
          ariaLabel="Max tokens"
        />
      </span>
      <span id="sampling-max-tokens-help" class="sr-only">{HELP.maxTokens}</span>
    </div>
    {#if thinkingSupported}
    <div
      class="control toggle"
      class:forced={thinkingForced}
      use:describedBy={"sampling-thinking-help"}
    >
      <span class="control-label">
        <span class="label think-label">Thinking{thinkingForced ? " (always on)" : ""}</span>
        <InfoTip
          text={!thinkingOptional
            ? "This model always uses its reasoning phase, so this setting cannot be changed."
            : HELP.thinking}
          label="About Thinking"
        />
      </span>
      <Checkbox
        checked={thinkingForced ? true : thinkingView}
        disabled={!ready || thinkingForced}
        onchange={onThinking}
        ariaLabel="Thinking"
      />
      <span id="sampling-thinking-help" class="sr-only">
        {!thinkingOptional
          ? "This model always uses its reasoning phase, so this setting cannot be changed."
          : HELP.thinking}
      </span>
    </div>
    {/if}
  </div>

  <div class="row actions">
    <button
      type="button"
      class="sys-btn"
      disabled={!ready}
      onclick={openAdvanced}
    >Sampling settings</button>

    {#if !sessionState.info?.is_base_model}<button
      type="button"
      class="sys-btn"
      disabled={!ready}
      onclick={openSystemPrompt}
    >System prompt</button>{/if}
  </div>
  <ResetSettings />
</section>

<style>
  .sampling-strip {
    container-type: inline-size;
    display: flex;
    flex-direction: column;
    gap: var(--space-6);
    padding: 0 var(--panel-padding) var(--panel-padding);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    color: var(--fg-strong);
  }

  /* Each row uses equal tracks so labels and controls stay aligned. */
  .row {
    display: grid;
    grid-auto-flow: row;
    grid-template-columns: minmax(0, 1fr);
    align-items: center;
    gap: var(--space-6);
    width: 100%;
  }
  .row > .control {
    min-width: 0;
  }
  /* Keep the two secondary actions balanced across the panel. */
  .row.actions {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-2);
    padding-top: var(--space-3);
  }
  .row.actions > * {
    min-width: 0;
  }
  .row.actions > :only-child { grid-column: 1 / -1; }
  .control {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-height: 2.6rem;
    padding: 0;
    background: transparent;
    white-space: nowrap;
    min-width: 0;
  }

  /* Boxed inputs fill their (grown) control after the fixed-width label.
   * The inner <input> is width:100%, so the cell owns the sizing. */
  .slider-cell,
  .num-cell {
    display: inline-flex;
    flex: 1 1 0;
    min-width: 0;
  }

  .control-label {
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    gap: var(--space-1);
    min-width: 10.25rem;
  }

  .label {
    flex: 1 1 auto;
    min-width: 0;
    color: var(--fg-strong);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    text-align: start;
  }

  .value {
    flex: 0 0 auto;
    color: var(--fg-strong);
    font-family: var(--font-data);
    font-variant-numeric: tabular-nums;
    min-width: 2.5em;
    text-align: end;
  }

  /* Forced-thinking toggle: locked-on visual.  The checkbox is disabled
     so the browser already dims it; we keep the label dim too so the
     "(forced)" suffix reads as informational rather than interactive. */
  .control.toggle.forced .label {
    color: var(--fg-dim);
    font-style: italic;
  }
  /* The think label hugs its checkbox rather than reserving the numeric
   * gutter — it sits in the action row, not a field column. */
  .think-label {
    min-width: 0;
    text-align: start;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .sys-btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-1) var(--space-4);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    line-height: 1.3;
    min-height: var(--control-field);
  }
  .sys-btn:hover:not(:disabled) {
    background: var(--glass-strong);
    color: var(--accent);
  }
  .sys-btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }

  @container (max-width: 26rem) {
    .control {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--space-2) var(--space-3);
      white-space: normal;
    }

    .control-label {
      grid-column: 1 / -1;
      min-width: 0;
      width: 100%;
    }

    .slider-cell,
    .num-cell { grid-column: 1; }

    .num-cell { grid-column: 1 / -1; }
    .value { grid-column: 2; }

  }
  /* (Role-input styles moved to Chat.svelte's cast row.) */
</style>
