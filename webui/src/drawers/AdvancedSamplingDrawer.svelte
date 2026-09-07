<script lang="ts">
  import { slidingSelection } from "../lib/slidingSelection";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import {
    closeDrawer,
    samplingState,
    setSampling,
    patchSessionDefaults,
    genUiMode,
    setGenUiMode,
    sessionState,
  } from "../lib/stores.svelte";
  import NumberInput from "../lib/NumberInput.svelte";
  import InfoTip from "../lib/ui/InfoTip.svelte";
  import Disclosure from "../lib/Disclosure.svelte";
  import { getRuntimeClient } from "../lib/runtime/registry";
  import {
    clampTokenAlternativeCount,
    tokenAlternativeLimit,
  } from "../lib/runtime/samplingCapabilities";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  // Render mode — two-state: chat renders bubbles + roles, raw renders a
  // single flat completion buffer.  The mode is seeded from the model's
  // ``is_base_model`` flag the first time the model is seen, then it's a
  // plain user toggle.  Toggling never mutates the loom tree.
  const RENDER_MODES: { value: "chat" | "raw"; label: string }[] = [
    { value: "chat", label: "chat" },
    { value: "raw", label: "raw" },
  ];
  const isBaseModel = $derived(sessionState.info?.is_base_model === true);
  const runtimeMode = getRuntimeClient().mode;
  const alternativesMax = tokenAlternativeLimit(runtimeMode);
  const alternativesAvailable = alternativesMax > 0;
  const TOP_K_MIN = 1;
  const TOP_K_MAX = 4096;
  const PENALTY_MIN = -2;
  const PENALTY_MAX = 2;
  let technicalOpen = $state(false);

  const HELP = {
    topK: "Top K limits sampling to the K most likely next tokens. Leave it blank to use the model's default.",
    frequencyPenalty: "Frequency penalty reduces the probability of tokens in proportion to how often they have already appeared.",
    presencePenalty: "Presence penalty reduces the probability of any token that has already appeared, regardless of frequency.",
    returnTopK: runtimeMode === "browser"
      ? "Return top K retains up to five alternative tokens so you can inspect or branch from them later."
      : "Return top K retains alternative tokens so you can inspect or branch from them later.",
    seed: "Seed fixes the random-number sequence so repeated runs are easier to compare.",
  } as const;

  function onTopK(raw: number | null): void {
    if (raw === null) {
      setSampling("top_k", null);
      void patchSessionDefaults({ top_k: null });
      return;
    }
    const value = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, Math.floor(raw)));
    setSampling("top_k", value);
    void patchSessionDefaults({ top_k: value });
  }

  function onPenalty(
    key: "presence_penalty" | "frequency_penalty",
    raw: number | null,
  ): void {
    if (raw === null) return;
    setSampling(key, Math.max(PENALTY_MIN, Math.min(PENALTY_MAX, raw)));
  }

  function onAlternatives(raw: number | null): void {
    if (raw === null) return;
    setSampling("return_top_k", clampTokenAlternativeCount(raw, runtimeMode));
  }

  function onSeed(raw: number | null): void {
    setSampling("seed", raw === null ? null : Math.floor(raw));
  }

  const logitBiasValid = $derived.by(() => {
    const raw = samplingState.logit_bias_text.trim();
    if (!raw) return true;
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed);
    } catch {
      return raw
        .split(/\r?\n/)
        .filter(Boolean)
        .every((line) =>
          /^\s*-?\d+\s*[:=,\s]\s*-?\d+(?:\.\d+)?\s*$/.test(line),
        );
    }
  });
</script>

<section class="drawer-shell" aria-label="Sampling settings drawer">
  <header class="header">
    <div>
      <h2 class="title">Sampling settings</h2>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="body">
    <section class="panel">
      <div class="panel-heading">
        <div>
          <h3>Sampling filters and penalties</h3>
        </div>
      </div>
      <div class="setting-grid">
        <div class="setting">
          <span class="setting-label">Top K <InfoTip text={HELP.topK} label="About Top K" /></span>
          <NumberInput
            value={samplingState.top_k}
            min={TOP_K_MIN}
            max={TOP_K_MAX}
            step={1}
            placeholder="Model default"
            allowEmpty
            onchange={onTopK}
            ariaLabel="Top K"
          />
        </div>
        <div class="setting">
          <span class="setting-label">Frequency penalty <InfoTip text={HELP.frequencyPenalty} label="About Frequency penalty" /></span>
          <NumberInput
            value={samplingState.frequency_penalty}
            min={PENALTY_MIN}
            max={PENALTY_MAX}
            step={0.05}
            onchange={(value) => onPenalty("frequency_penalty", value)}
            ariaLabel="Frequency penalty"
          />
        </div>
        <div class="setting">
          <span class="setting-label">Presence penalty <InfoTip text={HELP.presencePenalty} label="About Presence penalty" /></span>
          <NumberInput
            value={samplingState.presence_penalty}
            min={PENALTY_MIN}
            max={PENALTY_MAX}
            step={0.05}
            onchange={(value) => onPenalty("presence_penalty", value)}
            ariaLabel="Presence penalty"
          />
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <h3>Captured output and reproducibility</h3>
        </div>
      </div>
      <div class="setting-grid two">
        <div class="setting">
          <span class="setting-label">Return top K <InfoTip text={HELP.returnTopK} label="About Return top K" /></span>
          <NumberInput
            value={samplingState.return_top_k}
            min={0}
            max={alternativesMax}
            step={1}
            disabled={!alternativesAvailable}
            onchange={onAlternatives}
            ariaLabel="Return top K"
          />
        </div>
        <div class="setting">
          <span class="setting-label">Seed <InfoTip text={HELP.seed} label="About Seed" /></span>
          <NumberInput
            value={samplingState.seed}
            min={0}
            step={1}
            placeholder="Not fixed"
            allowEmpty
            onchange={onSeed}
            ariaLabel="Seed"
          />
        </div>
      </div>
    </section>

    <Disclosure bind:expanded={technicalOpen} summary="Additional parameters">
      <div class="technical-stack">
        <section class="technical-section">
          <h3>Conversation format</h3>
          <div class="mode-row" role="group" aria-label="Conversation format" use:slidingSelection>
            {#each RENDER_MODES as m (m.label)}
              <button
                type="button"
                class="mode-opt"
                class:active={genUiMode.mode === m.value}
                aria-pressed={genUiMode.mode === m.value}
                disabled={isBaseModel && m.value === "chat"}
                onclick={() => setGenUiMode(m.value)}
              >{m.value === "chat" ? "Chat template" : "Raw completion"}</button>
            {/each}
          </div>
          <p class="hint">{isBaseModel ? "Base models continue the exact text in the buffer, without chat roles or a system prompt." : "This model normally opens in chat template mode."}</p>
        </section>

        <label class="technical-section">
          <span class="setting-label">Stop sequences</span>
          <textarea
            rows="4"
            value={samplingState.stop_sequences}
            oninput={(ev) => setSampling("stop_sequences", (ev.currentTarget as HTMLTextAreaElement).value)}
            placeholder={"One sequence per line\n###\n<|eot_id|>"}
            aria-label="Stop sequences"
          ></textarea>
        </label>

        <label class="technical-section">
          <span class="setting-label">Logit bias</span>
          <textarea
            rows="5"
            class:invalid={!logitBiasValid}
            value={samplingState.logit_bias_text}
            oninput={(ev) => setSampling("logit_bias_text", (ev.currentTarget as HTMLTextAreaElement).value)}
            placeholder={'{"198": -4, "220": 1.5}'}
            aria-label="Logit bias"
            aria-invalid={!logitBiasValid}
            aria-describedby="logit-bias-help"
          ></textarea>
          <p id="logit-bias-help" class:error={!logitBiasValid} class="hint">
            {logitBiasValid
              ? "Use a JSON object or one token ID and value per line."
              : "Use JSON or write one token ID and value per line."}
          </p>
        </label>
      </div>
    </Disclosure>
  </div>
</section>

<style>
  .drawer-shell {
    container: sampling-settings / inline-size;
    min-height: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
    background: transparent;
  }
  .header {
    display: flex;
    flex: 0 0 auto;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-6);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    background: transparent;
  }
  .title {
    color: var(--accent);
    font-family: var(--font-structure);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-structure);
  }
  p {
    margin: var(--space-2) 0 0;
    color: var(--fg-muted);
    line-height: 1.5;
    text-wrap: pretty;
  }
  .body {
    display: grid;
    grid-auto-rows: max-content;
    align-content: start;
    gap: var(--drawer-section-gap);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    min-height: 0;
    min-width: 0;
    overflow: auto;
    overscroll-behavior: contain;
  }
  .panel {
    min-width: 0;
  }
  h3 {
    margin: 0;
    color: var(--fg);
    font-size: var(--text-sm);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
    letter-spacing: 0;
  }
  .panel-heading {
    margin-bottom: var(--space-5);
  }
  .setting-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-6);
  }
  .setting-grid.two {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .setting,
  .technical-section {
    display: grid;
    gap: var(--space-2);
    min-width: 0;
  }
  .setting-label {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-strong);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
  }
  .technical-stack {
    display: grid;
    gap: var(--space-6);
  }
  .technical-section + .technical-section {
    padding-top: var(--space-5);
    box-shadow: inset 0 1px var(--glass-line);
  }
  textarea {
    width: 100%;
    border: 1px solid transparent;
    border-radius: var(--radius);
    background: var(--input-well);
    color: var(--fg);
    padding: var(--space-4);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    letter-spacing: 0;
    resize: vertical;
    line-height: 1.45;
  }
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .invalid {
    border-color: var(--accent-red);
  }
  .hint {
    font-size: var(--text-xs);
    line-height: 1.35;
    color: var(--fg-muted);
  }
  .error {
    color: var(--accent-red);
  }
  .mode-row {
    display: flex;
    gap: var(--space-1);
    border-radius: var(--radius-group);
    background: var(--surface-sheen), var(--glass);
    padding: var(--space-1);
    margin-top: var(--space-3);
  }
  .mode-opt {
    min-height: var(--control-target);
    flex: 1 1 0;
    background: transparent;
    color: var(--fg-muted);
    border: 0;
    border-radius: var(--radius-inset);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .mode-opt:hover {
    color: var(--fg);
  }
  .mode-opt.active {
    background: var(--accent-subtle);
    color: var(--accent);
  }

  @container sampling-settings (max-width: 40rem) {
    .setting-grid,
    .setting-grid.two { grid-template-columns: minmax(0, 1fr); }
  }
</style>
