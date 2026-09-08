<script lang="ts">
  import TemplatePreview from "../lib/ui/TemplatePreview.svelte";
  import MorphText from "../lib/ui/MorphText.svelte";
  import FluentIcon from "../lib/ui/FluentIcon.svelte";
  import Select from "../lib/Select.svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // TemplateLabDrawer — the standalone templated-completion artifact.
  //
  // Two tabs:
  //   * score — pick a template, optionally steer, run; render the per-context
  //     restricted-choice value distribution (the logit read). With a steering
  //     expression this is the distributional before/after.
  //   * build — author a template: a slot token, candidate values, and one or
  //     more multi-turn contexts (history turns + the slotted final assistant
  //     turn). The slot lives only in the assistant turn.
  //
  // Reached from the command palette, or deep-linked to the build tab
  // (``params: { tab: "build" }``) from the manifold builder's template
  // path — this is the ONE template editor, and the builder derives a
  // manifold from what it authors (`drowse manifold from-template`).

  import { onMount, tick, untrack } from "svelte";
  import { isFittingCancellation } from "../lib/runtime/fittingCancellation";
  import { getHostedController } from "../lib/runtime/registry";
  import { apiTemplates, describeError } from "../lib/runtime/services";
  import { validateTemplateDraft } from "../lib/templates";
  import { closeDrawer } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import Button from "../lib/ui/Button.svelte";
  import type {
    ChoiceScores,
    TemplateContextSpec,
    TemplateSummary,
    TemplateDetail,
    TemplateTurn,
  } from "../lib/types";

  let { params }: { params?: unknown } = $props();

  type Tab = "score" | "build";
  // Opening tab: ``score`` unless a caller deep-linked the editor.  Read
  // once at mount — the drawer host remounts this component on every
  // open, so a later params change isn't a case that exists.
  let tab: Tab = $state(
    untrack(
      () =>
        (params as { tab?: unknown } | null | undefined)?.tab === "build"
          ? "build"
          : "score",
    ),
  );

  const TAB_ITEMS: Array<{ value: Tab; label: string; title: string }> = [
    { value: "score", label: "score", title: "restricted-choice scores" },
    { value: "build", label: "build", title: "new template" },
  ];

  // ----- shared: template catalog --------------------------------------
  let templates: TemplateSummary[] = $state([]);
  let loading = $state(false);
  let deleteCandidate = $state<string | null>(null);
  let deleting = $state(false);

  async function loadTemplates(): Promise<void> {
    loading = true;
    try {
      templates = (await apiTemplates.list()).templates;
    } catch (e) {
      pushToast(`couldn't load templates: ${describeError(e)}`, { kind: "error" });
    } finally {
      loading = false;
    }
  }
  onMount(loadTemplates);

  // ----- score tab -----------------------------------------------------
  let selectedKey = $state("");
  let steerExpr = $state("");
  let scoring = $state(false);
  let scoreBy: "sum" | "mean" = $state("sum");
  let baseline: ChoiceScores[] | null = $state(null);
  let steered: ChoiceScores[] | null = $state(null);
  let scoredKey = $state("");
  let cancellingScore = $state(false);
  const hostedController = getHostedController();

  const selectedTemplate = $derived(
    templates.find((t) => `${t.namespace}/${t.name}` === selectedKey) ?? null,
  );

  let previewTemplate = $state<TemplateDetail | null>(null);
  let previewError = $state("");
  $effect(() => {
    const template = selectedTemplate;
    previewTemplate = null;
    previewError = "";
    if (!template) return;
    let current = true;
    void apiTemplates.get(template.namespace, template.name).then(detail => {
      if (current) previewTemplate = detail;
    }).catch(() => { if (current) previewError = "Preview unavailable. Scoring is still available."; });
    return () => { current = false; };
  });

  function probOf(c: { prob_sum: number; prob_mean: number }): number {
    return scoreBy === "sum" ? c.prob_sum : c.prob_mean;
  }

  async function runScore(): Promise<void> {
    if (!selectedTemplate || scoring) return;
    scoring = true;
    cancellingScore = false;
    baseline = null;
    steered = null;
    const { namespace, name } = selectedTemplate;
    try {
      baseline = (await apiTemplates.score(namespace, name, null)).contexts;
      const expr = steerExpr.trim();
      if (expr) {
        steered = (await apiTemplates.score(namespace, name, expr)).contexts;
      }
      scoredKey = selectedKey;
    } catch (e) {
      baseline = null;
      steered = null;
      scoredKey = "";
      if (isFittingCancellation(e)) {
        pushToast("scoring cancelled", { kind: "info" });
      } else {
        pushToast(`scoring failed: ${describeError(e)}`, { kind: "error" });
      }
    } finally {
      scoring = false;
      cancellingScore = false;
    }
  }

  async function cancelScore(): Promise<void> {
    if (!hostedController || !scoring || cancellingScore) return;
    cancellingScore = true;
    try {
      await hostedController.cancelFitting();
    } catch (e) {
      pushToast(`couldn't cancel scoring: ${describeError(e)}`, {
        kind: "error",
        ttlMs: null,
      });
    } finally {
      cancellingScore = false;
    }
  }

  /** Rows for one context: label + baseline prob + optional steered prob,
   *  sorted by the active baseline probability descending. */
  function rows(ctxIdx: number): {
    label: string;
    base: number;
    steer: number | null;
  }[] {
    const b = baseline?.[ctxIdx];
    if (!b) return [];
    const s = steered?.[ctxIdx] ?? null;
    const out = b.choices.map((c, i) => ({
      label: c.label,
      base: probOf(c),
      steer: s ? probOf(s.choices[i]) : null,
    }));
    out.sort((x, y) => Math.max(y.base, y.steer ?? -1) - Math.max(x.base, x.steer ?? -1));
    return out;
  }

  // ----- build tab -----------------------------------------------------
  let bName = $state("");
  let bSlot = $state("[DAY]");
  let bValuesText = $state("");
  let bContexts = $state<TemplateContextSpec[]>([
    { turns: [{ role: "user", content: "" }], assistant: "" },
  ]);
  let building = $state(false);
  let buildSubmitted = $state(false);
  let buildForm: HTMLFormElement | null = $state(null);

  const bValues = $derived(
    bValuesText
      .split(/[\n,]+/)
      .map((v) => v.trim())
      .filter(Boolean),
  );

  function addContext(): void {
    bContexts = [
      ...bContexts,
      { turns: [{ role: "user", content: "" }], assistant: "" },
    ];
  }
  function removeContext(i: number): void {
    bContexts = bContexts.filter((_, idx) => idx !== i);
  }
  function addTurn(ci: number): void {
    const ctx = bContexts[ci];
    const lastRole = ctx.turns[ctx.turns.length - 1]?.role;
    const role: TemplateTurn["role"] = lastRole === "user" ? "assistant" : "user";
    ctx.turns = [...ctx.turns, { role, content: "" }];
    bContexts = [...bContexts];
  }
  function removeTurn(ci: number, ti: number): void {
    bContexts[ci].turns = bContexts[ci].turns.filter((_, idx) => idx !== ti);
    bContexts = [...bContexts];
  }

  // Blank turn rows are the editor's "add a turn" affordance, not part of
  // the draft — they're dropped here exactly as ``submitBuild`` drops them.
  const buildDraft = $derived({
    slot: bSlot,
    values: bValues,
    contexts: bContexts.map((c) => ({
      turns: c.turns.filter((t) => t.content.trim()),
      assistant: c.assistant,
    })),
  });

  const buildValidation = $derived([
    ...(bName.trim() ? [] : ["name required"]),
    ...validateTemplateDraft(buildDraft),
  ]);
  const nameInvalid = $derived(!bName.trim());
  const slotInvalid = $derived(buildValidation.includes("slot required"));
  const valuesInvalid = $derived(
    buildValidation.some((error) => error === "≥ 2 values" || error.startsWith('value "')),
  );
  const contextInvalid = (index: number) =>
    buildValidation.some((error) => error.startsWith(`context ${index + 1}:`));
  const validationDescription = (invalid: boolean) =>
    buildSubmitted && invalid ? "template-build-errors" : undefined;

  async function submitBuild(ev: Event): Promise<void> {
    ev.preventDefault();
    if (buildValidation.length) {
      buildSubmitted = true;
      await tick();
      buildForm?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
      return;
    }
    building = true;
    try {
      await apiTemplates.create({
        namespace: "local",
        name: bName.trim(),
        slot: bSlot.trim(),
        values: bValues,
        contexts: buildDraft.contexts,
      });
      pushToast(`template ${bName.trim()} created`, { kind: "info" });
      await loadTemplates();
      selectedKey = `local/${bName.trim()}`;
      tab = "score";
    } catch (e) {
      pushToast(`create failed: ${describeError(e)}`, { kind: "error" });
    } finally {
      building = false;
    }
  }

  async function deleteTemplate(t: TemplateSummary): Promise<void> {
    if (deleting) return;
    deleting = true;
    try {
      await apiTemplates.delete(t.namespace, t.name);
      pushToast(`removed ${t.namespace}/${t.name}`, { kind: "info" });
      if (selectedKey === `${t.namespace}/${t.name}`) selectedKey = "";
      await loadTemplates();
      deleteCandidate = null;
    } catch (e) {
      pushToast(`delete failed: ${describeError(e)}`, { kind: "error" });
    } finally {
      deleting = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Template lab">
  <header class="header">
    <div class="title">
      <h2 class="eyebrow">Prompt template lab</h2>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="toolbar">
    <SegmentedTabs items={TAB_ITEMS} bind:value={tab} ariaLabel="Template lab view" />
  </div>

  <div class="body">
    {#if tab === "score"}
      <p class="hint">Compare the model's preference among a fixed set of answers.</p>

      {#if loading}
        <p class="muted loading-pulse loading-placeholder" role="status">Loading templates…</p>
      {:else if templates.length === 0}
        <p class="muted">no templates</p>
      {:else}
        <label class="field">
          <span class="label">template</span>
          <Select bind:value={selectedKey} disabled={scoring} ariaLabel="Template"
            options={[{ value: "", label: "Select…" }, ...templates.map(t => ({
              value: `${t.namespace}/${t.name}`, label: `${t.namespace}/${t.name} · ${t.n_values} values × ${t.n_contexts} ctx`,
            }))]} />
        </label>

        {#if previewTemplate}
          {#each previewTemplate.contexts as context, index (index)}
            <TemplatePreview sentence={context.assistant} slot={previewTemplate.slot} values={previewTemplate.values} />
          {/each}
        {:else if previewError}<p class="hint">{previewError}</p>{/if}
        <label class="field">
          <span class="label">steering</span>
          <input type="text" placeholder="0.5 patient.hurried" bind:value={steerExpr}
            disabled={scoring} autocomplete="off" spellcheck="false" />
        </label>

        <div class="controls">
          <label class="byrow">
            <span class="label">rank by</span>
            <Select bind:value={scoreBy} ariaLabel="Rank by" options={[{ value: "sum", label: "sum" }, { value: "mean", label: "mean" }]} />
          </label>
          {#if hostedController && scoring}
            <Button variant="ghost" disabled={cancellingScore} onclick={cancelScore}>
              <MorphText text={cancellingScore ? "cancelling…" : "cancel"} />
            </Button>
          {/if}
          <Button variant="solid" busy={scoring} disabled={!selectedTemplate || scoring} onclick={runScore}>
            <MorphText text={scoring ? "scoring…" : "score"} />
          </Button>
        </div>

        {#if baseline && scoredKey === selectedKey && !scoring}
          {#each baseline as _ctx, ci (ci)}
            <div class="ctx-card">
              <div class="ctx-head">context {ci + 1}{steered ? " · base → steered" : ""}</div>
              <div class="mass-row"><span>Baseline mass</span><div class="mass-bar" role="img" aria-label={rows(ci).map(r => `${r.label}: ${(r.base * 100).toFixed(1)}%`).join(", ")}>{#each rows(ci) as r, index (r.label)}<span style:width={`${r.base * 100}%`} style:opacity={.35 + (index % 4) * .2} {...{ "aria-description": (`${r.label}: ${(r.base * 100).toFixed(1)}%`) }}></span>{/each}</div></div>
              {#if steered}<div class="mass-row"><span>Steered mass</span><div class="mass-bar" role="img" aria-label={rows(ci).map(r => `${r.label}: ${((r.steer ?? 0) * 100).toFixed(1)}%`).join(", ")}>{#each rows(ci) as r, index (r.label)}<span style:width={`${(r.steer ?? 0) * 100}%`} style:opacity={.35 + (index % 4) * .2} {...{ "aria-description": (`${r.label}: ${((r.steer ?? 0) * 100).toFixed(1)}%`) }}></span>{/each}</div></div>{/if}
              {#each rows(ci) as r (r.label)}
                <div class="bar-row">
                  <span class="bar-label" {...{ "aria-description": (r.label) }}>{r.label}</span>
                  <div class="bars">
                    <div class="bar base" style={`width:${(r.base * 100).toFixed(1)}%`}></div>
                    {#if r.steer !== null}
                      <div class="bar steer" style={`width:${(r.steer * 100).toFixed(1)}%`}></div>
                    {/if}
                  </div>
                  <span class="bar-num">
                    <MorphText text={`${(r.base * 100).toFixed(0)}%`} />{#if r.steer !== null}<span class="arrow">→</span><MorphText text={`${(r.steer * 100).toFixed(0)}%`} />{/if}
                  </span>
                </div>
              {/each}
            </div>
          {/each}
        {/if}
      {/if}

    {:else}
      <form bind:this={buildForm} class="form" aria-busy={building} onsubmit={submitBuild}>
        <label class="field">
          <span class="label">name</span>
          <input type="text" placeholder="weekday" bind:value={bName} disabled={building}
            autocomplete="off" spellcheck="false"
            aria-invalid={buildSubmitted && nameInvalid}
            aria-describedby={validationDescription(nameInvalid)} />
        </label>
        <label class="field">
          <span class="label">slot token</span>
          <input type="text" placeholder="[DAY]" bind:value={bSlot} disabled={building}
            autocomplete="off" spellcheck="false"
            aria-invalid={buildSubmitted && slotInvalid}
            aria-describedby={validationDescription(slotInvalid)} />
        </label>
        <label class="field">
          <span class="label">values</span>
          <textarea rows="3" placeholder={"Monday\nTuesday\nWednesday"}
            bind:value={bValuesText} disabled={building}
            aria-invalid={buildSubmitted && valuesInvalid}
            aria-describedby={validationDescription(valuesInvalid)}></textarea>
        </label>

        <fieldset class="contexts">
          <legend>contexts <span class="optional">({bContexts.length})</span></legend>
          {#each bContexts as ctx, ci (ci)}
            <div class="ctx-build">
              <div class="ctx-build-head">
                <span>context {ci + 1}</span>
                {#if bContexts.length > 1}
                  <button
                    type="button"
                    class="mini"
                    aria-label={`Remove context ${ci + 1}`}
                    onclick={() => removeContext(ci)}
                  >remove</button>
                {/if}
              </div>
              {#each ctx.turns as turn, ti (ti)}
                <div class="turn-row">
                  <Select
                    bind:value={turn.role}
                    disabled={building}
                    ariaLabel={`Context ${ci + 1}, turn ${ti + 1} role`}
                    invalid={buildSubmitted && contextInvalid(ci)}
                    ariaDescribedby={validationDescription(contextInvalid(ci))}
                    options={[{ value: "user", label: "user" }, { value: "assistant", label: "assistant" }, { value: "system", label: "system" }]}
                  />
                  <input type="text" placeholder="turn content" bind:value={turn.content}
                    disabled={building} autocomplete="off"
                    aria-invalid={buildSubmitted && contextInvalid(ci)}
                    aria-describedby={validationDescription(contextInvalid(ci))}
                    aria-label={`Context ${ci + 1}, turn ${ti + 1} content`} />
                  {#if ctx.turns.length > 1}
                    <button
                      type="button"
                      class="mini"
                      aria-label={`Remove turn ${ti + 1} from context ${ci + 1}`}
                      onclick={() => removeTurn(ci, ti)}
                    ><FluentIcon name="dismiss" /></button>
                  {/if}
                </div>
              {/each}
              <button
                type="button"
                class="mini add"
                aria-label={`Add turn to context ${ci + 1}`}
                onclick={() => addTurn(ci)}
              >+ turn</button>
              <label class="field assistant-field">
                <span class="label">assistant · slot</span>
                <input type="text" placeholder={`today is ${bSlot}`} bind:value={ctx.assistant}
                  disabled={building} autocomplete="off"
                  aria-invalid={buildSubmitted && contextInvalid(ci)}
                  aria-describedby={validationDescription(contextInvalid(ci))} />
              </label>
              <TemplatePreview sentence={ctx.assistant} slot={bSlot} values={bValuesText.split("\n").map(value => value.trim()).filter(Boolean)} />
            </div>
          {/each}
          <button type="button" class="mini add" onclick={addContext}>+ context</button>
        </fieldset>

        {#if buildSubmitted && buildValidation.length}
          <ul id="template-build-errors" class="errs" role="status" aria-live="polite">
            {#each buildValidation as e (e)}<li>{e}</li>{/each}
          </ul>
        {/if}

        <footer class="foot">
          <Button variant="ghost" onclick={closeDrawer}>cancel</Button>
          <Button
            type="submit"
            variant="solid"
            busy={building}
            disabled={building}
          >
            <MorphText text={building ? "creating…" : "create template"} />
          </Button>
        </footer>
      </form>
    {/if}

    {#if templates.length > 0}
      <div class="catalog">
        <div class="catalog-head">installed</div>
        {#each templates as t (`${t.namespace}/${t.name}`)}
          <div class="cat-row">
            <span class="cat-name">{t.namespace}/{t.name}</span>
            <span class="cat-sub">{t.slot} · {t.n_values}×{t.n_contexts}</span>
            {#if deleteCandidate === `${t.namespace}/${t.name}`}
              <div class="delete-confirmation">
                <p>Delete {t.namespace}/{t.name}? This removes the saved template and cannot be undone.</p>
                <Button variant="ghost" size="sm" disabled={deleting} onclick={() => (deleteCandidate = null)}>Cancel</Button>
                <Button variant="danger" size="sm" disabled={deleting} onclick={() => deleteTemplate(t)}><MorphText text={deleting ? "Deleting…" : "Delete template"} /></Button>
              </div>
            {:else}
              <Button
                variant="danger"
                size="sm"
                disabled={deleting}
                ariaLabel={`Delete template ${t.namespace}/${t.name}`}
                onclick={() => (deleteCandidate = `${t.namespace}/${t.name}`)}
              >Delete…</Button>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  .mass-row { display: grid; grid-template-columns: 8em 1fr; align-items: center; gap: var(--space-2); font-size: var(--text-xs); margin-block: var(--space-2); }
  .mass-bar { display: flex; height: 12px; overflow: hidden; border-radius: var(--radius-pill); }
  .mass-bar span { background: var(--accent); }
  .delete-confirmation { grid-column: 1 / -1; }
  .delete-confirmation p { margin: 0 0 var(--space-3); line-height: 1.5; overflow-wrap: anywhere; }
  /* v2 sheet interior — the host paints the sheet surface, so the root
   * stays transparent and chrome speaks sans (values/identifiers stay
   * mono). Templates carry no pillar hue — chrome stays achromatic. */
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .toolbar {
    padding: var(--space-5) var(--drawer-gutter-inline);
  }

  .body {
    flex: 1;
    overflow-y: auto;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: column;
    gap: var(--drawer-section-gap);
  }
  .hint {
    font-size: var(--text-sm);
    color: var(--fg-muted);
    margin: 0;
    line-height: 1.5;
    max-width: 62ch;
  }
  .muted { color: var(--fg-muted); font-size: var(--text-sm); }
  .field { display: flex; flex-direction: column; gap: var(--space-2); }
  .label { font-size: var(--text-sm); color: var(--fg-muted); }
  .optional { color: var(--fg-subtle); }
  input,
  textarea {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  input:focus,
  textarea:focus {
    outline: none;
    border-color: var(--fg-muted);
  }
  textarea { resize: vertical; }
  .controls { display: flex; align-items: flex-end; gap: var(--space-5); }
  .byrow { display: flex; flex-direction: column; gap: var(--space-2); flex: 1; }

  /* Data well — the per-context restricted-choice distribution. */
  .ctx-card {
    border-radius: var(--radius);
    background: var(--bg);
    padding: var(--surface-padding);
  }
  .ctx-head {
    font-size: var(--text-xs);
    color: var(--fg-muted);
    margin-bottom: var(--space-3);
  }
  .bar-row {
    display: grid;
    grid-template-columns: 5.5em 1fr 4.5em;
    align-items: center;
    gap: var(--space-3);
    margin: var(--space-1) 0;
  }
  .bar-label {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bars { display: flex; flex-direction: column; gap: var(--data-mark-gap); }
  .bar { height: 7px; border-radius: var(--radius-sm); min-width: 1px; }
  /* Achromatic before/after: the steered bar reads brighter above the
   * muted baseline — no pillar hue borrowed for a non-pillar surface. */
  .bar.base { background: var(--fg-muted); }
  .bar.steer { background: var(--accent); }
  .bar-num {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--fg-dim);
    text-align: end;
    font-variant-numeric: tabular-nums;
  }
  .arrow { color: var(--fg-subtle); margin: 0 var(--space-1); }

  .form { display: flex; flex-direction: column; gap: var(--space-4); }
  .contexts {
    border-radius: var(--radius);
    background: var(--glass);
    box-shadow: var(--shadow-well);
    padding: var(--surface-padding);
    margin: 0;
  }
  legend {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0 var(--space-2);
  }
  .ctx-build {
    padding: var(--space-4) 0;
  }
  .ctx-build-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: var(--text-xs);
    color: var(--fg-muted);
    margin-bottom: var(--space-3);
  }
  .turn-row {
    display: grid;
    grid-template-columns: 6em 1fr auto;
    gap: var(--space-2);
    margin: var(--space-2) 0;
  }
  .assistant-field { margin-top: var(--space-3); }

  /* Compact row actions use the shared small-control shape. */
  .mini {
    background: var(--glass);
    border: 1px solid transparent;
    color: var(--fg-muted);
    border-radius: var(--radius-sm);
    padding: var(--space-2) var(--space-4);
    cursor: pointer;
    font: inherit;
    font-size: var(--text-2xs);
    line-height: 1;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .mini:hover {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .mini.add { margin-top: var(--space-3); }
  .errs {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: 0;
    padding-inline-start: var(--space-sm);
  }
  .foot { display: flex; justify-content: flex-end; gap: var(--space-3); }

  .catalog {
    padding-top: var(--space-4);
  }
  .catalog-head {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-3);
  }
  .cat-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) 0;
    font-size: var(--text-sm);
  }
  .cat-name { font-family: var(--font-mono); color: var(--fg); overflow-wrap: anywhere; }
  .cat-sub {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  @media (max-width: 600px) {
    .cat-row { grid-template-columns: minmax(0, 1fr) auto; }
    .cat-name { grid-column: 1 / -1; }
  }
</style>
