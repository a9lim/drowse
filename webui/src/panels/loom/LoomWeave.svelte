<script lang="ts">
  import { animatedDetails } from "../../lib/animatedDetails";
  import Disclosure from "../../lib/Disclosure.svelte";
  import Select from "../../lib/Select.svelte";
  import { tick, untrack } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import {
    effectiveRawMode, genStatus, loomNavigate, loomStar, loomTree,
    sendFork, sendGenerate, sendStop, sendSubmit,
  } from "../../lib/stores.svelte";
  import type { LoomNodeJSON } from "../../lib/types";
  import { userFacingError } from "../../lib/runtime/userFacingError";

  let { active, oninspect, onwrite }: {
    active: boolean;
    oninspect: (node: LoomNodeJSON, index: number) => void;
    onwrite: (node: LoomNodeJSON, trigger: HTMLElement) => void;
  } = $props();

  let anchorId = $state<string | null>(null);
  let count = $state(3);
  let prompt = $state("");
  let startingPrompt = $state<string | null>(null);
  const expandedTokens = new SvelteSet<string>();
  let dispatching = $state(false);
  let error = $state("");
  let choicesHeading: HTMLHeadingElement;
  const busy = $derived(dispatching || genStatus.active);
  const anchor = $derived(anchorId ? loomTree.nodes.get(anchorId) : undefined);
  const choices = $derived((anchorId ? loomTree.children_of.get(anchorId) ?? [] : [])
    .map(id => loomTree.nodes.get(id)).filter((node): node is LoomNodeJSON => !!node));
  const path = $derived(loomTree.activePath.map(id => loomTree.nodes.get(id))
    .filter((node): node is LoomNodeJSON => !!node && !(node.parent_id === null && !node.text)));
  const empty = $derived(path.length === 0);
  const chosen = $derived(choices.find(node => loomTree.activePath.includes(node.id)));
  const parent = $derived(anchor?.parent_id ? loomTree.nodes.get(anchor.parent_id) : undefined);

  $effect(() => {
    if (!active) return;
    if (anchorId && loomTree.nodes.has(anchorId)) return;
    const current = untrack(() => loomTree.nodes.get(loomTree.active_node_id ?? ""));
    anchorId = current?.recipe && current.parent_id ? current.parent_id
      : current?.id ?? loomTree.root_id;
  });

  function role(node: LoomNodeJSON): string {
    return node.role_label || node.role;
  }

  async function explore(node: LoomNodeJSON): Promise<void> {
    anchorId = node.id;
    error = "";
    await tick();
    choicesHeading?.focus({ preventScroll: true });
    choicesHeading?.scrollIntoView({ block: "nearest" });
  }

  async function choose(node: LoomNodeJSON): Promise<void> {
    if (busy) return;
    dispatching = true;
    try { await loomNavigate(node.id); }
    finally { dispatching = false; }
  }

  async function generate(event?: SubmitEvent): Promise<void> {
    event?.preventDefault();
    if (busy || !anchorId) return;
    dispatching = true;
    error = "";
    try {
      if (empty) {
        if (!prompt.trim()) {
          error = "Enter some starting text before generating alternatives.";
          return;
        }
        startingPrompt = prompt;
        await sendSubmit(prompt, "user", "assistant", {
          parent_node_id: anchorId, n: count, raw: effectiveRawMode(),
        });
      } else {
        const generatedRole = choices.find(node => node.recipe !== null)?.role;
        await sendGenerate({
          parent_node_id: anchorId, n: count, raw: effectiveRawMode(),
          append_same_role: false,
          generate_seat: generatedRole === "user" ? "user" : "assistant",
        });
      }
    } catch (cause) {
      startingPrompt = null;
      error = userFacingError(cause, "Could not generate alternatives. Please try again.");
    } finally { dispatching = false; }
  }

  function lastBoundary(node: LoomNodeJSON): number {
    const index = (node.tokens?.length ?? 0) - 1;
    const token = node.tokens?.[index];
    return node.recipe && token?.raw_index != null && token.token_id != null ? index : -1;
  }

  async function extend(node: LoomNodeJSON): Promise<void> {
    const token = node.tokens?.[lastBoundary(node)];
    if (busy || token?.raw_index == null || token.token_id == null) return;
    dispatching = true;
    error = "";
    try {
      await sendFork(node.id, token.raw_index, token.token_id, true);
    } catch (cause) {
      error = userFacingError(cause, "Could not continue this text. Please try again.");
    } finally { dispatching = false; }
  }

  // A first prompt creates its own branch point; subsequent generations keep
  // that point fixed while each new sibling arrives.
  $effect(() => {
    if (startingPrompt === null) return;
    const first = choices.find(node => node.role === "user" && node.text === startingPrompt);
    if (first) { anchorId = first.id; prompt = ""; startingPrompt = null; }
  });
</script>

<div class="weave" class:empty>
  <section class="reading" aria-labelledby="weave-reading-title">
    <header class="section-heading">
      <h2 id="weave-reading-title">Current text</h2>
      <span>{path.length} {path.length === 1 ? "turn" : "turns"}</span>
    </header>
    {#if empty}
      <p class="hint">Type a prompt to get started.</p>
    {:else}
      <div class="reading-path" role="list" aria-label="Weave current path">
        {#each path as node, index (node.id)}
          <article role="listitem" class="passage" class:branch-point={anchorId === node.id} data-weave-passage={node.id}>
            <header>
              <span class="role">{role(node)}</span>
              {#if anchorId === node.id}<span class="marker">branch point</span>{/if}
              <button type="button" class="quiet" onclick={() => void explore(node)}
                aria-label={`Explore here after turn ${index + 1}`} aria-pressed={anchorId === node.id}>Explore here</button>
            </header>
            <div class="passage-text" dir="auto">{node.text}</div>
            {#if node.tokens?.length}
              <Disclosure summary="Token branch points" flush bind:expanded={() => expandedTokens.has(node.id), (open) => { if (open) expandedTokens.add(node.id); else expandedTokens.delete(node.id); }}>
                <div class="tokens">
                  {#each node.tokens as token, tokenIndex}
                    <button type="button" data-cursor="inspect" disabled={busy} {...{ "aria-description": "Inspect or branch from this token" }}
                      aria-label={`Inspect token ${tokenIndex + 1}: ${token.text.trim() || "whitespace"}`}
                      onclick={() => oninspect(node, tokenIndex)}>{token.text || "∅"}</button>
                  {/each}
                </div>
              </Disclosure>
            {/if}
          </article>
        {/each}
      </div>
    {/if}
  </section>

  <section class="possibilities" aria-labelledby="weave-choices-title">
    <header class="section-heading">
      <h2 id="weave-choices-title" tabindex="-1" bind:this={choicesHeading}>Continuations</h2>
      {#if parent}<button type="button" onclick={() => void explore(parent)}>Back one point</button>{/if}
    </header>
    {#if !empty && anchor}
      <details use:animatedDetails class="context">
        <summary>After {role(anchor)} · {anchor.text.slice(0, 70) || "Start of conversation"}{anchor.text.length > 70 ? "…" : ""}</summary>
        <div class="passage-text">{anchor.text}</div>
      </details>
    {/if}
    <form onsubmit={generate}>
      {#if empty}
        <label class="prompt-label" for="weave-prompt">Starting text</label>
        <textarea id="weave-prompt" name="prompt" dir="auto" bind:value={prompt} required rows="5"
          placeholder="Type a prompt…" disabled={busy}
          onkeydown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}></textarea>
      {/if}
      <div class="generation-controls">
        <label>Alternatives
          <Select bind:value={count} disabled={busy} ariaLabel="Alternatives" options={[1, 2, 3, 4, 6, 8].map(value => ({ value, label: String(value) }))} />
        </label>
        <button type="submit" class="primary" disabled={busy || !loomTree.loaded}
          {...{ "aria-description": "Create sibling turns using the current model and response controls" }}>Generate {count}</button>
        <button type="button" disabled={!genStatus.active} onclick={sendStop}>Stop</button>
      </div>
    </form>
    <p class="status" role="status">{genStatus.active ? "Generating alternatives…" : `${choices.length} available${chosen ? " · one chosen" : ""}`}</p>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    {#if !empty && choices.length === 0}
      <p class="hint">No continuations at this point yet. Generate some to explore.</p>
    {/if}
    <div class="choice-list" role="list" aria-label="Weave alternatives">
      {#each choices as node, index (node.id)}
        <article class="choice" class:chosen={chosen?.id === node.id} class:generation-active={genStatus.active && loomTree.pendingNodeId === node.id} role="listitem" data-weave-choice={node.id}>
          <header>
            <span class="role">Option {index + 1} · {role(node)}</span>
            {#if genStatus.active && loomTree.pendingNodeId === node.id}<span class="generation-label" role="status">Writing…</span>{/if}
            {#if chosen?.id === node.id}<span class="marker">chosen</span>{/if}
            <button type="button" class="quiet" aria-label={`${node.starred ? "Starred" : "Star"} option ${index + 1}`} aria-pressed={node.starred}
              onclick={() => void loomStar(node.id, !node.starred)}>{node.starred ? "Starred" : "Star"}</button>
          </header>
          <div class="passage-text" dir="auto">{node.text || (genStatus.active ? "Generating…" : "Empty turn")}</div>
          <footer>
            <button type="button" disabled={busy || chosen?.id === node.id}
              onclick={() => void choose(node)}>{chosen?.id === node.id ? "Chosen" : "Choose"}</button>
            <button type="button" disabled={busy} onclick={async () => { await choose(node); if (loomTree.active_node_id === node.id) await explore(node); }}
              {...{ "aria-description": "Choose this path and explore its next turns" }}>Explore next</button>
            <button type="button" disabled={busy || lastBoundary(node) < 0}
              {...{ "aria-description": (lastBoundary(node) < 0
                ? "This turn has no saved token boundary to continue from"
                : "Keep every existing token and sample more text with this turn’s saved settings; the original stays intact") }}
              onclick={() => void extend(node)}>Continue text</button>
            <button type="button" disabled={busy || !node.parent_id} onclick={(event) => onwrite(node, event.currentTarget)}>Write alternative</button>
          </footer>
        </article>
      {/each}
    </div>
  </section>
</div>

<style>
  .weave { display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 0.85fr); flex: 1; min-height: 0; min-width: 0; overflow: hidden; }
  section { min-width: 0; overflow-y: auto; padding: var(--surface-padding); overscroll-behavior: contain; }
  .possibilities { background: var(--input-well); }
  .section-heading, article header, footer, .generation-controls { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); }
  .section-heading { justify-content: space-between; margin-block-end: var(--space-4); }
  h2 { margin: 0; font-size: var(--text-md); font-weight: var(--weight-display); }
  .section-heading > span, .hint, .status { color: var(--fg-muted); font-size: var(--text-xs); }
  .reading-path, .choice-list { display: flex; flex-direction: column; gap: var(--space-4); }
  article { min-width: 0; padding: var(--surface-padding); border: 1px solid var(--glass-line); border-radius: var(--radius-lg); }
  .passage { border-color: transparent; border-inline-start: 2px solid var(--glass-line); border-radius: 0; }
  .passage.branch-point { border-inline-start-color: var(--accent); background: var(--glass); }
  .choice { background: var(--bg-elev); }
  .choice.chosen { border-color: var(--accent); }
  article header { margin-block-end: var(--space-2); }
  article header .quiet { margin-inline-start: auto; }
  .role, .marker { font-size: var(--text-2xs); color: var(--fg-muted); }
  .marker { color: var(--accent); }
  .passage-text { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.65; font-size: var(--text); }
  footer { margin-block-start: var(--space-4); }
  button, textarea { font: inherit; color: var(--fg); border: 1px solid var(--glass-line); border-radius: var(--radius); background: var(--glass); }
  button { min-height: var(--control-target); padding: var(--space-2) var(--space-3); font-size: var(--text-xs); cursor: pointer; touch-action: manipulation; }
  button:hover:not(:disabled) { background: var(--glass-bright); }
  button.primary { background: var(--accent); color: var(--text-on-accent); border-color: transparent; }
  button.primary:hover:not(:disabled) { background: var(--accent); }
  button:disabled { opacity: var(--disabled-opacity); cursor: not-allowed; }
  button.quiet { background: transparent; border-color: transparent; }
  :is(button, textarea, summary):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .generation-controls label { display: flex; align-items: center; gap: var(--space-2); font-size: var(--text-xs); }
  .prompt-label { display: block; margin-block-end: var(--space-2); }
  textarea { box-sizing: border-box; width: 100%; resize: vertical; min-height: 100px; padding: var(--space-3); margin-block-end: var(--space-3); background: var(--input-well); }
  summary { cursor: pointer; min-height: var(--control-target); align-content: center; font-size: var(--text-xs); overflow-wrap: anywhere; }
  details { color: var(--fg-muted); }
  .context { margin-block-end: var(--space-3); }
  .tokens { display: flex; flex-wrap: wrap; gap: var(--space-1); padding: var(--space-1); }
  .tokens button { white-space: pre-wrap; overflow-wrap: anywhere; max-width: 100%; }
  .error { color: var(--accent-red); }
  @media (max-width: 960px), (max-height: 700px) {
    .weave { display: block; overflow-y: auto; }
    section { overflow: visible; padding: var(--surface-padding); }
    .reading { max-height: none; }
    textarea { font-size: max(16px, var(--text-sm)); }
  }
</style>
