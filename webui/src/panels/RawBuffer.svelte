<script lang="ts">
  import { tokenInspectorUi } from "../lib/stores/drawers.svelte";
  // Flat completion buffer — the chat surface for base (non-chat)
  // models.  No bubbles, no role labels: the loom active path is joined
  // into a single continuous editable text surface, ``white-space:
  // pre-wrap``.  Per-token tinting is preserved for the assistant span
  // when a highlight probe is selected.
  //
  // "generate" runs with ``raw: true`` from the active leaf — the
  // whole buffer is effectively a prefill.  A mid-buffer edit lands as
  // a ``loomEdit`` (active leaf) or ``loomBranch`` (interior node)
  // first; toggling render mode never mutates the tree, only generation
  // and explicit edits do.

  import { onMount, tick } from "svelte";
  import StatusFooter from "./StatusFooter.svelte";
  import PendingBubbles from "./PendingBubbles.svelte";
  import {
    chatLog,
    loomTree,
    genStatus,
    sendSubmit,
    sendGenerate,
    pendingActions,
    refreshLoomTree,
    sendStop,
    loomNavigate,
    highlightState,
    openDrawer,
    beginTokenHover,
    endTokenHover,
  } from "../lib/stores.svelte";
  import Button from "../lib/ui/Button.svelte";
  import BaseModelTag from "../lib/ui/BaseModelTag.svelte";
  import { sessionState } from "../lib/stores/session.svelte";
  import { apiTree } from "../lib/runtime/services";
  import { completionAnchor } from "../lib/completionSelection";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import SegmentedTabs from "../lib/ui/SegmentedTabs.svelte";
  import Select from "../lib/Select.svelte";
  import TokenLogitsPopover from "../lib/ui/TokenLogitsPopover.svelte";
  import { completionTokenViews, projectEditedTokens, type RawTokenView } from "../lib/rawTokenInspection";
  import { tokenRowToScore } from "../lib/stores/loom.svelte";
  import {
    ENTROPY_TARGET,
    PROBABILITY_TARGET,
    SURPRISE_TARGET,
  } from "../lib/tokens";
  import {
    highlightScoreFor,
    highlightStyleString,
  } from "../lib/highlight";

  // ---------- buffer text ----------

  /** The conversation flattened to one string — every turn's text
   *  concatenated in order. This is the raw prefix the model extends. */
  const bufferText = $derived(
    chatLog.turns.map((t) => t.text ?? "").join(""),
  );

  // Local editable mirror.  Synced from ``bufferText`` whenever the
  // tree changes and the user isn't mid-edit; user edits write here
  // and send / append land them on the tree.
  let draft = $state("");
  let dirty = $state(false);
  // Set the moment a send/append is fired: the tree hasn't caught
  // up to the locally-edited draft yet, so the buffer→draft sync below
  // is held until ``bufferText`` reaches the draft — without this the
  // draft would briefly snap back to the pre-edit text (a flash of the
  // user's typed tail vanishing) for the server round-trip.
  let committing = $state(false);
  let submissionFinishedAt: number | null = $state(null);
  let textareaRef: HTMLTextAreaElement | null = $state(null);
  let editError = $state<string | null>(null);
  let editFocusOrigin = $state<Element | null>(null);
  let editedTokenViews = $state<RawTokenView[] | null>(null);
  let selection = $state<{ start: number; end: number; text: string; nodeId: string | null } | null>(null);
  let preparingSelection = $state(false);

  // Re-sync the draft from the tree when the buffer changes and the
  // user has no pending edit.  ``dirty`` guards against clobbering an
  // in-progress edit; ``committing`` holds the just-sent draft until
  // the server-side node lands, after which streamed tokens flow in.
  $effect(() => {
    const text = bufferText;
    if (preparingSelection) return;
    if (committing) {
      // Hold until the tree genuinely carries the submitted draft as a
      // prefix — a *content* check, not a length one.  A length compare
      // releases early when a fast-streaming generation fills the
      // buffer to the right size before the authored span has landed
      // (or under a transient wrong-parent active path), snapping the
      // draft onto text that doesn't contain what the user wrote.
      if (!text.startsWith(draft)) {
        if (!genStatus.active && genStatus.finishedAt !== null && genStatus.finishedAt !== submissionFinishedAt) {
          committing = false;
          dirty = true;
        }
        return;
      }
      committing = false;
    }
    if (!dirty) {
      draft = text;
      editedTokenViews = null;
    }
  });

  function onInput(ev: Event): void {
    selection = null;
    editError = null;
    const next = (ev.currentTarget as HTMLTextAreaElement).value;
    editedTokenViews = next === bufferText ? null : projectEditedTokens(editedTokenViews ?? tokenViews, next);
    draft = next;
    dirty = draft !== bufferText;
    // Any keystroke supersedes a pending submission — resume live sync.
    committing = false;
  }

  function rememberSelection(): void {
    if (!textareaRef || genStatus.active || committing || preparingSelection) return;
    selection = { start: textareaRef.selectionStart, end: textareaRef.selectionEnd, text: draft, nodeId: activeLeaf };
  }

  async function completeFromSelection(recomplete = false): Promise<void> {
    if (!selectionCurrent || !selection || genStatus.active || committing || preparingSelection || dirty || pendingActions.queue.length || !loomTree.root_id) return;
    const offset = recomplete ? selection.start : selection.end;
    const original = draft;
    const originalNodeId = activeLeaf;
    const originalRootId = loomTree.root_id;
    preparingSelection = true;
    editError = null;
    editFocusOrigin = document.activeElement;
    try {
      const anchor = completionAnchor(chatLog.turns, loomTree.root_id, offset);
      let parentNodeId = anchor.parentNodeId;
      if (anchor.branch) {
        const result = await apiTree.branch(anchor.branch.nodeId, anchor.branch.text, undefined, "user");
        parentNodeId = result.node_id;
      } else {
        await apiTree.navigate(parentNodeId);
      }
      await refreshLoomTree();
      if (loomTree.root_id !== originalRootId || loomTree.active_node_id !== parentNodeId) {
        throw new Error("The selected completion point changed. Select the text again.");
      }
      draft = original.slice(0, offset);
      editedTokenViews = null;
      selection = null;
      scrolledUp = false;
      committing = true;
      submissionFinishedAt = genStatus.finishedAt;
      await sendGenerate({ raw: true, parent_node_id: parentNodeId, append_same_role: false, n: 1 });
    } catch (error) {
      if (loomTree.root_id !== originalRootId) {
        committing = false;
        return;
      }
      draft = original;
      if (originalNodeId) {
        try {
          await apiTree.navigate(originalNodeId);
          await refreshLoomTree();
        } catch {
          // Keep the original text editable if the connection also prevents returning.
        }
      }
      restoreFailedEdit(error);
    } finally {
      preparingSelection = false;
    }
  }

  $effect(() => {
    if (!editFocusOrigin || committing || genStatus.active || preparingSelection) return;
    const origin = editFocusOrigin;
    editFocusOrigin = null;
    void tick().then(() => {
      if (document.activeElement === origin || document.activeElement === document.body) {
        textareaRef?.focus({ preventScroll: true });
      }
    });
  });

  // ---------- send / generate / append ----------
  //
  // Flat mode is non-linear: editing text anywhere in the buffer is the
  // same operation as appending to the end.  Both collapse to a single
  // "divergence" — the first character that differs from the committed
  // tree text — and everything from there becomes one new span.  The
  // node containing the divergence (and its whole subtree) is preserved
  // untouched as the original branch; the edited tail is recorded as a
  // fresh span branched at that point.  Appending is just the special
  // case where the divergence sits at the very end of the buffer.

  /** The active leaf node — what a clean generation extends. */
  const activeLeaf = $derived(
    loomTree.loaded ? (loomTree.active_node_id ?? null) : null,
  );
  const selectionCurrent = $derived(selection !== null && selection.text === draft && selection.nodeId === activeLeaf);

  interface Divergence {
    /** New text — from the divergence offset to the end of the draft. */
    tail: string;
    /** Where the new span hangs: the diverging node's parent for a
     *  mid-buffer edit, the active leaf for a pure append.
     *  ``undefined`` means the buffer is clean — generate from the leaf
     *  with no new span at all. */
    parentNodeId: string | null | undefined;
  }

  /** Diff the draft against the committed buffer and locate the single
   *  span the change collapses to. */
  function resolveDivergence(): Divergence {
    if (!dirty) return { tail: "", parentNodeId: undefined };
    const turns = chatLog.turns;
    let startOffset = 0;
    for (let i = 0; i < turns.length; i++) {
      const turnText = turns[i].text ?? "";
      const slice = draft.slice(startOffset, startOffset + turnText.length);
      if (slice !== turnText) {
        // Divergence inside turns[i].  That node + its subtree stay as
        // the original branch; the tail (this node's start → end of
        // draft) becomes a new span branched as its sibling — i.e. a
        // child of the node's parent.
        const nid = turns[i].nodeId ?? null;
        const parentNodeId = nid
          ? (loomTree.nodes.get(nid)?.parent_id ?? null)
          : (activeLeaf ?? null);
        return { tail: draft.slice(startOffset), parentNodeId };
      }
      startOffset += turnText.length;
    }
    // No node diverged — the draft runs past the joined buffer.  The
    // appended tail hangs under the active leaf as a fresh child.
    return { tail: draft.slice(startOffset), parentNodeId: activeLeaf ?? null };
  }

  async function submitBuffer(): Promise<void> {
    if (genStatus.active || committing || preparingSelection) return;
    mode = "edit";
    scrolledUp = false;
    const d = resolveDivergence();
    // The divergence tail is an authored user span; generation occupies the
    // assistant role. A clean buffer omits the authored half entirely.
    const text = d.tail === "" ? null : d.tail;
    committing = true;
    editError = null;
    submissionFinishedAt = genStatus.finishedAt;
    dirty = false;
    try {
      await sendSubmit(text, text === null ? null : "user", "assistant",
        { raw: true, parent_node_id: d.parentNodeId });
    } catch (error) {
      restoreFailedEdit(error);
    }
  }

  // ---------- append without generating ----------

  /** Land the pending edit on the tree without generating — the same
   *  divergence branch ``submitBuffer`` would take, minus the decode. */
  async function appendEdit(): Promise<void> {
    if (!dirty || genStatus.active || committing || preparingSelection) return;
    editFocusOrigin = document.activeElement;
    const d = resolveDivergence();
    if (d.tail === "") {
      // A boundary deletion selects the shorter path; it must not silently
      // restore the discarded suffix on the next tree update.
      committing = true;
      try {
        await loomNavigate(d.parentNodeId ?? loomTree.root_id!);
        dirty = draft !== bufferText;
      } catch (error) {
        restoreFailedEdit(error);
      } finally {
        committing = false;
      }
      return;
    }
    dirty = false;
    committing = true;
    editError = null;
    submissionFinishedAt = genStatus.finishedAt;
    try {
      await sendSubmit(d.tail, "user", null,
        { raw: true, parent_node_id: d.parentNodeId ?? null });
    } catch (error) {
      restoreFailedEdit(error);
    }
  }

  function restoreFailedEdit(error: unknown): void {
    committing = false;
    dirty = draft !== bufferText;
    editError = userFacingError(error, "The edit could not be saved.");
  }

  function revertEdit(): void {
    draft = bufferText;
    dirty = false;
    committing = false;
    editError = null;
    editedTokenViews = null;
    void tick().then(() => textareaRef?.focus({ preventScroll: true }));
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey) && !ev.isComposing) {
      ev.preventDefault();
      submitBuffer();
      return;
    }
    if (ev.key === "Escape" && genStatus.active) {
      ev.preventDefault();
      sendStop();
    }
  }

  // ---------- token inspection ----------

  const tokenViews = $derived(completionTokenViews(chatLog.turns));
  const currentTokenViews = $derived(editedTokenViews ?? tokenViews);
  const earlierGenerations = $derived(
    [...loomTree.nodes.values()]
      .filter(node => node.recipe && node.tokens?.length && !loomTree.activePath.includes(node.id))
      .sort((a, b) => b.created_at - a.created_at),
  );
  let inspectionSource = $state("current");
  const inspectionSources = $derived([
    { value: "current", label: dirty ? "Current draft" : "Current text" },
    ...(dirty && tokenViews.some(view => view.tok) ? [{ value: "recorded", label: "Original text before edit" }] : []),
    ...earlierGenerations.map((node, index) => ({
      value: node.id,
      label: `Earlier generation ${earlierGenerations.length - index} · ${node.text.slice(0, 48).replace(/\s+/g, " ")}`,
    })),
  ]);
  const inspectedTokenViews = $derived.by(() => {
    if (inspectionSource === "recorded") return tokenViews;
    const node = earlierGenerations.find(node => node.id === inspectionSource);
    if (node) return completionTokenViews([{
      nodeId: node.id, role: node.role, text: node.text, generated: true,
      tokens: node.tokens!.map(tokenRowToScore),
    }]).map(view => ({ ...view, turnIdx: null }));
    return currentTokenViews;
  });
  $effect(() => {
    if (!inspectionSources.some(source => source.value === inspectionSource)) inspectionSource = "current";
  });
  let tokenPopup = $state<{ view: RawTokenView; anchor: HTMLElement } | null>(null);

  // ---------- edit / inspect mode ----------
  //
  // Editing changes the draft projection, never the recorded token rows.
  // Inspection can show either that projection or an original generation.
  type Mode = "edit" | "inspect";
  let mode: Mode = $state<Mode>("edit");

  const showColorMirror = $derived(
    mode === "edit" && !committing && (dirty || (highlightState.target !== null && currentTokenViews.some(view => view.tok))) &&
    currentTokenViews.map((token) => token.text).join("") === (dirty ? draft : bufferText),
  );
  const colorNotice = $derived.by(() => {
    if (!highlightState.target) return null;
    if (dirty) return "Retained tokens keep their original readings; edited text has no recorded probabilities.";
    if (committing || !hasClickableTokens) return null;
    const targets = [highlightState.target, ...(highlightState.compareTwo && highlightState.compareTarget
      ? [highlightState.compareTarget] : [])];
    const scores = tokenViews.flatMap(({ tok }) => targets.flatMap((target) => {
      const score = tok ? highlightScoreFor(tok, target) : undefined;
      return score === undefined ? [] : [score];
    }));
    if (scores.length === 0) return "No readings were recorded for this color.";
    if (scores.some((score) => score !== 0)) return null;
    const reading = targets.length > 1 ? "values are" : highlightState.target === SURPRISE_TARGET ? "token surprisal is"
      : highlightState.target === ENTROPY_TARGET ? "sampler entropy is" : "values are";
    return `Recorded ${reading} zero, so these tokens stay uncolored.`;
  });

  /** Whether any clickable (generated) token exists to inspect. */
  const hasClickableTokens = $derived(
    currentTokenViews.some(view => view.tok !== null),
  );
  const inspectableTokenIndices = $derived(
    inspectedTokenViews.flatMap((token, index) => token.tok === null ? [] : [index]),
  );
  let focusedTokenViewIndex = $state(-1);

  $effect(() => {
    const indices = inspectableTokenIndices;
    if (indices.length === 0) {
      focusedTokenViewIndex = -1;
    } else if (!indices.includes(focusedTokenViewIndex)) {
      focusedTokenViewIndex = indices[0];
    }
  });

  const hasRecordedTokens = $derived(hasClickableTokens || tokenViews.some(view => view.tok) || earlierGenerations.length > 0);
  const canInspect = $derived(hasRecordedTokens && !committing && !genStatus.active && !preparingSelection);
  const modeItems = $derived([
    { value: "edit" as const, label: "Edit text", title: "Edit the completion text" },
    {
      value: "inspect" as const,
      label: "Inspect tokens",
      disabled: mode !== "inspect" && !canInspect,
      title: canInspect || mode === "inspect"
        ? "Choose a token to inspect readings or branch"
        : genStatus.active
          ? "Stop or finish generation before inspecting"
          : committing
          ? "Wait for the edit to finish saving"
          : "Continue text to record tokens",
    },
  ]);

  /** Fall back to edit when nothing remains to inspect (buffer cleared,
   *  conversation reset) so the surface never strands on an empty view. */
  $effect(() => {
    if (mode === "inspect" && !hasRecordedTokens) mode = "edit";
  });

  function changeMode(next: Mode): void {
    tokenPopup = null;
    if (next === "inspect" && !hasClickableTokens) {
      inspectionSource = tokenViews.some(view => view.tok) && dirty ? "recorded" : earlierGenerations[0]?.id ?? "current";
    }
    mode = next;
  }

  function inspectTooltip(v: RawTokenView): string {
    const tok = v.tok;
    if (!tok) return "";
    const parts: string[] = [];
    const sc = highlightScoreFor(tok, highlightState.target);
    if (sc !== undefined && highlightState.target) {
      if (highlightState.target === SURPRISE_TARGET) {
        if (tok.logprob != null) {
          parts.push(`token surprisal ${(-tok.logprob).toFixed(3)} nats`);
        }
      } else if (highlightState.target === PROBABILITY_TARGET && tok.logprob != null) {
        parts.push(`token probability ${(Math.exp(Math.min(0, tok.logprob)) * 100).toFixed(1)}%`);
      } else if (highlightState.target === ENTROPY_TARGET && tok.samplerEntropy != null) {
        parts.push(`sampler entropy ${tok.samplerEntropy.toFixed(3)} nats`);
      } else {
        parts.push(`${highlightState.target} ${sc >= 0 ? "+" : ""}${sc.toFixed(3)}`);
      }
    }
    const n = tok.topAlts?.length ?? 0;
    parts.push(n > 0 ? `Select to see ${n} alternatives` : "Select to see the recorded probability");
    return parts.join(" · ");
  }

  function tokenAccessibleName(v: RawTokenView): string {
    const token = v.text.trim() || "whitespace";
    const alternatives = v.tok?.topAlts?.length ?? 0;
    return alternatives > 0
      ? `Inspect token ${token}, ${alternatives} alternatives`
      : `Inspect token ${token}`;
  }

  function focusInspectableToken(index: number): void {
    focusedTokenViewIndex = index;
    queueMicrotask(() => {
      logRef?.querySelector<HTMLElement>(`[data-token-view-index="${index}"]`)?.focus();
    });
  }

  function onInspectableTokenKeydown(ev: KeyboardEvent, viewIndex: number, token: RawTokenView): void {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      openToken(token, ev.currentTarget as HTMLElement);
      return;
    }
    const current = inspectableTokenIndices.indexOf(viewIndex);
    if (current < 0) return;
    let next = current;
    if (ev.key === "ArrowRight" || ev.key === "ArrowDown") {
      next = (current + 1) % inspectableTokenIndices.length;
    } else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") {
      next = (current - 1 + inspectableTokenIndices.length) % inspectableTokenIndices.length;
    } else if (ev.key === "Home") {
      next = 0;
    } else if (ev.key === "End") {
      next = inspectableTokenIndices.length - 1;
    } else {
      return;
    }
    ev.preventDefault();
    focusInspectableToken(inspectableTokenIndices[next]);
  }

  function showTokenDetails(v: RawTokenView): void {
    if (v.turnIdx === null) return;
    openDrawer("token_drilldown", {
      turnIdx: v.turnIdx,
      tokenIdx: v.tokenIdx,
      isThinking: v.isThinking,
      initialTab: "logits",
    });
  }

  function openToken(view: RawTokenView, anchor: HTMLElement): void {
    if (!view.tok) return;
    if (tokenInspectorUi.docked && view.turnIdx !== null) {
      tokenPopup = null;
      openDrawer("token_drilldown", { turnIdx: view.turnIdx, tokenIdx: view.tokenIdx, isThinking: view.isThinking });
      return;
    }
    tokenPopup = { view, anchor };
  }

  function hoverToken(v: RawTokenView): void {
    if (!v.tok) return;
    beginTokenHover(v.tok, v.nodeId ?? undefined);
  }

  let logRef: HTMLDivElement | null = $state(null);
  let colorMirrorRef: HTMLDivElement | null = $state(null);
  let scrolledUp = $state(false);
  function syncColorScroll(): void {
    if (!colorMirrorRef || !textareaRef) return;
    colorMirrorRef.scrollTop = textareaRef.scrollTop;
    colorMirrorRef.scrollLeft = textareaRef.scrollLeft;
  }
  function onScroll(ev: Event): void {
    const el = ev.currentTarget as HTMLElement;
    scrolledUp = el.scrollHeight - el.scrollTop - el.clientHeight >= 8;
    if (mode === "edit") syncColorScroll();
  }
  function followLatest(): void {
    scrolledUp = false;
    const el = mode === "edit" ? textareaRef : logRef;
    if (el) el.scrollTop = el.scrollHeight;
    syncColorScroll();
  }
  $effect(() => {
    if (showColorMirror) void tick().then(syncColorScroll);
  });
  $effect(() => {
    void bufferText;
    if (!dirty && !scrolledUp) {
      void tick().then(() => {
        if (!dirty && !scrolledUp) followLatest();
      });
    }
  });

  onMount(() => {
    draft = bufferText;
  });
</script>

<div class="raw-buffer" aria-label="Completion buffer">
  <div class="raw-head">
    <span class="head-label"><label for="completion-buffer">Text completion</label> {#if sessionState.info?.is_base_model}<BaseModelTag />{/if}</span>
    <SegmentedTabs value={mode} onchange={changeMode} items={modeItems} ariaLabel="Buffer mode" />
  </div>
  <p class="completion-hint" id="completion-hint">{mode === "inspect"
    ? "Click a recorded token to see its probabilities. Arrow keys move between tokens; editing stays in Edit text."
    : "Continue from the end, or place the cursor or select text to complete from another point. The original stays saved in Loom."}{#if colorNotice}<span class="color-notice" role="status">{colorNotice}</span>{/if}</p>

  {#if hasRecordedTokens || dirty}
    <div class="token-legend" aria-label="Text origin legend">
      <span class="origin-user">User text</span>
      <span class="origin-model">Model tokens</span>
      {#if dirty}<span class="origin-draft">Unsaved edit</span>{/if}
    </div>
  {/if}
  {#if mode === "inspect" && inspectionSources.length > 1}
    <div class="inspection-source">
      <span>Inspect</span>
      <Select value={inspectionSource} options={inspectionSources} ariaLabel="Text to inspect"
        onchange={(value) => { tokenPopup = null; inspectionSource = value; }} />
    </div>
    {#if inspectionSource !== "current"}<p class="completion-hint">Original recorded text. Your current text and unsaved edits are unchanged.</p>{/if}
  {/if}

  <div class="surface" class:inspecting={mode === "inspect"} class:loading-pulse={genStatus.active} bind:this={logRef} onscroll={mode === "inspect" ? onScroll : undefined}>
    {#if mode === "inspect"}
      <div class="inspect" dir="auto" aria-label="Completion tokens">
        {#each inspectedTokenViews as v, i (i)}
          {#if !v.tok}
            <span class="seg plain" class:origin-user={v.source === "user"} class:origin-draft={v.source === "draft"}
              title={v.source === "draft" ? "Unsaved user edit · no recorded probabilities" : v.source === "model" ? "Model text · no recorded probabilities" : "User text · no recorded probabilities"}>{v.text}</span>
          {:else}
            <span
              class="seg tok clickable"
              class:origin-user={v.source === "user"}
              class:origin-model={v.source === "model"}
              class:context-changed={v.contextChanged}
              data-cursor="inspect"
              class:tinted={highlightState.target !== null}
              class:has-alts={(v.tok?.topAlts?.length ?? 0) > 0}
              style={v.tok ? highlightStyleString(v.tok) : ""}
              title={inspectTooltip(v)}
              role="button"
              tabindex={i === focusedTokenViewIndex ? 0 : -1}
              data-token-view-index={i}
              aria-label={tokenAccessibleName(v)}
              aria-haspopup="dialog"
              aria-expanded={tokenPopup?.anchor.dataset.tokenViewIndex === String(i)}
              onpointerenter={() => hoverToken(v)}
              onpointerleave={endTokenHover}
              onfocus={() => hoverToken(v)}
              onblur={endTokenHover}
              onclick={(event) => {
                focusedTokenViewIndex = i;
                openToken(v, event.currentTarget);
              }}
              onkeydown={(ev) => onInspectableTokenKeydown(ev, i, v)}
            >{v.text}</span>
          {/if}
        {/each}
      </div>
    {:else}
      {#if showColorMirror}
        <div class="color-mirror" bind:this={colorMirrorRef} dir="auto" aria-hidden="true">
          {#each currentTokenViews as token, i (i)}<span class:origin-draft={token.source === "draft"} style={token.tok ? highlightStyleString(token.tok) : ""}>{token.text}</span>{/each}<span>{"\u200b"}</span>
        </div>
      {/if}
      <textarea
        id="completion-buffer"
        class="buffer-text"
        bind:this={textareaRef}
        dir="auto"
        value={draft}
        oninput={onInput}
        onkeydown={onKeydown}
        onselect={rememberSelection}
        onkeyup={rememberSelection}
        onpointerup={rememberSelection}
        onscroll={onScroll}
        spellcheck="false"
        placeholder="The door opened onto…"
        aria-label="Editable completion buffer"
        aria-describedby={editError ? "completion-hint completion-error" : "completion-hint"}
        aria-keyshortcuts="Meta+Enter Control+Enter"
        aria-invalid={editError ? true : undefined}
        readonly={genStatus.active || committing || preparingSelection}
      ></textarea>
    {/if}
    {#if scrolledUp && !dirty && mode === "edit"}
      <div class="latest-action"><Button onclick={() => { followLatest(); textareaRef?.focus({ preventScroll: true }); }}>Jump to latest text</Button></div>
    {/if}
  </div>

  {#if mode === "edit" && selectionCurrent && selection && draft.length > 0}
    <div class="selection-actions" role="group" aria-label="Complete from selected text" aria-describedby="selection-hint">
      <div class="selection-buttons">
        <Button onclick={() => completeFromSelection()} disabled={dirty || genStatus.active || committing || preparingSelection || pendingActions.queue.length > 0}>
          {selection.start === selection.end ? "Continue from cursor" : "Continue after selection"}
        </Button>
        {#if selection.start !== selection.end}
          <Button onclick={() => completeFromSelection(true)} disabled={dirty || genStatus.active || committing || preparingSelection || pendingActions.queue.length > 0}>Re-complete from selection</Button>
        {/if}
      </div>
      <p id="selection-hint" class="completion-hint">{dirty ? "Save your edit first to complete from this point." : selection.start === selection.end ? "Text after the cursor is regenerated in a new branch. The original stays saved." : "Continue keeps the selection; re-complete starts before it. Text after that point is regenerated in a new branch."}</p>
    </div>
  {/if}

  {#if tokenPopup?.view.tok}
    {@const view = tokenPopup.view}
    {#key tokenPopup.anchor}
      <TokenLogitsPopover token={tokenPopup.view.tok} anchor={tokenPopup.anchor}
        source={inspectionSource !== "current" ? "Original recorded token" : tokenPopup.view.source === "user" ? "User text · recorded" : "Model token"}
        contextChanged={tokenPopup.view.contextChanged}
        onclose={() => tokenPopup = null}
        ondetails={view.turnIdx === null ? undefined : () => showTokenDetails(view)} />
    {/key}
  {/if}

  {#if editError}<p class="completion-error" id="completion-error" role="alert">{editError} Your text is still here.</p>{/if}
  <StatusFooter savedEdit={!genStatus.active && genStatus.finishReason === "stop" && genStatus.tokensSoFar === 0 && chatLog.turns.at(-1)?.generated === false} />
  <PendingBubbles />

  <div class="actions">
    {#if dirty}
    <div class="edit-actions">
      <span class="dirty-flag" role="status">
        Unsaved edit
      </span>
      <Button
        onclick={() => void appendEdit()}
        disabled={genStatus.active || committing || preparingSelection}
        title="Save this edit without generating"
      >
        Save edit
      </Button>
      <Button onclick={revertEdit} disabled={genStatus.active || committing || preparingSelection}>
        Discard edit
      </Button>
    </div>
    {/if}
    <div class="generation-actions">
    <span class="keyboard-hint" aria-hidden="true">⌘ / Ctrl + Enter</span>
    <Button
      variant="solid"
      onclick={submitBuffer}
      disabled={genStatus.active || committing || preparingSelection}
      busy={committing && !genStatus.active}
      title="Continue text (Cmd/Ctrl+Enter)"
    >Continue text</Button>
    <Button
      variant="danger"
      onclick={sendStop}
      disabled={!genStatus.active}
      title="Esc"
    >Stop</Button>
    </div>
  </div>
</div>

<style>
  .selection-actions { display: grid; gap: var(--space-sm); }
  .selection-buttons { display: flex; flex-wrap: wrap; gap: var(--space-sm); }
  .completion-hint, .completion-error, .color-notice { margin: 0; color: var(--fg-muted); font-family: var(--font-ui); font-size: var(--text-sm); line-height: 1.5; text-wrap: pretty; }
  .color-notice { display: block; }
  .token-legend { display: flex; flex-wrap: wrap; gap: var(--space-md); color: var(--fg-dim); font-family: var(--font-ui); font-size: var(--text-xs); }
  .origin-user { color: var(--fg-dim); }
  .origin-model { text-decoration: underline solid var(--fg-dim); text-underline-offset: 3px; }
  .origin-draft { background: var(--accent-subtle); text-decoration: underline dashed var(--accent); text-underline-offset: 3px; }
  .inspection-source { display: flex; align-items: center; gap: var(--space-sm); min-width: 0; color: var(--fg-dim); font-family: var(--font-ui); font-size: var(--text-sm); }
  .inspection-source :global(.sk-select) { min-width: 0; max-width: min(30rem, 100%); }
  .completion-error { color: var(--accent-red); }
  .raw-buffer {
    display: flex;
    flex: 1 0 auto;
    flex-direction: column;
    min-height: 0;
    gap: var(--composer-space, var(--space-6));
    font-family: var(--font-mono);
    font-size: var(--text);
    color: var(--fg);
  }
  .surface {
    flex: 1 0 calc(2lh + var(--surface-padding) * 2);
    position: relative;
    overflow: clip;
    min-height: calc(2lh + var(--surface-padding) * 2);
    /* Recessed input well — this is the editable completion surface. */
    background: var(--workspace-buffer-bg, var(--input-well));
    border: 0;
    border-radius: var(--radius);
  }
  .surface.inspecting { overflow-y: auto; }
  .latest-action { position: absolute; inset-inline-end: var(--space-3); bottom: var(--space-3); z-index: 2; }
  .buffer-text,
  .color-mirror,
  .inspect {
    margin: 0;
    padding: var(--surface-padding);
    font-family: var(--font-mono);
    font-size: var(--text);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    box-sizing: border-box;
  }
  .buffer-text, .color-mirror { scrollbar-gutter: stable; }
  .color-mirror {
    position: absolute;
    inset: 0;
    overflow: hidden;
    color: transparent;
    pointer-events: none;
  }
  /* Inspect mode shares the editor's wrapping and exposes generated tokens. */
  .inspect {
    min-height: 100%;
    color: var(--fg-strong);
  }
  .inspect .plain {
    /* Prompt text (user / system turns) — present but not interactive. */
    color: var(--fg-dim);
  }
  .inspect .tok {
    display: inline;
    min-width: 0;
    min-height: 0;
    padding: 0;
    border: 0;
    background: transparent;
    box-shadow: none;
    font: inherit;
    color: inherit;
    white-space: inherit;
    text-align: inherit;
    vertical-align: baseline;
    border-radius: var(--radius);
  }
  .inspect .tok.origin-user { color: var(--fg-dim); }
  .inspect .context-changed { text-decoration-style: dashed; }
  .inspect .clickable {
    cursor: crosshair;
  }
  .inspect .clickable:hover {
    outline: 1px solid var(--fg-muted);
    outline-offset: -1px;
    border-radius: var(--radius);
  }
  .inspect .clickable:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  /* Tokens that captured top-K alternatives get a faint underline so the
   * forkable ones are discoverable at a glance. */
  .inspect .has-alts {
    text-decoration: underline dotted var(--fg-dim);
    text-underline-offset: 2px;
  }

  /* Header: label + edit/inspect toggle. */
  .raw-head {
    display: flex;
    flex: 0 0 auto;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .head-label {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-transform: none;
    letter-spacing: 0;
  }
  .buffer-text {
    display: block;
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 100%;
    background: transparent;
    color: var(--fg-strong);
    border: 0;
    resize: none;
  }
  .buffer-text:focus {
    outline: none;
  }
  .surface:focus-within { box-shadow: var(--shadow-field-focus); }
  .buffer-text:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -3px; }

  .actions, .edit-actions, .generation-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--composer-space, var(--space-6));
  }
  .actions { flex: 0 0 auto; justify-content: space-between; }
  .generation-actions { margin-inline-start: auto; }
  .keyboard-hint { color: var(--fg-muted); font-size: var(--text-xs); }
  .dirty-flag {
    color: var(--accent-yellow);
    font-size: var(--text-xs);
    margin-inline-end: auto;
  }
  @media (max-width: 40rem) {
    .keyboard-hint { display: none; }
    .generation-actions { gap: var(--space-2); }
    .surface, .buffer-text, .color-mirror, .inspect { font-size: max(16px, var(--text)); }
  }
</style>
