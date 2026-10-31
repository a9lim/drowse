<script lang="ts">
  // The loom's dialogs: regenerate, edit, branch, delete, clear, note,
  // go-to-turn, search, guided fan-out, and regenerate-with-variation.  The
  // sidebar opens one through ``open`` (bound instance) and gets focus back
  // through ``onclosed``; while a dialog is open the rest of the workspace
  // is inert.
  import { tick } from "svelte";
  import { fade, fly } from "svelte/transition";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import MorphText from "../../lib/ui/MorphText.svelte";
  import NumberInput from "../../lib/NumberInput.svelte";
  import Select from "../../lib/Select.svelte";
  import { containTab } from "../../lib/focusTrap";
  import { modalIn, modalOut, scrimIn, scrimOut } from "../../lib/motion";
  import { apiTree } from "../../lib/runtime/services";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { flushConversationAutosave, savedConversationState } from "../../lib/stores/savedConversations.svelte";
  import {
    applyTreeFilter,
    clearNodeSelection,
    clearTreeFilter,
    filterState,
    genStatus,
    LOOM_DELETE_DURING_GENERATION_MESSAGE,
    loomBranch,
    loomContinueFromCommitted,
    loomDelete,
    loomEdit,
    loomNavigate,
    loomNodeIntersectsGeneration,
    loomNote,
    loomRegenerateActive,
    loomTree,
    pushToast,
    refreshLoomTree,
  } from "../../lib/stores.svelte";
  import { parseStrengthList, resolveNodeIdPrefix } from "./loomCommands";

  let {
    background = null,
    onsearchmatch,
    onbranchcreated,
    onclosed,
  }: {
    /** The loom panel; it and its workspace siblings go inert while open. */
    background?: HTMLElement | null;
    /** A search matched: reveal the first match. */
    onsearchmatch?: () => Promise<void>;
    onbranchcreated?: (nodeId: string) => void;
    /** The dialog closed; ``trigger`` is the control that opened it. */
    onclosed?: (trigger: HTMLElement | null) => void;
  } = $props();

  interface ModalState {
    kind:
      | null
      | "regenerate"
      | "edit"
      | "branch"
      | "delete"
      | "clear"
      | "note"
      | "navpicker"
      | "search"
      | "fanout"
      | "regen_mode";
    nodeId: string | null;
    text: string;
    n: number;
    /** Steering selector for the fan-out modal. */
    vector?: string;
    /** Mode for the regen-with-modifier modal. */
    mode?: string;
    /** Inline validation error.  Non-empty means commit failed; the
     *  modal stays open with this message rendered below the input. */
    error?: string;
  }
  let modal: ModalState = $state({
    kind: null,
    nodeId: null,
    text: "",
    n: 1,
    vector: "",
    mode: "unsteered",
    error: "",
  });
  /** Focusable ref shared by every modal input variant — the regenerate
   *  / regen_mode number-of-siblings field swaps in a themed
   *  ``NumberInput`` which exposes a structurally-matching
   *  ``focus()`` / ``select()`` pair, so the modal-open auto-focus
   *  works regardless of which branch is rendered. */
  type FocusableRef = {
    focus: () => void;
    select?: () => void;
  };
  let modalInput: FocusableRef | null = $state(null);
  let modalEl: HTMLElement | null = $state(null);
  let modalCancelButton: HTMLButtonElement | null = $state(null);
  let modalTrigger: HTMLElement | null = null;
  let backgroundStates: Array<{
    element: HTMLElement;
    inert: boolean;
    ariaHidden: string | null;
  }> = [];

  function hideModalBackground(): void {
    if (backgroundStates.length > 0) return;
    const zone = background?.closest<HTMLElement>(".loom-zone");
    const layout = zone?.parentElement;
    if (!zone || !layout) return;
    backgroundStates = Array.from(layout.children)
      .filter((element): element is HTMLElement => element instanceof HTMLElement && element !== zone)
      .map((element) => ({
        element,
        inert: element.hasAttribute("inert"),
        ariaHidden: element.getAttribute("aria-hidden"),
      }));
    for (const state of backgroundStates) {
      state.element.setAttribute("inert", "");
      state.element.setAttribute("aria-hidden", "true");
    }
  }

  function restoreModalBackground(): void {
    for (const state of backgroundStates) {
      if (!state.element.isConnected) continue;
      if (!state.inert) state.element.removeAttribute("inert");
      if (state.ariaHidden === null) state.element.removeAttribute("aria-hidden");
      else state.element.setAttribute("aria-hidden", state.ariaHidden);
    }
    backgroundStates = [];
  }

  function focusInitialModalControl(): void {
    if (modalInput) {
      modalInput.focus();
      modalInput.select?.();
    } else {
      modalCancelButton?.focus();
    }
  }

  async function openModal(
    kind: ModalState["kind"],
    nodeId: string | null,
    initialText: string = "",
    initialN: number = 1,
    trigger?: HTMLElement | null,
  ): Promise<void> {
    if (kind === "delete" && nodeId && loomNodeIntersectsGeneration(nodeId)) {
      pushToast(LOOM_DELETE_DURING_GENERATION_MESSAGE, { kind: "warning" });
      return;
    }
    modalTrigger = trigger ??
      (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    modalInput = null;
    modal = {
      kind,
      nodeId,
      text: initialText,
      n: initialN,
      vector: "",
      mode: "unsteered",
      error: "",
    };
    await tick();
    focusInitialModalControl();
    background?.setAttribute("inert", "");
    background?.setAttribute("aria-hidden", "true");
    hideModalBackground();
  }

  function closeModal(): void {
    const trigger = modalTrigger;
    background?.removeAttribute("inert");
    background?.removeAttribute("aria-hidden");
    restoreModalBackground();
    modal = {
      kind: null,
      nodeId: null,
      text: "",
      n: 1,
      vector: "",
      mode: "unsteered",
      error: "",
    };
    modalInput = null;
    modalEl = null;
    modalCancelButton = null;
    modalTrigger = null;
    void tick().then(() => onclosed?.(trigger));
  }

  function onModalKey(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      closeModal();
      return;
    }
    if (modalEl) containTab(ev, modalEl);
  }

  function setModalError(message: string): void {
    // Clear input focus so the error reads; modal stays open.
    modal = { ...modal, error: message };
  }

  let modalBusy = $state(false);
  async function commitModal(): Promise<void> {
    if (modalBusy) return;
    modalBusy = true;
    try { await applyModal(); }
    finally { modalBusy = false; }
  }

  async function applyModal(): Promise<void> {
    const m = modal;
    if (!m.kind || !m.nodeId) return closeModal();
    switch (m.kind) {
      case "clear":
        if (genStatus.active) { setModalError("Stop the current reply before clearing the loom."); return; }
        try {
          await flushConversationAutosave();
          await apiTree.reset();
          savedConversationState.activeId = null;
          savedConversationState.avatarSeed = null;
          savedConversationState.accent = "purple";
          await refreshLoomTree();
          clearNodeSelection();
          clearTreeFilter();
          pushToast("Loom cleared. The previous conversation is still in Saved chats.", { kind: "info" });
        } catch (error) {
          setModalError(userFacingError(error, "The loom could not be cleared. Your saved chats were not deleted."));
          return;
        }
        break;
      case "regenerate":
        await loomRegenerateActive(Math.max(1, Math.floor(m.n)));
        break;
      case "edit":
        await loomEdit(m.nodeId, m.text);
        break;
      case "branch": {
        const newId = await loomBranch(m.nodeId, m.text);
        if (newId) { await loomNavigate(newId); onbranchcreated?.(newId); }
        break;
      }
      case "delete":
        if (loomNodeIntersectsGeneration(m.nodeId)) {
          setModalError(LOOM_DELETE_DURING_GENERATION_MESSAGE);
          return;
        }
        if (!await loomDelete(m.nodeId)) {
          setModalError(loomTree.error ?? "The branch could not be removed. Try again.");
          return;
        }
        break;
      case "note":
        await loomNote(m.nodeId, m.text);
        break;
      case "navpicker": {
        // Resolve a node by id-prefix or full id.  Ambiguity keeps the
        // modal open with a list of matches so the user can disambiguate
        // rather than navigating to a random first match (Map insertion
        // order leaks the bug otherwise).
        const r = resolveNodeIdPrefix(m.text, loomTree.nodes.keys(), loomTree.root_id);
        if (r.id) {
          await loomNavigate(r.id);
          break;
        }
        if (r.matches.length === 0) {
          setModalError(`no node matches '${m.text}'`);
          return;
        }
        const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
        setModalError(
          `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")",
        );
        return;
      }
      case "search": {
        await applyTreeFilter(m.text, "text");
        if (!filterState.matchingIds?.size) {
          setModalError(`No text match for '${m.text}'.`);
          return;
        }
        closeModal();
        await onsearchmatch?.();
        return;
      }
      case "fanout": {
        // Fan-out: anchor on the user node, send one regen per alpha as a
        // sibling.  We use a sequential dispatch
        // rather than the engine's ``generate_sweep`` because the
        // active rack already carries the rest of the steering
        // context — we only need to overlay the swept vector's α
        // per call.
        const vector = (m.vector ?? "").trim();
        if (!vector) {
          setModalError("vector name required");
          return;
        }
        const alphas = parseStrengthList(m.text);
        if (alphas.length === 0) {
          setModalError(
            "couldn't parse alphas, try a comma list (0.0, 0.3, 0.7) " +
            "or linspace(-1, 1, 5)",
          );
          return;
        }
        const userId = m.nodeId;
        for (const alpha of alphas) {
          // Recipe override carries the per-row alpha for the swept
          // vector.  The engine's modifier resolver accepts a partial
          // recipe expression on the ``steering`` axis.
          await loomContinueFromCommitted(userId, {
            n: 1,
            recipe_override: `steering=${alpha} ${vector}`,
          });
        }
        break;
      }
      case "regen_mode": {
        // Manual regen-with-modifier: regenerate a generated node, or
        // author fresh continuations under a committed node, and dispatch N
        // siblings under the
        // chosen mode.  Modes are resolved engine-side.
        const node = loomTree.nodes.get(m.nodeId);
        const mode = (m.mode ?? "unsteered").trim();
        const N = Math.max(1, Math.floor(m.n));
        if (node?.recipe === null) {
          await loomContinueFromCommitted(m.nodeId, {
            n: N,
            recipe_override: mode,
          });
        } else {
          // Generated node — set it active then regenerate in its own seat.
          if (loomTree.active_node_id !== m.nodeId) {
            await loomNavigate(m.nodeId);
          }
          await loomRegenerateActive(N, { recipe_override: mode });
        }
        break;
      }
    }
    closeModal();
  }

  // Unmounting with a dialog open must not leave the workspace inert.
  $effect(() => () => restoreModalBackground());

  export function open(
    kind: Exclude<ModalState["kind"], null>,
    nodeId: string | null,
    initialText = "",
    initialN = 1,
    trigger?: HTMLElement | null,
  ): Promise<void> {
    return openModal(kind, nodeId, initialText, initialN, trigger);
  }

  export function close(): void {
    closeModal();
  }

  export function isOpen(): boolean {
    return modal.kind !== null;
  }
</script>

{#if modal.kind}
  <div
    class="loom-modal-backdrop"
    aria-hidden="true"
    onclick={closeModal}
    in:fade={scrimIn()}
    out:fade={scrimOut()}
  ></div>
  <div
    class="loom-modal"
    role="dialog"
    aria-modal="true"
    aria-labelledby="loom-modal-title"
    tabindex="-1"
    bind:this={modalEl}
    onkeydown={onModalKey}
    in:fly={modalIn(8)}
    out:fly={modalOut(4)}
  >
    <header class="modal-header">
      <h2 id="loom-modal-title">
        {#if modal.kind === "regenerate"}Generate alternatives
        {:else if modal.kind === "edit"}Edit turn
        {:else if modal.kind === "branch"}Start a branch
        {:else if modal.kind === "delete"}Delete branch
        {:else if modal.kind === "clear"}Clear loom
        {:else if modal.kind === "note"}Add note
        {:else if modal.kind === "navpicker"}Go to turn
        {:else if modal.kind === "search"}Search conversation
        {:else if modal.kind === "fanout"}Create guided variations
        {:else if modal.kind === "regen_mode"}Generate several replies
        {/if}
      </h2>
      <button type="button" class="icon-btn" onclick={closeModal} aria-label="Close"><FluentIcon name="dismiss" /></button>
    </header>
    <div class="modal-body">
      {#if modal.kind === "regenerate"}
        <label>
          <span>Number of replies</span>
          <NumberInput
            bind:this={modalInput}
            value={modal.n}
            min={1}
            max={16}
            step={1}
            oninput={(v) => { if (v !== null) modal.n = v; }}
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Uses the current model settings.</p>
      {:else if modal.kind === "delete"}
        <p>Remove this turn and every continuation below it?</p>
        <blockquote>{loomTree.nodes.get(modal.nodeId ?? "")?.text.slice(0, 240) || "Empty turn"}</blockquote>
        <p class="hint danger">Earlier turns and sibling branches stay. This also updates the saved chat and cannot be undone. Download a copy first if you want a backup.</p>
      {:else if modal.kind === "clear"}
        <p>Save this conversation and start with an empty loom?</p>
        <p class="hint">The full conversation, including all branches, stays in Saved chats. Model files and response settings are kept.</p>
      {:else if modal.kind === "navpicker" || modal.kind === "search"}
        <input
          bind:this={modalInput as HTMLInputElement}
          bind:value={modal.text}
          type="text"
          aria-label={modal.kind === "navpicker" ? "Node ID prefix" : "Search node text"}
          placeholder={modal.kind === "navpicker" ? "node id prefix (or 'root')" : "search node text"}
          onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
        />
      {:else if modal.kind === "fanout"}
        <label>
          <span>Steering direction</span>
          <input
            bind:this={modalInput as HTMLInputElement}
            bind:value={modal.vector}
            type="text"
            placeholder="e.g. honest, calm, deer.wolf"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <label>
          <span>Strengths</span>
          <input
            bind:value={modal.text}
            type="text"
            placeholder="0.0, 0.3, 0.6 · linspace(-1, 1, 5) · 0:1:0.25"
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Enter a comma-separated list, linspace(), or start:stop:step.</p>
      {:else if modal.kind === "regen_mode"}
        <label>
          <span>Variation style</span>
          <Select
            value={modal.mode ?? "unsteered"}
            options={[
              { value: "unsteered", label: "unsteered" },
              { value: "inverted", label: "inverted" },
              { value: "reseed", label: "reseed" },
              { value: "cool", label: "cool" },
              { value: "hot", label: "hot" },
            ]}
            onchange={(v) => { modal.mode = v; }}
            ariaLabel="regen mode"
          />
        </label>
        <label>
          <span>Number of replies</span>
          <NumberInput
            bind:this={modalInput}
            value={modal.n}
            min={1}
            max={16}
            step={1}
            oninput={(v) => { if (v !== null) modal.n = v; }}
            onkeydown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); void commitModal(); } }}
          />
        </label>
        <p class="hint">Keeps this branch's settings and applies the selected variation.</p>
      {:else}
        <textarea
          bind:this={modalInput as HTMLTextAreaElement}
          bind:value={modal.text}
          rows="6"
          aria-label={modal.kind === "edit" ? "Node text" : modal.kind === "branch" ? "Branch text" : "Node note"}
          placeholder={modal.kind === "branch" ? "(empty = branch from blank)" : ""}
          onkeydown={(ev) => {
            if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
              void commitModal();
            }
            // Prevent browser bold-formatting on Ctrl+B inside the
            // textarea for branch buffers.
            if (ev.key === "b" && (ev.metaKey || ev.ctrlKey)) {
              ev.preventDefault();
            }
          }}
        ></textarea>
        <p class="hint">⌃⏎ / ⌘⏎</p>
      {/if}
      {#if modal.error}
        <p class="modal-error" role="alert">{modal.error}</p>
      {/if}
    </div>
    <footer class="modal-footer">
      <button
        type="button"
        class="cancel"
        bind:this={modalCancelButton}
        onclick={closeModal}
      >cancel</button>
      <button
        type="button"
        class={modal.kind === "delete" || modal.kind === "clear" ? "danger" : "primary"}
        onclick={() => void commitModal()}
        disabled={modalBusy || (modal.kind === "clear" && genStatus.active) || (modal.kind === "delete" && modal.nodeId !== null &&
          loomNodeIntersectsGeneration(modal.nodeId))}
        {...{ "aria-description": (modal.kind === "delete" && modal.nodeId !== null &&
          loomNodeIntersectsGeneration(modal.nodeId)
          ? LOOM_DELETE_DURING_GENERATION_MESSAGE
          : "") }}
      >
        <MorphText text={modalBusy ? (modal.kind === "branch" ? "Creating branch…" : "Applying…")
          : modal.kind === "delete" ? "Delete branch"
          : modal.kind === "clear" ? "Save and clear loom"
          : modal.kind === "edit" || modal.kind === "note" ? "Save"
          : modal.kind === "branch" ? "Create branch"
          : modal.kind === "navpicker" ? "Go"
          : modal.kind === "search" ? "Search" : "Generate"} numbers={false} />
      </button>
    </footer>
  </div>
{/if}

<style>
  .loom-modal-backdrop {
    position: fixed;
    inset: 0;
    background: var(--scrim-strong);
    z-index: var(--z-modal);
    border: 0;
    cursor: pointer;
  }
  .loom-modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: calc(var(--z-modal) + 1);
    background: var(--surface-sheen), var(--popup-bg);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    box-shadow: var(--popup-shadow);
    width: min(640px, calc(100vw - 2rem));
    min-width: 0;
    max-height: calc(100dvh - 2rem);
    display: flex;
    flex-direction: column;
    font-family: var(--font-reading);
    font-size: var(--text);
    color: var(--fg);
  }
  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    color: var(--accent);
    text-transform: lowercase;
    letter-spacing: 0;
  }
  .modal-header h2 {
    margin: 0;
    font: inherit;
    font-weight: var(--weight-medium);
  }
  .modal-body {
    padding: var(--panel-padding) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
    overflow-y: auto;
  }
  .modal-body label {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .modal-body input,
  .modal-body textarea {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-reading);
    resize: vertical;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
  }
  .modal-body input:focus,
  .modal-body textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .modal-body blockquote {
    margin: 0;
    overflow-wrap: anywhere;
    color: var(--accent-yellow);
    background: var(--surface-sheen), var(--bg-elev);
    padding: var(--surface-padding);
    border-radius: var(--radius);
  }
  .modal-body .danger {
    color: var(--accent-red);
  }
  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: var(--space-3);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .modal-footer button {
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius);
    color: var(--fg-strong);
    padding: var(--space-3) var(--space-5);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-mono);
  }
  .modal-footer .primary {
    background: var(--accent);
    color: var(--text-on-accent);
  }
  .modal-footer .primary:hover {
    background: var(--accent-light);
  }
  .modal-footer .danger {
    background: var(--danger-bg);
    color: var(--accent-red);
  }
  .modal-footer .danger:hover {
    background: var(--danger-hover);
  }
  .modal-footer .cancel:hover {
    background: var(--glass-strong);
  }
  .modal-error {
    font-size: var(--text-sm);
    margin-top: var(--space-3);
    color: var(--error-text);
    background: var(--surface-sheen), var(--popup-bg);
    border-radius: var(--radius-sm);
    padding: var(--surface-padding);
    font-family: var(--font-mono);
  }
  @media (prefers-contrast: more) {
    .modal-error { background: var(--bg); }
  }
  @media (forced-colors: active) {
    .modal-error { background: Canvas; color: CanvasText; }
  }
  @media (max-width: 760px), (max-height: 600px) {
    .modal-body input,
    .modal-body textarea {
      font-size: var(--text-input-touch);
    }
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
</style>
