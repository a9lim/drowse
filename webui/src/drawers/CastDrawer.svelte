<script lang="ts">
  import { onDestroy } from "svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Cast manager (phase 3 of the cast model) — the tree's roster of
  // named labels, each with a standing steering recipe.  A member's
  // recipe is the *weakest* tier at generation: it fills only fields
  // the send left unset, so the rack and per-send controls always win.
  // Identity is auto-derived from labels observed anywhere in the tree;
  // configuration adds only a standing recipe/notes layer.  Mutations and
  // new observed labels reconcile through the inlined effective roster.
  //
  // This drawer also owns the two active structural-role labels.  The
  // composer chooses between those labels without separately exposing their
  // canonical user/assistant roles; a turn labeled with a member's slug
  // generates under its recipe.

  import { apiTree } from "../lib/runtime/services";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import {
    castState,
    pushToast,
    samplingState,
    sessionState,
  } from "../lib/stores.svelte";
  import { closeDrawer } from "../lib/stores/drawers.svelte";
  import InfoTip from "../lib/ui/InfoTip.svelte";
  import type { CastMemberJSON } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let steering = $state("");
  let notes = $state("");
  let editingLabel = $state<string | null>(null);
  let editingKey = $state<string | null>(null);
  let busy = $state(false);
  let err = $state<string | null>(null);
  let lastSavedKey = $state<string | null>(null);
  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  let saveRequest: Promise<boolean> | null = null;

  const defaultUserRole = $derived(
    sessionState.info?.default_user_role ?? "user",
  );
  const defaultAssistantRole = $derived(
    sessionState.info?.default_assistant_role ?? "assistant",
  );
  const activeRoles = $derived.by(() => [...new Set([
    defaultUserRole,
    defaultAssistantRole,
    samplingState.user_role.trim(),
    samplingState.assistant_role.trim(),
  ].filter(Boolean))]);
  // The server roster is keyed by Drowse's structural roles. Present those
  // two entries through the model's actual chat-template vocabulary, while
  // retaining the hidden key so structural cast recipes still round-trip.
  const roster = $derived.by(() => {
    const byLabel = new Map<
      string,
      { key: string; label: string; member: CastMemberJSON | null }
    >();
    for (const [key, member] of Object.entries(castState.roster)) {
      const displayLabel =
        key === "user"
          ? defaultUserRole
          : key === "assistant"
            ? defaultAssistantRole
            : key;
      const existing = byLabel.get(displayLabel);
      const structural = key === "user" || key === "assistant";
      if (!existing || structural) {
        byLabel.set(displayLabel, { key, label: displayLabel, member });
      }
    }
    for (const role of activeRoles) {
      if (byLabel.has(role)) continue;
      const key = role === defaultUserRole
        ? "user"
        : role === defaultAssistantRole
          ? "assistant"
          : role;
      byLabel.set(role, {
        key,
        label: role,
        member: castState.roster[key] ?? null,
      });
    }
    const standard = activeRoles
      .map((role) => byLabel.get(role))
      .filter((row) => row !== undefined);
    const custom = [...byLabel.values()]
      .filter((row) => !activeRoles.includes(row.label))
      .sort((a, b) => a.label.localeCompare(b.label));
    return [...standard, ...custom];
  });

  const selectedMember = $derived(
    editingKey ? (castState.roster[editingKey] ?? null) : null,
  );
  const dirty = $derived(
    editingKey !== null && (
      steering.trim() !== (selectedMember?.recipe?.steering ?? "")
      || notes.trim() !== (selectedMember?.notes ?? "")
    ),
  );
  const saveStatus = $derived(
    busy
      ? "Saving…"
      : dirty
        ? "Changes save automatically"
        : lastSavedKey === editingKey
          ? "Saved"
          : "Changes save automatically",
  );

  function loadMember(row: (typeof roster)[number]): void {
    editingLabel = row.label;
    editingKey = row.key;
    steering = row.member?.recipe?.steering ?? "";
    notes = row.member?.notes ?? "";
    err = null;
  }

  function cancelScheduledSave(): void {
    if (saveTimer === null) return;
    clearTimeout(saveTimer);
    saveTimer = null;
  }

  function scheduleSave(): void {
    err = null;
    lastSavedKey = null;
    cancelScheduledSave();
    saveTimer = setTimeout(() => {
      saveTimer = null;
      void saveCurrent();
    }, 450);
  }

  $effect(() => {
    const rows = roster;
    if (editingKey !== null && rows.some((row) => row.key === editingKey)) return;
    if (rows[0]) loadMember(rows[0]);
  });

  async function saveCurrent(): Promise<boolean> {
    cancelScheduledSave();
    if (saveRequest) {
      const saved = await saveRequest;
      if (!saved) return false;
    }
    const target = editingKey;
    if (!target || !dirty) return true;
    const targetLabel = editingLabel;
    const nextSteering = steering.trim();
    const nextNotes = notes.trim();
    busy = true;
    err = null;
    const request = (async (): Promise<boolean> => {
      try {
        const r = await apiTree.castPut(target, {
          steering: nextSteering === "" ? null : nextSteering,
          notes: nextNotes,
        });
        // Optimistic merge — the ``op="cast"`` frame confirms shortly.
        castState.roster = { ...castState.roster, [target]: r.member };
        if (editingKey === target) lastSavedKey = target;
        return true;
      } catch (e) {
        err = userFacingError(e, `Unable to save ${targetLabel ?? "this role"}. Check the fields and try again.`);
        return false;
      } finally {
        busy = false;
      }
    })();
    saveRequest = request;
    const saved = await request;
    if (saveRequest === request) saveRequest = null;
    if (saved && editingKey === target && dirty) {
      return saveCurrent();
    }
    return saved;
  }

  async function selectMember(row: (typeof roster)[number]): Promise<void> {
    if (row.key === editingKey) return;
    if (!(await saveCurrent())) return;
    loadMember(row);
  }

  async function closeWithSave(): Promise<void> {
    if (await saveCurrent()) closeDrawer();
  }

  async function remove(slug: string): Promise<void> {
    try {
      await apiTree.castDelete(slug);
      const nextRoster = { ...castState.roster };
      delete nextRoster[slug];
      castState.roster = nextRoster;
      if (editingKey === slug) {
        steering = "";
        notes = "";
        lastSavedKey = null;
      }
    } catch (e) {
      pushToast(userFacingError(e, "Unable to remove this speaker. Try again."), {
        kind: "error",
      });
    }
  }

  async function clearSelectedSettings(): Promise<void> {
    cancelScheduledSave();
    if (saveRequest && !(await saveRequest)) return;
    if (editingKey) await remove(editingKey);
  }

  onDestroy(() => {
    cancelScheduledSave();
    if (dirty) void saveCurrent();
  });
</script>

<section class="drawer-shell" aria-label="Role settings drawer">
  <header class="header">
    <div>
      <h2 class="title">Role settings</h2>
    </div>
    <DrawerCloseButton onclick={() => void closeWithSave()} />
  </header>

  <div class="body">
    <section class="panel role-picker" aria-labelledby="conversation-roles-title">
      <div class="section-heading">
        <div>
          <h3 id="conversation-roles-title">Conversation roles</h3>
        </div>
      </div>
      <ul class="roster" aria-label="Conversation roles">
        {#each roster as row (row.label)}
          <li class="member" class:editing={editingKey === row.key}>
            <button
              type="button"
              class="member-main"
              aria-pressed={editingKey === row.key}
              onclick={() => void selectMember(row)}
            >
              <span class="glyph" aria-hidden="true">{row.label.slice(0, 1).toUpperCase()}</span>
              <span class="member-text">
                <span class="member-label">{row.label}</span>
                {#if row.member?.recipe?.steering}
                  <span class="member-recipe">Custom guidance</span>
                {:else}
                  <span class="member-notes">Uses the current reply settings</span>
                {/if}
                {#if row.member?.notes}
                  <span class="member-notes">{row.member.notes}</span>
                {/if}
              </span>
            </button>
          </li>
        {/each}
      </ul>
    </section>

    {#if editingKey && editingLabel}
      <section class="panel form" aria-labelledby="selected-role-title">
        <div class="section-heading">
          <div>
            <h3 id="selected-role-title">{editingLabel}</h3>
          </div>
        </div>
        <div class="field">
          <span class="field-heading">
            <span class="label">Response guidance</span>
            <InfoTip
              label="About role guidance"
              text="Guidance used whenever this role writes. Settings for one reply take priority."
            />
          </span>
          <input
            class="input mono"
            bind:value={steering}
            placeholder="Leave blank for normal behavior"
            spellcheck="false"
            autocomplete="off"
            aria-label={`Response guidance for ${editingLabel}`}
            oninput={scheduleSave}
          />
        </div>
        <label class="field">
          <span class="label">Private note</span>
          <input
            class="input"
            bind:value={notes}
            placeholder="What should you remember about this role?"
            autocomplete="off"
            oninput={scheduleSave}
          />
        </label>
        {#if err}
          <p class="error" role="alert">{err}</p>
        {/if}
        <div class="form-actions">
          <p class="save-status" role="status" aria-live="polite">{saveStatus}</p>
          {#if selectedMember?.origin === "configured"}
            <button
              type="button"
              class="btn quiet"
              disabled={busy}
              onclick={() => void clearSelectedSettings()}
            >Clear settings</button>
          {/if}
        </div>
      </section>
    {/if}
  </div>
</section>

<style>
  .drawer-shell {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-6);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    color: var(--accent);
    font-family: var(--font-structure);
    font-size: var(--text-md);
    font-weight: var(--weight-structure);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: grid;
    grid-auto-rows: max-content;
    align-content: start;
    gap: var(--drawer-section-gap);
    min-height: 0;
  }
  .panel {
    padding: var(--panel-padding);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--glass);
    box-shadow: var(--shadow-well);
  }
  .section-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-4);
    margin-bottom: var(--space-5);
  }
  h3 {
    margin: 0;
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text);
    font-weight: var(--weight-structure);
  }
  .roster {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .member {
    min-width: 0;
    border-radius: var(--radius);
  }
  .member-main {
    width: 100%;
    min-height: var(--control-field);
    display: flex;
    align-items: center;
    gap: var(--space-3);
    background: var(--bg-elev);
    border: 1px solid transparent;
    border-radius: var(--radius);
    color: inherit;
    font: inherit;
    text-align: start;
    padding: var(--surface-padding);
    cursor: pointer;
    min-width: 0;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .member-main:hover {
    background: var(--surface-hi);
  }
  .member-main:active {
    transform: scale(var(--press-scale));
  }
  .member-main[aria-pressed="true"] {
    background: var(--accent-subtle);
    border-color: var(--accent-strong);
  }
  .glyph {
    flex: none;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--glass-bright);
    color: var(--fg-strong);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .member-text {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
    min-width: 0;
  }
  .member-label {
    font-family: var(--font-mono);
    color: var(--fg-strong);
  }
  .member-recipe,
  .member-notes {
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .member-recipe { color: var(--accent); }
  .member-notes {
    color: var(--fg-muted);
  }
  .form {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-strong);
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
  }
  .field-heading {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .input {
    min-height: var(--control-field);
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    font: inherit;
  }
  .input.mono {
    font-family: var(--font-mono);
  }
  .input:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 1px;
  }
  .error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
  }
  .form-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: var(--space-3);
    margin-top: var(--space-2);
  }
  .save-status {
    margin: 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .btn {
    min-height: var(--control-field);
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-5);
    font: inherit;
    border-radius: var(--radius);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
    cursor: pointer;
    transition:
      background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .btn:hover:not(:disabled) {
    background: var(--glass-strong);
  }
  .btn.quiet {
    background: transparent;
    color: var(--fg-muted);
  }
  .btn.quiet:hover:not(:disabled) {
    color: var(--accent-red);
  }
  .btn:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }

  @media (max-width: 560px) {
    .roster { grid-template-columns: minmax(0, 1fr); }
  }
</style>
