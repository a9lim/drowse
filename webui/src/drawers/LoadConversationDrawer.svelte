<script lang="ts">
  import BaseModelTag from "../lib/ui/BaseModelTag.svelte";
  import { chatAccentStyle } from "../lib/chatAccent";
  import { flushConversationAutosave } from "../lib/stores/savedConversations.svelte";
  import { Blobatar } from "@blobatar/svelte";
  import { onMount, tick } from "svelte";
  import { downloadChatBackup, importChatBackup } from "../lib/chatBackup";

  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import {
    conversationLibrary,
    closeDrawer,
    genStatus,
    openDrawer,
    requestPersistentConversationStorage,
    savedConversationState,
    sessionState,
  } from "../lib/stores.svelte";
  import {
    displayModelName,
    randomAvatarSeed,
    summarizeConversation,
    type SavedConversationIssue,
    type SavedConversationRecord,
    type SavedConversationSummary,
  } from "../lib/conversationLibrary";
  import { restoreConversationSnapshot } from "../lib/conversationWorkspace";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { pushToast } from "../lib/stores/toasts.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => void _drawerProps.params);

  let fileInput: HTMLInputElement | null = $state(null);
  let conversations: SavedConversationSummary[] = $state([]);
  let issues: SavedConversationIssue[] = $state([]);
  let query = $state("");
  let loading = $state(true);
  let error = $state<string | null>(null);
  let openingId = $state<string | null>(null);
  let changingId = $state<string | null>(null);
  let duplicatingId = $state<string | null>(null);
  let confirmDeleteId = $state<string | null>(null);
  let editingId = $state<string | null>(null);
  let editingName = $state("");
  let importing = $state(false);
  let renameInput: HTMLInputElement | null = $state(null);
  let persistenceState: "checking" | "protected" | "unprotected" | "unavailable" = $state("checking");
  let persistenceRequesting = $state(false);
  let persistenceAttempted = $state(false);

  const currentModelId = $derived(sessionState.info?.model_id ?? null);
  const filtered = $derived.by(() => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return conversations;
    return conversations.filter((record) =>
      record.name.toLocaleLowerCase().includes(needle) ||
      displayModelName(record.modelId).toLocaleLowerCase().includes(needle)
    );
  });

  onMount(() => {
    void refresh();
    void refreshPersistenceState();
  });

  async function refreshPersistenceState(): Promise<void> {
    if (!navigator.storage?.persisted) {
      persistenceState = "unavailable";
      return;
    }
    try {
      persistenceState = await navigator.storage.persisted()
        ? "protected"
        : "unprotected";
    } catch {
      persistenceState = "unavailable";
    }
  }

  async function protectStorage(): Promise<void> {
    persistenceRequesting = true;
    try {
      const protectedStorage = await requestPersistentConversationStorage();
      persistenceState = protectedStorage === true
        ? "protected"
        : protectedStorage === false
          ? "unprotected"
          : "unavailable";
      persistenceAttempted = protectedStorage !== true;
      if (protectedStorage) pushToast("Saved chats are protected from automatic browser cleanup.", { kind: "info" });
    } finally {
      persistenceRequesting = false;
    }
  }

  async function refresh(): Promise<void> {
    loading = true;
    error = null;
    try {
      const result = await conversationLibrary.listSummaries();
      conversations = result.conversations;
      issues = result.issues;
    } catch (cause) {
      error = userFacingError(cause, "Saved conversations could not be opened.");
    } finally {
      loading = false;
    }
  }

  async function openConversation(record: SavedConversationSummary): Promise<void> {
    if (record.modelId !== currentModelId || genStatus.active) return;
    openingId = record.id;
    error = null;
    try {
      await flushConversationAutosave();
      const current = await conversationLibrary.get(record.id);
      await restoreConversationSnapshot(current.snapshot);
      savedConversationState.activeId = record.id;
      savedConversationState.avatarSeed = current.avatarSeed;
      savedConversationState.accent = current.accent ?? "purple";
      pushToast(`Opened “${record.name}”.`, { kind: "info" });
      closeDrawer();
    } catch (cause) {
      error = userFacingError(
        cause,
        "This conversation could not be opened. Your current work was left unchanged.",
      );
    } finally {
      openingId = null;
    }
  }

  async function regenerateAvatar(record: SavedConversationSummary): Promise<void> {
    changingId = record.id;
    error = null;
    try {
      const updated = await conversationLibrary.update(record.id, {
        avatarSeed: randomAvatarSeed(),
      });
      if (savedConversationState.activeId === record.id) savedConversationState.avatarSeed = updated.avatarSeed;
      replaceConversation(updated);
    } catch (cause) {
      error = userFacingError(cause, "The avatar could not be changed.");
    } finally {
      changingId = null;
    }
  }

  function beginRename(record: SavedConversationSummary): void {
    confirmDeleteId = null;
    editingId = record.id;
    editingName = record.name;
    void tick().then(() => {
      renameInput?.focus();
      renameInput?.select();
    });
  }

  async function saveRename(record: SavedConversationSummary): Promise<void> {
    changingId = record.id;
    error = null;
    try {
      const updated = await conversationLibrary.update(record.id, { name: editingName });
      replaceConversation(updated);
      editingId = null;
    } catch (cause) {
      error = userFacingError(cause, "The conversation could not be renamed.");
    } finally {
      changingId = null;
    }
  }

  function onRenameKey(event: KeyboardEvent, record: SavedConversationSummary): void {
    if (event.key === "Enter") {
      event.preventDefault();
      if (editingName.trim() && changingId !== record.id) void saveRename(record);
    } else if (event.key === "Escape") {
      event.preventDefault();
      editingId = null;
    }
  }

  function requestDelete(id: string): void {
    editingId = null;
    confirmDeleteId = id;
  }

  async function deleteConversation(id: string): Promise<void> {
    changingId = id;
    error = null;
    try {
      await conversationLibrary.delete(id);
      conversations = conversations.filter((record) => record.id !== id);
      issues = issues.filter((issue) => issue.id !== id);
      confirmDeleteId = null;
      pushToast("Saved conversation deleted.", { kind: "info" });
    } catch (cause) {
      error = userFacingError(cause, "The saved conversation could not be deleted.");
    } finally {
      changingId = null;
    }
  }

  async function download(summary: SavedConversationSummary): Promise<void> {
    changingId = summary.id;
    error = null;
    try {
      await flushConversationAutosave();
      await downloadChatBackup(conversationLibrary, summary.id);
    } catch (cause) {
      error = userFacingError(cause, "This conversation could not be exported. Your saved chat was not changed.");
    } finally {
      changingId = null;
    }
  }

  async function importFile(event: Event): Promise<void> {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0] ?? null;
    target.value = "";
    if (!file || importing) return;
    importing = true;
    error = null;
    try {
      const record = await importChatBackup(file, conversationLibrary);
      conversations = [summarizeConversation(record), ...conversations];
      pushToast(`Imported “${record.name}” as a separate chat.`, { kind: "info" });
      void requestPersistentConversationStorage();
    } catch (cause) {
      error = userFacingError(cause, "This conversation file could not be imported.");
    } finally { importing = false; }
  }

  async function duplicateConversation(record: SavedConversationSummary): Promise<void> {
    if (changingId !== null || importing || openingId !== null) return;
    changingId = record.id;
    duplicatingId = record.id;
    error = null;
    try {
      await flushConversationAutosave();
      const copy = await conversationLibrary.duplicate(record.id);
      conversations = [summarizeConversation(copy), ...conversations];
      pushToast(`Created “${copy.name}”. The original chat is unchanged.`, { kind: "info" });
    } catch (cause) {
      error = userFacingError(cause, "The chat could not be duplicated. The original chat was not changed.");
    } finally {
      changingId = null;
      duplicatingId = null;
    }
  }

  function replaceConversation(updated: SavedConversationRecord): void {
    conversations = conversations.map((record) => record.id === updated.id ? summarizeConversation(updated) : record);
  }

  function relativeTime(timestamp: number): string {
    const elapsed = timestamp - Date.now();
    const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
      ["year", 365 * 24 * 60 * 60 * 1000],
      ["month", 30 * 24 * 60 * 60 * 1000],
      ["day", 24 * 60 * 60 * 1000],
      ["hour", 60 * 60 * 1000],
      ["minute", 60 * 1000],
    ];
    for (const [unit, duration] of units) {
      if (Math.abs(elapsed) >= duration || unit === "minute") {
        const value = Math.round(elapsed / duration);
        return new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(value, unit);
      }
    }
    return "just now";
  }

  function absoluteTime(timestamp: number): string {
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" })
      .format(new Date(timestamp));
  }
</script>

<section class="drawer-shell" aria-label="Saved conversations drawer">
  <header class="header">
    <div>
      <h2 class="title">Saved chats</h2>
      <p>Named conversations stay on this device until you delete them. Click an avatar for a new one.</p>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="toolbar">
    <label class="search-field">
      <span class="sr-only">Search saved chats</span>
      <input bind:value={query} type="search" placeholder="Search saved chats…" />
    </label>
    <input
      bind:this={fileInput}
      class="file-input"
      type="file"
      accept=".drowsechat,.polytheticchat,.saklaschat,.json,application/json"
      aria-label="Import Drowse conversation file"
      onchange={importFile}
      tabindex="-1"
    />
    <div class="toolbar-actions">
      <button
        type="button"
        class="button primary"
        disabled={genStatus.active}
        onclick={() => openDrawer("save_conversation")}
      >Save current chat</button>
      <button type="button" class="button" disabled={importing} aria-busy={importing} onclick={() => fileInput?.click()}>{importing ? "Importing…" : "Import file"}</button>
    </div>
  </div>

  {#if !loading && conversations.length > 0 && persistenceState === "unprotected"}
    <div class="storage-protection" role="status">
      <div>
        <strong>Protect saved chats</strong>
        <p>{persistenceAttempted
          ? "Your browser did not grant protection. Your chats are still saved locally."
          : "Automatic browser cleanup protection is off."}</p>
      </div>
      <button
        type="button"
        class="quiet-button"
        disabled={persistenceRequesting}
        onclick={() => void protectStorage()}
      >{persistenceRequesting ? "Checking…" : persistenceAttempted ? "Try again" : "Protect storage"}</button>
    </div>
  {/if}

  <div class="body">
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    {#if genStatus.active}<p class="notice">Wait for the current reply to finish before opening another chat.</p>{/if}

    {#if loading}
      <p class="empty">Loading saved chats…</p>
    {:else if filtered.length === 0 && issues.length === 0}
      <div class="empty-state">
        <strong>{query.trim() ? "No matches" : "No saved chats yet"}</strong>
        <p>{query.trim() ? "Try a different name or model." : "Save the current chat to name it and keep its full loom."}</p>
      </div>
    {:else}
      <div class="conversation-list" role="list" aria-label="Saved conversations">
        {#each filtered as record (record.id)}
          {@const matchesModel = record.modelId === currentModelId}
          <article
            class="conversation-card"
            role="listitem"
            class:current={savedConversationState.activeId === record.id}
            data-saved-conversation={record.id}
            data-chat-accent={record.accent ?? "purple"}
            style={chatAccentStyle(record.accent)}
          >
            <div class="card-main">
              <button
                type="button"
                class="avatar-button"
                aria-busy={changingId === record.id}
                onclick={() => void regenerateAvatar(record)}
                disabled={changingId === record.id}
                aria-label={`Generate another avatar for ${record.name}`}
                title="Generate another avatar"
              >
                <Blobatar name={record.avatarSeed} size={62} background="circle" alt="" />
              </button>

              <div class="card-copy">
                {#if editingId === record.id}
                  <label class="rename-field">
                    <span class="sr-only">Conversation name</span>
                    <input
                      bind:this={renameInput}
                      bind:value={editingName}
                      maxlength="120"
                      onkeydown={(event) => onRenameKey(event, record)}
                      disabled={changingId === record.id}
                    />
                  </label>
                {:else}
                  <strong title={record.name} dir="auto">{record.name}</strong>
                {/if}
                <span class="card-meta">
                  <span class="model">{displayModelName(record.modelId)}{#if record.modelType === "base" || (matchesModel && sessionState.info?.is_base_model)}<BaseModelTag />{/if}</span>
                  {#if savedConversationState.activeId === record.id}<span class="current-label">Open now</span>{/if}
                </span>
              </div>

              <time datetime={new Date(record.updatedAt).toISOString()} title={absoluteTime(record.updatedAt)}>
                {relativeTime(record.updatedAt)}
              </time>
            </div>

            {#if editingId === record.id}
              <div class="card-actions">
                <button type="button" class="quiet-button" onclick={() => (editingId = null)}>Cancel</button>
                <button
                  type="button"
                  class="button compact"
                  onclick={() => void saveRename(record)}
                  disabled={!editingName.trim() || changingId === record.id}
                >Save name</button>
              </div>
            {:else if confirmDeleteId === record.id}
              <div class="confirm-row" role="group" aria-label={`Confirm deletion of ${record.name}`}>
                <p>Delete “{record.name}”? This cannot be undone.</p>
                <button type="button" class="quiet-button" onclick={() => (confirmDeleteId = null)}>Cancel</button>
                <button
                  type="button"
                  class="danger-button"
                  onclick={() => void deleteConversation(record.id)}
                  disabled={changingId === record.id}
                >Delete</button>
              </div>
            {:else}
              <div class="card-actions">
                <button
                  type="button"
                  class="button primary compact"
                  onclick={() => void openConversation(record)}
                  disabled={!matchesModel || genStatus.active || openingId !== null}
                  title={matchesModel ? undefined : `Load ${displayModelName(record.modelId)} to open this chat`}
                >{openingId === record.id ? "Opening…" : matchesModel ? "Open" : "Different model"}</button>
                <button type="button" class="quiet-button" onclick={() => beginRename(record)}>Rename</button>
                <button type="button" class="quiet-button" disabled={changingId !== null} onclick={() => void download(record)}>Download</button>
                <button type="button" class="quiet-button" disabled={changingId !== null || importing || openingId !== null}
                  aria-label={`Duplicate ${record.name}`} aria-busy={duplicatingId === record.id}
                  onclick={() => void duplicateConversation(record)}>Duplicate</button>
                <button type="button" class="quiet-button danger-text" onclick={() => requestDelete(record.id)}>Delete</button>
              </div>
            {/if}
          </article>
        {/each}

        {#each issues as issue (issue.id)}
          <article class="conversation-card unavailable" role="listitem">
            <div class="card-main">
              <div class="unavailable-avatar" aria-hidden="true">?</div>
              <div class="card-copy">
                <strong>{issue.name || "Unavailable saved chat"}</strong>
                <span class="model">The saved data is still in storage.</span>
              </div>
            </div>
            {#if confirmDeleteId === issue.id}
              <div class="confirm-row" role="group" aria-label="Confirm deletion of unavailable saved chat">
                <p>Delete this saved data? This cannot be undone.</p>
                <button type="button" class="quiet-button" onclick={() => (confirmDeleteId = null)}>Cancel</button>
                <button type="button" class="danger-button" onclick={() => void deleteConversation(issue.id)}>Delete</button>
              </div>
            {:else}
              <div class="card-actions">
                <span
                  class="issue-reason"
                  title={userFacingError(issue.reason, "This saved chat could not be read. The data is still in storage.")}
                >Could not read this chat</span>
                <button type="button" class="quiet-button danger-text" onclick={() => requestDelete(issue.id)}>Delete</button>
              </div>
            {/if}
          </article>
        {/each}
      </div>
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
  }
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .header div { min-width: 0; }
  .title { margin: 0; color: var(--fg); font-size: var(--text-lg); font-weight: var(--weight-medium); }
  .header p { margin: var(--space-2) 0 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.45; }
  .toolbar {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: var(--space-2);
    padding: 0 var(--drawer-gutter-inline) var(--space-4);
  }
  .toolbar-actions {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .search-field { min-width: 0; }
  .search-field input {
    width: 100%;
    min-height: var(--control-field);
    box-sizing: border-box;
    border: 0;
    border-radius: var(--radius);
    background: var(--input-well);
    box-shadow: var(--shadow-well);
    color: var(--fg);
    padding: var(--space-3) var(--space-4);
    font: inherit;
  }
  .search-field input:focus { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  .body {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding: 0 var(--drawer-gutter-inline) var(--drawer-gutter-block);
  }
  .storage-protection {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-3);
    margin: 0 var(--drawer-gutter-inline) var(--space-4);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--warning-bg);
    box-shadow: inset 0 0 0 1px var(--warning-border);
  }
  .storage-protection strong { color: var(--warning-ink); font-size: var(--text-sm); }
  .storage-protection p { margin: var(--space-1) 0 0; color: var(--warning-ink); font-size: var(--text-xs); line-height: 1.4; }
  .storage-protection .quiet-button { color: var(--warning-action-ink); background: var(--warning-action); box-shadow: none; }
  .storage-protection .quiet-button:hover:not(:disabled) { background: var(--warning-action-hover); }
  .conversation-list { display: grid; gap: var(--space-3); }
  .conversation-card {
    min-width: 0;
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--glass);
    box-shadow: var(--shadow-rack);
    transition: background-color 140ms ease, box-shadow 140ms ease;
  }
  .conversation-card:hover { background: var(--surface-sheen), var(--glass-strong); box-shadow: var(--shadow-rack-hover); }
  .conversation-card.current { box-shadow: var(--shadow-card-active); }
  .card-main {
    min-width: 0;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-3);
  }
  .avatar-button, .unavailable-avatar {
    width: 64px;
    height: 64px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    padding: 0;
    border: 0;
    border-radius: 50%;
    background: var(--bg-elev);
    box-shadow: var(--shadow-control);
  }
  .avatar-button {
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease;
  }
  .avatar-button :global(img) { display: block; width: 100%; height: 100%; min-width: 0; border-radius: inherit; }
  .avatar-button:hover:not(:disabled) { box-shadow: var(--shadow-control-hover); }
  .avatar-button:active:not(:disabled) { transform: scale(0.96); }
  .avatar-button:disabled { opacity: 0.55; cursor: wait; }
  .avatar-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 3px; }
  .unavailable-avatar { color: var(--fg-muted); font-family: var(--font-mono); }
  .card-copy { min-width: 0; display: grid; gap: var(--space-xs); }
  .card-copy strong {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--fg);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .model { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg-dim); font-size: var(--text-sm); }
  .card-meta { display: flex; align-items: center; flex-wrap: wrap; gap: var(--space-2); min-width: 0; }
  .current-label {
    flex: 0 0 auto;
    padding: var(--space-xs) var(--space-xs);
    border-radius: var(--radius-pill);
    background: color-mix(in srgb, var(--live) 12%, transparent);
    color: var(--live);
    font-family: var(--font-data);
    font-size: var(--text-2xs);
  }
  time { align-self: start; color: var(--fg-muted); font-size: var(--text-xs); white-space: nowrap; }
  .card-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-4);
    margin-block-start: var(--space-4);
    margin-inline-start: calc(var(--space-unit) * 10);
  }
  .button, .quiet-button, .danger-button {
    min-width: var(--control-target);
    min-height: var(--control-field);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-size: var(--text-sm);
    cursor: pointer;
    transition: background-color 120ms ease, color 120ms ease, transform 120ms ease;
  }
  .button { background: var(--glass-strong); color: var(--fg-strong); box-shadow: var(--shadow-control); }
  .button:hover:not(:disabled) { background: var(--glass-bright); }
  .button.primary { background: var(--action-bg); color: var(--action-ink); }
  .button.primary:hover:not(:disabled) { background: var(--action-hover); }
  .button.compact { padding: var(--space-2) var(--space-4); }
  .quiet-button { background: transparent; color: var(--fg-dim); }
  .quiet-button:hover:not(:disabled) { background: var(--bg-hover); color: var(--fg); }
  .danger-text { color: var(--accent-red); }
  .danger-button { background: var(--danger-bg); color: var(--accent-red); }
  .danger-button:hover:not(:disabled) { background: var(--danger-hover); }
  .button:active:not(:disabled), .quiet-button:active:not(:disabled), .danger-button:active:not(:disabled) { transform: scale(0.96); }
  .button:disabled, .quiet-button:disabled, .danger-button:disabled { opacity: 0.45; cursor: not-allowed; }
  .button:focus-visible, .quiet-button:focus-visible, .danger-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .rename-field { display: block; min-width: 0; }
  .rename-field input {
    width: 100%;
    min-width: 0;
    min-height: var(--control-field);
    box-sizing: border-box;
    border: 0;
    border-radius: var(--radius);
    background: var(--input-well);
    box-shadow: var(--shadow-well);
    color: var(--fg);
    padding: var(--space-2) var(--space-3);
    font: inherit;
  }
  .rename-field input:focus { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  .confirm-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    align-items: center;
    gap: var(--space-2);
    margin-top: var(--space-3);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--surface-sheen), color-mix(in srgb, var(--accent-red) 8%, var(--input-well));
  }
  .confirm-row p { margin: 0; color: var(--fg-strong); font-size: var(--text-sm); line-height: 1.4; }
  .issue-reason { margin-inline-end: auto; color: var(--accent-red); font-size: var(--text-sm); }
  .notice, .error, .empty { margin: 0 0 var(--space-3); color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.45; }
  .notice { color: var(--accent-amber); }
  .error { color: var(--accent-red); }
  .empty-state {
    display: grid;
    gap: var(--space-2);
    place-items: center;
    padding: var(--space-8) var(--space-5);
    text-align: center;
    color: var(--fg-dim);
  }
  .empty-state strong { color: var(--fg); font-size: var(--text-md); }
  .empty-state p { margin: 0; max-width: 34ch; font-size: var(--text-sm); line-height: 1.5; }
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
  .file-input { display: none; }
  @media (max-width: 520px) {
    .toolbar { grid-template-columns: 1fr; }
    .toolbar-actions { display: grid; grid-template-columns: minmax(0, 1fr) auto; }
    .storage-protection { grid-template-columns: 1fr; }
    .storage-protection .quiet-button { width: 100%; }
    .avatar-button, .unavailable-avatar { width: 56px; height: 56px; }
    .card-actions { margin-inline-start: 0; }
    .confirm-row { grid-template-columns: 1fr auto auto; }
    .confirm-row p { grid-column: 1 / -1; }
  }
</style>
