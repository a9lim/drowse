<script lang="ts">
  import { registerInterfaceController } from "../lib/workspaceController";
  import { objectSchema, ToolError } from "../lib/webmcp/types";

  import MorphText from "../lib/ui/MorphText.svelte";
  import { Blobatar } from "@blobatar/svelte";
  import { onMount } from "svelte";
  import ChatAccentPicker from "../lib/ui/ChatAccentPicker.svelte";
  import { CHAT_ACCENTS, type ChatAccent } from "../lib/chatAccent";

  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import {
    conversationLibrary,
    closeDrawer,
    genStatus,
    requestPersistentConversationStorage,
    savedConversationState,
  } from "../lib/stores.svelte";
  import {
    defaultConversationName,
    displayModelName,
    randomAvatarSeed,
    ConversationLibraryError,
    SAVED_CONVERSATION_SCHEMA_VERSION,
    type SavedConversationRecord,
  } from "../lib/conversationLibrary";
  import { captureConversationSnapshot } from "../lib/conversationWorkspace";
  import { downloadPreparedChatBackup, encodeChatBackup } from "../lib/chatBackup";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import { flushConversationAutosave, onConversationLibraryChanged } from "../lib/stores/savedConversations.svelte";

  let { embedded = false }: { params?: unknown; embedded?: boolean } = $props();

  let draftBaseline = $state("");
  let current: SavedConversationRecord | null = $state(null);
  let name = $state("");
  let avatarSeed = $state(randomAvatarSeed());
  let accent: ChatAccent = $state("purple");
  const metadataDirty = $derived(draftBaseline !== "" && JSON.stringify([name, avatarSeed, accent]) !== draftBaseline);

  let error = $state<string | null>(null);
  let loading = $state(true);
  let saving = $state(false);
  let modelName = $state("Current model");
  let turnCount = $state(0);

  onMount(() => {
    void initialize();
  });
  onMount(() => onConversationLibraryChanged(() => { if (!saving && !metadataDirty) return initialize(); }));

  async function initialize(): Promise<void> {
    loading = true;
    error = null;
    try {
      await flushConversationAutosave().catch(() => undefined);
      const snapshot = captureConversationSnapshot();
      modelName = displayModelName(snapshot.model_id);
      turnCount = snapshot.tree.nodes.filter((node) => node.parent_id !== null).length;
      name = defaultConversationName();
      const activeId = savedConversationState.activeId;
      if (activeId) {
        try {
          const saved = await conversationLibrary.get(activeId);
          if (saved.modelId === snapshot.model_id) {
            current = saved;
            name = saved.name;
            avatarSeed = saved.avatarSeed;
            accent = saved.accent ?? "purple";
          } else {
            savedConversationState.activeId = null;
            savedConversationState.avatarSeed = null;
            savedConversationState.accent = "purple";
          }
        } catch (cause) {
          if (cause instanceof ConversationLibraryError && cause.code === "NOT_FOUND") {
            savedConversationState.activeId = null;
            savedConversationState.avatarSeed = null;
            savedConversationState.accent = "purple";
          } else {
            throw cause;
          }
        }
      }
    } catch (cause) {
      error = userFacingError(cause, "This conversation is not ready to save yet.");
    } finally {
      draftBaseline = JSON.stringify([name, avatarSeed, accent]);
      loading = false;
    }
  }

  function regenerateAvatar(): void {
    avatarSeed = randomAvatarSeed();
  }

  async function save(asNew = false): Promise<void> {
    error = null;
    if (genStatus.active) {
      error = "Wait for the current reply to finish before saving.";
      return;
    }
    saving = true;
    try {
      await flushConversationAutosave().catch(() => undefined);
      const persistenceProtected = await requestPersistentConversationStorage();
      const snapshot = captureConversationSnapshot();
      const activeId = current?.id ?? savedConversationState.activeId;
      const record = activeId && !asNew
        ? await conversationLibrary.update(activeId, { name, avatarSeed, accent, snapshot })
        : await conversationLibrary.create({ name, avatarSeed, accent, snapshot });
      current = record;
      draftBaseline = JSON.stringify([record.name, record.avatarSeed, record.accent ?? "purple"]);
      name = record.name;
      avatarSeed = record.avatarSeed;
      savedConversationState.activeId = record.id;
      savedConversationState.avatarSeed = record.avatarSeed;
      savedConversationState.accent = record.accent ?? "purple";
      savedConversationState.status = "saved";
      savedConversationState.error = null;
      pushToast(
        persistenceProtected === false
          ? `Saved “${record.name}”. Browser cleanup protection is off.`
          : `Saved “${record.name}”.`,
        { kind: persistenceProtected === false ? "warning" : "info" },
      );
      if (!embedded) closeDrawer();
    } catch (cause) {
      error = userFacingError(cause, "This conversation could not be saved.");
    } finally {
      saving = false;
    }
  }

  let backupState = $state("idle");
  $effect(() => { if (backupState !== "ready") return; const timer = setTimeout(() => backupState = "idle", 1800); return () => clearTimeout(timer); });
  async function downloadCopy(): Promise<void> {
    if (saving || loading || genStatus.active) return;
    backupState = "preparing";
    saving = true;
    error = null;
    try {
      const snapshot = captureConversationSnapshot();
      const now = Date.now();
      const record: SavedConversationRecord = {
        schemaVersion: SAVED_CONVERSATION_SCHEMA_VERSION,
        id: current?.id ?? crypto.randomUUID(),
        name: name.trim() || defaultConversationName(now),
        avatarSeed,
        accent,
        modelId: snapshot.model_id,
        createdAt: current?.createdAt ?? now,
        updatedAt: Math.max(current?.updatedAt ?? now, now),
        snapshot,
      };
      const blob = new Blob([await encodeChatBackup(record)], { type: "application/json" });
      downloadPreparedChatBackup(blob, record.name);
      backupState = "ready";
    } catch (cause) {
      backupState = "idle";
      error = userFacingError(cause, "A backup copy could not be created.");
    } finally { saving = false; }
  }

  onMount(() => registerInterfaceController("chat_identity", {
    schema: objectSchema({ name: { type: "string", maxLength: 120 }, avatar_seed: { type: "string", maxLength: 256 }, accent: { type: "string", enum: CHAT_ACCENTS.map(row => row.id) } }),
    read: () => ({ busy: loading || saving, dirty: metadataDirty, chat_id: current?.id ?? savedConversationState.activeId, values: { name, avatar_seed: avatarSeed, accent } }),
    update: change => {
      if (loading || saving) throw new ToolError("BUSY", "Wait for the saved-chat form.");
      if (change.name !== undefined) name = change.name as string;
      if (change.avatar_seed !== undefined) avatarSeed = change.avatar_seed as string;
      if (change.accent !== undefined) accent = change.accent as ChatAccent;
    },
  }));
</script>

<section class="drawer-shell" class:embedded aria-label={embedded ? "Save and name chat" : "Save conversation drawer"}>
  <header class="header">
    <div>
      <h2 class="title"><MorphText text={embedded ? "Save and name chat" : current ? "Update saved chat" : "Save chat"} /></h2>
      <p>Keep the full loom, response settings, and readings on this device.</p>
    </div>
    {#if !embedded}<DrawerCloseButton onclick={closeDrawer} />{/if}
  </header>

  <div class="body">
    <div class="identity-card">
      <div class="avatar-picker">
        <button
          type="button"
          class="avatar-button"
          onclick={regenerateAvatar}
          aria-label="Generate another avatar"

        >
          <Blobatar name={avatarSeed} size={76} background="circle" alt="" />
        </button>
        <span>Click for a new avatar</span>
      </div>
      <label class="name-field">
        <span>Name</span>
        <input
          bind:value={name}
          maxlength="120"
          autocomplete="off"
          placeholder="Untitled conversation"
          disabled={loading || saving}
        />
      </label>
    </div>

    <div class="summary" aria-label="Conversation summary">
      <span>{modelName}</span>
      <span>{turnCount} {turnCount === 1 ? "turn" : "turns"}</span>
    </div>

    <ChatAccentPicker value={accent} disabled={loading || saving} onchange={(value) => { accent = value; }} />

    <p class="storage-note">
      Stored in this browser. Drowse removes a saved chat only after you confirm Delete.
    </p>

    {#if genStatus.active}
      <p class="notice">Wait for the current reply to finish before saving.</p>
    {/if}
    {#if error}
      <p class="error" role="alert">{error}</p>
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="text-button" onclick={downloadCopy} disabled={loading || saving || genStatus.active}>
      <MorphText text={backupState === "preparing" ? "Preparing copy…" : backupState === "ready" ? "Download started" : "Download copy"} numbers={false} />
    </button>
    <div class="footer-actions">
      {#if current}
        <button type="button" class="button" onclick={() => void save(true)} disabled={loading || saving || genStatus.active}>
          Save as new
        </button>
      {/if}
      <button
        type="button"
        class="button primary"
        class:loading-pulse={saving}
        aria-busy={saving}
        onclick={() => void save(false)}
        disabled={loading || saving || genStatus.active || !name.trim()}
      >
        <MorphText text={saving ? "Saving…" : current ? "Update" : "Save"} />
      </button>
    </div>
  </footer>
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
  .drawer-shell.embedded {
    --drawer-gutter-inline: var(--surface-gutter);
    --drawer-gutter-block: var(--panel-padding);
    height: auto;
  }
  .embedded .body {
    flex: none;
    overflow: visible;
  }
  .header div { min-width: 0; }
  .title {
    margin: 0;
    color: var(--fg);
    font-size: var(--text-lg);
    font-weight: var(--weight-medium);
  }
  .header p {
    margin: var(--space-2) 0 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.45;
  }
  .body {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--drawer-gutter-block);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .identity-card {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: var(--space-4);
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--glass);
    box-shadow: var(--shadow-rack);
  }
  .avatar-button {
    width: 80px;
    height: 80px;
    display: grid;
    place-items: center;
    padding: 0;
    border: 0;
    border-radius: 50%;
    background: var(--bg-elev);
    box-shadow: var(--shadow-control);
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease;
  }
  .avatar-picker {
    display: grid;
    justify-items: center;
    gap: var(--space-2);
  }
  .avatar-picker > span {
    max-width: 10ch;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    line-height: 1.3;
    text-align: center;
  }
  .avatar-button:hover { box-shadow: var(--shadow-control-hover); }
  .avatar-button :global(img) { display: block; width: 100%; height: 100%; min-width: 0; border-radius: inherit; }
  .avatar-button:active { transform: scale(0.96); }
  .avatar-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 3px; }
  .name-field {
    min-width: 0;
    display: grid;
    gap: var(--space-2);
  }
  .name-field span {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-family: var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .name-field input {
    width: 100%;
    min-width: 0;
    min-height: 46px;
    box-sizing: border-box;
    border: 0;
    border-radius: var(--radius);
    background: var(--input-well);
    box-shadow: var(--shadow-well);
    color: var(--fg);
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-size: var(--text-md);
  }
  .name-field input:focus { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  .summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    color: var(--fg-dim);
    font-size: var(--text-sm);
  }
  .summary span:first-child { color: var(--fg-strong); }
  .storage-note, .notice, .error {
    margin: 0;
    font-size: var(--text-sm);
    line-height: 1.5;
  }
  .storage-note { color: var(--fg-muted); }
  .notice { color: var(--accent-amber); }
  .error { color: var(--accent-red); }
  .footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--drawer-gutter-block);
    padding: 0 var(--drawer-gutter-inline) var(--drawer-gutter-block);
  }
  .footer-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--drawer-gutter-block); }
  .button, .text-button {
    min-height: var(--control-field);
    border: 0;
    border-radius: var(--radius);
    font: inherit;
    cursor: pointer;
    transition: background-color 120ms ease, color 120ms ease, transform 120ms ease;
  }
  .button {
    padding: var(--space-3) var(--space-5);
    background: var(--glass-strong);
    color: var(--fg-strong);
    box-shadow: var(--shadow-control);
  }
  .button:hover:not(:disabled) { background: var(--glass-bright); }
  .button.primary { background: var(--action-bg); color: var(--action-ink); }
  .button.primary:hover:not(:disabled) { background: var(--action-hover); }
  .button:active:not(:disabled), .text-button:active:not(:disabled) { transform: scale(0.96); }
  .text-button { padding: var(--space-2); background: transparent; color: var(--fg-dim); }
  .text-button:hover:not(:disabled) { color: var(--fg); background: var(--bg-hover); }
  .button:disabled, .text-button:disabled { opacity: var(--disabled-opacity); cursor: not-allowed; }
  .button:focus-visible, .text-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (max-width: 520px) {
    .drawer-shell.embedded { --drawer-gutter-inline: var(--panel-padding); }
    .footer { align-items: stretch; flex-direction: column-reverse; }
    .footer-actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
    .text-button { align-self: flex-start; }
  }
</style>
