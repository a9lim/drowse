<script lang="ts">
  import { onMount } from "svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import { closeDrawer, genStatus } from "../lib/stores.svelte";
  import { conversationLibrary, savedConversationState } from "../lib/stores/savedConversations.svelte";
  import { captureConversationSnapshot } from "../lib/conversationWorkspace";
  import { ConversationLibraryError, defaultConversationName, randomAvatarSeed, SAVED_CONVERSATION_SCHEMA_VERSION } from "../lib/conversationLibrary";
  import { chatBackupFilename, downloadPreparedChatBackup, encodeChatBackup } from "../lib/chatBackup";
  import { userFacingError } from "../lib/runtime/userFacingError";

  let { params: _params }: { params?: unknown } = $props();
  let name = $state("");
  let blob = $state<Blob | null>(null);
  let error = $state<string | null>(null);
  let loading = $state(true);
  let downloading = $state(false);
  const filename = $derived(chatBackupFilename(name));

  onMount(() => { void prepare(); });

  async function prepare(): Promise<void> {
    loading = true;
    error = null;
    blob = null;
    try {
      if (genStatus.active) throw new Error("Wait for the reply to finish, then try again.");
      const snapshot = captureConversationSnapshot();
      const activeId = savedConversationState.activeId;
      let saved = null;
      if (activeId) {
        try { saved = await conversationLibrary.get(activeId); }
        catch (cause) {
          if (!(cause instanceof ConversationLibraryError && cause.code === "NOT_FOUND")) throw cause;
        }
      }
      if (saved?.modelId !== snapshot.model_id) saved = null;
      const now = Date.now();
      const record = {
        schemaVersion: SAVED_CONVERSATION_SCHEMA_VERSION,
        id: saved?.id ?? crypto.randomUUID(),
        name: saved?.name ?? defaultConversationName(now),
        avatarSeed: saved?.avatarSeed ?? randomAvatarSeed(),
        accent: saved?.accent ?? savedConversationState.accent,
        modelId: snapshot.model_id,
        createdAt: saved?.createdAt ?? now,
        updatedAt: Math.max(saved?.updatedAt ?? now, now),
        snapshot,
      };
      name = record.name;
      blob = new Blob([await encodeChatBackup(record)], { type: "application/json" });
    } catch (cause) {
      error = userFacingError(cause, "This chat could not be prepared for download.");
    } finally { loading = false; }
  }

  function download(event: SubmitEvent): void {
    event.preventDefault();
    if (!blob || !name.trim() || downloading) return;
    downloading = true;
    try {
      downloadPreparedChatBackup(blob, name);
      closeDrawer();
    } catch (cause) {
      error = userFacingError(cause, "The download could not start. Try again.");
      downloading = false;
    }
  }
</script>

<form class="download-chat" onsubmit={download}>
  <header>
    <h2>Download chat</h2>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>
  <p>All messages, Loom branches, settings, and recorded readings. Model files aren’t included.</p>
  {#if loading}
    <p role="status">Preparing your backup…</p>
  {:else if blob}
    <div class="name-field">
      <label for="chat-download-name">File name</label>
      <input id="chat-download-name" class="field-focus" bind:value={name} maxlength="120" required autocomplete="off" />
    </div>
    <div class="file-summary">
      <span class="filename">{filename}</span>
      <span class="size" aria-label="Backup size" data-bytes={blob.size}>{blob.size.toLocaleString()} bytes</span>
    </div>
    <p class="note">This is a copy. Your saved chat’s name stays unchanged.</p>
  {/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  <footer>
    <button type="button" onclick={closeDrawer}>Cancel</button>
    {#if error && !blob}
      <button type="button" class="primary" onclick={() => void prepare()} disabled={loading}>Try again</button>
    {:else}
      <button type="submit" class="primary" disabled={!blob || !name.trim() || downloading}>Download</button>
    {/if}
  </footer>
</form>

<style>
  .download-chat { display: grid; gap: var(--space-md); padding: var(--surface-padding); min-width: 0; }
  header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-4); }
  h2 { margin: 0; font-size: var(--text-lg); }
  p { margin: 0; color: var(--fg-dim); text-wrap: pretty; }
  label { font-weight: var(--weight-structure); }
  .name-field { display: grid; gap: var(--space-4); min-width: 0; }
  input { width: 100%; min-width: 0; box-sizing: border-box; padding: var(--space-4); border: 1px solid var(--glass-line); border-radius: var(--radius); background: var(--input-well); color: var(--fg); font: inherit; }
  .file-summary { display: grid; gap: var(--space-2); padding: var(--surface-padding); border-radius: var(--radius); background: var(--glass); }
  .filename { overflow-wrap: anywhere; }
  .size { color: var(--accent); font-variant-numeric: tabular-nums; }
  .note { font-size: var(--text-sm); }
  .error { color: var(--accent-red); }
  footer { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: var(--space-md); }
  button { min-height: var(--control-target); padding: var(--space-xs) var(--space-sm); border: 0; border-radius: var(--radius); background: var(--glass); color: var(--fg); font: inherit; cursor: pointer; transition: background var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out); }
  button:hover:not(:disabled) { background: var(--accent-subtle); }
  button:active:not(:disabled) { transform: scale(0.96); }
  button.primary { background: var(--accent); color: var(--bg); }
  button.primary:hover:not(:disabled) { background: color-mix(in srgb, var(--accent) 88%, var(--fg)); }
  button:disabled { opacity: 0.45; cursor: default; }
</style>
