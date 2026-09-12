<script lang="ts">
  import { onMount } from "svelte";
  import { registerSystemPromptController } from "../lib/workspaceController";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // System-prompt drawer — edit the session's default system prompt.
  // Saves via PATCH /sessions/{id} (``patchSessionDefaults``); cancel
  // closes without writing.

  import {
    sessionState,
    patchSessionDefaults,
    closeDrawer,
  } from "../lib/stores.svelte";
  import { ApiError } from "../lib/runtime/services";
  import { userFacingError } from "../lib/runtime/userFacingError";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let value = $state(sessionState.info?.config.system_prompt ?? "");
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  onMount(() => registerSystemPromptController({
    read: () => ({ busy, dirty: value !== (sessionState.info?.config.system_prompt ?? ""), values: { text: value } }),
    sync: () => { value = sessionState.info?.config.system_prompt ?? ""; errorMsg = null; },
  }));

  async function save(): Promise<void> {
    if (busy) return;
    busy = true;
    errorMsg = null;
    try {
      await patchSessionDefaults({ system_prompt: value });
      closeDrawer();
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        errorMsg = userFacingError(e, `Unable to save the system prompt. ${detail}`);
      } else {
        errorMsg = userFacingError(e, "Unable to save the system prompt. Try again.");
      }
    } finally {
      busy = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="System prompt drawer">
  <header class="header">
    <h2 class="title">System prompt</h2>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="body">
    <p class="hint">
      Sets the default system prompt for new generations in this session.
      Per-request OpenAI or Ollama system messages take precedence. Leaving it
      empty clears the system prompt.
    </p>

    <label class="field">
      <span class="label">System prompt</span>
      <textarea
        class="textarea"
        rows="12"
        bind:value={value}
        disabled={busy}
        placeholder="No system prompt"
        spellcheck="false"
      ></textarea>
      <span class="char-count">{value.length} char{value.length === 1 ? "" : "s"}</span>
    </label>

    {#if errorMsg}
      <p class="error" role="alert">{errorMsg}</p>
    {/if}
  </div>

  <footer class="footer">
    <button
      type="button"
      class="btn"
      onclick={closeDrawer}
      disabled={busy}
    >Cancel</button>
    <button
      type="button"
      class="btn primary"
      class:loading-pulse={busy}
      aria-busy={busy}
      onclick={save}
      disabled={busy}
    >{busy ? "Saving…" : "Save system prompt"}</button>
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
    font-size: var(--text);
  }
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    color: var(--accent);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-height: 0;
  }
  .hint {
    margin: 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    line-height: 1.4;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    flex: 1 1 auto;
    min-height: 0;
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  .textarea {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-4);
    font: inherit;
    font-family: var(--font-reading);
    line-height: 1.4;
    resize: vertical;
    min-height: 200px;
  }
  .textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .char-count {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    align-self: flex-end;
  }
  .error {
    color: var(--accent-red);
    margin: 0;
    font-size: var(--text-sm);
    word-break: break-word;
  }
  .footer {
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: var(--drawer-gutter-block);
    padding: 0 var(--drawer-gutter-inline) var(--drawer-gutter-block);
    color: var(--fg-muted);
  }
  .btn {
    background: var(--glass);
    color: var(--fg-strong);
    border: 1px solid transparent;
    padding: var(--space-3) var(--space-5);
    min-height: var(--control-target);
    font-family: var(--font-structure);
    font-size: inherit;
    font-weight: var(--weight-structure);
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    background: var(--glass-strong);
  }
  .btn:disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--action-bg);
    color: var(--action-ink);
    border-color: transparent;
  }
  .btn.primary:hover:not(:disabled) {
    background: var(--action-hover);
  }
  .btn.primary:disabled {
    background: var(--bg-elev);
  }
</style>
