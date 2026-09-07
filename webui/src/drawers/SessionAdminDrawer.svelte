<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import { getApiKey, setApiKey } from "../lib/runtime/http-auth";
  import { apiSessions } from "../lib/runtime/services";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import type { SessionInfo } from "../lib/types";
  import { closeDrawer, refreshSession, sessionState } from "../lib/stores.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  let key = $state(getApiKey() ?? "");
  let sessions: SessionInfo[] = $state([]);
  let busy = $state(false);
  let errorMsg: string | null = $state(null);
  let saved = $state(false);
  const keyId = $props.id();

  async function loadSessions(): Promise<void> {
    busy = true;
    errorMsg = null;
    try {
      const r = await apiSessions.list();
      sessions = r.sessions;
    } catch (e) {
      errorMsg = userFacingError(e, "Unable to load server sessions. Check the server connection and try again.");
    } finally {
      busy = false;
    }
  }

  async function saveKey(): Promise<void> {
    setApiKey(key);
    saved = true;
    await refreshSession();
    await loadSessions();
  }
</script>

<section class="drawer-shell" aria-label="Session and auth drawer">
  <header class="header">
    <div>
      <h2 class="title">API access</h2>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="body">
    <section class="panel">
      <h3><label for={keyId}>API key</label></h3>
      <form class="key-row" onsubmit={(event) => { event.preventDefault(); void saveKey(); }}>
        <input
          id={keyId}
          type="password"
          bind:value={key}
          placeholder="DROWSE_API_KEY"
          autocomplete="off"
          aria-label="Drowse API key"
          oninput={() => (saved = false)}
        />
        <button type="submit">Apply key</button>
        <button type="button" onclick={() => { key = ""; void saveKey(); }}>Clear key</button>
      </form>
      <p class="hint">Changes apply only to this tab until you reload.</p>
      <p class="hint" role="status">{saved ? "API key updated for this tab." : ""}</p>
    </section>

    <section class="panel">
      <div class="section-head">
        <h3>Server sessions</h3>
        <button type="button" class:loading-pulse={busy} aria-busy={busy} disabled={busy} onclick={loadSessions}>
          {busy ? "Loading…" : "Refresh sessions"}
        </button>
      </div>
      {#if errorMsg}
        <p class="error" role="alert">{errorMsg}</p>
      {/if}
      <div class="sessions">
        {#if sessions.length === 0}
          <div class="empty">Select Refresh sessions to list the models running on this server.</div>
        {:else}
          {#each sessions as s (s.id)}
            <article class:active={sessionState.info?.id === s.id}>
              <strong>{s.id}</strong>
              {#if sessionState.info?.id === s.id}<span>Current session</span>{/if}
              <code>{s.model_id}</code>
              <span>{s.device}/{s.dtype} · {s.profiles.length} profiles · {s.probes.length} probes</span>
            </article>
          {/each}
        {/if}
      </div>
    </section>

  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; background: transparent; }
  .header { display: flex; justify-content: space-between; gap: var(--space-6); padding: var(--drawer-gutter-block) var(--drawer-gutter-inline); background: transparent; }
  .title { color: var(--fg); letter-spacing: 0; font-size: var(--text-md); font-weight: var(--weight-medium); }
  .hint, .panel p { margin: var(--space-1) 0 0; color: var(--fg-muted); line-height: 1.45; }
  .body { display: grid; gap: var(--drawer-section-gap); padding: var(--drawer-gutter-block) var(--drawer-gutter-inline); overflow: auto; }
  .panel {
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--glass);
    box-shadow: var(--shadow-well);
    padding: var(--panel-padding);
  }
  h3 { margin: 0 0 var(--space-4); color: var(--fg); font-size: var(--text-sm); letter-spacing: 0; }
  .key-row { display: flex; flex-wrap: wrap; gap: var(--space-3); }
  input { min-width: 0; flex: 1 1 100%; min-height: var(--control-target); border: 1px solid transparent; border-radius: var(--radius); background: var(--input-well); color: var(--fg); padding: var(--space-4); font-family: var(--font-mono); }
  button { min-height: var(--control-target); border: 1px solid transparent; border-radius: var(--radius); background: var(--glass); color: var(--fg); padding: var(--space-3) var(--space-5); }
  button:hover:not(:disabled) { background: var(--glass-strong); color: var(--accent); }
  .section-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-4); }
  .sessions { display: grid; gap: var(--space-3); }
  article { display: grid; gap: var(--space-2); border: 1px solid transparent; border-radius: var(--radius); background: var(--surface-sheen), var(--bg-elev); padding: var(--surface-padding); }
  article.active { border-color: var(--accent); background: var(--surface-sheen), var(--accent-subtle); }
  strong { color: var(--fg); overflow-wrap: anywhere; }
  code { color: var(--fg-strong); font-family: var(--font-mono); overflow-wrap: anywhere; }
  article span { color: var(--fg-muted); }
  .panel .error { color: var(--accent-red); }
  .empty { color: var(--fg-muted); background: var(--surface-sheen), var(--bg); border-radius: var(--radius); padding: var(--panel-padding); text-align: center; }
</style>
