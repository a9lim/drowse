<script lang="ts">
  import { slidingSelection } from "../lib/slidingSelection";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Transcript export / import drawer — phase 5.  Two tabs:
  //
  //   * export — render the path ending at the chosen node as
  //     transcript YAML and offer it as a .yaml download.  Defaults
  //     to the active node; the user can pick any node id via a
  //     short id-prefix.
  //   * import — paste YAML or upload a file, pick a mode
  //     (default / here / merge), tick strict, fire the load.
  //     Guard warnings (model / system-prompt / probe drift) surface
  //     in a banner with the diff list.

  import { apiTree, describeError } from "../lib/runtime/services";
  import {
    closeDrawer,
    loomTree,
    refreshLoomTree,
  } from "../lib/stores.svelte";
  import Radio from "../lib/Radio.svelte";
  import Checkbox from "../lib/Checkbox.svelte";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  type Tab = "export" | "import";
  let tab: Tab = $state("export");

  function selectTab(next: Tab): void {
    tab = next;
  }

  function onTabKeydown(ev: KeyboardEvent): void {
    const tabs = ["export", "import"] as const;
    const current = tabs.indexOf(tab);
    let next = current;
    if (ev.key === "ArrowRight") next = (current + 1) % tabs.length;
    else if (ev.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = tabs.length - 1;
    else return;
    ev.preventDefault();
    selectTab(tabs[next]);
    document.getElementById(`transcript-${tabs[next]}-tab`)?.focus();
  }

  // -------------------------------------------------- export state ---

  let exportTargetId = $state(""); // empty = active node
  let exportYaml = $state("");
  let exportError: string | null = $state(null);
  let exportBusy = $state(false);
  let exportLeafId: string | null = $state(null);

  async function runExport(): Promise<void> {
    if (exportBusy) return;
    exportBusy = true;
    exportError = null;
    exportYaml = "";
    exportLeafId = null;
    try {
      const id = exportTargetId.trim();
      let resolved: string | null = null;
      if (id) {
        const r = resolveByPrefix(id);
        if (r.id) {
          resolved = r.id;
        } else if (r.matches.length === 0) {
          exportError = `no node matches prefix "${id}"`;
          return;
        } else {
          const preview = r.matches.slice(0, 6).map((s) => s.slice(0, 8)).join(", ");
          exportError =
            `ambiguous: ${r.matches.length} matches (${preview}` +
            (r.matches.length > 6 ? ", …" : "") + ")";
          return;
        }
      }
      const r = await apiTree.transcriptExport(resolved);
      exportYaml = r.yaml;
      exportLeafId = r.node_id;
    } catch (e) {
      exportError = describeError(e);
    } finally {
      exportBusy = false;
    }
  }

  function downloadYaml(): void {
    if (!exportYaml) return;
    const blob = new Blob([exportYaml], {
      type: "application/yaml;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const tsId = exportLeafId ? exportLeafId.slice(0, 8) : "active";
    a.download = `drowse-transcript-${tsId}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function copyYaml(): Promise<void> {
    if (!exportYaml) return;
    try {
      await navigator.clipboard.writeText(exportYaml);
    } catch {
      // Clipboard unavailable — leave the textarea content for the user.
    }
  }

  interface PrefixResolution {
    /** Single unambiguous match — caller can use immediately. */
    id: string | null;
    /** All node ids whose ulid starts with the prefix; ``length > 1``
     *  signals ambiguity. */
    matches: string[];
  }

  function resolveByPrefix(prefix: string): PrefixResolution {
    const p = prefix.trim();
    if (!p) return { id: null, matches: [] };
    if (p === "root") {
      const root = loomTree.root_id;
      return root ? { id: root, matches: [root] } : { id: null, matches: [] };
    }
    const matches: string[] = [];
    for (const id of loomTree.nodes.keys()) {
      if (id === p) return { id, matches: [id] };
      if (id.startsWith(p)) matches.push(id);
    }
    if (matches.length === 1) return { id: matches[0], matches };
    return { id: null, matches };
  }

  // -------------------------------------------------- import state ---

  let importYaml = $state("");
  let importMode: "default" | "here" | "merge" = $state("default");
  let importStrict = $state(false);
  let importError: string | null = $state(null);
  let importBusy = $state(false);
  let importGuards: string[] = $state([]);
  let importLeafId: string | null = $state(null);

  let fileInputRef: HTMLInputElement | null = $state(null);

  async function onFileChange(ev: Event): Promise<void> {
    const target = ev.target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) return;
    importYaml = await file.text();
  }

  async function runImport(): Promise<void> {
    if (importBusy) return;
    const yaml = importYaml.trim();
    if (!yaml) {
      importError = "paste a transcript YAML or upload a file";
      return;
    }
    importBusy = true;
    importError = null;
    importGuards = [];
    importLeafId = null;
    try {
      const r = await apiTree.transcriptLoad(yaml, importMode, importStrict);
      importGuards = r.guards;
      importLeafId = r.leaf_id;
      // Refresh the tree so the new branch shows up in the sidebar.
      await refreshLoomTree();
    } catch (e) {
      importError = describeError(e);
    } finally {
      importBusy = false;
    }
  }
</script>

<section class="drawer-shell" aria-label="Transcript drawer">
  <header class="header">
    <h2 class="title">Conversation transcript</h2>
    <div
      class="tabs"
      use:slidingSelection
      role="tablist"
      aria-label="Transcript action"
    >
      <button
        id="transcript-export-tab"
        type="button"
        class="tab"
        class:active={tab === "export"}
        role="tab"
        aria-selected={tab === "export"}
        aria-controls="transcript-export-panel"
        tabindex={tab === "export" ? 0 : -1}
        onclick={() => selectTab("export")}
        onkeydown={onTabKeydown}
      >export</button>
      <button
        id="transcript-import-tab"
        type="button"
        class="tab"
        class:active={tab === "import"}
        role="tab"
        aria-selected={tab === "import"}
        aria-controls="transcript-import-panel"
        tabindex={tab === "import" ? 0 : -1}
        onclick={() => selectTab("import")}
        onkeydown={onTabKeydown}
      >import</button>
    </div>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="body">
    {#if tab === "export"}
      <div
        id="transcript-export-panel"
        class="tab-panel"
        role="tabpanel"
        aria-labelledby="transcript-export-tab"
        tabindex="0"
      >
      <p class="hint">
        Export the active conversation path, or end at a specific node.
      </p>

      <label class="field">
        <span class="label">end at node <span class="optional">optional</span></span>
        <input
          type="text"
          class="input"
          bind:value={exportTargetId}
          placeholder={`Active node · ${loomTree.active_node_id?.slice(0, 12) ?? "-"}`}
          autocomplete="off"
          spellcheck="false"
        />
      </label>

      <div class="form-actions">
        <button
          type="button"
          class="btn primary"
          onclick={runExport}
          disabled={exportBusy}
        >{exportBusy ? "preparing…" : "preview YAML"}</button>
      </div>

      {#if exportError}
        <p class="error" role="alert">{exportError}</p>
      {/if}

      {#if exportYaml}
        <textarea
          class="yaml"
          readonly
          rows="20"
          value={exportYaml}
          aria-label="Rendered transcript YAML"
        ></textarea>
        <div class="form-actions">
          <button type="button" class="btn" data-cursor="copy" onclick={copyYaml}>copy</button>
          <button type="button" class="btn primary" onclick={downloadYaml}
            >download .yaml</button>
        </div>
      {/if}
      </div>
    {:else}
      <div
        id="transcript-import-panel"
        class="tab-panel"
        role="tabpanel"
        aria-labelledby="transcript-import-tab"
        tabindex="0"
      >
      <p class="hint">
        Import a Drowse transcript from a local YAML file or pasted text.
      </p>

      <div class="field" role="radiogroup" aria-labelledby="import-mode-label">
        <span class="label" id="import-mode-label">place imported turns</span>
        <Radio bind:group={importMode} value="default" label="start a new path at the root" />
        <Radio bind:group={importMode} value="here" label="continue from the active node" />
        <Radio bind:group={importMode} value="merge" label="merge after the deepest matching turn" />
      </div>

      <span class="mode-opt">
        <Checkbox bind:checked={importStrict} label="Require matching reading settings" />
      </span>

      <label class="field">
        <span class="label">file</span>
        <input
          type="file"
          accept=".yaml,.yml,application/x-yaml,text/yaml"
          bind:this={fileInputRef}
          onchange={onFileChange}
        />
      </label>

      <textarea
        class="yaml"
        rows="14"
        bind:value={importYaml}
        placeholder="paste transcript YAML here…"
        spellcheck="false"
        aria-label="Transcript YAML to import"
      ></textarea>

      <div class="form-actions">
        <button
          type="button"
          class="btn primary"
          onclick={runImport}
          disabled={importBusy}
        >{importBusy ? "importing…" : "import"}</button>
      </div>

      {#if importError}
        <p class="error" role="alert">{importError}</p>
      {/if}

      {#if importGuards.length > 0}
        <div class="banner">
          <span class="banner-title">guards</span>
          <ul>
            {#each importGuards as g (g)}
              <li>{g}</li>
            {/each}
          </ul>
          <p class="hint">saved as root notes</p>
        </div>
      {/if}
      {#if importLeafId && !importError}
        <p class="ok">
          imported · leaf
          <code>{importLeafId.slice(0, 12)}</code>
        </p>
      {/if}
      </div>
    {/if}
  </div>

  <footer class="footer">
    <button type="button" class="btn" onclick={closeDrawer}>close</button>
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
    gap: var(--space-4);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    margin: 0;
    min-width: 0;
    color: var(--fg);
    letter-spacing: 0;
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
  }
  .tabs {
    display: flex;
    gap: var(--space-2);
    flex: 1 1 auto;
    justify-content: center;
  }
  @media (max-width: 680px) {
    .header { display: grid; grid-template-columns: minmax(0, 1fr) auto; }
    .tabs { grid-column: 1 / -1; grid-row: 2; justify-content: flex-start; }
    .tab { flex: 1; }
  }
  .tab {
    background: var(--glass);
    color: var(--fg-dim);
    border: 1px solid transparent;
    padding: var(--space-1) var(--space-4);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    min-height: var(--control-target);
    border-radius: var(--radius-sm);
  }
  .tab:hover {
    color: var(--fg-strong);
  }
  .tab.active {
    color: var(--accent);
    border-color: var(--accent);
    background: var(--accent-subtle);
  }
  .body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    min-height: 0;
  }
  .tab-panel {
    display: flex;
    flex-direction: column;
    gap: var(--drawer-gutter-block);
  }
  .hint {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-transform: lowercase;
  }
  .optional {
    color: var(--fg-subtle);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    margin-inline-start: var(--space-2);
  }
  .input {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    min-height: var(--control-field);
  }
  .input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .yaml {
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    padding: var(--surface-padding);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    resize: vertical;
    min-height: 8em;
    white-space: pre;
    overflow: auto;
  }
  .mode-opt {
    display: inline-flex;
    gap: var(--space-2);
    align-items: center;
    color: var(--fg-strong);
    font-size: var(--text-sm);
    margin: var(--space-1) 0;
  }
  .form-actions {
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: var(--drawer-gutter-block);
  }
  .error {
    color: var(--accent-red);
    font-size: var(--text-sm);
    margin: 0;
  }
  .ok {
    color: var(--accent-green);
    font-size: var(--text-sm);
    margin: 0;
  }
  .banner {
    background: color-mix(in srgb, var(--accent-amber) 14%, transparent);
    border: 1px solid transparent;
    padding: var(--surface-padding);
    color: var(--accent-yellow);
    font-size: var(--text-sm);
  }
  .banner-title {
    text-transform: lowercase;
    font-weight: var(--weight-medium);
    display: block;
    margin-bottom: var(--space-1);
  }
  .banner ul {
    margin: var(--space-1) 0 var(--space-1) var(--space-sm);
    padding: 0;
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
    padding: var(--space-2) var(--space-5);
    font: inherit;
    font-family: var(--font-mono);
    cursor: pointer;
    min-height: var(--control-target);
    border-radius: var(--radius-sm);
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
