<script lang="ts">
  import MorphText from "../lib/ui/MorphText.svelte";
  import { slidingSelection } from "../lib/slidingSelection";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // ManifoldPacksDrawer — local manifold catalog and HF search/install.
  //
  // Hosted search accepts only repositories with one immutable, provenance-
  // bound .drowse; the Python runtime also supports legacy folder repos.
  //
  // The rack browser is the active-session surface (steer, probe, fit,
  // delete); this drawer is the catalog surface (list local, browse HF,
  // install). It is reachable from the command palette.

  import { onMount } from "svelte";
  import { ApiError, apiManifoldInstallStream, apiManifolds } from "../lib/runtime/services";
  import { runtimeClient } from "../lib/runtime/client";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import {
    closeDrawer,
    probeRack,
    steerRack,
    refreshManifoldList,
  } from "../lib/stores.svelte";
  import { pushToast } from "../lib/stores/toasts.svelte";
  import type { HostedManifoldPackInfo } from "../lib/runtime/contracts";
  import type { ManifoldInfo, RemoteManifoldInfo } from "../lib/types";
  import { manifoldUsage, manifoldUsageMessage } from "../lib/manifolds/selectors";

  type Tab = "installed" | "search";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  let tab: Tab = $state("installed");

  // ----- search-tab state ----------------------------------------------
  let query = $state("");
  let searchResults: RemoteManifoldInfo[] = $state([]);
  let searchLoading = $state(false);
  let searchError: string | null = $state(null);
  let installing: string | null = $state(null);
  // Latest ``progress`` frame of the in-flight install, shown in place of a
  // bare spinner — an HF manifold repo can be hundreds of megabytes.
  let installStage: string | null = $state(null);
  const browserMode = runtimeClient.mode !== "http";
  const artifactAvailability = runtimeOperationAvailability("manifold_artifacts");
  const hfDiscoveryAvailable = !browserMode || artifactAvailability.available;
  let installedTabButton: HTMLButtonElement | null = $state(null);
  let searchTabButton: HTMLButtonElement | null = $state(null);
  let localPacks: HostedManifoldPackInfo[] = $state([]);
  let localPacksLoading = $state(false);
  let archiveBusy: string | null = $state(null);
  let archiveError: string | null = $state(null);
  let archiveProgress: string | null = $state(null);
  let pendingReplacement: File | null = $state(null);
  let confirmLocalDelete: string | null = $state(null);

  // Redo searches 300ms after the user stops typing.
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  function scheduleSearch(): void {
    if (debounceTimer) clearTimeout(debounceTimer);
    const q = query.trim();
    if (!q) {
      searchResults = [];
      searchError = null;
      searchLoading = false;
      return;
    }
    searchLoading = true;
    debounceTimer = setTimeout(() => {
      void runSearch(q);
    }, 300);
  }

  async function runSearch(q: string): Promise<void> {
    try {
      const r = await apiManifolds.search(q, 20);
      searchResults = r.results ?? [];
      searchError = null;
    } catch (e) {
      searchResults = [];
      if (e instanceof ApiError) {
        if (e.status === 503) {
          searchError = browserMode
            ? "Hugging Face search is unavailable. Check your connection and try again."
            : "huggingface_hub isn't installed on the server. Run `pip install -e \".[serve]\"` and restart.";
        } else if (e.status === 502) {
          searchError = userFacingError(
            e,
            "Hugging Face search was interrupted. Check your connection and try again.",
          );
        } else {
          searchError = userFacingError(e, "Unable to search Hugging Face. Try again.");
        }
      } else {
        searchError = userFacingError(e, "Unable to search Hugging Face. Try again.");
      }
    } finally {
      searchLoading = false;
    }
  }

  async function installRow(row: RemoteManifoldInfo): Promise<void> {
    const repository = typeof row.repository === "string"
      ? row.repository
      : `${row.namespace}/${row.name}`;
    const displayTarget = selectorOf(row);
    const requestTarget = typeof row.revision === "string"
      ? `${repository}@${row.revision}`
      : repository;
    installing = displayTarget;
    installStage = null;
    try {
      // Streaming client, like the fit / generate flows: the terminal
      // ``error`` frame surfaces as a plain Error (no HTTP status), so the
      // ApiError branches below only fire for a pre-stream rejection.
      const installed = await apiManifoldInstallStream({ target: requestTarget }, (ev) => {
        if (ev.event === "progress") {
          const progress = ev.data as {
            message?: string;
            phase?: string;
            downloadedBytes?: number;
            totalBytes?: number | null;
          } | null;
          if (progress?.message) {
            installStage = progress.message;
          } else if (progress?.phase === "downloading" && progress.downloadedBytes !== undefined) {
            const downloaded = `${(progress.downloadedBytes / 1_000_000).toFixed(1)} MB`;
            const total = progress.totalBytes
              ? ` of ${(progress.totalBytes / 1_000_000).toFixed(1)} MB`
              : "";
            installStage = `Downloading ${downloaded}${total}`;
          } else if (progress?.phase) {
            installStage = `${progress.phase[0].toUpperCase()}${progress.phase.slice(1)} archive`;
          }
        }
      });
      await refreshManifoldList();
      tab = "installed";
      pushToast(`installed ${installed.namespace}/${installed.name}`, { kind: "info" });
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 503) {
          pushToast(
            browserMode
              ? "Hugging Face install is unavailable. Check your connection and try again."
              : "huggingface_hub isn't installed on the server",
            { kind: "error", ttlMs: null },
          );
        } else if (e.status === 502) {
          pushToast(userFacingError(
            e,
            "The Hugging Face download was interrupted. Check your connection and try again.",
          ), {
            kind: "error", ttlMs: null,
          });
        } else if (e.status === 409) {
          pushToast(`${displayTarget} is already installed`, {
            kind: "error", ttlMs: null,
          });
        } else {
          pushToast(userFacingError(
            e,
            `Unable to install ${displayTarget}. Check the pack and try again.`,
          ), {
            kind: "error", ttlMs: null,
          });
        }
      } else {
        pushToast(userFacingError(
          e,
          `Unable to install ${displayTarget}. Check the pack and try again.`,
        ), {
          kind: "error", ttlMs: null,
        });
      }
    } finally {
      installing = null;
      installStage = null;
    }
  }

  async function refreshLocalPacks(): Promise<void> {
    if (!browserMode || !artifactAvailability.available) return;
    localPacksLoading = true;
    try {
      localPacks = (await apiManifolds.drowseArchiveList()).packs;
      archiveError = null;
    } catch (error) {
      archiveError = describe(error);
    } finally {
      localPacksLoading = false;
    }
  }

  async function importArchive(file: File, force = false): Promise<void> {
    if (!file.name.toLowerCase().endsWith(".drowse")) {
      archiveError = "Choose a .drowse file.";
      return;
    }
    archiveBusy = "import";
    archiveError = null;
    archiveProgress = "Inspecting archive";
    try {
      const installed = await apiManifolds.drowseArchiveInstall(
        file,
        { force },
        (event) => {
          if (!event.data || typeof event.data !== "object") return;
          const progress = event.data as { phase?: string; path?: string | null };
          archiveProgress = progress.path
            ? `${progress.phase ?? "Verifying"}: ${progress.path}`
            : progress.phase ?? "Verifying archive";
        },
      );
      pendingReplacement = null;
      await refreshLocalPacks();
      await refreshManifoldList();
      pushToast(`installed ${installed.namespace}/${installed.name}`, { kind: "info" });
    } catch (error) {
      const message = describe(error);
      archiveError = message;
      pendingReplacement = /already installed/i.test(message) && !force ? file : null;
    } finally {
      archiveBusy = null;
      archiveProgress = null;
    }
  }

  let exportedPack = $state<string | null>(null);
  $effect(() => { if (!exportedPack) return; const timer = setTimeout(() => exportedPack = null, 1800); return () => clearTimeout(timer); });
  async function exportArchive(pack: HostedManifoldPackInfo): Promise<void> {
    archiveBusy = `export:${pack.id}`;
    archiveError = null;
    try {
      const archive = await apiManifolds.drowseArchiveExport(pack.id);
      const url = URL.createObjectURL(archive);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${pack.name}.drowse`;
      link.click();
      exportedPack = pack.id;
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (error) {
      archiveError = describe(error);
    } finally {
      archiveBusy = null;
    }
  }

  async function deleteArchive(pack: HostedManifoldPackInfo): Promise<void> {
    const guard = manifoldUsageMessage(
      pack,
      manifoldUsage(pack, steerRack.entries, probeRack.entries),
    );
    if (guard) {
      archiveError = guard;
      return;
    }
    archiveBusy = `delete:${pack.id}`;
    archiveError = null;
    try {
      await apiManifolds.drowseArchiveDelete(pack.id);
      confirmLocalDelete = null;
      await refreshLocalPacks();
      await refreshManifoldList();
    } catch (error) {
      archiveError = describe(error);
    } finally {
      archiveBusy = null;
    }
  }

  function describe(error: unknown): string {
    return userFacingError(
      error,
      "Unable to update local direction packs. Check the file and try again.",
    );
  }

  function selectTab(next: Tab, focus = false): void {
    tab = next;
    if (!focus) return;
    queueMicrotask(() => {
      (next === "installed" ? installedTabButton : searchTabButton)?.focus();
    });
  }

  function onTabKeydown(event: KeyboardEvent): void {
    if (!hfDiscoveryAvailable) return;
    let next: Tab | null = null;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      next = tab === "installed" ? "search" : "installed";
    } else if (event.key === "Home") {
      next = "installed";
    } else if (event.key === "End") {
      next = "search";
    }
    if (next === null) return;
    event.preventDefault();
    selectTab(next, true);
  }

  // ----- installed-tab state -------------------------------------------
  function selectorOf(row: ManifoldInfo | RemoteManifoldInfo): string {
    if ("repository" in row && typeof row.repository === "string") return row.repository;
    return `${row.namespace}/${row.name}`;
  }
  function fitBadge(row: ManifoldInfo | RemoteManifoldInfo): string | null {
    if (row.fit_mode && row.fit_mode !== "authored") return row.fit_mode;
    return null;
  }

  // The store keeps the local catalog hot — refresh on mount in case
  // we landed here without having visited ManifoldDrawer first.
  onMount(() => {
    void refreshManifoldList();
    void refreshLocalPacks();
  });
</script>

<div class="drawer-shell">
  <header class="header">
    <h2 class="title">Downloaded response controls</h2>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>

  <div class="tabs" role="tablist" aria-label="Manifold pack views" use:slidingSelection>
    <button
      bind:this={installedTabButton}
      id="packs-tab-installed"
      type="button"
      role="tab"
      aria-selected={tab === "installed"}
      aria-controls="packs-panel"
      tabindex={tab === "installed" ? 0 : -1}
      class:active={tab === "installed"}
      onclick={() => selectTab("installed")}
      onkeydown={onTabKeydown}
    >Installed</button>
    {#if hfDiscoveryAvailable}
      <button
        bind:this={searchTabButton}
        id="packs-tab-search"
        type="button"
        role="tab"
        aria-selected={tab === "search"}
        aria-controls="packs-panel"
        tabindex={tab === "search" ? 0 : -1}
        class:active={tab === "search"}
        onclick={() => selectTab("search")}
        onkeydown={onTabKeydown}
      >Hugging Face</button>
    {/if}
  </div>

  <div
    class="body"
    id="packs-panel"
    role="tabpanel"
    aria-labelledby={tab === "installed" ? "packs-tab-installed" : "packs-tab-search"}
  >
    <p class="catalog-count"><MorphText text={tab === "installed" ? `${steerRack.catalog.length} installed manifolds` : searchResults === null ? "Search compatible manifolds" : `${searchResults.length} found`} /></p>
    {#if tab === "installed"}
      {#if browserMode}
        <section class="archive-tools" aria-labelledby="archive-tools-title">
          <div class="archive-heading">
            <div>
              <h2 id="archive-tools-title">Portable control files</h2>
              <p>Checksums verify archive integrity, not the publisher’s identity.</p>
            </div>
            {#if artifactAvailability.available}
              <label class="file-action" class:disabled={archiveBusy !== null}>
                <span><MorphText text={archiveBusy === "import" ? "verifying…" : "import pack"} /></span>
                <input
                  type="file"
                  accept=".drowse,application/zip"
                  disabled={archiveBusy !== null}
                  onchange={(event) => {
                    const input = event.currentTarget;
                    const file = input.files?.[0];
                    if (file) void importArchive(file);
                    input.value = "";
                  }}
                />
              </label>
            {/if}
          </div>
          {#if !artifactAvailability.available}
            <p class="muted">{artifactAvailability.reason}</p>
          {:else if archiveProgress}
            <p class="install-stage" aria-live="polite"><MorphText text={archiveProgress.split(": ")[0]} numbers={false} />{#if archiveProgress.includes(": ")} · {archiveProgress.slice(archiveProgress.indexOf(": ") + 2)}{/if}</p>
          {/if}
          {#if artifactAvailability.available && archiveError}<p class="error" role="alert">{archiveError}</p>{/if}
          {#if artifactAvailability.available && pendingReplacement}
            <div class="replace-warning" role="alert">
              <span>This replaces the installed pack and its included templates. The current files stay in place until the replacement passes verification.</span>
              <button type="button" disabled={archiveBusy !== null} onclick={() => void importArchive(pendingReplacement!, true)}>replace installed pack</button>
              <button type="button" disabled={archiveBusy !== null} onclick={() => (pendingReplacement = null)}>keep current pack</button>
            </div>
          {/if}
          {#if artifactAvailability.available && localPacksLoading}
            <p class="muted">loading portable packs…</p>
          {:else if artifactAvailability.available && localPacks.length > 0}
            <ul class="archive-list" role="list">
              {#each localPacks as pack (pack.id)}
                <li>
                  <div>
                    <strong>{pack.namespace}/{pack.name}</strong>
                    <span>{pack.source.repository ? `${pack.source.repository}@${pack.source.revision}` : pack.source.uri}</span>
                    <span>Publisher unverified</span>
                  </div>
                  <div class="actions">
                    <button type="button" disabled={archiveBusy !== null} onclick={() => void exportArchive(pack)}><MorphText text={archiveBusy === `export:${pack.id}` ? "Preparing…" : exportedPack === pack.id ? "Download started" : "export"} numbers={false} /></button>
                    {#if confirmLocalDelete === pack.id}
                      <span>Delete {pack.namespace}/{pack.name} from this device? Export a backup first if you want to keep it.</span>
                      <button type="button" class="danger" disabled={archiveBusy !== null} onclick={() => void deleteArchive(pack)}>Delete pack</button>
                      <button type="button" disabled={archiveBusy !== null} onclick={() => (confirmLocalDelete = null)}>cancel</button>
                    {:else}
                      <button type="button" disabled={archiveBusy !== null} onclick={() => (confirmLocalDelete = pack.id)}>delete</button>
                    {/if}
                  </div>
                </li>
              {/each}
            </ul>
          {/if}
        </section>
      {/if}
      {#if steerRack.loading && steerRack.catalog.length === 0}
        <p class="muted">loading manifolds…</p>
      {:else if steerRack.catalog.length === 0}
        <p class="muted">
          {artifactAvailability.available
            ? browserMode
              ? "No manifolds installed. Import a .drowse or search Hugging Face."
              : "No manifolds installed. Search Hugging Face to find compatible sources."
            : "No manifolds installed."}
        </p>
      {:else}
        <ul class="rows" role="list">
          {#each steerRack.catalog as m (selectorOf(m))}
            {@const key = selectorOf(m)}
            {@const badge = fitBadge(m)}
            <li class="row" {...{ "aria-description": (m.description || key) }}>
              <div class="meta">
                <span class="row-name">{key}</span>
                <span class="row-sub">
                  {m.domain_label} · {m.node_count} nodes
                  {#if badge}
                    <span class="fit-badge fit-{badge}">{badge}</span>
                  {/if}
                  {#if m.fitted_for_session}
                    <span class="fit-tag">fitted</span>
                  {/if}
                  {#if m.stale}
                    <span class="stale">stale</span>
                  {/if}
                </span>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    {:else}
      <div class="search">
        <label class="search-label">
          <span class="vh">search query</span>
          <input
            type="search"
            placeholder="Search Hugging Face…"
            aria-label="Search HF for drowse-manifold repos"
            bind:value={query}
            oninput={scheduleSearch}
          />
        </label>
        {#if !query.trim()}
          <p class="muted">Only public, browser-compatible Drowse packs appear here.</p>
        {:else if searchLoading}
          <p class="muted">searching…</p>
        {:else if searchError}
          <p class="error" role="alert">{searchError}</p>
        {:else if searchResults.length === 0}
          <p class="muted">No Hugging Face packs match “{query.trim()}”. Try a different search.</p>
        {:else}
          <ul class="rows" role="list">
            {#each searchResults as row (selectorOf(row))}
              {@const target = selectorOf(row)}
              {@const inFlight = installing === target}
              {@const badge = fitBadge(row)}
              <li class="row" {...{ "aria-description": (row.description || target) }}>
                <div class="meta">
                  <span class="row-name">{target}</span>
                  <span class="row-sub">
                    {row.domain_label} · {row.node_count} nodes
                    {#if badge}
                      <span class="fit-badge fit-{badge}">{badge}</span>
                    {/if}
                    {#if row.tensor_models.length > 0}
                      <span class="hf-fit-count">
                        · {row.tensor_models.length} fit{row.tensor_models.length === 1 ? "" : "s"}
                      </span>
                    {/if}
                  </span>
                  {#if inFlight && installStage}
                    <span class="install-stage" aria-live="polite"><MorphText text={installStage} /></span>
                  {/if}
                </div>
                <div class="actions">
                  <button
                    type="button"
                    class="act install"
                    disabled={inFlight}
                    onclick={() => void installRow(row)}
                    {...{ "aria-description": (`install ${target}`) }}
                  ><MorphText text={inFlight ? "…" : "install"} /></button>
                </div>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  /* Shared installed/search layout: compact tabs, catalog rows, and a
   * purple install accent for the manifold family. */
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

  .tabs {
    display: flex;
    gap: 0;
    padding: 0 var(--space-5);
  }
  .tabs button {
    min-height: 44px;
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: var(--space-3) var(--space-4);
    color: var(--fg-dim);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      color var(--dur) var(--ease-out),
      border-color var(--dur) var(--ease-out);
  }
  .tabs button.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .tabs button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
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

  .search {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .search-label {
    display: flex;
    flex-direction: column;
  }
  .vh {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
  }
  .search input[type="search"] {
    min-height: 44px;
    background: var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
  .search input[type="search"]:focus-visible {
    outline: 1px solid var(--pillar-manifold);
    outline-offset: -1px;
  }

  .muted {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    margin: 0;
    line-height: 1.4;
  }
  .error {
    margin: 0;
    color: var(--accent-red);
    font-size: var(--text-sm);
    word-break: break-word;
  }

  .archive-tools {
    display: grid;
    gap: var(--space-3);
    padding: var(--surface-padding);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--glass);
  }
  .archive-heading,
  .archive-list li,
  .replace-warning {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
  }
  .archive-heading h2 {
    margin: 0;
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .archive-heading p {
    margin: var(--space-1) 0 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .file-action,
  .archive-tools button {
    display: inline-flex;
    min-height: 44px;
    align-items: center;
    justify-content: center;
    padding: var(--space-2) var(--space-3);
    border: 1px solid transparent;
    border-radius: var(--radius);
    color: var(--pillar-manifold);
    background: var(--bg-elev);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
  }
  .file-action:hover:not(.disabled),
  .archive-tools button:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-manifold) 12%, var(--bg-elev));
  }
  .file-action.disabled,
  .archive-tools button:disabled {
    cursor: not-allowed;
    opacity: 0.48;
  }
  .file-action input {
    position: absolute;
    inset: -1px;
    cursor: pointer;
    opacity: 0;
  }
  .file-action { position: relative; }
  .file-action.disabled input { cursor: not-allowed; }
  .file-action:focus-within,
  .archive-tools button:focus-visible {
    outline: 2px solid var(--pillar-manifold);
    outline-offset: 2px;
  }
  .archive-list {
    display: grid;
    gap: var(--space-2);
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .archive-list li {
    padding-top: var(--space-3);
    border-top: 1px solid var(--glass-line);
  }
  .archive-list li > div:first-child {
    display: grid;
    min-width: 0;
    gap: var(--space-1);
  }
  .archive-list strong,
  .archive-list span {
    overflow-wrap: anywhere;
  }
  .archive-list span {
    color: var(--fg-muted);
    font-size: var(--text-2xs);
  }
  .archive-tools button.danger {
    color: var(--accent-red);
  }
  .replace-warning {
    padding: var(--surface-padding);
    border-radius: var(--radius);
    color: var(--warning-ink);
    background: var(--warning-bg);
    font-size: var(--text-xs);
  }

  @media (max-width: 38rem) {
    .archive-heading,
    .archive-list li,
    .replace-warning {
      align-items: stretch;
      flex-direction: column;
    }
  }

  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .row {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: var(--space-3);
    background: var(--surface-sheen), var(--bg-deep);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--surface-padding);
    transition: background var(--dur) var(--ease-out);
  }
  .row:hover { background: color-mix(in srgb, var(--pillar-manifold) 8%, var(--bg-deep)); }
  .meta {
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .row-name {
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-sub {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .fit-badge {
    display: inline-block;
    margin-inline-start: var(--space-2);
    padding: 0 var(--space-2);
    border-radius: var(--radius);
    text-transform: uppercase;
    font-size: var(--text-2xs);
    letter-spacing: 0.04em;
    border: 1px solid transparent;
    color: var(--pillar-manifold);
    background: color-mix(in srgb, var(--pillar-manifold) 12%, transparent);
  }
  .fit-spectral {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .fit-tag {
    color: var(--accent-green);
    margin-inline-start: var(--space-2);
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .stale {
    color: var(--accent-yellow);
    margin-inline-start: var(--space-2);
  }
  .hf-fit-count {
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  /* Per-stage install narration — replaces the bare spinner while a
     multi-hundred-megabyte HF pull lands. */
  .install-stage {
    display: block;
    color: var(--accent);
    font-size: var(--text-2xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .actions {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .act {
    min-height: 44px;
    background: var(--glass);
    color: var(--pillar-manifold);
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    cursor: pointer;
    transition:
      background var(--dur) var(--ease-out),
      color var(--dur) var(--ease-out);
  }
  .act:hover:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-manifold) 12%, transparent);
  }
  .act:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
</style>
