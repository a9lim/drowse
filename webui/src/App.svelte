<script lang="ts">
  import FluentIcon from "./lib/ui/FluentIcon.svelte";
  import SidebarIcon from "./lib/ui/SidebarIcon.svelte";
  import { tokenInspectorUi, dockTokenDetails, hideTokenDetails, toolPresentation, setToolPresentation, restoreToolPresentation, openToolInSidebar } from "./lib/stores/drawers.svelte";
  import ManifoldBuilderDrawer from "./drawers/ManifoldBuilderDrawer.svelte";
  import ManifoldJobProgress from "./lib/ui/ManifoldJobProgress.svelte";
  import { manifoldJobs } from "./lib/stores/manifoldJobs.svelte";
  import ToolDirectory from "./panels/ToolDirectory.svelte";
  import TokenDrilldownDrawer from "./drawers/TokenDrilldownDrawer.svelte";
  import BottomSheet from "./lib/ui/BottomSheet.svelte";
  import { MOBILE_SHEET_QUERY } from "./lib/bottomSheet";
  import "./lib/style/workspace.css";
  import WorkspaceBackground from "./lib/ui/WorkspaceBackground.svelte";
  import { slidingSelection } from "./lib/slidingSelection";
  // Drowse workbench shell. Conversation owns its response-shaping tools;
  // the loom remains a separate workspace for navigating branches.

  import { onMount, tick } from "svelte";
  import { sessionState } from "./lib/stores/session.svelte";
  import { attachPersistence } from "./lib/stores/persistence.svelte";
  import { fade, fly } from "svelte/transition";

  import ControlsPanel, { type ControlsSection } from "./panels/ControlsPanel.svelte";
  import Chat from "./panels/Chat.svelte";
  import LoomSidebar from "./panels/loom/LoomSidebar.svelte";
  import CommandPalette from "./panels/CommandPalette.svelte";
  import WorkbenchCard from "./panels/WorkbenchCard.svelte";
  import Toaster from "./lib/Toaster.svelte";
  import PageHeader from "./hosted/ui/PageHeader.svelte";
  import WorkbenchMenu from "./lib/ui/WorkbenchMenu.svelte";
  import TabIdentity from "./lib/ui/TabIdentity.svelte";
  import ConversationAutosave from "./lib/ui/ConversationAutosave.svelte";
  import ChatAccentTheme from "./lib/ui/ChatAccentTheme.svelte";
  import { workbenchTabState, type TabState } from "./lib/tabIdentity";
  import { getHostedController } from "./lib/runtime/registry";
  import { runtimeClient } from "./lib/runtime/client";
  import {
    panelIn,
    panelOut,
    scrimIn,
    scrimOut,
  } from "./lib/motion";
  import { userFacingError } from "./lib/runtime/userFacingError";
  import { restoreConversationSnapshot } from "./lib/conversationWorkspace";
  import { resetSettings } from "./lib/stores/settingsReset.svelte";
  import {
    conversationLibrary,
    savedConversationState,
  } from "./lib/stores/savedConversations.svelte";
  import {
    clearConversationOpen,
    peekConversationOpen,
  } from "./hosted/runtime/entryExperience";
  import {
    paletteState,
    closePalette,
    openPalette,
  } from "./lib/stores/palette.svelte";

  // One typed registry row per drawer — component, params, sizing.  The
  // host below renders whichever row ``drawerState.open`` names, so there
  // is no per-drawer branch here to fall out of sync with the union.
  import { DRAWERS, drawerParams } from "./drawers";

  import {
    bootstrap,
    ensureRuntimeChannel,
    drawerState,
    closeDrawer,
    genStatus,
    chatLog,
    sendStop,
    loomTree,
    currentLoomTreeSnapshot,
    loomUiState,
    loomRegenerateActive,
    requestLoomModal,
    openDrawer,
    pushToast,
  } from "./lib/stores.svelte";

  let { onhome, onmodels }: { onhome?: () => Promise<void>; onmodels?: () => Promise<void> } = $props();
  let returningHome = $state(false);

  async function returnHome(destination: "chats" | "models" = "chats"): Promise<void> {
    if (returningHome) return;
    if (creationActive()) {
      pushToast("Finish or cancel creation before leaving this model. Your progress stays visible above the workspace.", { kind: "warning" });
      return;
    }
    if (!onhome) {
      openDrawer("load_conversation");
      return;
    }
    if (genStatus.active && !window.confirm(`Stop the current reply and ${destination === "models" ? "open Models" : "return to Your chats"}? Your conversation will be saved on this device.`)) return;
    returningHome = true;
    try { await (destination === "models" && onmodels ? onmodels() : onhome()); }
    catch (error) {
      pushToast(userFacingError(error, `${destination === "models" ? "Models" : "Your chats"} could not open. Stay here and try again.`), { kind: "error" });
    } finally { returningHome = false; }
  }

  type BootStatus = "loading" | "ready" | "failed";
  type WorkspaceView = "conversation" | "branches" | "controls" | "tools";
  let bootStatus: BootStatus = $state("loading");
  $effect(() => {
    if (bootStatus === "ready") return attachPersistence();
  });
  let bootError: string | null = $state(null);
  let runtimeTabState = $state<TabState | null>(null);
  onMount(() => getHostedController()?.subscribe(snapshot => {
    runtimeTabState = snapshot.lifecycle === "failed" || snapshot.generation.phase === "failed" ? "error"
      : snapshot.lifecycle === "loading" ? "loading"
      : snapshot.lifecycle === "checking" ? "checking"
      : snapshot.fitting.phase === "running" ? "training"
      : snapshot.download.phase === "running" ? "downloading" : null;
  }));
  let drawerEl: HTMLElement | null = $state(null);
  let leftSidebarVisible = $state(true);
  let compactNavigation = $state(false);
  let restoreNavigationButton: HTMLButtonElement | null = $state(null);
  let collapseNavigationButton: HTMLButtonElement | null = $state(null);
  async function setMobileNavigation(visible: boolean) {
    leftSidebarVisible = visible;
    await tick();
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    focusWithoutScrolling(visible ? collapseNavigationButton : restoreNavigationButton);
  }
  let narrowScreen = $state(false);
  let compactTools = $state(false);
  let builderVisited = $state(false);
  let builderParams: unknown = $state(null);
  $effect(() => {
    if (drawerState.open === "manifold_builder") {
      builderVisited = true;
      builderParams = drawerState.params;
    }
  });
  onMount(() => {
    restoreToolPresentation();
    const query = window.matchMedia("(max-width: 1100px)");
    const update = () => { compactTools = query.matches; };
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  });
  onMount(() => {
    const navigationQuery = window.matchMedia("(max-width: 760px)");
    const updateNavigation = () => { compactNavigation = navigationQuery.matches; };
    updateNavigation();
    navigationQuery.addEventListener("change", updateNavigation);
    return () => navigationQuery.removeEventListener("change", updateNavigation);
  });
  onMount(() => {
    const query = window.matchMedia(MOBILE_SHEET_QUERY);
    const update = () => { narrowScreen = query.matches; };
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  });
  const dockedTokenDetails = $derived(tokenInspectorUi.visible);
  const mobileTokenDrawer = $derived(narrowScreen && drawerState.open === "token_drilldown");
  const sideToolOpen = $derived(drawerState.open !== null && drawerState.open !== "token_drilldown" && toolPresentation.mode === "sidebar");
  const toolTakesWorkspace = $derived(sideToolOpen && (compactTools || toolPresentation.expanded));
  const modalDrawerOpen = $derived((drawerState.open !== null && !sideToolOpen) || (narrowScreen && dockedTokenDetails && drawerState.open === null));
  let workspaceView: WorkspaceView = $state("conversation");
  let conversationToolsVisible = $state(false);
  let loomToolsVisible = $state(false);
  const headersVisible = $derived.by(() => workspaceView === "branches" ? loomToolsVisible : conversationToolsVisible);

  function toggleViewTools() {
    if (workspaceView === "branches") loomToolsVisible = !loomToolsVisible;
    else conversationToolsVisible = !conversationToolsVisible;
  }
  let controlsSection: ControlsSection = $state("response");
  const pendingTurn = $derived(chatLog.pendingIndex === null ? null : chatLog.turns[chatLog.pendingIndex]);
  const tabState = $derived(workbenchTabState({
    boot: bootStatus, runtime: runtimeTabState, active: genStatus.active,
    replay: genStatus.replay !== null && genStatus.replay !== undefined,
    thinking: Boolean(pendingTurn?.thinkingTokens?.length && !pendingTurn?.tokens?.length),
    view: (workspaceView as WorkspaceView) === "tools" ? "controls" : workspaceView as Exclude<WorkspaceView, "tools">, section: controlsSection, drawer: drawerState.open, saveStatus: savedConversationState.status,
  }));
  let drawerTrigger: HTMLElement | null = null;
  let previousDrawer: string | null = null;

  function creationActive(): boolean {
    return manifoldJobs.current !== null && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status);
  }

  function beforeLeave(event: BeforeUnloadEvent): void {
    if (!creationActive()) return;
    event.preventDefault();
    event.returnValue = "";
  }

  const workspacePage = $derived(
    workspaceView === "conversation" ? "1" : workspaceView === "branches" ? "2" : workspaceView === "controls" ? "3" : "4",
  );

  function selectWorkspace(view: WorkspaceView): void {
    workspaceView = view;
    if (toolTakesWorkspace) closeDrawer();
  }

  function onWorkspaceRequest(event: Event): void {
    const requested = (event as CustomEvent<
      WorkspaceView | { view: WorkspaceView; section?: ControlsSection }
    >).detail;
    if (typeof requested === "object") {
      if (requested.section === "response" || requested.section === "model" || requested.section === "chat") {
        controlsSection = requested.section;
      }
      selectWorkspace(requested.view);
      return;
    }
    if (requested === "conversation" || requested === "branches" || requested === "controls" || requested === "tools") {
      selectWorkspace(requested);
    }
  }

  const FOCUSABLE = [
    "button:not([disabled])",
    "[href]",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");

  function focusWithoutScrolling(element: HTMLElement | null | undefined): void {
    element?.focus({ preventScroll: true });
  }

  $effect(() => {
    const open = drawerState.open;
    const modal = open !== null && !mobileTokenDrawer && !sideToolOpen;
    if (open !== null && previousDrawer === null) {
      drawerTrigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    }
    if (open !== null && !mobileTokenDrawer && (open !== previousDrawer || modal && !drawerEl?.contains(document.activeElement))) {
      void tick().then(() => {
        const first = [...(drawerEl?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [])].find(el => el.offsetParent !== null);
        focusWithoutScrolling(first ?? drawerEl);
      });
    } else if (open === null && previousDrawer !== null) {
      const trigger = drawerTrigger;
      void tick().then(() => {
        if (trigger?.isConnected) focusWithoutScrolling(trigger);
        else focusWithoutScrolling(document.querySelector<HTMLElement>('[aria-controls="workspace-sidebar"]'));
      });
      drawerTrigger = null;
    }
    previousDrawer = open;
  });

  function onDrawerKeydown(ev: KeyboardEvent): void {
    if (ev.key !== "Tab" || !drawerEl) return;
    const focusable = [...drawerEl.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
      (el) => el.offsetParent !== null,
    );
    if (focusable.length === 0) {
      ev.preventDefault();
      focusWithoutScrolling(drawerEl);
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (ev.shiftKey && (document.activeElement === first || document.activeElement === drawerEl)) {
      ev.preventDefault();
      focusWithoutScrolling(last);
    } else if (!ev.shiftKey && (document.activeElement === last || document.activeElement === drawerEl)) {
      ev.preventDefault();
      focusWithoutScrolling(first);
    }
  }

  function drawerLabel(name: string): string {
    const labels: Record<string, string> = {
      advanced_sampling: "Sampling settings",
      cast: "Role settings",
      compare: "Compare controls by layer",
      correlation: "Compare saved readings",
      health: "Model health",
      appearance: "Appearance",
      help: "Help and shortcuts",
      feedback: "Submit Feedback",
      load_conversation: "Saved chats",
      local_runtime: "Model settings",
      manifold_builder: "Create a concept or scale",
      surface_geometry: "Surface geometry",
      manifold_merge: "Combine response controls",
      manifold_pack: "Downloaded response controls",
      manifolds: "Add manifold",
      node_compare: "Compare conversation branches",
      probe_inspector: "Reading details",
      save_conversation: "Save chat",
      download_chat: "Download chat",
      session_admin: "API access",
      subspace: "Add subspace",
      system_prompt: "System prompt",
      template_lab: "Prompt template lab",
      token_drilldown: "Generated word details",
      transcript: "Conversation transcript",
    };
    return labels[name] ?? name.replaceAll("_", " ");
  }

  async function runBootstrap(): Promise<void> {
    bootStatus = "loading";
    bootError = null;
    try {
      await bootstrap();
      tokenInspectorUi.params = null;
      savedConversationState.activeId = null;
      savedConversationState.avatarSeed = null;
      savedConversationState.accent = "purple";
      savedConversationState.status = "idle";
      const pendingConversation = peekConversationOpen();
      if (pendingConversation) {
        const record = await conversationLibrary.get(pendingConversation.id);
        if (record.modelId !== pendingConversation.modelId) {
          throw new Error("The saved chat changed before it could be opened. Return to Chats and try again.");
        }
        await restoreConversationSnapshot(record.snapshot);
        savedConversationState.activeId = record.id;
        savedConversationState.avatarSeed = record.avatarSeed;
        savedConversationState.accent = record.accent ?? "purple";
        clearConversationOpen();
      } else if (runtimeClient.mode === "browser" && sessionState.info) {
        const tree = currentLoomTreeSnapshot();
        const record = tree && await conversationLibrary.findForTree(sessionState.info.model_id, tree.root_id);
        if (record) {
          await restoreConversationSnapshot({ ...record.snapshot, tree });
          savedConversationState.activeId = record.id;
          savedConversationState.avatarSeed = record.avatarSeed;
          savedConversationState.accent = record.accent ?? "purple";
        } else if (tree?.nodes.length === 1 && tree.active_node_id === tree.root_id) {
          await resetSettings(true);
        }
      }
      // Open the WS eagerly so the first generate doesn't pay connect
      // latency.  Failure here is non-fatal — we'll re-attempt on send.
      try {
        await ensureRuntimeChannel();
      } catch {
        /* ignore — the send paths reopen the socket on demand */
      }
      bootStatus = "ready";
    } catch (e) {
      bootError = userFacingError(
        e,
        "Unable to open the workbench. Reopen the model and try again.",
      );
      bootStatus = "failed";
    }
  }

  onMount(() => {
    void runBootstrap();
    window.addEventListener("drowse:workspace", onWorkspaceRequest);
    return () => window.removeEventListener("drowse:workspace", onWorkspaceRequest);
  });

  // Global keyboard accelerators. Esc stops generation; Cmd/Ctrl-Enter is
  // left for the chat input to handle locally.
  //
  // Loom (phase 3): Ctrl/Cmd+R/E/B/N/D fire the corresponding tree op
  // via the sidebar's modal flow.  Browser Ctrl+B (bold) is suppressed
  // via ``preventDefault`` per Decision 9.
  async function onWindowKey(ev: KeyboardEvent) {
    if (ev.defaultPrevented) return;
    // Escape priority (most-targeted close first):
    //   1. open loom modal / menu — let the sidebar's own Esc handler
    //      (LoomSidebar.svelte::onWindowKey) close it.  We DON'T
    //      preventDefault here so its listener still fires.
    //   2. open drawer — close it.
    //   3. fall-through: stop in-flight gen.
    //
    // The earlier order (gen-stop first) made Esc-during-stream-with-
    // modal-open stop the gen instead of closing the modal — surprising
    // for the n-way regen flow where a user might want to back out of
    // a follow-up modal without killing the stream.
    if (ev.key === "Escape") {
      // Palette overlays everything — its own input handler closes it when
      // focused; this catches Esc after focus wandered (backdrop click-arm,
      // devtools, etc.).
      if (paletteState.open) {
        closePalette();
        ev.preventDefault();
        return;
      }
      if (loomUiState.modalRequest.kind !== null) {
        return;
      }
      if (drawerState.open !== null && (!sideToolOpen || drawerEl?.contains(document.activeElement))) {
        closeDrawer();
        ev.preventDefault();
        return;
      }
      if (genStatus.active) {
        sendStop();
        ev.preventDefault();
        return;
      }
    }

    const mod = ev.ctrlKey || ev.metaKey;
    if (!mod) return;
    // Shift+ctrl combos fall through to the browser; the loom shortcuts
    // use bare Cmd/Ctrl+key.
    if (ev.shiftKey) return;
    const k = ev.key.toLowerCase();

    if (k === "r") {
      ev.preventDefault();
      // Ctrl+R = regenerate the active node when it carries a generation
      // receipt, independent of which seat it occupies.
      const active = loomTree.active_node_id;
      if (!active) return;
      const node = loomTree.nodes.get(active);
      if (node?.recipe) {
        await loomRegenerateActive(1);
      } else {
        // A committed node has no regeneration capability; open the modal
        // to author a fresh continuation instead.
        requestLoomModal("regenerate", { nodeId: active, n: 1 });
      }
      return;
    }
    if (k === "e") {
      ev.preventDefault();
      const active = loomTree.active_node_id;
      if (!active) return;
      const node = loomTree.nodes.get(active);
      requestLoomModal("edit", { nodeId: active, text: node?.text ?? "" });
      return;
    }
    if (k === "b") {
      ev.preventDefault();
      const active = loomTree.active_node_id;
      if (!active) return;
      const node = loomTree.nodes.get(active);
      requestLoomModal("branch", { nodeId: active, text: node?.text ?? "" });
      return;
    }
    if (k === "n") {
      ev.preventDefault();
      requestLoomModal("navpicker", { nodeId: loomTree.active_node_id });
      return;
    }
    if (k === "d") {
      ev.preventDefault();
      const active = loomTree.active_node_id;
      if (!active) return;
      requestLoomModal("delete", { nodeId: active });
      return;
    }
  }
</script>

<svelte:window onkeydown={onWindowKey} onbeforeunload={beforeLeave} />
{#if bootStatus === "ready"}<ChatAccentTheme />{/if}
<TabIdentity state={tabState} title={workspaceView === "tools" && drawerState.open === null && !runtimeTabState && !genStatus.active ? "Tools · Drowse" : undefined} />

{#if bootStatus === "failed"}
  <div class="boot-failed" role="alert">
    <h1>{runtimeClient.mode === "http" ? "offline" : "runtime unavailable"}</h1>
    <p class="message">{bootError}</p>
    <p class="hint">
      {#if runtimeClient.mode === "http"}
        start <code>drowse serve</code>
      {:else}
        retry the local model, or return to device setup
      {/if}
    </p>
    <button type="button" class="retry" onclick={runBootstrap}>retry</button>
    {#if runtimeClient.mode !== "http"}
      <button type="button" class="retry" onclick={() => window.location.assign("/app")}>model setup</button>
    {/if}
  </div>
{:else}
  <div class="shell workspace-material" class:loading={bootStatus === "loading"} class:loom-workspace={workspaceView === "branches"}>
    <a class="skip-link" href="#workspace-main">Skip to workspace</a>
    <div class="workspace-notices" inert={modalDrawerOpen || paletteState.open}>
      {#if bootStatus === "ready"}<ConversationAutosave />{/if}
      {#if !modalDrawerOpen}<ManifoldJobProgress compact />{/if}
    </div>
    <main
      class="layout"
      class:sidebar-collapsed={!leftSidebarVisible}
      class:has-token-sidebar={(dockedTokenDetails && !narrowScreen && !sideToolOpen) || (sideToolOpen && !toolTakesWorkspace)}
      id="workspace-main"
      inert={paletteState.open || bootStatus !== "ready"}
      aria-busy={bootStatus === "loading"}
    >
      <WorkspaceBackground />
      <div class="app-header" id="workbench-header" inert={modalDrawerOpen || returningHome}>
        <PageHeader current="workbench" compact siteNavigation={runtimeClient.mode !== "http"}
          homeHref={runtimeClient.mode !== "http" ? "/" : "#workspace-main"}>
          {#snippet leading()}
            <button type="button" class="sidebar-toggle" aria-expanded={leftSidebarVisible}
              aria-controls="workspace-sidebar" aria-label={leftSidebarVisible ? "Hide left sidebar" : "Show left sidebar"}

              onclick={() => (leftSidebarVisible = !leftSidebarVisible)}><SidebarIcon /></button>
            {#if compactNavigation && !leftSidebarVisible}
              <button type="button" class="sidebar-toggle navigation-restore" bind:this={restoreNavigationButton}
                aria-label="Show navigation bar" aria-expanded="false" aria-controls="workspace-sidebar"
                onclick={() => void setMobileNavigation(true)}><FluentIcon name="down" /></button>
            {/if}
          {/snippet}
          {#snippet actions()}
            {#if !leftSidebarVisible || compactNavigation}
            <WorkbenchMenu hosted={runtimeClient.mode !== "http"} busy={returningHome}
              hasChat={loomTree.nodes.size > 1} generating={genStatus.active}
              toolsLabel={workspaceView === "controls" ? null : workspaceView === "branches" ? "Loom tools" : "chat tools"}
              toolsVisible={headersVisible} onToggleTools={toggleViewTools}
              onChats={() => void returnHome()} onModels={() => void returnHome("models")}
              onDownload={() => openDrawer("download_chat")} onAllTools={openPalette} onHelp={() => openDrawer("help")} />
            {/if}
            <button type="button" class="sidebar-toggle" aria-expanded={dockedTokenDetails}
              aria-controls="workspace-token-sidebar" aria-label={dockedTokenDetails ? "Hide right sidebar" : "Show right sidebar"}

              onclick={(event) => { focusWithoutScrolling(event.currentTarget); if (sideToolOpen) { closeDrawer(); dockTokenDetails(); } else if (dockedTokenDetails) hideTokenDetails(); else dockTokenDetails(); }}><SidebarIcon side="right" /></button>
          {/snippet}
        </PageHeader>
      </div>

      <aside class="app-sidebar t-panel-slide" id="workspace-sidebar" data-open={leftSidebarVisible} aria-hidden={!leftSidebarVisible}
        inert={modalDrawerOpen || !leftSidebarVisible} aria-label="Workspace sidebar">
        <div class="workspace-navigation">
        {#if compactNavigation}
          <button type="button" class="sidebar-toggle mobile-navigation-action" aria-label="Back to Chats"
            disabled={returningHome} onclick={() => void returnHome()}><FluentIcon name="back" /></button>
        {/if}
        <nav class="sidebar-links sidebar-back" aria-label="Back to chats">
          <button type="button" disabled={returningHome} onclick={() => void returnHome()}><FluentIcon name="chats" /><span>Back to Chats</span></button>
          <hr />
        </nav>
        <nav class="workspace-nav" aria-label="Workspace" use:slidingSelection>
          <div class="workspace-group" class:current={workspaceView === "conversation" || workspaceView === "controls"}>
            <button
              type="button"
              class="workspace-parent"
              class:active={workspaceView === "conversation"}
              aria-current={workspaceView === "conversation" ? "page" : undefined}
              aria-describedby="workspace-conversation-description"
              onclick={() => selectWorkspace("conversation")}
            >
              <FluentIcon name="conversation" size={16} />
              <span class="nav-copy"><span class="wide-label">{sessionState.info?.is_base_model ? "Completion" : "Conversation"}</span>
                <span class="nav-description" id="workspace-conversation-description" aria-hidden="true">Write and inspect text</span></span>
              <span class="short-label">{sessionState.info?.is_base_model ? "Text" : "Chat"}</span>
            </button>
            <div class="workspace-subnav" aria-label="Conversation tools">
              <button
                type="button"
                class="workspace-child"
                class:active={workspaceView === "controls"}
                aria-current={workspaceView === "controls" ? "page" : undefined}
                aria-describedby="workspace-controls-description"
                onclick={() => selectWorkspace("controls")}
              >
                <FluentIcon name="controls" size={16} /><span class="nav-copy"><span>Controls</span><span class="nav-description" id="workspace-controls-description" aria-hidden="true">Shape and measure output</span></span>
              </button>
            </div>
          </div>

          <button
            type="button"
            class="workspace-parent"
            class:active={workspaceView === "branches"}
            aria-current={workspaceView === "branches" ? "page" : undefined}
            aria-describedby="workspace-loom-description"
            onclick={() => selectWorkspace("branches")}
          ><FluentIcon name="loom" size={16} /><span class="nav-copy"><span>Loom</span><span class="nav-description" id="workspace-loom-description" aria-hidden="true">Explore alternate paths</span></span></button>

          <button type="button" class="workspace-parent" class:active={workspaceView === "tools"}
            aria-current={workspaceView === "tools" ? "page" : undefined} onclick={() => selectWorkspace("tools")}
          ><FluentIcon name="controls" size={16} /><span class="nav-copy"><span>Tools</span><span class="nav-description" aria-hidden="true">Create, inspect, and customize</span></span></button>

        </nav>
        {#if compactNavigation}
          <button type="button" class="sidebar-toggle mobile-navigation-action" bind:this={collapseNavigationButton}
            aria-label="Hide navigation bar" aria-expanded="true" aria-controls="workspace-sidebar"
            onclick={() => void setMobileNavigation(false)}><FluentIcon name="up" /></button>
        {/if}
        <nav class="sidebar-links" aria-label="Library and tools">
          {#if runtimeClient.mode !== "http"}<button type="button" disabled={returningHome} onclick={() => void returnHome("models")}><FluentIcon name="models" /><span>Models</span></button>{/if}
          <span class="sidebar-label">Workspace</span>
          <button type="button" onclick={() => openToolInSidebar("manifold_builder")}><FluentIcon name="controls" /><span>Create a concept or scale</span></button>
          {#if workspaceView !== "controls"}
            <button type="button" aria-pressed={headersVisible} onclick={toggleViewTools}><FluentIcon name="controls" /><span>{headersVisible ? "Hide" : "Show"} {workspaceView === "branches" ? "Loom tools" : "chat tools"}</span></button>
          {/if}
          {#if loomTree.nodes.size > 1}<button type="button" disabled={genStatus.active} onclick={() => openDrawer("download_chat")}><FluentIcon name="download" /><span>Download chat</span></button>{/if}
          <button type="button" onclick={openPalette}><FluentIcon name="search" /><span>All tools</span></button>
          <button type="button" onclick={() => openDrawer("appearance")}><FluentIcon name="appearance" /><span>Appearance</span></button>
          <button type="button" onclick={() => openDrawer("help")}><FluentIcon name="help" /><span>Help and shortcuts</span></button>
          <button type="button" onclick={(event) => { focusWithoutScrolling(event.currentTarget); openDrawer("feedback"); }}><FluentIcon name="conversation" /><span>Submit Feedback</span></button>
        </nav>
        </div>

        <div class="sidebar-context" id="workbench-context">
          <button
            type="button"
            class="model-summary"
            aria-label="Open model controls"
            onclick={() => {
              controlsSection = "model";
              workspaceView = "controls";
            }}
          ><WorkbenchCard /></button>
        </div>
      </aside>

      <div class="workspace-frame t-page-slide" data-page={workspacePage} inert={toolTakesWorkspace} aria-hidden={toolTakesWorkspace}>
        <section
          class="workspace-page t-page conversation-page"
          data-page-id="1"
          aria-labelledby="conversation-title"
          aria-hidden={workspaceView !== "conversation"}
          inert={workspaceView !== "conversation" || modalDrawerOpen}
        >
          <h1 id="conversation-title" class="workspace-title-sr">{sessionState.info?.is_base_model ? "Continue text with your model" : "Talk with your model"}</h1>
          <div class="workspace-surface chat-zone"><Chat headersVisible={conversationToolsVisible} /></div>
        </section>

        <section
          class="workspace-page t-page branches-page"
          data-page-id="2"
          aria-labelledby="branches-title"
          aria-hidden={workspaceView !== "branches"}
          inert={workspaceView !== "branches" || modalDrawerOpen}
        >
          <h1 id="branches-title" class="workspace-title-sr">See every path</h1>
          <div class="workspace-surface loom-zone"><LoomSidebar active={workspaceView === "branches"} headersVisible={loomToolsVisible} /></div>
        </section>

        <section
          class="workspace-page t-page controls-page"
          data-page-id="3"
          aria-labelledby="controls-title"
          aria-hidden={workspaceView !== "controls"}
          inert={workspaceView !== "controls" || modalDrawerOpen}
        >
          <h1 id="controls-title" class="workspace-title-sr">{sessionState.info?.is_base_model ? "Completion" : "Response"}, model, and chat controls</h1>
          <div class="workspace-surface rack-zone">
            <ControlsPanel bind:section={controlsSection} />
          </div>
        </section>
        <section class="workspace-page t-page tools-page" data-page-id="4" aria-label="Tools"
          aria-hidden={workspaceView !== "tools"} inert={workspaceView !== "tools" || modalDrawerOpen}>
          <ToolDirectory />
        </section>
      </div>

      <BottomSheet enabled={narrowScreen} open={dockedTokenDetails && drawerState.open === null} onclose={hideTokenDetails}>
        {#snippet children(dismiss)}
        <div class="drawer token-details docked t-panel-slide" class:mobile-sheet-content={narrowScreen} id="workspace-token-sidebar"
        data-open={dockedTokenDetails && !sideToolOpen} aria-hidden={!dockedTokenDetails || sideToolOpen}
        inert={!dockedTokenDetails || drawerState.open !== null} role={narrowScreen ? undefined : "complementary"} aria-label={narrowScreen ? undefined : "Generated word details"}>
        {#if tokenInspectorUi.docked}
          <TokenDrilldownDrawer docked mobile={narrowScreen} onclose={dismiss} params={tokenInspectorUi.params} active={dockedTokenDetails && drawerState.open === null} />
        {/if}
        </div>
        {/snippet}
      </BottomSheet>

      {#if drawerState.open !== null || builderVisited}
        {@const entry = DRAWERS[drawerState.open ?? "manifold_builder"]}
        {#if drawerState.open !== null && !mobileTokenDrawer && !sideToolOpen}
        <div
          class="drawer-backdrop"
          aria-hidden="true"
          onclick={closeDrawer}
          in:fade={scrimIn()}
          out:fade={scrimOut()}
        ></div>
        {/if}
        <BottomSheet enabled={mobileTokenDrawer} open={drawerState.open !== null} onclose={closeDrawer}>
          {#snippet children(dismiss)}
        <div
          bind:this={drawerEl}
          class="drawer"
          class:docked={sideToolOpen}
          class:workspace-tool={toolTakesWorkspace}
          hidden={drawerState.open === null}
          class:narrow={entry.narrow}
          class:token-details={drawerState.open === "token_drilldown"}
          class:mobile-sheet-content={mobileTokenDrawer}
          class:download-confirm={drawerState.open === "download_chat"}
          role={drawerState.open === null || mobileTokenDrawer ? undefined : sideToolOpen ? "complementary" : "dialog"}
          aria-modal={drawerState.open === null || mobileTokenDrawer || sideToolOpen ? undefined : "true"}
          aria-label={drawerState.open === null || mobileTokenDrawer ? undefined : drawerLabel(drawerState.open)}
          tabindex="-1"
          onkeydown={event => { if (!mobileTokenDrawer && !sideToolOpen) onDrawerKeydown(event); }}
          in:fly={mobileTokenDrawer || sideToolOpen ? { duration: 0 } : panelIn(24)}
          out:fly={mobileTokenDrawer || sideToolOpen ? { duration: 0 } : panelOut(14)}
          onintrostart={(event) => { event.currentTarget.inert = false; }}
          onoutrostart={(event) => { event.currentTarget.inert = true; }}
        >
          {#if drawerState.open !== null && drawerState.open !== "token_drilldown"}
            <div class="tool-presentation" role="group" aria-label="Tool presentation">
              <div class="presentation-options">
                <button type="button" aria-pressed={sideToolOpen} onclick={() => setToolPresentation("sidebar")}>Side panel</button>
                <button type="button" aria-pressed={!sideToolOpen} onclick={() => setToolPresentation("dialog")}>Dialog</button>
              </div>
              {#if sideToolOpen && !compactTools}<button type="button" aria-pressed={toolPresentation.expanded} onclick={() => (toolPresentation.expanded = !toolPresentation.expanded)}>{toolPresentation.expanded ? "Restore width" : "Expand"}</button>{/if}
            </div>
          {/if}
          {#if modalDrawerOpen}<ManifoldJobProgress compact />{/if}
          {#if builderVisited}
            <div class="tool-content" hidden={drawerState.open !== "manifold_builder"} inert={drawerState.open !== "manifold_builder"}>
              <ManifoldBuilderDrawer params={builderParams} />
            </div>
          {/if}
          {#if drawerState.open !== null && drawerState.open !== "manifold_builder"}
            <entry.component params={drawerParams(drawerState.open, drawerState.params)} mobile={mobileTokenDrawer} onclose={dismiss} />
          {/if}
        </div>
          {/snippet}
        </BottomSheet>
      {/if}
    </main>

    {#if bootStatus === "loading"}
      <div
        class="boot-loading"
        role="status"
        aria-live="polite"
        in:fade={scrimIn()}
        out:fade={scrimOut()}
      >
        <span class="loading-pulse loading-placeholder">Opening workbench…</span>
      </div>
    {/if}

    <Toaster />
    <CommandPalette />
  </div>
{/if}

<style>
  .shell {
    --surface-padding: var(--workspace-surface-padding, 16px);
    --surface-gutter: var(--surface-padding);
    --panel-padding: var(--surface-padding);
    --drawer-gutter-inline: var(--surface-padding);
    --drawer-gutter-block: var(--surface-padding);
    --page-gutter: var(--surface-padding);
    --workspace-gutter: var(--surface-padding);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    height: 100vh;
    height: 100dvh;
    width: 100%;
    min-width: 0;
    min-height: 0;
    background: var(--bg);
    color: var(--fg);
    overflow: clip;
  }
  .workspace-notices {
    max-height: 25dvh;
    overflow-y: auto;
    overscroll-behavior: contain;
  }
  .shell.loading {
    cursor: wait;
    /* Keep the real frame visible for orientation while the explicit
     * readiness veil below prevents controls racing their source data. */
    color-scheme: inherit;
  }
  .shell.loading .layout {
    filter: saturate(0.72) brightness(0.82);
  }
  .boot-loading {
    position: fixed;
    inset: 0;
    z-index: calc(var(--z-drawer) + 20);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    color: var(--fg-strong);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    letter-spacing: 0.04em;
    text-transform: lowercase;
    background: var(--scrim-soft);
    backdrop-filter: blur(1px);
    pointer-events: none;
  }
  /* Sheet host — analysis tools float in from the right as a rounded
   * sheet inset from the frame edges (v2: no full-height wall).  Backdrop
   * blurs the bench underneath so the live data reads as "behind", not
   * "gone". */
  .drawer-backdrop {
    position: fixed;
    inset: 0;
    background: var(--scrim);
    backdrop-filter: blur(2px);
    z-index: var(--z-drawer);
    border: 0;
    cursor: pointer;
  }
  .drawer {
    position: fixed;
    top: var(--space-4);
    inset-inline-end: var(--space-4);
    bottom: var(--space-4);
    width: min(980px, 78%);
    background: var(--surface-sheen), color-mix(in srgb, var(--popup-bg) 90%, transparent);
    backdrop-filter: blur(12px);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    z-index: calc(var(--z-drawer) + 1);
    display: flex;
    flex-direction: column;
    box-shadow: var(--popup-shadow);
    overflow: hidden;
  }
  :global(:root:not([data-theme="light"])) .drawer.token-details {
    background: color-mix(in srgb, var(--bg-elev) 90%, transparent);
  }
  /* Forms / pickers — sized to their content rather than the wide
   * analysis panel. */
  .drawer.narrow {
    width: min(480px, 92%);
  }
  .drawer.download-confirm {
    inset: 0;
    margin: auto;
    width: min(480px, calc(100% - var(--surface-padding) * 2));
    height: fit-content;
    max-height: calc(100% - var(--surface-padding) * 2);
    overflow-y: auto;
  }
  .drawer[role="dialog"]:focus {
    /* Programmatic fallback focus must not make the whole dialog look selected. */
    outline: none !important;
  }

  /* Boot-failed gate — sits over the whole viewport since the rest of
   * the shell can't function without a session. */
  .boot-failed {
    position: fixed;
    inset: 0;
    background: var(--bg-deep);
    color: var(--fg);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-4);
    padding: var(--surface-padding);
    text-align: center;
  }
  .boot-failed h1 {
    color: var(--accent-red);
    margin: 0;
    font-size: var(--text-lg);
  }
  .boot-failed .message {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    margin: 0;
    max-width: 70ch;
    word-break: break-word;
  }
  .boot-failed .hint {
    color: var(--fg-muted);
    margin: 0;
    font-size: var(--text-sm);
  }
  .boot-failed code {
    background: var(--bg-elev);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius);
    color: var(--accent);
  }
  .retry {
    margin-top: var(--space-5);
    background: var(--bg-elev);
    color: var(--accent);
    border: 0;
    padding: var(--space-3) var(--space-6);
    border-radius: var(--radius);
    font-size: var(--text-sm);
    transition: background var(--dur) var(--ease-out);
  }
  .retry:hover {
    background: var(--accent-subtle);
  }

  /* Drawer base rules are declared after the compact layout block, so keep
   * this size override at the end of the cascade. */
  @media (max-width: 1279px) {
    .drawer {
      top: var(--space-2);
      inset-inline-end: var(--space-2);
      bottom: var(--space-2);
      width: calc(100% - 2 * var(--space-2));
      max-width: 980px;
    }
    .drawer.narrow {
      width: min(480px, calc(100% - 2 * var(--space-2)));
    }
  }

  @media (max-width: 760px) {
    .chat-zone {
      padding-bottom: max(var(--surface-padding), env(safe-area-inset-bottom));
    }
    .loom-zone,
    .rack-zone {
      box-sizing: border-box;
      padding-bottom: env(safe-area-inset-bottom);
    }
    .drawer,
    .drawer.narrow {
      top: max(var(--space-2), env(safe-area-inset-top));
      inset-inline-start: max(var(--space-2), env(safe-area-inset-left));
      inset-inline-end: max(var(--space-2), env(safe-area-inset-right));
      bottom: max(var(--space-2), env(safe-area-inset-bottom));
      width: auto;
    }
    .boot-failed {
      padding-top: max(var(--surface-padding), env(safe-area-inset-top));
      padding-inline-end: max(var(--surface-padding), env(safe-area-inset-right));
      padding-bottom: max(var(--surface-padding), env(safe-area-inset-bottom));
      padding-inline-start: max(var(--surface-padding), env(safe-area-inset-left));
    }
  }

  /* Drift-inspired focus shell: one clear task, calm chrome, and every
   * advanced surface one explicit step away. */
  .skip-link {
    position: fixed;
    z-index: calc(var(--z-modal) + 50);
    inset-block-start: var(--space-4);
    inset-inline-start: var(--space-4);
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius);
    background: var(--fg);
    color: var(--bg-deep);
    transform: translateY(-160%);
  }

  .skip-link:focus {
    transform: translateY(0);
  }

  .layout {
    --sidebar-duration: 300ms;
    --sidebar-ease: var(--panel-ease);
    --panel-open-dur: var(--sidebar-duration);
    --panel-close-dur: var(--sidebar-duration);
    --panel-blur: 0px;
    --left-sidebar-width: 13.25rem;
    --right-sidebar-width: clamp(20rem, 36vw, 30rem);
    position: relative;
    isolation: isolate;
    display: grid;
    grid-template-columns: var(--left-sidebar-width) minmax(0, 1fr) 0rem;
    grid-template-rows: max-content minmax(0, 1fr);
    gap: 0;
    min-width: 0;
    min-height: 0;
    overflow: clip;
    background: var(--bg);
    transition: grid-template-columns var(--sidebar-duration) var(--sidebar-ease), grid-template-rows var(--sidebar-duration) var(--sidebar-ease);
  }

  .app-header {
    grid-column: 1 / -1;
    min-height: min-content;
    justify-content: flex-start;
    z-index: 3;
    background: var(--workspace-panel-bg);
    border-bottom: 0;
  }

  .layout.sidebar-collapsed { grid-template-columns: 0rem minmax(0, 1fr) 0rem; }
  .layout.has-token-sidebar {
    grid-template-columns: var(--left-sidebar-width) minmax(0, 1fr) var(--right-sidebar-width);
  }
  .layout.sidebar-collapsed.has-token-sidebar { grid-template-columns: 0rem minmax(0, 1fr) var(--right-sidebar-width); }
  .sidebar-toggle {
    display: inline-grid;
    place-items: center;
    flex: none;
    width: var(--control-target);
    height: var(--control-target);
    padding: 0;
    border: 0;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-muted);
    transition: color var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out);
  }
  .sidebar-toggle:hover { color: var(--fg); background: var(--glass); }
  .sidebar-toggle:active { scale: var(--press-scale); }

  .app-sidebar {
    grid-column: 1;
    grid-row: 2;
    width: var(--left-sidebar-width);
    z-index: 2;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    padding: var(--surface-padding) var(--space-2);
    overflow-y: auto;
    border-inline-end: 0;
    background: color-mix(in srgb, var(--bg-elev) 90%, transparent);
  }

  .app-sidebar.t-panel-slide { transform: translateX(-100%); }
  .app-sidebar.t-panel-slide[data-open="true"] { transform: translateX(0); }
  .t-panel-slide { visibility: hidden; }
  .t-panel-slide[data-open="true"] { visibility: visible; }
  .t-panel-slide[data-open="false"] { transition: transform var(--sidebar-duration) var(--sidebar-ease), opacity var(--sidebar-duration) var(--sidebar-ease), filter var(--sidebar-duration) var(--sidebar-ease), visibility 0s var(--sidebar-duration); }

  .workspace-nav {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .workspace-navigation {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }

  .workspace-nav button {
    position: relative;
    width: 100%;
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-3);
    border: 0;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    text-align: start;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  .workspace-nav button:hover:not(.active) {
    background: var(--bg-hover);
    color: var(--fg-strong);
  }

  .workspace-nav button:active {
    transform: scale(var(--press-scale));
  }

  .workspace-nav button.active {
    background: var(--accent-subtle);
    color: var(--fg);
  }

  .workspace-group {
    display: grid;
    gap: 0;
  }

  .workspace-parent {
    font-weight: var(--weight-structure);
  }

  .workspace-nav button {
    display: inline-flex;
    align-items: center;
    gap: var(--control-label-gap);
  }

  .workspace-subnav {
    display: grid;
    gap: var(--space-1);
    padding-inline-start: 0;
  }

  .workspace-nav .workspace-child {
    min-height: 2.35rem;
    color: var(--fg-subtle);
    font-size: var(--text-sm);
  }

  .workspace-nav .workspace-child::before {
    display: none;
  }

  .workspace-nav .workspace-child.active::before {
    background: var(--accent);
  }

  .short-label { display: none; }
  .nav-copy { display: grid; gap: 0; min-width: 0; }
  .nav-description { color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-normal); line-height: 1.4; }
  .sidebar-links { display: grid; gap: var(--space-1); margin-top: var(--space-lg); }
  .sidebar-back { margin-top: 0; }
  .sidebar-back hr { margin: var(--space-2) var(--space-3); border: 0; border-block-start: 1px solid var(--glass-line); }
  .sidebar-label { padding: var(--space-sm) var(--space-3) var(--space-xs); color: var(--fg-muted); font-size: var(--text-xs); }
  .sidebar-links button { display: flex; align-items: center; gap: var(--control-label-gap); min-width: 0; min-height: var(--control-target); padding: var(--space-xs) var(--space-3); border: 0; border-radius: var(--radius); background: transparent; color: var(--fg-dim); font-size: var(--text-sm); text-align: start; }
  .sidebar-links button:hover:not(:disabled) { background: var(--bg-hover); color: var(--fg); }
  .sidebar-links button, .sidebar-toggle { transition: background var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .sidebar-links button:active:not(:disabled), .sidebar-toggle:active { scale: var(--press-scale); }

  .sidebar-context {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: var(--space-2);
    margin-block-start: auto;
    padding-block-start: var(--space-6);
    min-width: 0;
  }

  .model-summary {
    display: block;
    width: 100%;
    min-width: 0;
    padding: 0;
    border: 1px solid transparent;
    border-radius: var(--radius);
    background: transparent;
    color: inherit;
    text-align: start;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  .model-summary:hover {
    border-color: var(--glass-line);
    background: var(--glass);
  }

  .model-summary:active {
    transform: scale(var(--press-scale));
  }

  .workspace-frame {
    grid-column: 2;
    grid-row: 2;
    min-height: 0;
    overflow: clip;
  }

  .workspace-page {
    display: grid;
    grid-template-rows: minmax(0, 1fr);
    gap: 0;
    min-width: 0;
    min-height: 0;
    overflow: clip;
    padding: 0;
  }

  .conversation-page .workspace-surface {
    width: 100%;
  }

  .workspace-title-sr {
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

  .workspace-surface {
    min-width: 0;
    min-height: 0;
    overflow: clip;
    border: 0;
    border-radius: 0;
    background: transparent;
  }

  .chat-zone {
    --composer-space: var(--space-sm);
    display: flex;
    flex-direction: column;
    padding: var(--surface-padding);
    background: transparent;
  }

  .loom-zone,
  .rack-zone {
    display: flex;
    flex-direction: column;
    padding: 0;
    background: transparent;
  }

  .workspace-page.branches-page {
    padding: 0;
  }

  .workspace-surface.loom-zone {
    border-radius: 0;
    background: var(--bg);
    box-shadow: none;
  }

  /* transitions.dev — page side-by-side */
  .t-page-slide {
    position: relative;
  }
  .t-page-slide .t-page[data-page-id="1"] {
    --t-page-from-x: calc(var(--page-slide-distance) * -1);
  }
  .t-page-slide .t-page[data-page-id="2"] {
    --t-page-from-x: var(--page-slide-distance);
  }
  .t-page-slide .t-page {
    position: absolute;
    inset: 0;
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transform: translateX(calc(var(--t-page-from-x, 0px) * var(--page-exit-enabled)));
    filter: blur(calc(var(--page-blur) * var(--page-exit-enabled)));
    transition:
      opacity   var(--page-fade-dur)  var(--page-fade-ease),
      transform var(--page-slide-dur) var(--page-slide-ease),
      filter    var(--page-slide-dur) var(--page-slide-ease);
  }
  .t-page-slide[data-page="1"] .t-page[data-page-id="1"],
  .t-page-slide[data-page="2"] .t-page[data-page-id="2"] {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateX(0);
    filter: blur(0);
    transition-delay: var(--page-stagger);
  }

  .t-page-slide .t-page[data-page-id="3"] {
    --t-page-from-x: var(--page-slide-distance);
  }

  .t-page-slide[data-page="3"] .t-page[data-page-id="3"] {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateX(0);
    filter: blur(0);
    transition-delay: var(--page-stagger);
  }

  .t-page-slide[data-page="4"] .t-page[data-page-id="4"] {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: none;
    filter: none;
  }

  .drawer[hidden], .tool-content[hidden] { display: none !important; }
  .workspace-frame[aria-hidden="true"] { opacity: 0; }
  .tool-content { flex: 1; display: flex; flex-direction: column; min-height: 0; min-width: 0; }
  .tool-content > :global(*) { flex: 1; min-height: 0; }
  .tool-presentation { display: flex; justify-content: space-between; gap: var(--space-sm); padding: var(--surface-padding); padding-block-end: 0; flex: none; flex-wrap: wrap; }
  .presentation-options { display: flex; gap: var(--space-xs); padding: calc(var(--radius-group) - var(--radius)); border-radius: var(--radius-group); background: var(--workspace-field-bg); box-shadow: var(--shadow-well); }
  .tool-presentation button { min-width: var(--control-target); min-height: var(--control-target); padding: var(--space-xs) var(--space-sm); border: 1px solid transparent; border-radius: var(--radius); background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-sm); transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .tool-presentation button[aria-pressed="true"] { background: var(--workspace-neutral-bg); color: var(--fg); }
  .tool-presentation button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .tool-presentation > button:active { scale: var(--press-scale); }
  @media (hover: hover) { .tool-presentation button:hover { background: var(--workspace-neutral-hover); color: var(--fg); } }
  @media (forced-colors: active) { .tool-presentation button[aria-pressed="true"] { border-color: Highlight; } }

  .drawer {
    top: var(--surface-padding);
    inset-inline-end: var(--surface-padding);
    bottom: var(--surface-padding);
    width: min(66rem, calc(100% - var(--surface-padding) * 2));
  }

  .drawer.narrow { width: min(34rem, calc(100% - var(--surface-padding) * 2)); }

  .drawer.docked {
    position: relative;
    inset: auto;
    grid-column: 3;
    grid-row: 2;
    width: var(--right-sidebar-width);
    justify-self: end;
    min-width: 0;
    min-height: 0;
    border: 0;
    border-inline-start: 1px solid var(--glass-line);
    border-radius: 0;
    background: color-mix(in srgb, var(--bg-elev) 90%, transparent);
    box-shadow: none;
    z-index: 2;
    container-type: inline-size;
  }
  .drawer.docked.t-panel-slide { transform: translateX(100%); }
  .drawer.docked.t-panel-slide[data-open="true"] { transform: translateX(0); }
  :global(:root:not([data-theme="light"])) .drawer.token-details.docked {
    background: color-mix(in srgb, var(--bg-elev) 90%, transparent);
  }

  @media (prefers-reduced-motion: reduce) {
    .t-page-slide .t-page { transition: none !important; }
    .layout { transition: none !important; }
  }

  @media (prefers-reduced-transparency: reduce), (prefers-contrast: more), (forced-colors: active) {
    .drawer, .drawer.docked, .app-sidebar {
      background: Canvas !important;
      backdrop-filter: none;
    }
  }

  @media (max-width: 760px) {
    .layout, .layout.sidebar-collapsed, .layout.has-token-sidebar, .layout.sidebar-collapsed.has-token-sidebar {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: max-content minmax(0, calc(var(--control-target) + var(--space-2) * 2 + var(--space-1))) minmax(0, 1fr) minmax(0, 0fr);
    }
    .layout.sidebar-collapsed { grid-template-rows: max-content minmax(0, 0px) minmax(0, 1fr) minmax(0, 0fr); }
    .drawer.docked {
      grid-column: 1;
      grid-row: 4;
      width: 100%;
      border-inline-start: 0;
      border-top: 1px solid var(--glass-line);
    }
    .app-header {
      grid-column: 1;
      grid-row: 1;
    }
    .app-sidebar {
      grid-column: 1;
      grid-row: 2;
      width: 100%;
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: var(--space-4);
      min-height: 0;
      padding:
        max(var(--space-2), env(safe-area-inset-top))
        max(var(--surface-padding), env(safe-area-inset-right))
        var(--space-2)
        max(var(--surface-padding), env(safe-area-inset-left));
      overflow: clip;
      border-inline-end: 0;
      border-bottom: 1px solid var(--grid-line);
      background: color-mix(in srgb, var(--bg-elev) 90%, transparent);
    }
    .app-sidebar.t-panel-slide { transform: translateY(-12px); }
    .app-sidebar.t-panel-slide[data-open="true"] { transform: translateY(0); }
    .workspace-frame {
      grid-column: 1 / -1;
      grid-row: 3;
    }
    .workspace-nav {
      display: flex;
      flex-direction: row;
      align-items: center;
      width: fit-content;
      min-width: 0;
      gap: 0;
      padding: calc(var(--space-1) / 2);
      border-radius: var(--radius-lg);
      background: var(--glass);
    }
    .workspace-navigation {
      flex-direction: row;
      flex-wrap: nowrap;
      align-items: center;
      justify-content: center;
      width: 100%;
      max-width: 100%;
      flex: 1 1 auto;
    }
    .workspace-group,
    .workspace-subnav {
      display: contents;
    }
    .workspace-nav button {
      width: auto;
      border-radius: var(--radius);
      white-space: nowrap;
      text-align: center;
      justify-content: center;
    }
    .workspace-nav .workspace-child { min-height: var(--control-target); }
    .workspace-nav .workspace-child::before { display: none; }
    .sidebar-context { display: none; }
    .sidebar-links, .nav-description { display: none; }
  }

  @media (max-width: 680px) {
    .app-sidebar {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-2);
      padding-inline: max(var(--surface-padding), env(safe-area-inset-left)) max(var(--surface-padding), env(safe-area-inset-right));
    }
    .workspace-nav {
      display: grid;
      grid-template-columns: repeat(4, auto);
      width: 100%;
      max-width: 100%;
      margin-inline: auto;
    }
    .workspace-group,
    .workspace-subnav { display: contents; }
    .workspace-nav button { min-width: 0; padding-inline: var(--space-2); }
    .wide-label { display: none; }
    .short-label { display: inline; }
    .drawer,
    .drawer.narrow {
      top: max(var(--surface-padding), env(safe-area-inset-top));
      inset-inline-start: max(var(--surface-padding), env(safe-area-inset-left));
      inset-inline-end: max(var(--surface-padding), env(safe-area-inset-right));
      bottom: max(var(--surface-padding), env(safe-area-inset-bottom));
      width: auto;
    }
  }

  @media (max-width: 420px) {
    .app-sidebar { padding-inline: max(var(--space-xs), env(safe-area-inset-left)) max(var(--space-xs), env(safe-area-inset-right)); }
    .workspace-navigation { justify-content: center; gap: 0; }
    .workspace-nav { width: 100%; grid-template-columns: minmax(0, 1fr) max-content minmax(0, 1fr) minmax(0, 1fr); }
    .workspace-nav button { padding-inline: var(--space-1); }
    .workspace-nav :global(.fluent-icon) { display: none; }
  }

  @media (min-width: 621px) and (max-height: 600px) {
    .chat-zone:not(:has(:global(.chat-header))) { padding: 0; }
  }
  .drawer.docked.narrow { width: var(--right-sidebar-width); }
  .drawer.docked.workspace-tool {
    position: relative;
    inset: auto;
    grid-column: 2 / -1;
    grid-row: 2;
    width: 100%;
    max-width: none;
    height: 100%;
    z-index: 3;
    background: var(--workspace-panel-bg);
  }
  @media (max-width: 760px) {
    .drawer.docked.workspace-tool { grid-column: 1; grid-row: 3; width: 100%; border: 0; }
  }
</style>
