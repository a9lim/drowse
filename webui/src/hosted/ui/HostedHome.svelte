<script lang="ts">
  import { onConversationLibraryChanged } from "../../lib/stores/savedConversations.svelte";
  import { registerHomeController } from "../../lib/homeController";
  import MorphText from "../../lib/ui/MorphText.svelte";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import BaseModelTag from "../../lib/ui/BaseModelTag.svelte";
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import "../../lib/style/workspace.css";
  import { Blobatar } from "@blobatar/svelte";
  import { onDestroy, onMount, tick } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import ChatAccentPicker from "../../lib/ui/ChatAccentPicker.svelte";
  import { chatAccentStyle, type ChatAccent } from "../../lib/chatAccent";
  import { animatedDetails } from "../../lib/animatedDetails";
  import { slide } from "svelte/transition";
  import { flip } from "svelte/animate";
  import { cubicOut } from "svelte/easing";
  import { collapseIn, collapseOut, motionDuration } from "../../lib/motion";

  import { ConversationLibraryError, displayModelName, randomAvatarSeed, summarizeConversation, type SavedConversationIssue, type SavedConversationRecord, type SavedConversationSummary } from "../../lib/conversationLibrary";
  import { conversationLibrary, flushConversationAutosave, requestPersistentConversationStorage } from "../../lib/stores/savedConversations.svelte";
  import { downloadAllChatBackups, downloadChatBackup, findImportedChat, readChatBackupFiles, saveChatImport } from "../../lib/chatBackup";
  import Select from "../../lib/Select.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import SavedChatMenu from "./SavedChatMenu.svelte";
  import ModelProviderLogo from "./ModelProviderLogo.svelte";
  import ModelDownloadChoices from "./ModelDownloadChoices.svelte";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { reloadInstructions, appRefreshSafetyMessage } from "../runtime/chunkRecovery";
  import { formatEtaRange } from "../../lib/runtime/eta";
  import {
    clearConversationOpen,
    queueConversationOpen,
    readEntryMemory,
  } from "../runtime/entryExperience";
  import type { HostedModelOption, HostedShellController, HostedShellSnapshot } from "./types";

  let {
    controller,
    snapshot,
    onChooseModels,
    onChats,
    workbenchError = null,
    workbenchNeedsReload = false,
    workbenchReloading = false,
    onRetryWorkbench,
  }: {
    controller: HostedShellController;
    snapshot: HostedShellSnapshot;
    onChooseModels: (modelVariantId?: string, conversation?: SavedConversationSummary) => void;
    /** Runs when the current "Chats" link is selected; keeps it from reloading the page. */
    onChats?: () => void;
    workbenchError?: string | null;
    workbenchNeedsReload?: boolean;
    workbenchReloading?: boolean;
    onRetryWorkbench?: () => void;
  } = $props();

  let conversations: SavedConversationSummary[] = $state([]);
  let issues: SavedConversationIssue[] = $state([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let openingId = $state<string | null>(null);
  let changingId = $state<string | null>(null);
  let duplicatingId = $state<string | null>(null);
  const changingAvatars = new SvelteSet<string>();
  const changingAccents = new SvelteSet<string>();
  let revealedAvatarId = $state<string | null>(null);
  let confirmDeleteId = $state<string | null>(null);
  let editingId = $state<string | null>(null);
  let editingName = $state("");
  let renameInput: HTMLInputElement | null = $state(null);
  let newChatModelId = $state<string | null>(null);
  let persistenceRequesting = $state(false);
  let persistenceAttempted = $state(false);
  let storageGuideOpen = $state(false);
  let importing = $state(false);
  let importInput: HTMLInputElement | null = $state(null);
  let backupNotice = $state("");
  const selectedRecoveryIds = new SvelteSet<string>();
  const approvedRecoveryIds = new Set<string>();
  let recoveryQueue: string[] = $state([]);
  let activeRecoveryId = $state<string | null>(null);
  let recovering = $state(false);
  let recoveryError = $state<string | null>(null);
  let recoveryWarning = $state<HostedModelOption | null>(null);
  let recoveryChat: SavedConversationSummary | null = null;
  let recoveryNotice: HTMLElement | null = $state(null);
  let destroyed = false;
  const entryMemory = readEntryMemory();
  const persistenceRequestAvailable = typeof navigator.storage?.persist === "function";

  const SORT_KEY = "drowse.chat-sort.v1";
  const LAST_BACKUP_KEY = "drowse.last-backup.v1";
  const RECOVERY_DISMISSED_KEY = "drowse.recovery-dismissed.v1";
  const UNDO_MS = 10_000;
  type ChatSort = "recent" | "name" | "created";
  type UndoItem = {
    key: string;
    message: string;
    undo: () => Promise<void> | void;
    commit?: () => Promise<void> | void;
    timer: number;
  };
  type ImportDuplicate = { label: string; name: string; existing: SavedConversationSummary };

  let now = $state(Date.now());
  let backingUpAll = $state(false);
  let lastBackupAt = $state(readStoredNumber(LAST_BACKUP_KEY));
  let searchQuery = $state("");
  let sortOrder = $state<ChatSort>(readStoredSort());
  let modelFilter = $state("all");
  let undoItems: UndoItem[] = $state([]);
  const pendingDeleteIds = new SvelteSet<string>();
  let importDuplicates: ImportDuplicate[] = $state([]);
  let recoveryDismissedFor = $state(readStoredString(RECOVERY_DISMISSED_KEY));

  function readStoredString(key: string): string | null {
    try { return localStorage.getItem(key); } catch { return null; }
  }
  function readStoredNumber(key: string): number | null {
    const value = Number(readStoredString(key));
    return Number.isSafeInteger(value) && value > 0 ? value : null;
  }
  function readStoredSort(): ChatSort {
    const value = readStoredString(SORT_KEY);
    return value === "name" || value === "created" ? value : "recent";
  }
  function writeStored(key: string, value: string | null): void {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {}
  }

  const installedModels = $derived(
    snapshot.models.filter((model) =>
      model.setupComplete && model.fit !== "blocked" && snapshot.runtime.available
    ),
  );
  const continueModel = $derived(
    installedModels.find((model) => model.id === entryMemory?.lastModelVariantId) ??
      installedModels.find((model) => model.id === snapshot.selectedModelVariantId) ??
      installedModels[0] ?? null,
  );
  // Until the device check finishes, model install state is unknown. The page
  // still lists chats, but open and recovery actions wait for the check.
  const deviceChecking = $derived(snapshot.phase === "idle" || snapshot.phase === "checking");
  const visibleConversations = $derived(conversations.filter(record => !pendingDeleteIds.has(record.id)));
  const missingChats = $derived(
    deviceChecking ? [] : visibleConversations.filter((record) => modelForConversation(record)?.setupComplete !== true),
  );
  const missingModels = $derived(
    snapshot.models.filter(model => !model.setupComplete
      && missingChats.some(chat => modelForConversation(chat)?.id === model.id)),
  );
  const recoverySignature = $derived(missingChats.map(chat => chat.id).sort().join(","));
  const showRecovery = $derived(
    recoveryQueue.length > 0 || (missingChats.length > 0 && recoveryDismissedFor !== recoverySignature),
  );
  const chatModelOptions = $derived.by(() => {
    const seen = new Map<string, string>();
    for (const record of visibleConversations) {
      if (!seen.has(record.modelId)) seen.set(record.modelId, modelForConversation(record)?.name ?? displayModelName(record.modelId));
    }
    return [...seen].map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label));
  });
  const shownConversations = $derived.by(() => {
    const query = searchQuery.trim().normalize("NFC").toLocaleLowerCase();
    const rows = visibleConversations.filter(record =>
      (modelFilter === "all" || record.modelId === modelFilter)
      && (!query || record.name.normalize("NFC").toLocaleLowerCase().includes(query)));
    if (sortOrder === "name") return [...rows].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true }));
    if (sortOrder === "created") return [...rows].sort((a, b) => b.createdAt - a.createdAt || a.id.localeCompare(b.id));
    return rows;
  });
  const filtering = $derived(searchQuery.trim() !== "" || modelFilter !== "all");
  const recoveryModel = $derived(snapshot.models.find(model => model.id === activeRecoveryId));
  const recoveryDownload = $derived(snapshot.download.modelVariantId === activeRecoveryId ? snapshot.download : null);
  const recoveryProgress = $derived(recoveryDownload?.progress);
  const recoveryPercent = $derived(recoveryProgress && recoveryProgress.bytesTotal > 0
    ? Math.min(100, Math.round(recoveryProgress.bytesReceived / recoveryProgress.bytesTotal * 100)) : 0);
  const downloadBusy = $derived(["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase));
  const recoveryBlocked = $derived(recovering || downloadBusy || openingId !== null || !snapshot.download.available);
  const recoveryStatus = $derived(
    recoveryError ? "Download failed"
      : recoveryDownload?.phase === "paused" ? "Download paused"
      : recoveryDownload?.phase === "cancelling" ? "Pausing…"
      : recoveryDownload?.phase === "requesting_persistence" ? "Preparing download…"
      : recoveryProgress?.offline ? "Offline. Waiting for a connection"
      : recoveryProgress?.stalled ? "Waiting for data…"
      : recoveryProgress && recoveryPercent === 100 ? "Verifying files…"
      : recoveryProgress?.etaSeconds ? `${formatEtaRange(recoveryProgress.etaSeconds)} remaining`
      : "Downloading…",
  );
  const storageUnprotected = $derived(snapshot.storage !== undefined && snapshot.storage.persisted !== true);
  const newChatModel = $derived(
    installedModels.find((model) => model.id === newChatModelId) ?? null,
  );
  const openingModel = $derived(snapshot.models.find(model =>
    openingId === `new:${model.id}` || openingId === `continue:${model.id}`
    || (openingId !== null && snapshot.runtime.modelVariantId === model.id),
  ));
  const openingStatus = $derived(snapshot.runtime.phase === "ready"
    ? "Opening your chat…" : `Loading ${openingModel?.name ?? "your model"}…`);

  onMount(() => {
    void refreshChats();
  });

  // Keep relative times ("2 minutes ago") current while the page is visible.
  onMount(() => {
    let timer: number | undefined;
    const sync = () => {
      window.clearInterval(timer);
      timer = undefined;
      if (document.hidden) return;
      now = Date.now();
      timer = window.setInterval(() => { now = Date.now(); }, 30_000);
    };
    sync();
    document.addEventListener("visibilitychange", sync);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", sync);
    };
  });

  // Leaving the page finishes any deletes still waiting for Undo.
  onMount(() => {
    const flush = () => flushUndo();
    window.addEventListener("pagehide", flush);
    return () => window.removeEventListener("pagehide", flush);
  });

  onDestroy(() => {
    destroyed = true;
    flushUndo();
  });

  function pushUndo(item: Omit<UndoItem, "timer">): void {
    dismissUndo(item.key, true);
    const timer = window.setTimeout(() => dismissUndo(item.key, true), UNDO_MS);
    undoItems = [...undoItems, { ...item, timer }];
  }

  function dismissUndo(key: string, commit: boolean): UndoItem | null {
    const item = undoItems.find(candidate => candidate.key === key) ?? null;
    if (!item) return null;
    window.clearTimeout(item.timer);
    undoItems = undoItems.filter(candidate => candidate.key !== key);
    if (commit) void item.commit?.();
    return item;
  }

  function flushUndo(): void {
    for (const item of [...undoItems]) dismissUndo(item.key, true);
  }

  async function runUndo(key: string): Promise<void> {
    const item = dismissUndo(key, false);
    if (!item) return;
    try {
      await item.undo();
    } catch (cause) {
      error = userFacingError(cause, "That change could not be undone.");
    }
  }

  async function recoverModels(ids: string[], chat: SavedConversationSummary | null = null): Promise<void> {
    if (recoveryBlocked || recoveryQueue.length > 0 || ids.length === 0) return;
    recoveryQueue = [...new Set(ids)];
    approvedRecoveryIds.clear();
    recoveryChat = chat;
    recoveryError = null;
    await tick();
    recoveryNotice?.scrollIntoView({ block: "nearest", behavior: "instant" });
    await runRecovery();
  }

  async function runRecovery(): Promise<void> {
    if (recovering || downloadBusy || destroyed) return;
    recovering = true;
    recoveryError = null;
    recoveryWarning = null;
    try {
      while (recoveryQueue.length > 0 && !destroyed) {
        const id = recoveryQueue[0];
        const model = controller.current().models.find(candidate => candidate.id === id);
        activeRecoveryId = id;
        if (!model || model.fit === "blocked") {
          recoveryError = model?.reason ?? "This model is no longer available.";
          return;
        }
        if (!model.setupComplete) {
          if ((model.fit === "uncertain" || model.requiresOomRetry) && !approvedRecoveryIds.has(id)) {
            recoveryWarning = model;
            return;
          }
          await controller.download(id, { explicitUnsafeOverride: approvedRecoveryIds.has(id) });
          if (destroyed) return;
          const next = controller.current();
          if (!next.models.find(candidate => candidate.id === id)?.setupComplete) {
            if (next.download.phase !== "paused") recoveryError = next.download.reason;
            return;
          }
        }
        selectedRecoveryIds.delete(id);
        recoveryQueue = recoveryQueue.slice(1);
      }
      if (destroyed) return;
      activeRecoveryId = null;
      const chat = recoveryChat;
      recoveryChat = null;
      if (chat) {
        await tick();
        await openConversation(chat);
      }
    } catch (cause) {
      recoveryError = userFacingError(cause, "Download failed. Try again.");
    } finally {
      recovering = false;
    }
  }

  async function pauseRecovery(): Promise<void> {
    try {
      await controller.cancelDownload();
    } catch (cause) {
      recoveryError = userFacingError(cause, "The download could not be paused. Try again.");
    }
  }

  function cancelRecovery(): void {
    if (recovering) return;
    recoveryQueue = [];
    recoveryWarning = null;
    recoveryError = null;
    activeRecoveryId = null;
    recoveryChat = null;
  }

  function modelForConversation(record: SavedConversationSummary): HostedModelOption | null {
    return snapshot.models.find((model) =>
      model.id === record.modelId || model.modelId === record.modelId
    ) ?? null;
  }

  onMount(() => onConversationLibraryChanged(refreshChats));
  onMount(() => registerHomeController({
    metadataDraft: () => editingId ? { id: editingId, name: editingName, dirty: editingName !== conversations.find(row => row.id === editingId)?.name } : null,
    metadataState: () => conversations.map(row => [row.id, row.metadataVersion]),
    busy: () => openingId !== null || changingId !== null || importing || downloadBusy || recovering,
    open: async (id) => {
      if (openingId !== null || changingId !== null || importing || downloadBusy || recovering) throw new Error("Finish the current chat or model operation before opening another chat.");
      if (deviceChecking) throw new Error("Drowse is still checking this device. Try again in a moment.");
      const record = summarizeConversation(await conversationLibrary.get(id));
      const model = modelForConversation(record);
      if (!model?.setupComplete || model.fit === "blocked" || !snapshot.runtime.available) {
        onChooseModels(model?.id, record);
        return { state: "model_required", chat_id: id, model_id: record.modelId, model_variant_id: model?.id ?? null };
      }
      await openConversation(record);
      if (error) throw new Error(error);
      return { state: "opening", chat_id: id, model_id: record.modelId, model_variant_id: model.id };
    },
  }));

  async function refreshChats(): Promise<void> {
    loading = true;
    error = null;
    try {
      const result = await conversationLibrary.listSummaries();
      conversations = result.conversations;
      issues = result.issues;
    } catch (cause) {
      error = userFacingError(cause, "Saved chats could not be opened. Your stored data was not changed.");
    } finally {
      loading = false;
    }
  }

  async function downloadBackup(record: SavedConversationSummary): Promise<void> {
    changingId = record.id;
    error = null;
    try {
      await flushConversationAutosave();
      await downloadChatBackup(conversationLibrary, record.id);
      markBackedUp();
    } catch (cause) {
      error = userFacingError(cause, "This chat could not be downloaded. Your saved chat was not changed.");
    } finally { changingId = null; }
  }

  function markBackedUp(): void {
    lastBackupAt = Date.now();
    writeStored(LAST_BACKUP_KEY, String(lastBackupAt));
  }

  async function backUpAllChats(): Promise<void> {
    if (backingUpAll || visibleConversations.length === 0) return;
    backingUpAll = true;
    error = null;
    backupNotice = "";
    try {
      await flushConversationAutosave();
      const count = await downloadAllChatBackups(conversationLibrary, visibleConversations.map(record => record.id));
      markBackedUp();
      backupNotice = `Downloaded a backup of ${count} ${count === 1 ? "chat" : "chats"}. Use Import chats to restore it.`;
    } catch (cause) {
      error = userFacingError(cause, "Your chats could not be backed up. Your saved chats were not changed.");
    } finally {
      backingUpAll = false;
    }
  }

  async function duplicateConversation(record: SavedConversationSummary): Promise<void> {
    if (changingId !== null || duplicatingId !== null) return;
    changingId = record.id;
    duplicatingId = record.id;
    error = null;
    backupNotice = "";
    try {
      await flushConversationAutosave();
      const copy = await conversationLibrary.duplicate(record.id);
      conversations = [summarizeConversation(copy), ...conversations];
      backupNotice = `Created “${copy.name}”. The original chat is unchanged.`;
    } catch (cause) {
      error = userFacingError(cause, "The chat could not be duplicated. The original chat was not changed.");
    } finally {
      changingId = null;
      duplicatingId = null;
    }
  }

  async function importBackup(event: Event): Promise<void> {
    const input = event.currentTarget as HTMLInputElement;
    const files = [...(input.files ?? [])];
    input.value = "";
    if (files.length === 0 || importing) return;
    importing = true;
    error = null;
    backupNotice = "";
    importDuplicates = [];
    const imported: SavedConversationRecord[] = [];
    const duplicates: ImportDuplicate[] = [];
    const failures: string[] = [];
    const known = [...conversations];
    try {
      for (const file of files) {
        let entries: Awaited<ReturnType<typeof readChatBackupFiles>>;
        try {
          entries = await readChatBackupFiles(file);
        } catch (cause) {
          failures.push(`${file.name}: ${userFacingError(cause, "This file could not be read.")}`);
          continue;
        }
        for (const failure of entries.failures) {
          failures.push(`${failure.label}: ${userFacingError(failure.error, "This chat could not be read.")}`);
        }
        for (const { label, candidate } of entries.candidates) {
          try {
            const existing = await findImportedChat(candidate, conversationLibrary, known);
            if (existing) {
              duplicates.push({ label, name: candidate.name, existing });
              continue;
            }
            const record = await saveChatImport(candidate, conversationLibrary);
            imported.push(record);
            known.push(summarizeConversation(record));
          } catch (cause) {
            failures.push(`${label}: ${userFacingError(cause, "This chat could not be imported.")}`);
          }
        }
      }
      if (imported.length > 0) {
        await refreshChats();
        void requestPersistentConversationStorage();
      }
      importDuplicates = duplicates;
      backupNotice = imported.length === 1
        ? `Imported “${imported[0].name}”.`
        : imported.length > 1 ? `Imported ${imported.length} chats.` : "";
      if (failures.length > 0) {
        error = `${failures.length === 1 ? "One backup" : `${failures.length} backups`} could not be imported. Existing chats were not changed. ${failures.join(" ")}`;
      }
    } finally { importing = false; }
  }

  async function importDuplicateCopy(duplicate: ImportDuplicate): Promise<void> {
    if (importing) return;
    importing = true;
    error = null;
    try {
      const copy = await conversationLibrary.duplicate(duplicate.existing.id);
      importDuplicates = importDuplicates.filter(item => item !== duplicate);
      await refreshChats();
      backupNotice = `Imported a copy as “${copy.name}”.`;
    } catch (cause) {
      error = userFacingError(cause, "The copy could not be imported. Existing chats were not changed.");
    } finally { importing = false; }
  }

  function openDuplicate(duplicate: ImportDuplicate): void {
    importDuplicates = importDuplicates.filter(item => item !== duplicate);
    const current = conversations.find(record => record.id === duplicate.existing.id) ?? duplicate.existing;
    void openConversation(current);
  }

  function beginNewChat(): void {
    newChatModelId = "choose";
  }

  async function openCurrent(model: HostedModelOption): Promise<void> {
    if (openingId !== null) return;
    openingId = `continue:${model.id}`;
    error = null;
    clearConversationOpen();
    try {
      await controller.open(model.id);
    } catch (cause) {
      error = userFacingError(cause, `Unable to open ${model.name}. Check the model files and try again.`);
      openingId = null;
    }
  }

  async function startNewChat(model: HostedModelOption): Promise<void> {
    if (openingId !== null) return;
    openingId = `new:${model.id}`;
    error = null;
    clearConversationOpen();
    try {
      await controller.open(model.id, { resetSession: true });
    } catch (cause) {
      error = userFacingError(cause, `Unable to start a new chat with ${model.name}. Try again.`);
      openingId = null;
    }
  }

  async function openConversation(record: SavedConversationSummary): Promise<void> {
    if (deviceChecking) return;
    const model = modelForConversation(record);
    if (model && !model.setupComplete && model.fit !== "blocked" && snapshot.runtime.available) {
      await recoverModels([model.id], record);
      return;
    }
    if (!queueConversationOpen(record.id, record.modelId)) {
      error = "This browser could not prepare the saved chat. Reload Drowse and try again; the chat was not changed.";
      return;
    }
    if (!model?.setupComplete || model.fit === "blocked" || !snapshot.runtime.available) {
      onChooseModels(model?.id, record);
      return;
    }
    openingId = record.id;
    error = null;
    try {
      await controller.open(model.id);
    } catch (cause) {
      clearConversationOpen();
      error = userFacingError(cause, `Unable to open “${record.name}”. Your saved chat was not changed.`);
      openingId = null;
    }
  }

  async function protectStorage(): Promise<void> {
    persistenceRequesting = true;
    try {
      persistenceAttempted = !(await controller.retryPersistence());
    } catch {
      persistenceAttempted = true;
    } finally {
      persistenceRequesting = false;
      if (persistenceAttempted) storageGuideOpen = true;
    }
  }

  async function setAvatar(id: string, avatarSeed: string): Promise<string> {
    const updated = await conversationLibrary.update(id, { avatarSeed });
    const summary = conversations.find((conversation) => conversation.id === id);
    if (summary) summary.avatarSeed = updated.avatarSeed;
    return updated.avatarSeed;
  }

  async function regenerateAvatar(record: SavedConversationSummary): Promise<void> {
    if (changingAvatars.has(record.id) || changingId === record.id) return;
    changingAvatars.add(record.id);
    error = null;
    const previousSeed = record.avatarSeed;
    try {
      await setAvatar(record.id, randomAvatarSeed());
      revealedAvatarId = record.id;
      pushUndo({
        key: `avatar:${record.id}`,
        message: `New avatar for “${record.name}”.`,
        undo: async () => { await setAvatar(record.id, previousSeed); },
      });
    } catch (cause) {
      error = userFacingError(cause, "The avatar could not be changed.");
    } finally {
      changingAvatars.delete(record.id);
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

  async function changeAccent(record: SavedConversationSummary, accent: ChatAccent): Promise<void> {
    if (changingAccents.has(record.id)) return;
    changingAccents.add(record.id);
    error = null;
    try {
      const updated = await conversationLibrary.update(record.id, { accent });
      const summary = conversations.find(conversation => conversation.id === record.id);
      if (summary) summary.accent = updated.accent;
    } catch (cause) {
      error = userFacingError(cause, "The chat color could not be saved.");
    } finally {
      changingAccents.delete(record.id);
    }
  }

  async function saveRename(record: SavedConversationSummary): Promise<void> {
    if (changingId !== null || editingId !== record.id || !editingName.trim()) return;
    changingId = record.id;
    error = null;
    try {
      replaceConversation(await conversationLibrary.update(record.id, { name: editingName }));
      editingId = null;
    } catch (cause) {
      error = userFacingError(cause, "The chat could not be renamed.");
    } finally {
      changingId = null;
    }
  }

  function focusAfterRemoval(source: HTMLElement): void {
    const card = source.closest(".chat-card");
    const adjacent = card?.nextElementSibling ?? card?.previousElementSibling;
    void tick().then(() => {
      const focusTarget = adjacent?.isConnected ? adjacent.querySelector<HTMLElement>(".chat-menu-trigger") : null;
      (focusTarget ?? document.getElementById("home-title"))?.focus();
    });
  }

  // Saved chats are hidden first and removed for good after the Undo window.
  function deleteConversation(record: SavedConversationSummary, source: HTMLElement): void {
    if (pendingDeleteIds.has(record.id)) return;
    focusAfterRemoval(source);
    pendingDeleteIds.add(record.id);
    confirmDeleteId = null;
    error = null;
    if (recoveryChat?.id === record.id) recoveryChat = null;
    pushUndo({
      key: `delete:${record.id}`,
      message: `Deleted “${record.name}”.`,
      undo: () => { pendingDeleteIds.delete(record.id); },
      commit: async () => {
        try {
          const deleted = await conversationLibrary.delete(record.id);
          if (!deleted) throw new ConversationLibraryError("NOT_FOUND", "This saved chat no longer exists");
          conversations = conversations.filter((row) => row.id !== record.id);
        } catch (cause) {
          if (!destroyed) error = userFacingError(cause, "The saved chat could not be deleted.");
        } finally {
          pendingDeleteIds.delete(record.id);
        }
      },
    });
  }

  async function deleteIssue(id: string, source: HTMLElement): Promise<void> {
    changingId = id;
    error = null;
    try {
      const deleted = await conversationLibrary.delete(id);
      if (!deleted) throw new ConversationLibraryError("NOT_FOUND", "This saved chat no longer exists");
      issues = issues.filter((issue) => issue.id !== id);
      confirmDeleteId = null;
      focusAfterRemoval(source);
    } catch (cause) {
      error = userFacingError(cause, "The saved chat could not be deleted.");
    } finally {
      changingId = null;
    }
  }

  function dismissRecovery(): void {
    recoveryDismissedFor = recoverySignature;
    writeStored(RECOVERY_DISMISSED_KEY, recoverySignature);
  }

  function changeSort(value: ChatSort): void {
    sortOrder = value;
    writeStored(SORT_KEY, value);
  }

  function clearFilters(): void {
    searchQuery = "";
    modelFilter = "all";
  }

  function beginDelete(record: SavedConversationSummary): void {
    confirmDeleteId = record.id;
    const card = document.activeElement?.closest(".chat-card");
    void tick().then(() => card?.querySelector<HTMLButtonElement>("[data-cancel-delete]")?.focus());
  }

  function cancelDelete(event: MouseEvent): void {
    const card = (event.currentTarget as HTMLElement).closest(".chat-card");
    confirmDeleteId = null;
    void tick().then(() => card?.querySelector<HTMLButtonElement>(".chat-menu-trigger")?.focus());
  }

  function replaceConversation(updated: SavedConversationRecord): void {
    conversations = conversations.map((record) => record.id === updated.id ? summarizeConversation(updated) : record);
  }

  function onRenameKey(event: KeyboardEvent, record: SavedConversationSummary): void {
    if (event.key === "Enter") {
      event.preventDefault();
      if (editingName.trim()) void saveRename(record);
    } else if (event.key === "Escape") {
      event.preventDefault();
      editingId = null;
    }
  }

  function onRenameBlur(event: FocusEvent, record: SavedConversationSummary): void {
    if (event.relatedTarget instanceof Element && event.relatedTarget.closest("[data-cancel-rename]")) return;
    if (editingName.trim()) void saveRename(record);
    else editingId = null;
  }

  let viewportWidth = $state(1024);
  function relativeTime(timestamp: number, compact: boolean): string {
    const elapsed = timestamp - now;
    if (Math.abs(elapsed) < 60_000) return "just now";
    const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
      ["year", 365 * 24 * 60 * 60 * 1000],
      ["month", 30 * 24 * 60 * 60 * 1000],
      ["day", 24 * 60 * 60 * 1000],
      ["hour", 60 * 60 * 1000],
      ["minute", 60 * 1000],
    ];
    for (const [unit, duration] of units) {
      if (Math.abs(elapsed) >= duration || unit === "minute") {
        return new Intl.RelativeTimeFormat(undefined, { numeric: "auto", style: compact ? "narrow" : "long" })
          .format(Math.round(elapsed / duration), unit);
      }
    }
    return "just now";
  }
</script>

<svelte:window bind:innerWidth={viewportWidth} />

<TabIdentity state={openingId !== null && !workbenchError ? "loading" : "chats"}
  title={openingId !== null && !workbenchError ? `${openingStatus} · Drowse` : undefined} />

<a class="skip-link" href="#saved-chats">Skip to saved chats</a>

<div class="home-shell workspace-material workspace-library">
  <PageHeader current="chats" compact variant="app" onChats={onChats ?? (() => document.getElementById("home-title")?.focus())} onModels={() => onChooseModels()} />

  <main>
    <section class="hero" aria-labelledby="home-title" data-page-group="0">
      <div>
        <h1 id="home-title" tabindex="-1">Your chats</h1>
      </div>
      <div class="hero-actions">
        {#if continueModel}
          <button
            type="button"
            class="secondary"
            disabled={openingId !== null}
            onclick={() => void openCurrent(continueModel)}
          ><MorphText text={openingId === `continue:${continueModel.id}` ? "Opening…" : `Continue with ${continueModel.name}`} numbers={false} /></button>
        {/if}
        <button
          type="button"
          class="primary"
          disabled={openingId !== null || deviceChecking}
          onclick={beginNewChat}
        >{deviceChecking || installedModels.length > 0 ? "New chat" : "Download a model"}</button>
      </div>
    </section>

    {#if deviceChecking}
      <p class="device-check loading-pulse" role="status">Checking this device…</p>
    {/if}

    {#if workbenchError}
      <section class="notice danger" role="alert">
        <div>
          <h2>Your chat could not open</h2>
          <p>{workbenchError}</p>
          {#if workbenchNeedsReload}
            <p>{appRefreshSafetyMessage}</p>
            <p>{reloadInstructions()}</p>
          {/if}
        </div>
        <button type="button" class="secondary" disabled={workbenchReloading} onclick={onRetryWorkbench}>
          <MorphText text={workbenchReloading ? "Refreshing app…" : workbenchNeedsReload ? "Reload Drowse" : "Try opening chat again"} />
        </button>
      </section>
    {:else if openingId !== null}
      <section class="notice model-opening loading-pulse" role="status" aria-live="polite">
        <div>
          <h2>{openingStatus}</h2>
          <p>Your conversation will open here when the model is ready. Keep this tab open.</p>
        </div>
      </section>
    {/if}

    {#if storageUnprotected}
      <section class="notice storage-notice" role="status" aria-labelledby="storage-title" data-page-group="1">
        <div>
          <h2 id="storage-title">Keep your chats and models</h2>
          {#if visibleConversations.length > 0}
            <p class="backup-nudge">
              {lastBackupAt ? `Last backed up ${relativeTime(lastBackupAt, false)}.` : "You haven't backed up your chats in this browser yet."}
            </p>
          {/if}
          <p>
            {persistenceAttempted
              ? "Storage protection isn't confirmed. Your chats still save in this browser, but you should keep a backup."
              : persistenceRequestAvailable
                ? "Your chats and models are saved in this browser. Ask it to keep them during automatic cleanup, and back up your chats."
                : "This browser can't request storage protection. Back up your chats to keep a copy outside this browser."}
          </p>
          <details class="storage-guide" bind:open={storageGuideOpen} use:animatedDetails>
            <summary>How to keep your data</summary>
            <ol>
              <li>
                Select Back up all chats, or Download beside a saved chat, to keep a copy. Use Import chats to restore a backup. Autosave alone won't protect against cleared browser data.
              </li>
              <li>
                {persistenceRequestAvailable
                  ? "Select Protect storage to request protection from automatic cleanup. Your browser may decline the request."
                  : "This browser doesn't support storage protection. Back up your chats regularly."}
              </li>
              <li>
                Don't clear this site's data. Leave some free space on your device and use a regular browser window instead of private browsing.
              </li>
            </ol>
            <p>Back up your chats before installing the Safari web app. Your existing chats and models won't transfer to it.</p>
            <p class="storage-sources">
              Browser help:
              <a href="https://support.google.com/chrome/answer/14114868?hl=en" target="_blank" rel="noopener noreferrer">Chrome site data settings</a>
              · <a href="https://support.apple.com/en-us/104996" target="_blank" rel="noopener noreferrer">Safari web apps on Mac</a>.
            </p>
          </details>
        </div>
        {#if persistenceRequestAvailable}
          <button type="button" class="secondary" aria-busy={persistenceRequesting} disabled={persistenceRequesting} onclick={() => void protectStorage()}>
            {persistenceRequesting ? "Requesting protection…" : "Protect storage"}
          </button>
        {/if}
      </section>
    {/if}

    {#if newChatModelId !== null}
      <section class="new-chat-panel" aria-labelledby="new-chat-title" in:slide={collapseIn()} out:slide={collapseOut()}>
        {#if newChatModelId === "choose"}
          <div class="chooser-heading">
            <h2 id="new-chat-title">Choose a model</h2>
            <button type="button" class="quiet" disabled={openingId !== null} onclick={() => (newChatModelId = null)}>Cancel</button>
          </div>
          <ModelDownloadChoices {controller} {snapshot} disabled={recovering || recoveryQueue.length > 0 || openingId !== null}
            onSelect={(model) => (newChatModelId = model.id)} />
        {:else if newChatModel}
          <div>
            <h2 id="new-chat-title">Start a new chat with {newChatModel.name}?</h2>
            <p>This replaces that model’s current autosaved session. Named saved chats are not affected.</p>
          </div>
          <div class="panel-actions">
            <button type="button" class="quiet" disabled={openingId !== null} onclick={() => (newChatModelId = null)}>Cancel</button>
            <button type="button" class="primary" aria-busy={openingId === `new:${newChatModel.id}`} disabled={openingId !== null} onclick={() => void startNewChat(newChatModel)}>
              <MorphText text={openingId === `new:${newChatModel.id}` ? "Loading…" : "Start new chat"} />
            </button>
          </div>
        {/if}
      </section>
    {/if}

    {#if showRecovery}
      <section bind:this={recoveryNotice} class="notice model-recovery" aria-labelledby="missing-model-title">
        <div class="recovery-heading">
          <div>
            <h2 id="missing-model-title">{missingModels.length > 1 ? "Models need to be downloaded again" : missingModels.length === 1 || recoveryQueue.length > 0 ? "A model needs to be downloaded again" : "A model is no longer available"}</h2>
            {#if missingChats.length > 0}<p>{missingChats.length === 1 ? "1 saved chat uses it" : `${missingChats.length} saved chats use ${missingModels.length > 1 ? "them" : "it"}`}. Your saved chats are safe.</p>{/if}
          </div>
          {#if recoveryQueue.length === 0}
            <button type="button" class="quiet" onclick={dismissRecovery}>Dismiss</button>
          {/if}
        </div>
        {#if recoveryQueue.length > 0}
          <div class="recovery-progress">
            <strong>{recoveryModel?.name ?? "Model download"}</strong>
            {#if recoveryWarning}
              <p class="recovery-warning" role="alert">{recoveryWarning.reason} The browser may run out of memory.</p>
              <div class="panel-actions">
                <button type="button" class="secondary" onclick={cancelRecovery}>Cancel</button>
                <button type="button" class="primary" onclick={() => {
                  approvedRecoveryIds.add(recoveryWarning!.id);
                  void runRecovery();
                }}>Download anyway</button>
              </div>
            {:else}
              <div class="progress-track" role="progressbar" aria-label={`Download ${recoveryModel?.name ?? "model"}`}
                aria-valuemin="0" aria-valuemax="100" aria-valuenow={recoveryProgress ? recoveryPercent : undefined}
                aria-valuetext={recoveryProgress ? `${recoveryPercent}% · ${recoveryStatus}` : recoveryStatus}>
                <span class:indeterminate={!recoveryProgress} style:width={`${recoveryProgress ? recoveryPercent : 18}%`}></span>
              </div>
              <p class="progress-copy" role="status">{#if recoveryProgress}{recoveryPercent}% · {/if}{recoveryStatus}</p>
              {#if recoveryQueue.length > 1}<p>{recoveryQueue.length - 1} queued</p>{/if}
              {#if recoveryError}<p class="error" role="alert">{recoveryError}</p>{/if}
              <div class="panel-actions">
                {#if recovering}
                  <button type="button" class="secondary" disabled={recoveryDownload?.phase !== "downloading"} onclick={() => void pauseRecovery()}>
                    <MorphText text={recoveryDownload?.phase === "cancelling" ? "Pausing…" : "Pause download"} />
                  </button>
                {:else}
                  <button type="button" class="primary" disabled={!snapshot.download.available || downloadBusy} onclick={() => void runRecovery()}>{recoveryError ? "Retry download" : "Resume download"}</button>
                  <button type="button" class="quiet" onclick={cancelRecovery}>Cancel</button>
                {/if}
              </div>
            {/if}
          </div>
        {:else if missingModels.length > 1}
          <fieldset class="recovery-choices">
            <legend>Choose models to download</legend>
            {#each missingModels as model (model.id)}
              <label>
                <input type="checkbox" checked={selectedRecoveryIds.has(model.id)} disabled={recoveryBlocked || model.fit === "blocked"}
                  onchange={(event) => event.currentTarget.checked ? selectedRecoveryIds.add(model.id) : selectedRecoveryIds.delete(model.id)} />
                <span>{model.name}<small>{model.fit === "blocked" ? model.reason : model.size}</small></span>
              </label>
            {/each}
          </fieldset>
          <button type="button" class="primary" disabled={recoveryBlocked || !missingModels.some(model => selectedRecoveryIds.has(model.id) && model.fit !== "blocked")}
            onclick={() => void recoverModels(missingModels.filter(model => selectedRecoveryIds.has(model.id) && model.fit !== "blocked").map(model => model.id))}>Download selected</button>
        {:else if missingModels[0]}
          <p>{missingModels[0].name}</p>
          <button type="button" class="primary" disabled={recoveryBlocked || missingModels[0].fit === "blocked"}
            onclick={() => void recoverModels([missingModels[0].id])}>Download model</button>
          {#if missingModels[0].fit === "blocked"}<p>{missingModels[0].reason}</p>{/if}
        {:else}
          <p>This model is no longer in the catalog. You can still back up these chats or delete them.</p>
          <button type="button" class="secondary" onclick={() => onChooseModels()}>View models</button>
        {/if}
        {#if !snapshot.download.available}<p>{snapshot.download.reason}</p>{/if}
      </section>
    {/if}

    {#if snapshot.phase === "unsupported" || snapshot.phase === "failed"}
      <section class="notice danger" role="alert">
        <div>
          <h2>Drowse cannot open a model in this browser</h2>
          <p>{snapshot.detail}</p>
        </div>
        <button type="button" class="secondary" onclick={() => void controller.check()}>Check again</button>
      </section>
    {/if}

    {#if error}<p class="error" role="alert">{error}</p>{/if}

    <section id="saved-chats" class="chats-section" aria-labelledby="chats-title" data-page-group="2">
      <div class="section-heading">
        <div>
          <h2 id="chats-title">Saved chats</h2>
        </div>
        <div class="library-actions">
          <span>{visibleConversations.length} {visibleConversations.length === 1 ? "chat" : "chats"}</span>
          {#if visibleConversations.length > 0}
            <button type="button" class="secondary backup-action" disabled={backingUpAll || importing || changingId !== null} aria-busy={backingUpAll} onclick={() => void backUpAllChats()}>
              <FluentIcon name="download" size={18} />
              <MorphText text={backingUpAll ? "Preparing backup…" : "Back up all chats"} />
            </button>
          {/if}
          <button type="button" class="secondary backup-action" disabled={importing || changingId !== null} aria-busy={importing} onclick={() => importInput?.click()}>
            <FluentIcon name="upload" size={18} />
            <MorphText text={importing ? "Importing…" : "Import chats"} />
          </button>
          <input bind:this={importInput} type="file" multiple accept=".drowsechat,.json,.zip,application/json,application/zip" hidden aria-label="Import chat backup files" onchange={importBackup} />
        </div>
      </div>
      {#if backupNotice}<p class="backup-result" role="status">{backupNotice}</p>{/if}
      {#if importDuplicates.length > 0}
        <div class="import-duplicates" role="status">
          {#each importDuplicates as duplicate (duplicate)}
            <div class="import-duplicate">
              <span dir="auto">“{duplicate.name}” is already imported.</span>
              <div class="panel-actions">
                <button type="button" class="secondary compact" disabled={openingId !== null || deviceChecking} onclick={() => openDuplicate(duplicate)}>Open it</button>
                <button type="button" class="quiet compact" disabled={importing} onclick={() => void importDuplicateCopy(duplicate)}>Import as a copy</button>
              </div>
            </div>
          {/each}
          <button type="button" class="quiet compact" onclick={() => (importDuplicates = [])}>Dismiss</button>
        </div>
      {/if}

      {#if !loading && visibleConversations.length > 1}
        <div class="chat-tools" role="search" aria-label="Find saved chats">
          <label class="chat-search">
            <span class="sr-only">Search chats by name</span>
            <FluentIcon name="search" size={18} />
            <input type="search" bind:value={searchQuery} placeholder="Search chats" autocomplete="off" spellcheck="false" />
          </label>
          <span class="chat-tool">
            <span id="chat-sort-label" class="tool-label">Sort</span>
            <Select value={sortOrder} ariaLabelledby="chat-sort-label" onchange={changeSort}
              options={[{ value: "recent", label: "Recent" }, { value: "name", label: "Name" }, { value: "created", label: "Created" }]} />
          </span>
          {#if chatModelOptions.length > 1}
            <span class="chat-tool">
              <span id="chat-model-label" class="tool-label">Model</span>
              <Select bind:value={modelFilter} ariaLabelledby="chat-model-label"
                options={[{ value: "all", label: "All models" }, ...chatModelOptions]} />
            </span>
          {/if}
        </div>
      {/if}

      {#if loading}
        <p class="empty loading-pulse" role="status">Loading saved chats…</p>
      {:else if visibleConversations.length === 0 && issues.length === 0}
        <div class="empty">
          {#if !deviceChecking && installedModels.length === 0}
            <strong>No chats yet. Choose a model to start.</strong>
            <button type="button" class="primary" onclick={() => onChooseModels()}>Choose a model</button>
          {:else}
            <strong>No saved chats yet. Start a chat with a model and it autosaves here.</strong>
            <button type="button" class="primary" disabled={openingId !== null || deviceChecking} onclick={beginNewChat}>New chat</button>
          {/if}
        </div>
      {:else if shownConversations.length === 0 && issues.length === 0}
        <div class="empty" role="status">
          <strong>No chats match your search.</strong>
          <button type="button" class="secondary" onclick={clearFilters}>Clear search</button>
        </div>
      {:else}
        {#if filtering}<p class="sr-only" role="status">{shownConversations.length} of {visibleConversations.length} chats shown</p>{/if}
        <div class="chat-list" role="list" aria-label="Saved chats">
          {#each shownConversations as record (record.id)}
            {@const model = modelForConversation(record)}
            {@const canOpen = model?.setupComplete && model.fit !== "blocked" && snapshot.runtime.available}
            {@const modelActionUnavailable = deviceChecking || model === null || model.fit === "blocked" || !snapshot.runtime.available}
            <div animate:flip={{ duration: () => motionDuration(180), easing: cubicOut }} class="chat-card" role="listitem" data-saved-conversation={record.id} data-chat-accent={record.accent ?? "purple"} style={chatAccentStyle(record.accent)}>
              <button
                type="button"
                class="avatar"
                class:revealed={revealedAvatarId === record.id}
                disabled={changingId === record.id}
                aria-disabled={changingAvatars.has(record.id)}
                aria-busy={changingAvatars.has(record.id)}
                aria-label={`New avatar for ${record.name}. You can undo this.`}
                onpointerleave={() => { if (revealedAvatarId === record.id) revealedAvatarId = null; }}
                onblur={() => { if (revealedAvatarId === record.id) revealedAvatarId = null; }}
                onclick={() => void regenerateAvatar(record)}
              >
                <span class="avatar-image"><Blobatar name={record.avatarSeed} size={58} background="circle" alt="" /></span>
                <span class="avatar-shuffle" aria-hidden="true">
                  <FluentIcon name="shuffle" size={24} />
                </span>
              </button>
              <div class="chat-copy">
                <div class="chat-title-row">
                {#if editingId === record.id}
                  <label class="rename-field">
                    <span class="sr-only">Chat name</span>
                    <input bind:this={renameInput} bind:value={editingName} disabled={changingId === record.id} maxlength="120" onkeydown={(event) => onRenameKey(event, record)} onblur={(event) => onRenameBlur(event, record)} />
                  </label>
                {:else}
                  <button type="button" class="chat-name" data-cursor="text" aria-label={`Rename ${record.name}`} disabled={changingId !== null} onclick={() => beginRename(record)}><strong dir="auto"><MorphText text={record.name} numbers={false} /></strong></button>
                {/if}
                <div class="chat-meta">
                  <time datetime={new Date(record.updatedAt).toISOString()}><MorphText text={relativeTime(record.updatedAt, viewportWidth < 480)} /></time>
                  <SavedChatMenu name={record.name} disabled={changingId !== null || editingId === record.id || confirmDeleteId === record.id}
                    duplicateDisabled={importing} busy={duplicatingId === record.id}
                    onduplicate={() => void duplicateConversation(record)} ondelete={() => beginDelete(record)} />
                </div>
                </div>
                <span class="chat-model"><ModelProviderLogo modelId={record.modelId} /><span>{model?.name ?? displayModelName(record.modelId)}{#if (model?.modelType ?? record.modelType) === "base"}<BaseModelTag />{/if}</span></span>
                <span class="chat-counts"><MorphText text={`${record.messageCount} ${record.messageCount === 1 ? "message" : "messages"} · ${record.threadCount} Loom ${record.threadCount === 1 ? "thread" : "threads"}`} /></span>
                {#if !deviceChecking && !model?.setupComplete}<span class="missing">Download required</span>{/if}
              </div>
              <div class="chat-footer" class:confirming={editingId === record.id || confirmDeleteId === record.id}>
                {#if editingId !== record.id && confirmDeleteId !== record.id}
                  <button type="button" class="primary compact" disabled={openingId !== null || modelActionUnavailable || recovering || recoveryQueue.length > 0 || downloadBusy} onclick={() => void openConversation(record)}>
                    <MorphText text={openingId === record.id
                      ? "Opening…"
                      : deviceChecking
                        ? "Checking device…"
                      : canOpen
                        ? (model?.modelType ?? record.modelType) === "base" ? "Open completion" : "Open chat"
                        : model?.fit === "blocked"
                          ? "Model not supported"
                          : !snapshot.runtime.available
                            ? "Browser not compatible"
                            : model
                              ? "Download model"
                              : "Model unavailable"} />
                  </button>
                {/if}
                <div class="chat-color">
                  <ChatAccentPicker value={record.accent ?? "purple"} disabled={changingAccents.has(record.id)} onchange={(accent) => changeAccent(record, accent)} />
                </div>
                <div class="chat-actions">
                {#if editingId === record.id}
                  <button type="button" class="quiet" data-cancel-rename disabled={changingId !== null} onpointerdown={(event) => event.preventDefault()} onclick={() => (editingId = null)}>Cancel</button>
                  <button type="button" class="secondary" disabled={!editingName.trim() || changingId === record.id} onclick={() => void saveRename(record)}>Save name</button>
                {:else if confirmDeleteId === record.id}
                  <span>Delete this saved chat? You can undo for 10 seconds.</span>
                  <button type="button" class="quiet" data-cancel-delete onclick={cancelDelete}>Cancel</button>
                  <button type="button" class="delete" disabled={changingId === record.id} onclick={(event) => deleteConversation(record, event.currentTarget)}>Delete chat</button>
                {:else}
                  <button type="button" class="secondary backup-action" disabled={changingId !== null || importing} aria-label={`Download backup of ${record.name}`} aria-busy={changingId === record.id && duplicatingId === null} onclick={() => void downloadBackup(record)}>
                    <FluentIcon name="download" size={18} />
                    <span class="backup-label"><MorphText text={changingId === record.id && duplicatingId === null ? "Preparing…" : "Download"} /></span>
                  </button>
                {/if}
                </div>
              </div>
            </div>
          {/each}

          {#each issues as issue (issue.id)}
            <div class="chat-card unavailable" role="listitem">
              <div class="unavailable-avatar" aria-hidden="true">?</div>
              <div class="chat-copy">
                <strong>{issue.name || "Unreadable saved chat"}</strong>
                <span>The stored data has not been removed.</span>
              </div>
              <div class="chat-actions">
                {#if confirmDeleteId === issue.id}
                  <span>Delete this stored data?</span>
                  <button type="button" class="quiet" onclick={() => (confirmDeleteId = null)}>Cancel</button>
                  <button type="button" class="delete" onclick={(event) => void deleteIssue(issue.id, event.currentTarget)}>Delete data</button>
                {:else}
                  <span class="issue">This chat could not be read</span>
                  <button type="button" class="delete-control" onclick={() => (confirmDeleteId = issue.id)}>Delete</button>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </section>
  </main>

  <PageFooter />
</div>

<div class="undo-region" role="status" aria-live="polite">
  {#each undoItems as item (item.key)}
    <div class="undo-toast" in:slide={collapseIn()} out:slide={collapseOut()}>
      <span dir="auto">{item.message}</span>
      <button type="button" class="secondary compact" onclick={() => void runUndo(item.key)}>Undo</button>
    </div>
  {/each}
</div>

<style>
  .skip-link {
    position: fixed;
    z-index: 1000;
    top: var(--space-4);
    inset-inline-start: var(--space-4);
    padding: var(--space-3) var(--space-5);
    border-radius: var(--radius);
    color: var(--text-on-accent);
    background: var(--accent);
    transform: translateY(-160%);
  }
  .skip-link:focus { transform: translateY(0); }
  .home-shell {
    --workspace-surface-padding: 16px;
    --page-max: 90rem;
    --radius: 8px;
    --radius-lg: 12px;
    --text-md: var(--type-heading);
    --text-onboarding-heading: var(--text-page-title);
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
    color: var(--fg);
    background: var(--ambient-canvas);
  }
  main { min-width: 0; width: min(100%, var(--page-max)); margin-inline: auto; }
  main {
    padding-inline: max(var(--page-gutter), env(safe-area-inset-left)) max(var(--page-gutter), env(safe-area-inset-right));
  }
  .hero-actions, .panel-actions, .chat-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-5);
    min-width: 0;
  }
  main {
    flex: 1 1 auto;
    display: grid;
    align-content: start;
    gap: var(--surface-padding);
    padding-block: var(--page-top-space);
  }
  .hero {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    justify-content: space-between;
    gap: var(--space-8);
    padding-bottom: var(--space-6);
  }
  h1 { margin: 0; font-size: var(--text-onboarding-heading); font-weight: var(--weight-display); letter-spacing: -0.035em; line-height: 1.15; }
  .new-chat-panel p, .notice p {
    margin: var(--space-3) 0 0;
    color: var(--fg-dim);
    line-height: 1.5;
  }
  button {
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    min-height: var(--control-target);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-4) var(--space-6);
    color: var(--fg-strong);
    background: transparent;
    transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out);
  }
  button:active:not(:disabled) { transform: scale(0.96); }
  button:disabled { opacity: var(--disabled-opacity); }
  button.primary { color: var(--action-ink); background: var(--control-sheen), var(--action-bg); box-shadow: var(--shadow-control); }
  button.primary:hover:not(:disabled) { background: var(--control-sheen), var(--action-hover); box-shadow: var(--shadow-control-hover); }
  button.secondary { background: var(--control-sheen), var(--glass-strong); box-shadow: var(--shadow-control); }
  button.secondary:hover:not(:disabled), button.quiet:hover:not(:disabled) { color: var(--fg); background: var(--control-sheen), var(--bg-hover); }
  button.secondary:hover:not(:disabled) { box-shadow: var(--shadow-control-hover); }
  button:is(.primary, .secondary):active:not(:disabled) { box-shadow: var(--shadow-control-pressed); }
  button.quiet { padding-inline: var(--space-4); }
  button.compact { min-height: var(--control-compact); padding-inline: var(--space-5); }
  button.delete { color: var(--text-on-accent); background: var(--accent-red); }
  .new-chat-panel, .notice {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, auto);
    align-items: center;
    gap: var(--space-6);
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--surface-card);
    box-shadow: var(--shadow-rack);
  }
  .new-chat-panel { grid-template-columns: minmax(0, 1fr); gap: var(--space-7); padding: var(--space-8); }
  .new-chat-panel > *, .notice > * { min-width: 0; }
  .new-chat-panel h2, .notice h2 { overflow-wrap: anywhere; }
  .chooser-heading { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-5); }
  .new-chat-panel h2, .notice h2 { margin: 0; font-size: var(--text-onboarding-heading); letter-spacing: -0.035em; line-height: 1.25; }
  .storage-notice button.secondary {
    color: var(--warning-action-ink);
    background: var(--warning-action);
    box-shadow: none;
  }
  .storage-notice button.secondary:hover:not(:disabled) { color: var(--warning-action-ink); background: var(--warning-action-hover); }
  .storage-notice {
    align-items: start;
    background: var(--surface-card);
    color: var(--fg);
    box-shadow: var(--shadow-card);
  }
  .storage-notice > div { min-width: 0; flex: 1; }
  .storage-notice h2 { color: var(--fg-strong); }
  .storage-notice p { color: var(--fg-muted); }
  .storage-guide { margin-top: var(--space-3); overflow-wrap: anywhere; }
  .storage-guide summary { cursor: pointer; color: var(--fg); padding-block: var(--space-2); min-height: var(--control-target); }
  .storage-guide summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: var(--radius-sm); }
  .storage-guide ol { padding-inline-start: var(--space-5); margin-block: var(--space-2); color: var(--fg-muted); }
  .storage-guide li + li { margin-top: var(--space-2); }
  .storage-sources a { color: var(--fg); text-underline-offset: 0.2em; }
  .notice.danger { background: var(--danger-bg); }
  .error { margin: 0; color: var(--accent-red); }
  .chats-section {
    min-width: 0;
    display: grid;
    gap: var(--space-5);
    padding-top: var(--space-7);
  }
  .section-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: var(--space-5);
  }
  .section-heading h2 { margin: 0; font-size: var(--text-onboarding-heading); letter-spacing: -0.035em; line-height: 1.15; }
  .section-heading { flex-wrap: wrap; }
  .library-actions { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-5); }
  .library-actions > span { color: var(--fg-muted); font-family: var(--font-data); font-size: var(--text-sm); }
  .backup-action { display: inline-flex; align-items: center; justify-content: center; gap: var(--control-label-gap); }
  .backup-action :global(svg) { flex-shrink: 0; }
  .backup-result { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.5; max-width: 78ch; }
  .model-recovery { grid-template-columns: minmax(0, 1fr); justify-items: start; }
  .recovery-heading { width: 100%; display: flex; flex-wrap: wrap; align-items: start; justify-content: space-between; gap: var(--space-5); }
  .device-check { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .backup-nudge { color: var(--fg-strong); }
  .import-duplicates { display: grid; justify-items: start; gap: var(--space-4); padding: var(--space-5); border-radius: var(--radius-lg); background: var(--surface-card); box-shadow: var(--shadow-card); }
  .import-duplicate { width: 100%; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-4); }
  .import-duplicate > span { min-width: 0; overflow-wrap: anywhere; color: var(--fg-dim); font-size: var(--text-sm); }
  .chat-tools { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-5); min-width: 0; }
  .chat-search { flex: 1 1 14rem; min-width: 0; display: flex; align-items: center; gap: var(--control-label-gap); padding-inline: var(--space-4); min-height: var(--control-field); border-radius: var(--radius); color: var(--fg-muted); background: var(--input-well); box-shadow: var(--shadow-well); }
  .chat-search:focus-within { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .chat-search input { flex: 1; min-width: 0; min-height: var(--control-field); border: 0; padding: 0; color: var(--fg); background: transparent; font: inherit; outline: none; }
  .chat-tool { display: inline-flex; align-items: center; gap: var(--control-label-gap); min-width: 0; }
  .tool-label { color: var(--fg-muted); font-size: var(--text-sm); }
  .undo-region { position: fixed; z-index: calc(var(--z-drawer) - 2); inset-inline-start: calc(var(--space-6) + env(safe-area-inset-left, 0px)); bottom: calc(var(--space-6) + env(safe-area-inset-bottom, 0px)); display: grid; gap: var(--space-4); width: min(26rem, calc(100vw - var(--space-8))); pointer-events: none; }
  .undo-toast { display: flex; align-items: center; justify-content: space-between; gap: var(--space-5); padding: var(--space-4) var(--space-4) var(--space-4) var(--space-6); border: 1px solid var(--popup-border); border-radius: var(--popup-radius); color: var(--fg); background: var(--popup-bg); box-shadow: var(--popup-shadow); pointer-events: auto; }
  .undo-toast > span { min-width: 0; overflow-wrap: anywhere; font-size: var(--text-sm); }
  .recovery-progress { width: 100%; min-width: 0; display: grid; gap: var(--space-5); }
  .recovery-progress p { margin: 0; }
  .recovery-progress .error { color: var(--accent-red); }
  .recovery-warning { color: var(--accent-yellow); }
  .recovery-choices { display: grid; gap: var(--space-4); margin: 0; padding: 0; border: 0; min-width: 0; }
  .recovery-choices legend { margin-bottom: var(--space-5); color: var(--fg-muted); }
  .recovery-choices label { display: flex; align-items: center; gap: var(--space-5); min-height: var(--control-target); cursor: pointer; }
  .recovery-choices input { width: 18px; height: 18px; accent-color: var(--accent); }
  .recovery-choices span { min-width: 0; overflow-wrap: anywhere; }
  .recovery-choices small { display: block; color: var(--fg-muted); }
  .progress-track { width: min(520px, 100%); height: 7px; overflow: hidden; border-radius: var(--radius-pill); background: var(--glass-strong); }
  .progress-track span { display: block; height: 100%; border-radius: inherit; background: var(--accent); transition: width var(--dur) linear; }
  .progress-track .indeterminate { opacity: 0.6; }
  .progress-copy { font-family: var(--font-mono); font-size: var(--text-xs); }
  @media (prefers-reduced-motion: reduce) { .progress-track span { transition: none; } }
  .chat-list { display: grid; gap: var(--space-sm); }
  .chat-card {
    min-width: 0;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: var(--surface-padding);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--surface-card);
    box-shadow: var(--shadow-card);
  }
  button.delete-control { color: var(--accent-red); background: var(--danger-bg); box-shadow: var(--shadow-control); }
  button.delete-control:hover:not(:disabled) { background: var(--danger-hover); }
  .avatar, .unavailable-avatar {
    width: 62px;
    height: 62px;
    display: grid;
    place-items: center;
    padding: 0;
    border-radius: 50%;
    background: var(--bg-elev);
    box-shadow: var(--shadow-control);
  }
  .unavailable-avatar { color: var(--fg-muted); font-family: var(--font-data); }
  .avatar { position: relative; overflow: hidden; align-self: start; }
  .avatar-image { display: block; width: 100%; height: 100%; border-radius: inherit; transition: filter 200ms var(--ease-out); }
  .avatar-shuffle { position: absolute; inset: 0; display: grid; place-items: center; color: white; opacity: 0; scale: 0.25; filter: blur(4px); pointer-events: none; transition: opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1); }
  .avatar:focus-visible:not(.revealed) .avatar-image { filter: blur(3px) brightness(0.5); }
  .avatar:focus-visible:not(.revealed) .avatar-shuffle { opacity: 1; scale: 1; filter: blur(0); }
  @media (hover: hover) {
    .avatar:hover:not(.revealed) .avatar-image { filter: blur(3px) brightness(0.5); }
    .avatar:hover:not(.revealed) .avatar-shuffle { opacity: 1; scale: 1; filter: blur(0); }
  }
  .avatar :global(img) { display: block; width: 100%; height: 100%; min-width: 0; border-radius: inherit; }
  .chat-copy { min-width: 0; display: grid; gap: var(--space-4); }
  .chat-title-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; gap: var(--space-5); }
  .chat-meta { display: flex; align-items: center; justify-content: end; gap: var(--space-2); }
  .chat-title-row time { color: var(--fg-muted); font-family: var(--font-data); font-size: var(--text-xs); white-space: nowrap; }
  .chat-copy strong { display: block; min-width: 0; overflow-wrap: anywhere; font-size: var(--text-md); }
  button.chat-name { --control-sheen: none; appearance: none; min-width: 40px; width: fit-content; max-width: 100%; min-height: var(--control-target); padding: 0; text-align: start; justify-content: start; border-radius: var(--radius-sm); color: var(--fg-strong); background: transparent; box-shadow: none; }
  button.chat-name:hover:not(:disabled) { text-decoration: underline; text-underline-offset: 3px; }
  button.chat-name:active { transform: none; }
  .chat-counts { font-variant-numeric: tabular-nums; }
  .chat-copy > span { color: var(--fg-muted); font-size: var(--text-sm); }
  .chat-model, .chat-counts { font-weight: var(--weight-structure-bold); line-height: 24px; }
  .chat-model { display: flex; align-items: center; gap: var(--space-xs); }
  .chat-model > span { min-width: 0; overflow-wrap: anywhere; }
  .chat-copy .missing { color: var(--accent-yellow); }
  .panel-actions { gap: var(--space-6); }
  .chat-actions { grid-column: 2 / -1; justify-content: end; gap: var(--space-6); }
  .chat-footer { grid-column: 1 / -1; display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-sm); }
  .chat-color { min-width: 0; max-width: 100%; }
  .chat-footer .chat-actions { display: contents; }
  .chat-footer > .primary { margin-inline-end: auto; }
  .chat-footer.confirming .chat-actions { display: flex; flex-wrap: wrap; }
  .chat-actions > span, .issue { margin-inline-end: auto; color: var(--fg-dim); font-size: var(--text-sm); }
  .rename-field input {
    width: min(100%, 34rem);
    min-height: var(--control-field);
    border: 0;
    border-radius: var(--radius);
    padding: var(--space-3) var(--space-4);
    color: var(--fg);
    background: var(--input-well);
    box-shadow: var(--shadow-well);
  }
  .empty {
    margin: 0;
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    color: var(--fg-dim);
    background: var(--surface-sheen), var(--glass);
    text-align: center;
  }
  .empty { display: grid; justify-items: center; gap: var(--space-5); }
  .empty strong { color: var(--fg); font-family: var(--font-structure); }
  @media (min-width: 761px) and (pointer: fine) {
    .home-shell { --control-target: 40px; --control-field: 40px; --control-compact: 40px; }
  }
  @media (max-width: 700px) {
    .hero { align-items: stretch; flex-direction: column; gap: var(--space-6); }
    .hero-actions > button { flex: 1 1 9rem; }
    .new-chat-panel, .notice { grid-template-columns: 1fr; }
    .new-chat-panel > .panel-actions > button, .notice > button { width: 100%; }
    .chat-card { grid-template-columns: auto minmax(0, 1fr); }
    .chat-actions { grid-column: 1 / -1; }
    .chat-footer > .primary { flex: 1 1 auto; }
    .chat-footer .backup-action { min-width: var(--control-target); padding-inline: var(--space-5); }
  }
  @media (max-width: 430px) {
    .avatar, .unavailable-avatar { width: 54px; height: 54px; }
    .chat-title-row { grid-template-columns: minmax(0, 1fr); gap: var(--space-2); }
    .chat-meta { grid-row: 1; justify-self: end; }
    button.chat-name { min-height: var(--control-target); }
    .chat-actions button { flex: 1 1 auto; }
  }
  @media (prefers-reduced-motion: reduce) {
    button { transition: none; }
    .avatar-image, .avatar-shuffle { transition: none; }
    button:active:not(:disabled) { transform: none; }
  }
</style>
