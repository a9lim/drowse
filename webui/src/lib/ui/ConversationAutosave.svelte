<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { captureConversationSnapshot } from "../conversationWorkspace";
  import { loomTree } from "../stores/loom.svelte";
  import { genStatus } from "../stores/chat.svelte";
  import { sessionState } from "../stores/session.svelte";
  import { samplingState } from "../stores/sampling.svelte";
  import { steerRack } from "../stores/steering.svelte";
  import { probeRack, highlightState } from "../stores/probes.svelte";
  import { conversationLibrary, savedConversationState, registerConversationAutosave } from "../stores/savedConversations.svelte";
  import { userFacingError } from "../runtime/userFacingError";
  import { openDrawer, drawerState } from "../stores/drawers.svelte";
  import Button from "./Button.svelte";

  let timer: ReturnType<typeof setTimeout> | null = null;
  let tail: Promise<void> = Promise.resolve();
  let disposed = false;

  async function flush(): Promise<void> {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (!loomTree.loaded || genStatus.active) return tail;
    let snapshot;
    try { snapshot = captureConversationSnapshot(); }
    catch (error) {
      savedConversationState.status = "error";
      savedConversationState.error = userFacingError(error, "The chat could not be autosaved. Keep this page open and download a backup.");
      throw error;
    }
    const requestedId = savedConversationState.activeId;
    const modelType = sessionState.info?.is_base_model ? "base" : "chat";
    savedConversationState.status = "saving";
    const task = tail.catch(() => undefined).then(async () => {
      try {
        const record = await conversationLibrary.autosave(snapshot, requestedId, modelType);
        if (disposed || loomTree.root_id !== snapshot.tree.root_id || savedConversationState.activeId !== requestedId) return;
        savedConversationState.activeId = record?.id ?? null;
        savedConversationState.avatarSeed = record?.avatarSeed ?? null;
        savedConversationState.accent = record?.accent ?? "purple";
        savedConversationState.status = record ? "saved" : "idle";
        savedConversationState.error = null;
      } catch (error) {
        if (!disposed) {
          savedConversationState.status = "error";
          savedConversationState.error = userFacingError(error, "Autosave failed. Keep this page open and use Save chat to download a backup.");
        }
        throw error;
      }
    });
    tail = task;
    return task;
  }

  $effect(() => {
    void loomTree.rev;
    void loomTree.loaded;
    void genStatus.active;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (!loomTree.loaded || genStatus.active || drawerState.open === "load_conversation") return;
    JSON.stringify(samplingState);
    JSON.stringify([...steerRack.entries]);
    void steerRack.subspaceAlong;
    void steerRack.customExpression;
    JSON.stringify(probeRack.active.map(name => probeRack.entries.get(name)?.request));
    void probeRack.sortMode;
    JSON.stringify(highlightState);
    untrack(() => { savedConversationState.status = "pending"; });
    timer = setTimeout(() => { void flush().catch(() => undefined); }, 600);
  });

  onMount(() => {
    disposed = false;
    const unregister = registerConversationAutosave(flush);
    const saveWhenHidden = () => { if (document.hidden) void flush().catch(() => undefined); };
    document.addEventListener("visibilitychange", saveWhenHidden);
    return () => {
      void flush().catch(() => undefined);
      disposed = true;
      unregister();
      document.removeEventListener("visibilitychange", saveWhenHidden);
    };
  });
</script>

{#if savedConversationState.status === "error"}
  <div class="autosave-error" role="alert">
    <span>{savedConversationState.error}</span>
    <Button onclick={() => void flush().catch(() => undefined)}>Retry autosave</Button>
    <Button variant="solid" onclick={() => openDrawer("save_conversation")}>Save chat / backup</Button>
  </div>
{/if}

<style>
  .autosave-error { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-sm); padding: var(--surface-padding); color: var(--error-text); background: var(--danger-bg); font-size: var(--text-sm); }
  .autosave-error span { flex: 1 1 20rem; min-width: 0; overflow-wrap: anywhere; }
</style>
