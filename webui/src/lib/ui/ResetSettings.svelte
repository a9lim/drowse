<script lang="ts">
  import Button from "./Button.svelte";
  import { genStatus } from "../stores/chat.svelte";
  import { modelDefaultsState } from "../stores/sampling.svelte";
  import { resetSettings, settingsResetState } from "../stores/settingsReset.svelte";
  import { pushToast } from "../stores/toasts.svelte";
  import { userFacingError } from "../runtime/userFacingError";

  let { full = false }: { full?: boolean } = $props();
  let confirming = $state(false);
  let error = $state<string | null>(null);
  async function reset(): Promise<void> {
    error = null;
    try {
      await resetSettings(full);
      confirming = false;
      pushToast(full ? "Model settings reset." : "Generation settings reset.", { kind: "info" });
    } catch (cause) {
      error = userFacingError(cause, "Settings could not be fully reset. Try again.");
    }
  }
</script>

<div class="reset-settings">
  {#if full}
    <h3>Reset settings</h3>
    <p>Restore generation settings, system prompt, roles, steering, and readings. Chats and downloaded files stay saved.</p>
  {/if}
  {#if confirming}
    <p>Reset the current model’s settings to their starting values?</p>
    <div class="actions">
      <Button disabled={settingsResetState.busy} onclick={() => confirming = false}>Cancel</Button>
      <Button variant="solid" disabled={settingsResetState.busy || genStatus.active} onclick={() => void reset()}>Reset Settings</Button>
    </div>
  {:else}
    <Button disabled={!modelDefaultsState.info || settingsResetState.busy || genStatus.active}
      onclick={() => full ? confirming = true : void reset()}>
      {settingsResetState.busy ? "Resetting…" : full ? "Reset Settings" : "Reset to original"}
    </Button>
  {/if}
  {#if genStatus.active}<p>Available when the current reply finishes.</p>{/if}
  {#if error}<p role="alert">{error}</p>{/if}
</div>

<style>
  .reset-settings { display: grid; gap: var(--space-3); }
  h3 { margin: 0; font-size: var(--text-md); color: var(--fg-strong); }
  p { margin: 0; font-size: var(--text-sm); color: var(--fg-muted); line-height: 1.5; }
  .actions { display: flex; flex-wrap: wrap; gap: var(--space-2); }
  [role="alert"] { color: var(--accent-red); }
</style>
