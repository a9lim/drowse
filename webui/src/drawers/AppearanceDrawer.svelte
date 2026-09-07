<script lang="ts">
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import ThemeToggle from "../lib/ui/ThemeToggle.svelte";
  import Button from "../lib/ui/Button.svelte";
  import Select from "../lib/Select.svelte";
  import { closeDrawer } from "../lib/stores/drawers.svelte";
  import { appearanceState, updateBackground, uploadBackground, removeBackground } from "../lib/stores/appearance.svelte";
  let { params: _params }: { params?: unknown } = $props();

  const effects = [{ value: "original", label: "Original" }, { value: "pixel", label: "Pixels" }, { value: "dither", label: "Dither" }];
  const pixels = [1, 2, 3, 4, 6, 8, 12].map(value => ({ value: String(value), label: `${value} px` }));
  const visibility = [{ value: "0.04", label: "Subtle" }, { value: "0.1", label: "Balanced" }, { value: "0.14", label: "Visible" }];
</script>

<section class="drawer-shell" aria-label="Appearance settings">
  <header class="header"><h2>Appearance</h2><DrawerCloseButton onclick={closeDrawer} /></header>
  <div class="body">
    <div class="theme-row"><div><h3>Theme</h3><p>Choose a light or dark workspace.</p></div><ThemeToggle /></div>
    <section class="background-settings" aria-labelledby="background-title">
      <div><h3 id="background-title">Workspace background</h3><p>Optional. Images stay on this device and are never uploaded to a server.</p></div>
      <div class="upload-field">
        <label for="background-file">Background image</label>
        <input id="background-file" type="file" accept="image/png,image/jpeg,image/webp" disabled={!appearanceState.ready || appearanceState.busy}
          aria-describedby="background-file-help" onchange={(event) => {
            const file = event.currentTarget.files?.[0];
            if (file) void uploadBackground(file);
            event.currentTarget.value = "";
          }} />
        <p id="background-file-help">PNG, JPEG, or WebP. Up to 10 MB; resized to use less memory.</p>
      </div>
      {#if appearanceState.url}
        <div class="image-row"><span class="image-name">{appearanceState.name}</span><Button disabled={appearanceState.busy} onclick={() => void removeBackground()}>Remove image</Button></div>
        <div class="setting-row"><span>Treatment</span><Select ariaLabel="Background treatment" value={appearanceState.effect} options={effects} disabled={appearanceState.busy} onchange={value => void updateBackground({ effect: value as "original" | "pixel" | "dither" })} /></div>
        {#if appearanceState.effect !== "original"}<div class="setting-row"><span>Pixel size</span><Select ariaLabel="Background pixel size" value={String(appearanceState.pixelSize)} options={pixels} disabled={appearanceState.busy} onchange={value => void updateBackground({ pixelSize: Number(value) })} /></div>{/if}
        <div class="setting-row"><span>Visibility</span><Select ariaLabel="Background visibility" value={String(appearanceState.visibility)} options={visibility} disabled={appearanceState.busy} onchange={value => void updateBackground({ visibility: Number(value) })} /></div>
        <p>The image stays dim so text and token colors remain readable. The effect is static and does not animate during generation.</p>
      {/if}
      {#if appearanceState.error}<p class="error" role="alert">{appearanceState.error}</p>{/if}
      <p class="save-state" role="status">{appearanceState.busy ? "Saving on this device…" : appearanceState.url ? "Background saved on this device." : "Using the plain workspace background."}</p>
    </section>
  </div>
</section>

<style>
  .drawer-shell { display: flex; flex-direction: column; min-height: 0; height: 100%; }
  .header { display: flex; flex: none; align-items: center; justify-content: space-between; gap: var(--space-sm); padding: var(--surface-padding); }
  h2, h3, p { margin: 0; }
  h2 { font-size: var(--text-lg); font-weight: var(--weight-structure-bold); }
  h3, label { font-size: var(--text-sm); font-weight: var(--weight-structure-bold); }
  p { color: var(--fg-muted); font-size: var(--text-xs); line-height: 1.5; }
  .body { min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: var(--surface-padding); display: grid; align-content: start; gap: var(--space-lg); }
  .theme-row { display: flex; align-items: center; justify-content: space-between; gap: var(--space-sm); }
  .theme-row > div, .background-settings, .upload-field { display: grid; gap: var(--space-sm); }
  .background-settings { gap: var(--space-md); }
  input[type="file"] { display: block; min-width: 0; width: 100%; color: var(--fg-dim); font: inherit; font-size: var(--text-xs); }
  input::file-selector-button { min-height: var(--control-target); margin-inline-end: var(--space-sm); padding: var(--space-xs) var(--space-sm); background: var(--glass-strong); color: var(--fg); border: 0; border-radius: var(--radius); font: inherit; cursor: pointer; }
  .image-row { display: flex; align-items: center; gap: var(--space-sm); }
  .image-name { flex: 1; min-width: 0; overflow-wrap: anywhere; font-size: var(--text-xs); color: var(--fg-dim); }
  .setting-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(8rem, 1fr); align-items: center; gap: var(--space-sm); font-size: var(--text-sm); }
  .error { color: var(--accent-red); }
  .save-state { min-height: 1.5em; }
</style>
