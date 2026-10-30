<script lang="ts">
  // Themed checkbox — square box with `--radius`, accent fill on
  // checked, ✓ glyph rendered via CSS so we don't depend on a font.
  // Drop-in for ``<input type="checkbox">`` everywhere the webui has
  // them inline-next-to-label.
  //
  // The component renders the box only — wrap it in a host ``<label>``
  // (or pass ``label`` for a quick inline one) the way the existing
  // ``.check`` / ``.axis-check`` rows do.

  interface Props {
    checked: boolean;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    /** Optional inline label rendered next to the box.  Most callsites
     *  use their own ``<span>``-after-the-input layout; that still
     *  works — just leave this empty and place the checkbox where the
     *  native input used to sit. */
    label?: string;
    onchange?: (checked: boolean) => void;
  }

  let {
    checked = $bindable(),
    disabled = false,
    ariaLabel,
    title,
    label,
    onchange,
  }: Props = $props();

  function onChange(ev: Event): void {
    checked = (ev.currentTarget as HTMLInputElement).checked;
    onchange?.(checked);
  }
</script>

<label class="sk-checkbox-row" class:is-disabled={disabled}>
  <input
    type="checkbox"
    class="sk-checkbox-input"
    bind:checked
    aria-label={ariaLabel ?? undefined}
    {disabled}
    {...{ "aria-description": (title) }}
    onchange={onChange}
  />
  <span class="sk-checkbox" aria-hidden="true">
    <span class="sk-checkbox-box" aria-hidden="true">
      <span class="sk-checkbox-glyph" class:is-visible={checked}>✓</span>
    </span>
  </span>
  {#if label}
    <span class="sk-checkbox-label" class:is-disabled={disabled}>{label}</span>
  {/if}
</label>

<style>
  .sk-checkbox-row {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
    min-height: var(--control-target);
    color: var(--fg);
    cursor: pointer;
  }

  .sk-checkbox-input {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    opacity: 0;
    cursor: inherit;
  }
  .sk-checkbox {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: var(--control-target);
    height: var(--control-target);
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sk-checkbox-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    box-sizing: border-box;
    background: var(--bg-elev);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-sm);
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      scale var(--dur-fast) var(--ease-out);
  }
  .sk-checkbox-row:hover:not(.is-disabled) .sk-checkbox-box {
    border-color: var(--accent-strong);
  }
  .sk-checkbox-row:active:not(.is-disabled) .sk-checkbox-box {
    scale: var(--press-scale);
  }
  .sk-checkbox-input:focus-visible + .sk-checkbox {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .sk-checkbox-input:checked + .sk-checkbox .sk-checkbox-box {
    background: var(--accent);
    border-color: var(--accent);
  }
  .sk-checkbox-row:hover:not(.is-disabled) .sk-checkbox-input:checked + .sk-checkbox .sk-checkbox-box {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }
  .sk-checkbox-row.is-disabled {
    opacity: var(--disabled-opacity);
    cursor: not-allowed;
  }

  .sk-checkbox-glyph {
    font-size: var(--text-2xs);
    line-height: 1;
    color: var(--text-on-accent);
    font-weight: var(--weight-bold);
    pointer-events: none;
    opacity: 0;
    scale: 0.25;
    filter: blur(4px);
    transition:
      opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1),
      scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1),
      filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1);
  }
  .sk-checkbox-glyph.is-visible {
    opacity: 1;
    scale: 1;
    filter: blur(0);
  }

  .sk-checkbox-label {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    cursor: pointer;
  }
  .sk-checkbox-label.is-disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
</style>
