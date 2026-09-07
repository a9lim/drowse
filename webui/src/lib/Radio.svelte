<script lang="ts" generics="T extends string | number">
  // Themed radio button.  Used singly (one ``<Radio />`` per option,
  // sharing a ``group`` binding) or via the matching ``RadioGroup``
  // wrapper.  Same primitives as ``Checkbox`` — square turned circular,
  // ✓ glyph swapped for a centred dot.
  //
  // The component is generic over value type so callsites can keep
  // strongly-typed enums.

  interface Props {
    /** This radio's value — committed into ``group`` on click. */
    value: T;
    /** The shared selection — bind to the same variable across the
     *  group of radios. */
    group: T;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    label?: string;
    name?: string;
    onchange?: (value: T) => void;
  }

  let {
    value,
    group = $bindable(),
    disabled = false,
    ariaLabel,
    title,
    label,
    name,
    onchange,
  }: Props = $props();

  const selected = $derived(group === value);

  function pick(): void {
    if (disabled || selected) return;
    group = value;
    onchange?.(value);
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (disabled) return;
    if (ev.key === " " || ev.key === "Enter") {
      ev.preventDefault();
      pick();
      return;
    }
    const groupEl = (ev.currentTarget as HTMLElement).closest('[role="radiogroup"]');
    if (!groupEl) return;
    const radios = [...groupEl.querySelectorAll<HTMLButtonElement>('[role="radio"]')].filter(
      (radio) => !radio.disabled,
    );
    const current = radios.indexOf(ev.currentTarget as HTMLButtonElement);
    if (current < 0 || radios.length === 0) return;
    let next = current;
    const rtl = getComputedStyle(groupEl).direction === "rtl";
    if (ev.key === "ArrowDown" || ev.key === "ArrowRight") {
      const delta = ev.key === "ArrowRight" && rtl ? -1 : 1;
      next = (current + delta + radios.length) % radios.length;
    } else if (ev.key === "ArrowUp" || ev.key === "ArrowLeft") {
      const delta = ev.key === "ArrowLeft" && rtl ? 1 : -1;
      next = (current + delta + radios.length) % radios.length;
    }
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = radios.length - 1;
    else return;
    ev.preventDefault();
    radios[next].focus();
    radios[next].click();
  }
</script>

<button
  type="button"
  role="radio"
  class="sk-radio"
  class:is-selected={selected}
  class:is-disabled={disabled}
  aria-checked={selected}
  aria-label={ariaLabel}
  tabindex={selected && !disabled ? 0 : -1}
  data-name={name}
  {disabled}
  {title}
  onclick={pick}
  onkeydown={onKeydown}
>
  <span class="sk-radio-control" aria-hidden="true">
    <span class="sk-radio-box" aria-hidden="true">
      <span class="sk-radio-dot" class:is-visible={selected}></span>
    </span>
  </span>
  {#if label}
    <span class="sk-radio-label">{label}</span>
  {/if}
</button>

<style>
  .sk-radio {
    display: inline-flex;
    align-items: center;
    gap: var(--space-3);
    min-height: var(--control-target);
    padding-block: 0;
    padding-inline: 0 var(--space-2);
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    color: var(--fg);
    cursor: pointer;
    transition: scale var(--dur-fast) var(--ease-out);
  }
  .sk-radio:active:not(:disabled) { scale: var(--press-scale); }
  .sk-radio-control {
    flex: 0 0 var(--control-target);
    width: var(--control-target);
    height: var(--control-target);
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .sk-radio-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    box-sizing: border-box;
    background: var(--bg-elev);
    border: 1px solid var(--glass-line);
    border-radius: 50%;
    transition: background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sk-radio:hover:not(.is-disabled) .sk-radio-box {
    border-color: var(--accent-strong);
  }
  .sk-radio:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .sk-radio.is-selected .sk-radio-box {
    border-color: var(--accent);
  }
  .sk-radio.is-disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .sk-radio-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    pointer-events: none;
    opacity: 0;
    scale: 0.25;
    filter: blur(4px);
    transition:
      background var(--dur-fast) var(--ease-out),
      opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1),
      scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1),
      filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1);
  }
  .sk-radio-dot.is-visible {
    opacity: 1;
    scale: 1;
    filter: blur(0);
  }

  .sk-radio-label {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
  }
</style>
