<script lang="ts" generics="T extends string | number">
  import MorphText from "./ui/MorphText.svelte";
  import FluentIcon from "./ui/FluentIcon.svelte";
  // Themed single-select dropdown — replacement for native `<select>`
  // across the webui.  Built to match the flat-dark aesthetic in
  // ``tokens.css``: same border / radius / focus ring as `.input`,
  // popover laid on the highest surface tier (``--surface-hi``),
  // accent-tinted active row, no native chrome anywhere.
  //
  // Behaviour mirrors a native single-select listbox:
  //
  //   * click button → popover opens; click button again or outside →
  //     closes.  Esc closes.
  //   * ↑ / ↓ move the highlight, Enter / Space commit, Home / End
  //     jump to first / last.
  //   * type any character → first-letter typeahead (alpha-num only),
  //     buffer flushes after 600ms idle.
  //   * the listbox lives in the browser top layer, so scrollable racks
  //     cannot clip it; fixed placement flips and clamps to the viewport.
  //
  // The component is generic over value type: ``T extends string |
  // number``.  Options are declarative via the ``options`` prop —
  // ``{ value, label, disabled? }`` — so callsite replacement is
  // mechanical from ``<option value=…>…</option>``.

  import { onMount, tick } from "svelte";
  import { dropdownMotion } from "./dropdownMotion.svelte";

  interface Option<U> {
    value: U;
    label: string;
    disabled?: boolean;
  }

  interface Props {
    value: T;
    options: Option<T>[];
    placeholder?: string;
    disabled?: boolean;
    ariaLabel?: string;
    invalid?: boolean;
    ariaDescribedby?: string;
    title?: string;
    /** Optional callback fired on commit.  ``bind:value`` works without it. */
    onchange?: (value: T) => void;
  }

  let {
    value = $bindable(),
    options,
    placeholder = "",
    disabled = false,
    ariaLabel,
    invalid = false,
    ariaDescribedby,
    title,
    onchange,
  }: Props = $props();

  let open = $state(false);
  const presence = dropdownMotion();
  let highlight = $state(-1);
  let trigger: HTMLButtonElement | null = $state(null);
  let listbox: HTMLUListElement | null = $state(null);
  let popoverStyle = $state("");
  const uid = $props.id();

  // Typeahead buffer — alphanum chars within 600 ms accumulate; idle
  // flushes the buffer.
  let typeBuffer = $state("");
  let typeTimer: ReturnType<typeof setTimeout> | null = null;

  $effect(() => { if (disabled && open) closePopover(false); });

  const currentIndex = $derived(
    options.findIndex((opt) => opt.value === value),
  );
  const currentLabel = $derived(
    currentIndex >= 0 ? options[currentIndex].label : "",
  );

  async function commit(idx: number): Promise<void> {
    const opt = options[idx];
    if (!opt || opt.disabled) return;
    value = opt.value;
    onchange?.(opt.value);
    closePopover(true);
    await tick();
  }

  async function openPopover(): Promise<void> {
    if (disabled) return;
    open = true;
    presence.mount();
    highlight = currentIndex >= 0 ? currentIndex : firstEnabled(0, 1);
    await tick();
    if (!open) return;
    try {
      listbox?.showPopover();
    } catch {
      // Fixed positioning still escapes ordinary scroll containers on
      // browsers without the Popover API; the top layer is progressive.
    }
    placeListbox();
    await tick();
    if (!open || !listbox) return;
    presence.show(listbox);
    listbox?.focus();
  }

  function closePopover(restoreFocus: boolean): void {
    if (!open) return;
    open = false;
    presence.close(listbox);
    typeBuffer = "";
    if (restoreFocus) queueMicrotask(() => trigger?.focus());
  }

  function toggle(): void {
    if (open) closePopover(false);
    else void openPopover();
  }

  function firstEnabled(from: number, dir: 1 | -1): number {
    if (options.length === 0) return -1;
    let i = from;
    for (let n = 0; n < options.length; n++) {
      if (i >= 0 && i < options.length && !options[i].disabled) return i;
      i += dir;
      if (i < 0) i = options.length - 1;
      else if (i >= options.length) i = 0;
    }
    return -1;
  }

  function moveHighlight(dir: 1 | -1): void {
    if (options.length === 0) return;
    let i = highlight;
    for (let n = 0; n < options.length; n++) {
      i += dir;
      if (i < 0) i = options.length - 1;
      else if (i >= options.length) i = 0;
      if (!options[i].disabled) {
        highlight = i;
        scrollHighlightIntoView();
        return;
      }
    }
  }

  function scrollHighlightIntoView(): void {
    if (!listbox || highlight < 0) return;
    const row = listbox.children[highlight] as HTMLElement | undefined;
    if (!row) return;
    const top = row.offsetTop;
    const bot = top + row.offsetHeight;
    if (top < listbox.scrollTop) {
      listbox.scrollTop = top;
    } else if (bot > listbox.scrollTop + listbox.clientHeight) {
      listbox.scrollTop = bot - listbox.clientHeight;
    }
  }

  function handleTypeahead(key: string): void {
    if (key.length !== 1) return;
    const ch = key.toLowerCase();
    if (!/[a-z0-9]/.test(ch)) return;
    typeBuffer += ch;
    if (typeTimer) clearTimeout(typeTimer);
    typeTimer = setTimeout(() => {
      typeBuffer = "";
      typeTimer = null;
    }, 600);
    const needle = typeBuffer;
    // Start from highlight+1 so repeated presses of the same letter
    // cycle through matches.
    const start = highlight >= 0 ? (highlight + 1) % options.length : 0;
    for (let n = 0; n < options.length; n++) {
      const i = (start + n) % options.length;
      const opt = options[i];
      if (opt.disabled) continue;
      if (opt.label.toLowerCase().startsWith(needle)) {
        highlight = i;
        scrollHighlightIntoView();
        return;
      }
    }
  }

  function onTriggerKeydown(ev: KeyboardEvent): void {
    if (disabled) return;
    switch (ev.key) {
      case "ArrowDown":
      case "ArrowUp":
      case "Enter":
      case " ":
        ev.preventDefault();
        void openPopover();
        break;
    }
  }

  function onListKeydown(ev: KeyboardEvent): void {
    switch (ev.key) {
      case "ArrowDown":
        ev.preventDefault();
        moveHighlight(1);
        break;
      case "ArrowUp":
        ev.preventDefault();
        moveHighlight(-1);
        break;
      case "Home":
        ev.preventDefault();
        highlight = firstEnabled(0, 1);
        scrollHighlightIntoView();
        break;
      case "End":
        ev.preventDefault();
        highlight = firstEnabled(options.length - 1, -1);
        scrollHighlightIntoView();
        break;
      case "Enter":
      case " ":
        ev.preventDefault();
        if (highlight >= 0) void commit(highlight);
        break;
      case "Escape":
        ev.preventDefault();
        ev.stopPropagation();
        closePopover(true);
        break;
      case "Tab":
        // Keep native tab order: let the browser move past the listbox,
        // then remove the popover. Preventing Tab here trapped keyboard
        // users on the trigger for one extra keystroke.
        setTimeout(() => closePopover(false), 0);
        break;
      default:
        handleTypeahead(ev.key);
    }
  }

  function placeListbox(): void {
    if (!trigger || !listbox) return;
    const tr = trigger.parentElement!.getBoundingClientRect();
    const gutter = 8;
    const gap = 2;
    const viewport = window.visualViewport;
    const viewportLeft = viewport?.offsetLeft ?? 0;
    const viewportTop = viewport?.offsetTop ?? 0;
    const viewportWidth = viewport?.width ?? window.innerWidth;
    const viewportBottom = viewportTop + (viewport?.height ?? window.innerHeight);
    const below = Math.max(0, viewportBottom - tr.bottom - gutter - gap);
    const above = Math.max(0, tr.top - viewportTop - gutter - gap);
    const desired = Math.min(320, Math.max(40, listbox.scrollHeight));
    const flipUp = below < Math.min(desired, 200) && above > below;
    const available = flipUp ? above : below;
    const maxHeight = Math.max(40, Math.min(320, available));
    const renderedHeight = Math.min(desired, maxHeight);
    const popoverWidth = Math.min(tr.width, viewportWidth - gutter * 2);
    const left = Math.max(
      viewportLeft + gutter,
      Math.min(tr.left, viewportLeft + viewportWidth - popoverWidth - gutter),
    );
    const top = flipUp
      ? Math.max(viewportTop + gutter, tr.top - renderedHeight - gap)
      : Math.min(viewportBottom - renderedHeight - gutter, tr.bottom + gap);
    popoverStyle = `left:${left}px;top:${top}px;width:${popoverWidth}px;max-height:${maxHeight}px`;
    listbox.dataset.origin = flipUp ? "bottom-left" : "top-left";
  }

  function onDocumentMouseDown(ev: MouseEvent): void {
    if (!open) return;
    const t = ev.target as Node;
    if (trigger?.contains(t) || listbox?.contains(t)) return;
    closePopover(false);
  }

  function onWindowResize(): void {
    if (open) placeListbox();
  }

  onMount(() => {
    document.addEventListener("mousedown", onDocumentMouseDown, true);
    window.addEventListener("resize", onWindowResize);
    window.addEventListener("scroll", onWindowResize, true);
    window.visualViewport?.addEventListener("resize", onWindowResize);
    window.visualViewport?.addEventListener("scroll", onWindowResize);
    return () => {
      document.removeEventListener("mousedown", onDocumentMouseDown, true);
      window.removeEventListener("resize", onWindowResize);
      window.removeEventListener("scroll", onWindowResize, true);
      window.visualViewport?.removeEventListener("resize", onWindowResize);
      window.visualViewport?.removeEventListener("scroll", onWindowResize);
      if (typeTimer) clearTimeout(typeTimer);
      presence.destroy();
    };
  });
</script>

<div class="sk-select" class:is-open={open} class:is-disabled={disabled}>
  <button
    bind:this={trigger}
    type="button"
    class="sk-select-trigger field-focus"
    {disabled}
    {...{ "aria-description": (title) }}
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-controls={open ? `${uid}-listbox` : undefined}
    aria-label={ariaLabel}
    data-invalid={invalid || undefined}
    aria-describedby={ariaDescribedby}
    onclick={toggle}
    onkeydown={onTriggerKeydown}
  >
    <span class="sk-select-label" class:is-placeholder={currentIndex < 0}>
      <MorphText text={currentLabel || placeholder} numbers={false} />
    </span>
    <span class="sk-select-caret" aria-hidden="true"><FluentIcon name="down" /></span>
  </button>

  {#if presence.mounted}
    <ul
      bind:this={listbox}
      id={`${uid}-listbox`}
      class="sk-select-popover t-dropdown"
      popover="manual"
      style={popoverStyle}
      role="listbox"
      aria-invalid={invalid}
      tabindex="-1"
      aria-label={ariaLabel}
      aria-activedescendant={highlight >= 0
        ? `${uid}-opt-${highlight}`
        : undefined}
      onkeydown={onListKeydown}
    >
      {#each options as opt, i (i)}
        <li
          id={`${uid}-opt-${i}`}
          role="option"
          class="sk-select-opt"
          class:is-highlight={i === highlight}
          class:is-active={i === currentIndex}
          class:is-disabled={!!opt.disabled}
          aria-selected={i === currentIndex}
          aria-disabled={!!opt.disabled}
          onmouseenter={() => (opt.disabled ? null : (highlight = i))}
          onclick={(ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            void commit(i);
          }}
          onkeydown={onListKeydown}
        >
          {opt.label}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .sk-select {
    position: relative;
    display: inline-block;
    width: 100%;
    min-width: 0;
    max-width: 100%;
  }

  .sk-select-trigger {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: var(--control-target);
    width: 100%;
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
    gap: var(--control-label-gap);
    padding: var(--space-3) var(--space-4);
    /* Borderless input: recessed well fill is the "pick here" affordance;
     * the ring appears on focus only. */
    background: var(--control-sheen), var(--input-well);
    color: var(--fg);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font: inherit;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    text-align: start;
    cursor: pointer;
    box-shadow: var(--shadow-well);
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out);
  }
  .sk-select-trigger:hover:not(:disabled) {
    background: var(--control-sheen), var(--surface-hi);
    box-shadow: var(--shadow-control-hover);
  }
  .sk-select.is-open .sk-select-trigger {
    background: var(--control-sheen), var(--surface-hi);
  }
  .sk-select-trigger:disabled {
    opacity: var(--disabled-opacity);
    cursor: not-allowed;
  }
  .sk-select-trigger[data-invalid] { border-color: var(--accent-red); }

  .sk-select-label {
    flex: 1 1 0;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sk-select-label.is-placeholder {
    color: var(--fg-muted);
  }
  .sk-select-caret {
    flex: 0 0 auto;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    transition: transform var(--dur-fast) var(--ease-out);
  }
  .sk-select.is-open .sk-select-caret {
    transform: rotate(180deg);
    color: var(--accent);
  }

  .sk-select-popover {
    --popup-radius: var(--radius-lg);
    position: fixed;
    inset: auto;
    margin: 0;
    padding: var(--space-2);
    list-style: none;
    background: var(--surface-sheen), var(--popup-bg);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    box-shadow: var(--popup-shadow);
    box-sizing: border-box;
    overflow-y: auto;
    z-index: var(--z-modal);
    outline: none;
  }
  .sk-select-opt {
    display: flex;
    align-items: center;
    min-height: var(--control-option);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-sm);
    color: var(--fg-strong);
    cursor: pointer;
    font-size: var(--text-sm);
    line-height: 1.45;
    white-space: normal;
    overflow-wrap: anywhere;
  }
  .sk-select-opt.is-highlight:not(.is-disabled) {
    background: var(--control-sheen), var(--bg-hover);
    color: var(--fg);
  }
  .sk-select-opt.is-active {
    background: var(--control-sheen), var(--accent-subtle);
    color: var(--fg);
  }
  .sk-select-opt.is-active.is-highlight {
    background: var(--control-sheen), var(--accent-strong);
  }
  .sk-select-opt.is-disabled {
    color: var(--fg-muted);
    cursor: not-allowed;
  }
</style>
