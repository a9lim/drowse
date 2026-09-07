<script lang="ts">
  import FluentIcon from "./ui/FluentIcon.svelte";
  // Drowse editable combobox — arbitrary role labels plus a themed roster
  // picker. Unlike a native datalist, the popup stays inside the webui's
  // surface, type, focus, and keyboard system.
  import { onMount, tick } from "svelte";
  import { dropdownMotion } from "./dropdownMotion.svelte";

  interface Option { value: string; label: string; }
  interface Props {
    value: string;
    options: Option[];
    placeholder?: string;
    disabled?: boolean;
    invalid?: boolean;
    ariaLabel?: string;
    ariaDescribedby?: string;
    title?: string;
    spellcheck?: boolean;
    onchange?: (value: string) => void;
  }

  let {
    value = $bindable(), options, placeholder = "", disabled = false,
    invalid = false, ariaLabel, ariaDescribedby, title, spellcheck = false, onchange,
  }: Props = $props();

  let open = $state(false);
  const presence = dropdownMotion();
  let filtering = $state(false);
  let highlight = $state(-1);
  let input: HTMLInputElement | null = $state(null);
  let listbox: HTMLUListElement | null = $state(null);
  let popoverStyle = $state("");
  const uid = $props.id();

  $effect(() => { if (disabled && open) closePopover(); });

  const visible = $derived.by(() => {
    if (!filtering || !value.trim()) return options;
    const needle = value.trim().toLowerCase();
    return options.filter((opt) => opt.label.toLowerCase().includes(needle));
  });

  async function openPopover(filter = false): Promise<void> {
    if (disabled) return;
    filtering = filter;
    open = true;
    presence.mount();
    highlight = visible.findIndex((opt) => opt.value === value);
    if (highlight < 0 && visible.length > 0) highlight = 0;
    await tick();
    if (!open) return;
    try { listbox?.showPopover(); } catch { /* fixed fallback */ }
    placeListbox();
    await tick();
    if (open && listbox) presence.show(listbox);
  }

  function closePopover(): void {
    if (!open) return;
    open = false;
    presence.close(listbox);
  }

  async function commit(index: number): Promise<void> {
    const opt = visible[index];
    if (!opt) return;
    value = opt.value;
    onchange?.(value);
    closePopover();
    await tick();
    input?.focus();
    input?.setSelectionRange(value.length, value.length);
  }

  function onInput(ev: Event): void {
    value = (ev.currentTarget as HTMLInputElement).value;
    onchange?.(value);
    void openPopover(true);
  }

  function move(dir: 1 | -1): void {
    if (visible.length === 0) return;
    highlight = (highlight + dir + visible.length) % visible.length;
    const row = listbox?.children[highlight] as HTMLElement | undefined;
    row?.scrollIntoView({ block: "nearest" });
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      if (!open) void openPopover(false);
      else move(ev.key === "ArrowDown" ? 1 : -1);
    } else if (ev.key === "Enter" && open) {
      ev.preventDefault();
      if (highlight >= 0) void commit(highlight);
      else closePopover();
    } else if (ev.key === "Escape" && open) {
      ev.preventDefault();
      ev.stopPropagation();
      closePopover();
    } else if (ev.key === "Tab") {
      closePopover();
    }
  }

  function placeListbox(): void {
    if (!input || !listbox) return;
    const r = input.parentElement?.getBoundingClientRect()
      ?? input.getBoundingClientRect();
    const gutter = 8, gap = 2;
    const viewport = window.visualViewport;
    const viewportLeft = viewport?.offsetLeft ?? 0;
    const viewportTop = viewport?.offsetTop ?? 0;
    const viewportWidth = viewport?.width ?? window.innerWidth;
    const viewportBottom = viewportTop + (viewport?.height ?? window.innerHeight);
    const desired = Math.min(240, Math.max(32, listbox.scrollHeight));
    const below = viewportBottom - r.bottom - gutter - gap;
    const above = r.top - viewportTop - gutter - gap;
    const upward = below < Math.min(desired, 160) && above > below;
    const maxHeight = Math.max(32, Math.min(240, upward ? above : below));
    const height = Math.min(desired, maxHeight);
    const width = Math.min(r.width, viewportWidth - gutter * 2);
    const left = Math.max(viewportLeft + gutter, Math.min(r.left, viewportLeft + viewportWidth - width - gutter));
    const top = upward
      ? Math.max(viewportTop + gutter, r.top - height - gap)
      : Math.min(viewportBottom - height - gutter, r.bottom + gap);
    popoverStyle = `left:${left}px;top:${top}px;width:${width}px;max-height:${maxHeight}px`;
    listbox.dataset.origin = upward ? "bottom-left" : "top-left";
  }

  function onDocumentPointer(ev: PointerEvent): void {
    if (!open) return;
    const target = ev.target as Node;
    if (input?.parentElement?.contains(target) || listbox?.contains(target)) return;
    closePopover();
  }
  function reposition(): void { if (open) placeListbox(); }

  onMount(() => {
    document.addEventListener("pointerdown", onDocumentPointer, true);
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    window.visualViewport?.addEventListener("resize", reposition);
    window.visualViewport?.addEventListener("scroll", reposition);
    return () => {
      document.removeEventListener("pointerdown", onDocumentPointer, true);
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      window.visualViewport?.removeEventListener("resize", reposition);
      window.visualViewport?.removeEventListener("scroll", reposition);
      presence.destroy();
    };
  });
</script>

<div class="sk-combobox field-focus" class:is-open={open} class:is-invalid={invalid}>
  <input
    bind:this={input} bind:value {placeholder} {disabled} {title} {spellcheck}
    role="combobox" aria-label={ariaLabel} aria-autocomplete="list"
    aria-describedby={ariaDescribedby}
    aria-expanded={open} aria-controls={open ? `${uid}-listbox` : undefined}
    aria-activedescendant={open && highlight >= 0 ? `${uid}-option-${highlight}` : undefined}
    aria-invalid={invalid}
    oninput={onInput} onkeydown={onKeydown}
  />
  <button
    type="button" class="caret" aria-label={`Choose ${ariaLabel ?? "value"}`}
    tabindex="-1" {disabled}
    onclick={() => (open ? closePopover() : void openPopover(false))}
  ><FluentIcon name="down" /></button>

  {#if presence.mounted}
    <ul bind:this={listbox} id={`${uid}-listbox`} class="popover t-dropdown"
      popover="manual" style={popoverStyle} role="listbox" aria-label={ariaLabel}>
      {#each visible as opt, i (opt.value)}
        <li id={`${uid}-option-${i}`} role="option"
          class:highlight={i === highlight} class:active={opt.value === value}
          aria-selected={opt.value === value}
          tabindex="-1"
          onpointerenter={() => (highlight = i)}
          onpointerdown={(ev) => ev.preventDefault()}
          onkeydown={(ev) => {
            if (ev.key === "Enter" || ev.key === " ") void commit(i);
          }}
          onclick={() => void commit(i)}>{opt.label}</li>
      {:else}
        <li class="empty" role="option" aria-selected="false" aria-disabled="true">
          Custom role: “{value.trim()}”
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .sk-combobox { position: relative; display: flex; align-items: center; min-width: 0;
    min-height: var(--control-field);
    width: 100%; border-radius: var(--radius); background: var(--control-sheen), var(--input-well);
    box-shadow: var(--shadow-well);
    transition: background var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out); }
  .sk-combobox:hover, .sk-combobox.is-open {
    background: var(--control-sheen), var(--surface-hi);
    box-shadow: var(--shadow-control-hover);
  }
  .sk-combobox.is-invalid { outline: 1px solid var(--accent-red); }
  input { width: 100%; min-width: 0; min-height: var(--control-field);
    padding-block: var(--space-2);
    padding-inline: var(--space-sm) calc(var(--control-compact) + var(--space-xs));
    border: 0; border-radius: inherit; outline: 0; background: transparent; color: var(--fg); font: inherit;
    font-family: var(--font-mono); font-size: var(--text-sm); }
  input:disabled, input:disabled + .caret { opacity: .5; cursor: not-allowed; }
  .caret { position: absolute; inset-inline-end: 0; display: grid; place-items: center; width: var(--control-compact);
    height: 100%; padding: 0; border: 0; border-radius: inherit; background: transparent; background-image: none !important; color: var(--fg-muted);
    cursor: pointer; transition: color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out); }
  .is-open .caret { color: var(--accent); transform: rotate(180deg); }
  .popover { position: fixed; inset: auto; z-index: var(--z-modal); box-sizing: border-box;
    overflow-y: auto; margin: 0; padding: var(--surface-padding); list-style: none;
    border: 1px solid var(--popup-border); border-radius: var(--popup-radius); background: var(--surface-sheen), var(--popup-bg);
    box-shadow: var(--popup-shadow); }
  li { min-height: var(--control-target); padding: var(--space-2) var(--space-3);
    border-radius: var(--radius);
    color: var(--fg-strong); font-size: var(--text-sm); line-height: 1.45;
    overflow-wrap: anywhere; white-space: normal; cursor: pointer; }
  li.highlight { background: var(--control-sheen), var(--bg-hover); color: var(--fg); }
  li.active { background: var(--control-sheen), var(--accent-subtle); }
  li.active.highlight { background: var(--control-sheen), var(--accent-strong); }
  li.empty { color: var(--fg-muted); cursor: default; }
</style>
