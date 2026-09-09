<script lang="ts">
  import { onMount, tick } from "svelte";
  import { tokenProbabilityRows, visibleTokenText } from "../../lib/tokenProbabilities";
  import { scoreToRgb as highlightColor, surpriseScore } from "../../lib/tokens";
  import type { TokenScore } from "../../lib/types";

  let { tokens, highlights = true }: { tokens: TokenScore[]; highlights?: boolean } = $props();
  let active = $state<number | null>(null);
  let anchor: HTMLElement | null = null;
  let panel: HTMLDivElement;
  let pinned = false;
  let hoverAfter = 0;
  let hoverTimer: ReturnType<typeof setTimeout> | undefined;
  let closeTimer: ReturnType<typeof setTimeout> | undefined;
  const uid = $props.id();
  const token = $derived(active === null ? null : tokens[active]);
  const rows = $derived(token ? tokenProbabilityRows(token) : []);
  const groups = $derived.by(() => {
    const result: Array<{ space: string; pieces: Array<{ text: string; index: number }> }> = [];
    let group = { space: "", pieces: [] as Array<{ text: string; index: number }> };
    tokens.forEach((token, index) => {
      for (const part of token.text.match(/\s+|\S+/g) ?? []) {
        if (/^\s+$/.test(part)) {
          if (group.pieces.length) { result.push(group); group = { space: "", pieces: [] }; }
          group.space += part;
        } else group.pieces.push({ text: part, index });
      }
    });
    if (group.pieces.length) result.push(group);
    return result;
  });

  function place() {
    if (!anchor || active === null) return;
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0, top = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? innerWidth, height = viewport?.height ?? innerHeight;
    panel.style.maxHeight = `${height - 24}px`;
    const rect = anchor.getBoundingClientRect(), box = panel.getBoundingClientRect();
    const x = Math.max(left + 12, Math.min(rect.left, left + width - box.width - 12));
    const below = top + height - rect.bottom - 12;
    const y = below >= box.height || rect.top - top < below ? rect.bottom + 8 : rect.top - box.height - 8;
    panel.style.left = `${x}px`;
    panel.style.top = `${Math.max(top + 12, Math.min(y, top + height - box.height - 12))}px`;
  }

  function cancelTimers() { clearTimeout(hoverTimer); clearTimeout(closeTimer); }
  function close(restoreFocus = false) {
    cancelTimers();
    const trigger = anchor;
    active = null;
    pinned = false;
    hoverAfter = performance.now() + 250;
    panel?.hidePopover();
    if (restoreFocus) trigger?.focus({ preventScroll: true });
  }
  async function open(index: number, element: HTMLElement, pin = false, focus = false) {
    cancelTimers();
    anchor = element;
    active = index;
    pinned = pin;
    await tick();
    if (active !== index || anchor !== element) return;
    panel.showPopover();
    place();
    if (focus) panel.focus({ preventScroll: true });
  }
  function hover(event: PointerEvent, index: number) {
    if (event.pointerType !== "mouse" || pinned || performance.now() < hoverAfter) return;
    cancelTimers();
    const element = event.currentTarget as HTMLElement;
    hoverTimer = setTimeout(() => void open(index, element), 180);
  }
  function leave() {
    clearTimeout(hoverTimer);
    if (!pinned) closeTimer = setTimeout(() => close(), 220);
  }
  function move(event: KeyboardEvent) {
    const buttons = Array.from((event.currentTarget as HTMLElement).closest(".recorded-tokens")!.querySelectorAll<HTMLButtonElement>(".recorded-token"));
    const current = buttons.indexOf(event.currentTarget as HTMLButtonElement);
    const next = event.key === "ArrowRight" ? Math.min(current + 1, buttons.length - 1)
      : event.key === "ArrowLeft" ? Math.max(current - 1, 0)
      : event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : null;
    if (next !== null) { event.preventDefault(); buttons[next].focus(); }
  }

  onMount(() => {
    const outside = (event: PointerEvent) => {
      if (active !== null && !panel.contains(event.target as Node) && !anchor?.contains(event.target as Node)) close();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && active !== null) { event.preventDefault(); close(pinned); }
    };
    const focusOutside = (event: FocusEvent) => {
      if (active !== null && !panel.contains(event.target as Node) && !anchor?.contains(event.target as Node)) close();
    };
    const scroll = (event: Event) => { if (!(event.target instanceof Node) || !panel?.contains(event.target)) place(); };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    document.addEventListener("focusin", focusOutside);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", scroll, true);
    window.visualViewport?.addEventListener("resize", place);
    window.visualViewport?.addEventListener("scroll", place);
    return () => {
      cancelTimers();
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
      document.removeEventListener("focusin", focusOutside);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", scroll, true);
      window.visualViewport?.removeEventListener("resize", place);
      window.visualViewport?.removeEventListener("scroll", place);
    };
  });
</script>

<h2 class="message-title" aria-label={tokens.map(token => token.text).join("").trim()}><span class="recorded-tokens">
  {#each groups as group, groupIndex}{groupIndex === 0 ? "" : group.space}<span class="word">{#each group.pieces as piece}<button
    type="button" class="recorded-token" class:highlighted={highlights} class:selected={active === piece.index}
    style:--token-tint={highlightColor(surpriseScore(tokens[piece.index].logprob), undefined, "surprise")}
    aria-label={`Inspect token ${visibleTokenText(tokens[piece.index].text)}`}
    aria-haspopup="dialog" aria-expanded={active === piece.index} aria-controls={active === piece.index ? `${uid}-panel` : undefined}
    onpointerenter={event => hover(event, piece.index)} onpointerleave={leave}
    onclick={event => void open(piece.index, event.currentTarget, true, event.detail === 0)}
    onkeydown={move}
  >{piece.text}</button>{/each}</span>{/each}
</span></h2>

<div bind:this={panel} id={`${uid}-panel`} class="recorded-token-panel" popover="manual" role="dialog" tabindex="-1"
  aria-labelledby={`${uid}-title`} onpointerenter={cancelTimers} onpointerleave={leave}>
  {#if token}
    <header>
      <div><span class="source">Model token · Gemma 3 4B</span><h2 id={`${uid}-title`}>{visibleTokenText(token.text)}</h2></div>
      <button class="close" type="button" aria-label="Close token probabilities" onclick={() => close(true)}>Close</button>
    </header>
    <dl><div><dt>Token ID</dt><dd>{token.tokenId}</dd></div><div><dt>Surprisal</dt><dd>{(-(token.logprob ?? 0) / Math.LN2).toFixed(2)} bits</dd></div></dl>
    <table aria-label="Recorded token probabilities">
      <thead><tr><th scope="col">Token</th><th scope="col">Probability</th></tr></thead>
      <tbody>{#each rows as row, index (index)}<tr class:chosen={row.chosen}>
        <td><code>{visibleTokenText(row.text)}</code>{#if row.chosen}<span class="chosen-label">this token</span>{/if}</td>
        <td>{Math.exp(row.logprob) >= 0.001 ? Math.exp(row.logprob).toFixed(3) : Math.exp(row.logprob).toExponential(2)}</td>
      </tr>{/each}</tbody>
    </table>
    <p class="provenance">Recorded sampling probabilities, not raw logits. Each choice depends on the preceding tokens.</p>
  {/if}
</div>

<style>
  .message-title { margin: 0; font: inherit; letter-spacing: inherit; line-height: inherit; text-wrap: balance; }
  .recorded-tokens { white-space: pre-wrap; }
  .word { display: inline-block; white-space: nowrap; max-width: 100%; }
  .recorded-token { --control-sheen: none; display: inline-block; min-height: 44px; vertical-align: baseline; padding: 0; margin: 0; border: 0; border-radius: var(--data-mark-radius); background: transparent; color: inherit; font: inherit; letter-spacing: inherit; line-height: inherit; cursor: pointer; box-decoration-break: clone; -webkit-box-decoration-break: clone; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out); }
  .recorded-token.highlighted { background: var(--token-tint); }
  .recorded-token:hover, .recorded-token.selected { box-shadow: 0 0 0 2px var(--accent); }
  .recorded-token:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
  .recorded-token:active { transform: none; scale: 1; }
  .recorded-token-panel { position: fixed; inset: auto; margin: 0; box-sizing: border-box; width: min(24rem, calc(100vw - 24px)); padding: var(--surface-padding); overflow: auto; overscroll-behavior: contain; border: 0; border-radius: var(--popup-radius); background: var(--popup-bg); box-shadow: 0 0 0 1px var(--popup-border), var(--popup-shadow); color: var(--fg-strong); font: var(--text-sm)/1.5 var(--font-ui); letter-spacing: normal; }
  header { display: flex; justify-content: space-between; align-items: start; gap: var(--space-sm); }
  header > div { min-width: 0; }
  .recorded-token-panel h2 { margin: var(--space-xs) 0 var(--space-sm); font-family: var(--font-mono); font-size: var(--text-md); overflow-wrap: anywhere; text-wrap: balance; }
  .source, .provenance, dt { color: var(--fg-dim); font-size: var(--text-sm); }
  .close { min-width: 44px; min-height: 44px; padding: var(--space-xs) var(--space-sm); border: 0; border-radius: var(--radius); background: var(--bg-hover); color: var(--fg); font: inherit; cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .close:hover { background: var(--accent-subtle); }
  .close:active { scale: 0.96; }
  dl { display: flex; flex-wrap: wrap; gap: var(--space-sm) var(--space-lg); margin: var(--space-sm) 0; font-variant-numeric: tabular-nums; }
  dl div { display: flex; gap: var(--space-xs); }
  dd { margin: 0; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  th { font-weight: var(--weight-medium); color: var(--fg-dim); text-align: start; }
  th, td { padding: var(--space-sm) var(--space-xs); }
  th:last-child, td:last-child { width: 6rem; text-align: end; font-variant-numeric: tabular-nums; }
  td { border-top: 1px solid var(--glass-line); }
  code { overflow-wrap: anywhere; font-family: var(--font-mono); }
  .chosen { color: var(--accent); font-weight: var(--weight-medium); background: var(--accent-subtle); }
  .chosen-label { display: inline-block; margin-inline-start: var(--space-xs); font-family: var(--font-ui); font-size: var(--text-xs); }
  .provenance { margin-block: var(--space-sm) 0; text-wrap: pretty; }
  @media (prefers-reduced-motion: reduce) { .recorded-token, .close { transition: none; } .close:active { scale: 1; } }
</style>
