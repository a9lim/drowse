<script lang="ts">
  import { onMount, tick } from "svelte";
  import { flip } from "svelte/animate";
  import { motionDuration } from "../../lib/motion";
  import MorphText from "../../lib/ui/MorphText.svelte";
  import { tokenProbabilityRows, visibleTokenText } from "../../lib/tokenProbabilities";
  import { probabilityScore, scoreToRgb as highlightColor, surpriseScore } from "../../lib/tokens";
  import type { TokenScore } from "../../lib/types";

  let { tokens, highlights = true, layout = "message", coloring = "surprisal" }: {
    tokens: TokenScore[];
    highlights?: boolean;
    layout?: "message" | "inline";
    coloring?: string;
  } = $props();
  let active = $state<number | null>(null);
  let displayed = $state<number | null>(null);
  let anchor: HTMLElement | null = null;
  let panel: HTMLDivElement;
  let pinned = false;
  let fade: Animation | null = null;
  let hoverAfter = 0;
  let hoverTimer: ReturnType<typeof setTimeout> | undefined;
  let closeTimer: ReturnType<typeof setTimeout> | undefined;
  const uid = $props.id();
  const token = $derived(displayed === null ? null : tokens[displayed]);
  const rows = $derived(token ? tokenProbabilityRows(token) : []);
  $effect.pre(() => {
    tokens;
    cancelTimers();
    fade?.cancel();
    active = null;
    displayed = null;
    anchor = null;
    pinned = false;
    panel?.hidePopover();
  });
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
    if (active === null) return;
    const trigger = anchor;
    const opacity = getComputedStyle(panel).opacity;
    fade?.cancel();
    active = null;
    pinned = false;
    hoverAfter = performance.now() + 250;
    const exit = panel.animate([{ opacity }, { opacity: 0 }], {
      duration: matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 200,
      easing: "ease-out",
    });
    fade = exit;
    exit.onfinish = () => { if (fade === exit) { panel.hidePopover(); fade = null; } };
    if (restoreFocus) trigger?.focus({ preventScroll: true });
  }
  async function open(index: number, element: HTMLElement, pin = false, focus = false) {
    cancelTimers();
    anchor = element;
    active = index;
    displayed = index;
    pinned = pin;
    await tick();
    if (active !== index || anchor !== element) return;
    const opacity = panel.matches(":popover-open") ? getComputedStyle(panel).opacity : "0";
    fade?.cancel();
    panel.showPopover();
    place();
    fade = panel.animate([{ opacity }, { opacity: 1 }], {
      duration: matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 200,
      easing: "ease-out",
    });
    if (focus) panel.focus({ preventScroll: true });
  }
  function hover(event: PointerEvent, index: number) {
    if (event.pointerType !== "mouse" || pinned || performance.now() < hoverAfter) return;
    cancelTimers();
    const element = event.currentTarget as HTMLElement;
    hoverTimer = setTimeout(() => void open(index, element), 1000);
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
    const resize = new ResizeObserver(place);
    resize.observe(panel);
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
      fade?.cancel();
      resize.disconnect();
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

{#snippet tokenButton(text: string, index: number)}<button
    type="button" class="recorded-token" class:highlighted={highlights} class:selected={active === index} class:whitespace={text.trim().length === 0}
    style:--token-tint={highlightColor(coloring === "probability" ? probabilityScore(tokens[index].logprob) : surpriseScore(tokens[index].logprob), undefined, "surprise")}
    aria-label={`Inspect token ${visibleTokenText(tokens[index].text)}`}
    aria-haspopup="dialog" aria-expanded={active === index} aria-controls={active === index ? `${uid}-panel` : undefined}
    onpointerenter={event => hover(event, index)} onpointerleave={leave}
    onclick={event => void open(index, event.currentTarget, true, event.detail === 0)}
    onkeydown={move}
  >{#if layout === "inline"}<MorphText {text} numbers={false} duration={240} />{:else}{text}{/if}</button>{/snippet}

<svelte:element this={layout === "message" ? "h1" : "div"} class="message-title" class:inline={layout === "inline"} aria-label={tokens.map(token => token.text).join("").trim()}><span class="recorded-tokens">{#if layout === "inline"}{#each tokens as piece, index (index)}<span class="inline-piece" animate:flip={{ duration: motionDuration(240) }}>{@render tokenButton(piece.text, index)}</span>{/each}{:else}{#each [groups.slice(0, 2), groups.slice(2)] as section, sectionIndex}{sectionIndex === 0 ? "" : " "}<span class={sectionIndex === 0 ? "message-prefix" : "message-body"}>{#each section as group, groupIndex}{groupIndex === 0 ? "" : group.space}<span class="word">{#each group.pieces as piece}{@render tokenButton(piece.text, piece.index)}{/each}</span>{/each}</span>{/each}{/if}</span></svelte:element>

<svg class="glass-definitions" width="0" height="0" aria-hidden="true" focusable="false">
  <defs>
    <filter id="recorded-token-glass" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB">
      <feImage href="/images/landing-glass-map.svg" width="100%" height="100%" preserveAspectRatio="none" result="rim" />
      <feDisplacementMap in="SourceGraphic" in2="rim" scale="32" xChannelSelector="R" yChannelSelector="G" />
    </filter>
  </defs>
</svg>

<div bind:this={panel} id={`${uid}-panel`} class="recorded-token-panel" popover="manual" role="dialog" tabindex="-1"
  aria-labelledby={`${uid}-title`} aria-hidden={active === null} onpointerenter={cancelTimers} onpointerleave={leave}>
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
  {/if}
</div>

<style>
  .message-title { margin: 0; font: inherit; letter-spacing: inherit; line-height: inherit; text-wrap: balance; }
  .recorded-tokens { display: flex; flex-direction: column; white-space: pre-wrap; }
  .message-prefix { --weight-structure: var(--weight-display); font-size: var(--text-not-found-prefix); display: block; margin-bottom: var(--space-sm); font-family: var(--font-structure); font-weight: var(--weight-display); letter-spacing: -0.075em; white-space: nowrap; }
  .message-body { display: block; }
  .word { display: inline-block; white-space: nowrap; max-width: 100%; }
  .recorded-token { --control-sheen: none; display: inline-block; min-height: 44px; vertical-align: baseline; padding: 0; margin: 0; border: 0; border-radius: var(--data-mark-radius); background: transparent; color: inherit; font: inherit; letter-spacing: inherit; line-height: inherit; cursor: pointer; box-decoration-break: clone; -webkit-box-decoration-break: clone; transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out); }
  .recorded-token.highlighted { background: var(--token-tint); }
  .recorded-token:hover, .recorded-token.selected { box-shadow: 0 0 0 2px var(--accent); }
  .recorded-token:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
  .recorded-token:active { transform: none; scale: 1; }
  .inline { text-wrap: pretty; }
  .inline .recorded-tokens { display: inline; }
  .inline-piece { display: inline-block; max-width: 100%; vertical-align: baseline; }
  .inline .recorded-token { display: inline; min-height: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: inherit !important; font-weight: inherit !important; }
  .inline .recorded-token.whitespace { display: inline-block; min-width: 0.35em; white-space: pre; }
  .glass-definitions { position: absolute; pointer-events: none; }
  .recorded-token-panel { --fg: var(--landing-panel-ink); --fg-dim: var(--landing-panel-muted); position: fixed; inset: auto; margin: 0; box-sizing: border-box; width: min(20rem, calc(100vw - 24px)); padding: var(--space-sm); overflow: auto; overscroll-behavior: contain; border: 0; border-radius: var(--radius-lg); background: var(--recorded-token-panel-bg); color: var(--fg); text-shadow: var(--landing-panel-text-shadow); -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px); font: var(--text-sm)/1.4 var(--font-ui); letter-spacing: normal; }
  .recorded-token-panel[aria-hidden="true"] { pointer-events: none; }
  .recorded-token-panel { white-space: normal; overflow-wrap: normal; }
  header { display: flex; justify-content: space-between; align-items: start; gap: var(--space-sm); }
  header > div { min-width: 0; }
  .recorded-token-panel h2 { margin: var(--space-xs) 0; font-family: var(--font-mono); font-size: var(--text-md); overflow-wrap: anywhere; text-wrap: balance; }
  .source, dt { color: var(--fg-dim); font-size: var(--text-xs); }
  .close { min-width: 44px; min-height: 44px; padding: var(--space-xs) var(--space-sm); border: 0; border-radius: var(--radius); background: var(--bg-hover); color: var(--fg); font: inherit; cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .close:hover { background: var(--accent-subtle); }
  .close:active { scale: 0.96; }
  dl { display: flex; flex-wrap: wrap; gap: var(--space-xs) var(--space-sm); margin: var(--space-xs) 0; font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  dl div { display: flex; gap: var(--space-xs); }
  dd { margin: 0; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  th { font-weight: var(--weight-medium); color: var(--fg-dim); text-align: start; }
  th, td { padding: var(--space-xs); }
  th:last-child, td:last-child { width: 6rem; text-align: end; font-variant-numeric: tabular-nums; }
  td { border-top: 1px solid var(--glass-line); }
  code { overflow-wrap: anywhere; font-family: var(--font-mono); }
  .chosen { color: var(--accent); font-weight: var(--weight-medium); background: var(--accent-subtle); }
  .chosen-label { display: inline-block; margin-inline-start: var(--space-xs); font-family: var(--font-ui); font-size: var(--text-xs); }
  @supports (backdrop-filter: url("#recorded-token-glass")) and (not (-webkit-touch-callout: none)) {
    .recorded-token-panel { backdrop-filter: url("#recorded-token-glass") blur(12px); }
  }
  @media (prefers-contrast: more), (forced-colors: active), (prefers-reduced-transparency: reduce) {
    .recorded-token-panel { --fg: CanvasText; --fg-dim: CanvasText; background: Canvas; -webkit-backdrop-filter: none; backdrop-filter: none; text-shadow: none; }
  }
  @media (prefers-reduced-motion: reduce) { .recorded-token, .close, .recorded-token-panel { transition: none; } .close:active { scale: 1; } }
</style>
