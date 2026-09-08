<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { fade } from "svelte/transition";
  import { contentIn, contentOut } from "../motion";
  import type { TokenScore } from "../types";
  import { tokenProbabilityRows, visibleTokenText } from "../tokenProbabilities";
  import Button from "./Button.svelte";
  import { samplingState } from "../stores/sampling.svelte";
  import { getRuntimeClient } from "../runtime/registry";
  import { tokenAlternativeDefault } from "../runtime/samplingCapabilities";

  let { token, anchor, source = "Model token", contextChanged = false, onclose, ondetails }: {
    token: TokenScore;
    anchor: HTMLElement;
    source?: string;
    contextChanged?: boolean;
    onclose: () => void;
    ondetails?: () => void;
  } = $props();
  const uid = $props.id();
  const trigger = untrack(() => anchor);
  const rows = $derived(tokenProbabilityRows(token));
  const alternativeCount = tokenAlternativeDefault(getRuntimeClient().mode);
  let panel: HTMLDivElement;
  let closing = false;

  function close(restoreFocus = true) {
    if (closing) return;
    closing = true;
    panel.inert = true;
    if (restoreFocus && trigger.isConnected) trigger.focus({ preventScroll: true });
    onclose();
  }

  function place() {
    if (closing) return;
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? innerWidth;
    const height = viewport?.height ?? innerHeight;
    panel.style.maxWidth = `${width - 16}px`;
    panel.style.maxHeight = `${height - 16}px`;
    const rect = trigger.getBoundingClientRect();
    const box = panel.getBoundingClientRect();
    const x = Math.max(left + 8, Math.min(rect.left, left + width - box.width - 8));
    const below = Math.max(0, top + height - rect.bottom - 14);
    const above = Math.max(0, rect.top - top - 14);
    const opensBelow = box.height <= below || below >= above;
    const maxHeight = Math.min(height - 16, opensBelow ? below : above);
    const panelHeight = Math.min(box.height, maxHeight);
    const y = Math.max(top + 8, Math.min(opensBelow ? rect.bottom + 6 : rect.top - panelHeight - 6, top + height - panelHeight - 8));
    panel.style.left = `${x}px`;
    panel.style.top = `${y}px`;
    panel.style.maxHeight = `${maxHeight}px`;
  }

  onMount(() => {
    panel.showPopover();
    place();
    panel.focus({ preventScroll: true });
    const escape = (event: KeyboardEvent) => {
      if (closing) return;
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    const outside = (event: PointerEvent) => {
      if (!closing && !panel.contains(event.target as Node) && !trigger.contains(event.target as Node)) close(false);
    };
    const focusOutside = (event: FocusEvent) => {
      if (!closing && !panel.contains(event.target as Node) && !trigger.contains(event.target as Node)) close(false);
    };
    const scroll = (event: Event) => {
      if (event.target instanceof Node && panel.contains(event.target)) return;
      place();
    };
    document.addEventListener("keydown", escape, true);
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("focusin", focusOutside);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", scroll, true);
    window.visualViewport?.addEventListener("resize", place);
    window.visualViewport?.addEventListener("scroll", place);
    return () => {
      document.removeEventListener("keydown", escape, true);
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("focusin", focusOutside);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", scroll, true);
      window.visualViewport?.removeEventListener("resize", place);
      window.visualViewport?.removeEventListener("scroll", place);
    };
  });
</script>

<div bind:this={panel} class="token-logits-popover" popover="manual" role="dialog" tabindex="-1" aria-labelledby={`${uid}-title`}
  in:fade|global={contentIn()}
  out:fade|global={contentOut()}
  onoutrostart={() => { closing = true; panel.inert = true; }}
>
  <header>
    <div><span class="source">{source}</span><h2 id={`${uid}-title`}>{visibleTokenText(token.text)}</h2></div>
    <Button ariaLabel="Close token probabilities" onclick={() => close()}>Close</Button>
  </header>
  {#if contextChanged}<p class="context-note">Recorded before your edit, using the original preceding text.</p>{/if}
  {#if rows.length}
    <table aria-label="Recorded token probabilities">
      <thead><tr><th scope="col">Token</th><th scope="col">Probability</th></tr></thead>
      <tbody>{#each rows as row, index (index)}
        <tr class:chosen={row.chosen}>
          <td><code>{visibleTokenText(row.text)}</code>{#if row.chosen}<span class="chosen-label">this token</span>{/if}</td>
          <td {...{ "aria-description": (`log probability ${row.logprob.toFixed(4)}`) }}>{Math.exp(row.logprob) >= 0.001 ? Math.exp(row.logprob).toFixed(3) : Math.exp(row.logprob).toExponential(2)}</td>
        </tr>
      {/each}</tbody>
    </table>
  {/if}
  {#if !token.topAlts?.length}<p class="empty-note">Alternative probabilities weren’t recorded for this token.</p>{/if}
  {#if !token.topAlts?.length && alternativeCount > 0 && samplingState.return_top_k === 0}
    <Button onclick={() => samplingState.return_top_k = alternativeCount}>Record alternatives for future tokens</Button>
  {/if}
  <p class="provenance">Recorded sampling probabilities, not raw logits.</p>
  {#if ondetails}<Button onclick={() => { trigger.focus({ preventScroll: true }); ondetails?.(); onclose(); }}>Full token details</Button>{/if}
</div>

<style>
  .token-logits-popover {
    position: fixed; inset: auto; margin: 0; width: min(22rem, calc(100vw - 16px));
    box-sizing: border-box; padding: var(--surface-padding); overflow: auto; overscroll-behavior: contain;
    border: 1px solid var(--popup-border); border-radius: var(--popup-radius);
    background: var(--popup-bg); box-shadow: var(--popup-shadow); color: var(--fg-strong);
    font-family: var(--font-ui); font-size: var(--text-sm);
  }
  header { display: flex; justify-content: space-between; align-items: start; gap: var(--space-sm); }
  header > div { min-width: 0; }
  h2 { margin: var(--space-xs) 0 var(--space-sm); font-family: var(--font-mono); font-size: var(--text-md); overflow-wrap: anywhere; }
  .source, .provenance, .empty-note, .context-note { color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.5; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  th { font-weight: var(--weight-medium); color: var(--fg-dim); text-align: start; }
  th, td { padding: var(--space-sm) var(--space-xs); }
  th:last-child, td:last-child { width: 7rem; text-align: end; font-variant-numeric: tabular-nums; }
  td { border-top: 1px solid var(--glass-line); }
  code { overflow-wrap: anywhere; font-family: var(--font-mono); }
  .chosen { color: var(--accent); font-weight: var(--weight-medium); background: var(--accent-subtle); }
  .chosen-label { display: inline-block; margin-inline-start: var(--space-xs); font-family: var(--font-ui); font-size: var(--text-xs); }
  .provenance { margin-block: var(--space-sm); }
</style>
