<script lang="ts">
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  // A completion node in the loom canvas. Generated completions expose their
  // actual model-token rows directly; authored text remains one editable
  // snippet because it has no authoritative tokenizer record.

  import type { LoomNodeJSON } from "../../lib/types";
  import { onDestroy } from "svelte";
  import { Spring, prefersReducedMotion } from "svelte/motion";
  import { roleGlyphLetter, tokenRowToScore } from "../../lib/stores.svelte";
  import { highlightStyleString } from "../../lib/highlight";
  import { splitTokenSentences } from "./loomSentences";

  interface Props {
    node: LoomNodeJSON;
    displayId?: string;
    onmeasure?: (height: number) => void;
    tokenStart?: number;
    tokenEnd?: number;
    sharedCount?: number;
    sharedUsesActivePath?: boolean;
    /** Active path membership — bold + ring + accent. */
    onActivePath: boolean;
    /** Current roving-tab-stop node for sidebar keyboard navigation. */
    focused: boolean;
    /** Exact active node, distinct from active-path ancestry. */
    current: boolean;
    /** Multi-selection state used by branch comparison. */
    selected: boolean;
    /** One-based visual tree depth. */
    level: number;
    /** Whether this item owns visible child items. */
    hasChildren: boolean;
    /** Dead branch (not on active path) — retain readable text, quiet the chrome. */
    dead: boolean;
    /** In-flight target — pulse the node so the user sees streaming. */
    streaming: boolean;
    /** Click handler — navigate to this node. */
    onclick?: (ev: MouseEvent) => void;
    /** Focus handler keeps the sidebar's roving tab stop synchronized. */
    onfocus?: (ev: FocusEvent) => void;
    /** Tree keyboard handler supplied by the owning tree. */
    onkeydown?: (ev: KeyboardEvent) => void;
    /** Right-click handler — open the context menu. */
    oncontextmenu?: (ev: MouseEvent) => void;
    /** Visible action-menu trigger for pointer and touch users. */
    onactions?: (ev: MouseEvent) => void;
    /** Generate another continuation at this point. */
    ongrow?: (ev: MouseEvent) => void;
    /** Create an editable sibling path. */
    onbranch?: (ev: MouseEvent) => void;
    /** Select this exact model token and open its branch-point tools. */
    onselecttoken?: (tokenIndex: number, ev: MouseEvent) => void;
    /** Continue from the exact token at the end of a sentence as a sibling. */
    onbranchsentence?: (tokenIndex: number, ev: MouseEvent) => void;
    /** Runtime and node both support exact forced-token replay. */
    sentenceBranchAvailable?: boolean;
    /** Hide or reveal descendants in the canvas. */
    ontogglechildren?: (ev: MouseEvent) => void;
    /** Whether descendants are currently hidden in this view. */
    collapsed?: boolean;
    /** Optional probe ring fill in [-1, 1] (null when no probe aggregate). */
    ring?: number | null;
    /** Logit-pass: per-turn ``mean_logprob`` to render as a numeric
     *  badge.  Null (capture wasn't live) suppresses the badge. */
    weightBadge?: number | null;
    /** Steering-delta label for the edge into this node (e.g.
     *  ``0.45 angry.calm``).  Rendered as a trailing chip — it used to
   *  appears in the card so it stays readable at every zoom. */
    steerLabel?: string | null;
    /** How this sibling diverged from another reply. */
    forkLabel?: string | null;
  }

  let {
    node,
    displayId,
    onmeasure,
    tokenStart = 0,
    tokenEnd,
    sharedCount = 0,
    sharedUsesActivePath = false,
    onActivePath,
    focused,
    current,
    selected,
    level,
    hasChildren,
    dead,
    streaming,
    onclick,
    onfocus,
    onkeydown,
    oncontextmenu,
    onactions,
    ongrow,
    onbranch,
    onselecttoken,
    onbranchsentence,
    sentenceBranchAvailable = false,
    ontogglechildren,
    collapsed = false,
    ring = null,
    weightBadge = null,
    steerLabel = null,
    forkLabel = null,
  }: Props = $props();

  function measureCard(element: HTMLElement) {
    let frame = 0;
    let lastHeight = 0;
    const measure = () => {
      frame = 0;
      if (!element.clientWidth) return;
      const style = getComputedStyle(element);
      const children = [...element.children].filter(child => getComputedStyle(child).display !== "none") as HTMLElement[];
      let height = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
        + Math.max(0, children.length - 1) * parseFloat(style.rowGap);
      for (const child of children) {
        if (child.classList.contains("token-field")) {
          const field = getComputedStyle(child);
          const sentences = [...child.children] as HTMLElement[];
          height += parseFloat(field.paddingTop) + parseFloat(field.paddingBottom)
            + Math.max(0, sentences.length - 1) * parseFloat(field.rowGap)
            + sentences.reduce((sum, sentence) => sum + sentence.offsetHeight, 0);
        } else height += child.offsetHeight;
      }
      height = Math.ceil(height);
      if (height !== lastHeight) {
        lastHeight = height;
        onmeasure?.(height);
      }
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(measure); };
    const resize = new ResizeObserver(schedule);
    const observe = () => {
      resize.disconnect();
      resize.observe(element);
      element.querySelectorAll<HTMLElement>(":scope > *, .sentence-node").forEach(child => resize.observe(child));
      schedule();
    };
    const mutations = new MutationObserver(observe);
    mutations.observe(element, { childList: true, subtree: true, characterData: true });
    observe();
    return { destroy() { resize.disconnect(); mutations.disconnect(); cancelAnimationFrame(frame); } };
  }

  const PREVIEW_CHARS = 600;
  const glow = new Spring({ x: 0, y: 0 }, { stiffness: 0.2, damping: 0.85, precision: 0.01 });

  function followPointer(event: PointerEvent): void {
    if (event.pointerType !== "mouse" || event.buttons || prefersReducedMotion.current ||
      !window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    glow.target = {
      x: Math.max(-1, Math.min(1, ((event.clientX - rect.left) / rect.width - 0.5) * 2)),
      y: Math.max(-1, Math.min(1, ((event.clientY - rect.top) / rect.height - 0.5) * 2)),
    };
  }

  function resetGlow(): void {
    glow.target = { x: 0, y: 0 };
  }

  $effect(() => {
    if (prefersReducedMotion.current) void glow.set({ x: 0, y: 0 }, { instant: true });
  });
  onDestroy(() => { void glow.set({ x: 0, y: 0 }, { instant: true }); });

  // Glyph honors the node's per-turn role label (e.g. ``captain`` → ``C``);
  // default roles reduce to ``U`` / ``A`` / ``S``.
  function roleGlyph(node: LoomNodeJSON): string {
    return roleGlyphLetter(node.role, node.role_label);
  }

  const preview = $derived.by(() => {
    const t = (node.text ?? "").replace(/\s+/g, " ").trim();
    if (!t) return node.role === "system" && !node.parent_id ? "root" : "(empty)";
    return t.length > PREVIEW_CHARS ? t.slice(0, PREVIEW_CHARS) + "…" : t;
  });

  const roleName = $derived(node.role_label?.trim() || node.role);
  const visibleTokens = $derived(node.tokens?.slice(tokenStart, tokenEnd) ?? []);
  const sentences = $derived(splitTokenSentences(visibleTokens).map(sentence => ({
    ...sentence, start: sentence.start + tokenStart, end: sentence.end + tokenStart,
  })));

  function tokenName(text: string): string {
    const readable = text.replace(/\n/g, "↵").replace(/\t/g, "⇥").trim();
    return readable || "whitespace";
  }

  function tokenTitle(index: number, text: string, logprob: number | null): string {
    if (typeof logprob !== "number" || !Number.isFinite(logprob)) {
      return `Token ${index + 1}: ${tokenName(text)}`;
    }
    const probability = Math.min(1, Math.max(0, Math.exp(logprob)));
    return `Token ${index + 1}: ${tokenName(text)} · ${(probability * 100).toFixed(1)}%`;
  }

  // Ring color: caller passes ``ring`` in [-1,1] — negative red, positive
  // green; null when there's no probe aggregate to show.
  const ringColor = $derived(
    ring === null ? null : ring >= 0 ? "var(--accent-green)" : "var(--accent-red)",
  );
</script>

<div
  use:measureCard
  class="node"
  class:shared={sharedCount > 0}
  class:active={onActivePath}
  class:focused
  class:dead
  class:streaming
  class:generation-active={streaming}
  class:starred={!sharedCount && node.starred}
  class:user={node.role === "user"}
  class:assistant={node.role === "assistant"}
  class:system={node.role === "system"}
  role="treeitem"
  aria-label={sharedCount ? `Shared prefix across ${sharedCount} replies` : undefined}
  aria-level={level}
  aria-expanded={!sharedCount && hasChildren ? !collapsed : undefined}
  aria-current={current ? "true" : undefined}
  aria-selected={sharedCount ? undefined : selected}
  tabindex={focused ? 0 : -1}
  data-node-id={displayId ?? node.id}
  data-cursor={oncontextmenu ? "context-menu" : undefined}
  style:--glow-x={`${glow.current.x * 14}px`}
  style:--glow-y={`${glow.current.y * 3}px`}
  style:--glow-angle={`${glow.current.x * 4}deg`}
  onpointerenter={followPointer}
  onpointermove={followPointer}
  onpointerleave={resetGlow}
  onpointercancel={resetGlow}
  onpointerdown={resetGlow}
  {onclick}
  {onfocus}
  {onkeydown}
  {oncontextmenu}
>
  <span class="node-head">
    <span class="identity">
      <span class="glyph" aria-hidden="true">{roleGlyph(node)}</span>
      <span class="role-name">{sharedCount ? "shared prefix" : roleName}</span>
      {#if ringColor}
        <span class="ring" style="border-color: {ringColor}" aria-hidden="true"></span>
      {/if}
    </span>
    <span class="head-meta">
      {#if streaming}<span class="generation-label" role="status">Writing…</span>{/if}
      {#if sharedCount}<span class="current-label">{sharedCount} paths</span>{/if}
      {#if current}<span class="current-label">current</span>{/if}
      {#if !sharedCount}
      {#if node.starred}<span class="star" title="Saved branch" aria-hidden="true"><FluentIcon name="star" /></span>{/if}
      <button
        type="button"
        class="node-actions"
        aria-label={`Actions for ${roleName}: ${preview}`}
        title="Message actions"
        onclick={(ev) => {
          ev.stopPropagation();
          onactions?.(ev);
        }}
      >•••</button>
      {/if}
    </span>
  </span>
  {#if node.tokens && visibleTokens.length > 0}
    <div class="token-field" aria-label={`${sentences.length} sentence branch points`}>
      {#each sentences as sentence, sentenceIndex (`${sentence.start}:${sentence.end}`)}
        {@const boundary = node.tokens[sentence.end]}
        {@const canBranch = sentenceBranchAvailable &&
          boundary?.raw_index != null && boundary?.token_id != null}
        <div
          class="sentence-node"
          data-loom-sentence={sentenceIndex}
          data-sentence-start={sentence.start}
          data-sentence-end={sentence.end}
        >
          <div class="sentence-tokens">
            {#each node.tokens.slice(sentence.start, sentence.end + 1) as token, offset (`${token.raw_index ?? sentence.start + offset}:${token.token_id ?? token.text}`)}
              {@const tokenIndex = sentence.start + offset}
              <button
                type="button"
                class="token-node"
                data-cursor="inspect"
                style={sharedCount ? "" : highlightStyleString(tokenRowToScore(token))}
                data-loom-token-node={node.id}
                      data-token-index={tokenIndex}
                data-raw-index={token.raw_index ?? undefined}
                title={`${tokenTitle(tokenIndex, token.text, sharedCount ? null : token.logprob)} · Open branch-point tools`}
                aria-label={`Open token ${tokenIndex + 1}: ${tokenName(token.text)}`}
                aria-haspopup="dialog"
                onclick={(ev) => {
                  ev.stopPropagation();
                  onselecttoken?.(tokenIndex, ev);
                }}
              >{token.text || "∅"}</button>
            {/each}
          </div>
          <button
            type="button"
            class="sentence-branch"
            disabled={!canBranch}
            title={canBranch
              ? "Keep this sentence and generate a different continuation"
              : "This saved reply does not have an exact replay boundary"}
            aria-label={`Branch after sentence ${sentenceIndex + 1}: ${sentence.text.trim()}`}
            onclick={(ev) => {
              ev.stopPropagation();
              if (canBranch) onbranchsentence?.(sentence.end, ev);
            }}
          ><span class="branch-dot" aria-hidden="true"></span>Branch here</button>
        </div>
      {/each}
    </div>
  {:else}
    <span class="preview">{streaming ? "Preparing continuation…" : tokenStart > 0 ? "End of reply" : preview}</span>
  {/if}
  {#if sharedCount}
    <span class="shared-hint">Token tools use {sharedUsesActivePath ? "the current" : "the first listed"} path’s readings.</span>
  {/if}
  {#if !sharedCount && (forkLabel || steerLabel || weightBadge != null || node.notes)}
    <span class="node-meta">
      {#if forkLabel}<span class="fork" title="Where this reply diverged">{forkLabel}</span>{/if}
      {#if steerLabel}<span class="steer" title="Guidance changed on this branch">{steerLabel}</span>{/if}
      {#if weightBadge != null}
        <span class="weight" title="Average token log probability">
          <span aria-hidden="true">log p</span> {weightBadge.toFixed(2)}
        </span>
      {/if}
      {#if node.notes}<span class="note-mark" title={node.notes} aria-label="Has a note">●</span>{/if}
    </span>
  {/if}
  {#if !sharedCount}
  <span class="node-tools" aria-label="Branch actions">
    <button
      type="button"
      title="Generate another path from here"
      aria-label={`Generate another path from ${roleName}`}
      onclick={(ev) => { ev.stopPropagation(); ongrow?.(ev); }}
    >Generate</button>
    <button
      type="button"
      title="Write an alternative at this point"
      aria-label={`Write an alternative to ${roleName}`}
      onclick={(ev) => { ev.stopPropagation(); onbranch?.(ev); }}
    >Write</button>
    {#if hasChildren}
      <button
        type="button"
        title={collapsed ? "Show paths after this point" : "Hide paths after this point"}
        aria-label={collapsed ? "Show child paths" : "Hide child paths"}
        aria-pressed={collapsed}
        onclick={(ev) => { ev.stopPropagation(); ontogglechildren?.(ev); }}
      >{collapsed ? "Show" : "Hide"}</button>
    {/if}
  </span>
  {/if}
</div>

<style>
  .node {
    --node-outline: var(--glass-line);
    position: relative;
    isolation: isolate;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    width: 100%;
    height: 100%;
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    line-height: 1.3;
    color: var(--fg-strong);
    background:
      linear-gradient(
        145deg,
        color-mix(in srgb, var(--fg) 3%, var(--bg-elev)) 0%,
        color-mix(in srgb, var(--bg-elev) 96%, transparent) 54%,
        color-mix(in srgb, var(--bg) 18%, var(--bg-elev)) 100%
      );
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 6%, transparent),
      inset 0 0 0 1px var(--loom-node-outline, var(--node-outline)),
      var(--shadow-loom-node);
    transition:
      background var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
    min-width: 0;
    min-height: var(--control-target);
    overflow: hidden;
    user-select: none;
  }
  .node.shared { cursor: default; height: auto; max-height: 100%; }
  .shared .token-field { flex-grow: 0; }
  .node > * { flex-shrink: 0; }
  .shared .role-name { max-width: none; }
  .shared-hint { color: var(--fg-muted); font-size: var(--text-xs); line-height: 1.4; }
  .node::after {
    content: "";
    position: absolute;
    inset: 1px 1px auto;
    height: 40%;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    background: radial-gradient(ellipse at 50% 0%, color-mix(in srgb, var(--accent) 9%, transparent), transparent 75%);
    transform: translate(var(--glow-x, 0px), var(--glow-y, 0px)) rotate(var(--glow-angle, 0deg));
    transform-origin: 50% 0;
    pointer-events: none;
  }
  .node:hover {
    background: var(--glass-strong);
    transform: translateY(-2px);
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 8%, transparent),
      inset 0 0 0 1px var(--loom-node-outline, color-mix(in srgb, var(--fg-muted) 54%, var(--glass-line))),
      var(--shadow-loom-node-hover);
  }
  .node.focused {
    --node-outline: var(--focus-ring);
  }
  .node:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -1px;
  }
  .node.active {
    font-weight: var(--weight-medium);
    color: var(--fg);
    background: color-mix(in srgb, var(--accent-subtle) 46%, var(--bg-elev));
    --node-outline: color-mix(in srgb, var(--accent) 42%, var(--glass-line));
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 8%, transparent),
      inset 0 0 0 1px var(--loom-node-outline, var(--node-outline)),
      var(--shadow-loom-node-active);
  }
  .node[aria-current="true"] {
    transform: translateY(-3px);
    background:
      linear-gradient(
        145deg,
        color-mix(in srgb, var(--accent) 13%, var(--bg-elev)) 0%,
        color-mix(in srgb, var(--accent-subtle) 44%, var(--bg-elev)) 58%,
        color-mix(in srgb, var(--bg) 22%, var(--bg-elev)) 100%
      );
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 11%, transparent),
      inset 0 0 0 1px var(--loom-node-outline, color-mix(in srgb, var(--accent) 62%, var(--glass-line))),
      var(--shadow-loom-node-current),
      0 0 30px color-mix(in srgb, var(--accent) 9%, transparent);
  }
  .node.dead {
    color: var(--fg-muted);
    background: color-mix(in srgb, var(--bg-elev) 58%, transparent);
    --node-outline: color-mix(in srgb, var(--glass-line) 62%, transparent);
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 3%, transparent),
      inset 0 0 0 1px var(--loom-node-outline, var(--node-outline)),
      var(--shadow-loom-node-muted);
  }
  .node.dead:hover {
    color: var(--fg);
    background: var(--bg-elev);
  }
  .node.dead .glyph,
  .node.dead .ring {
    opacity: 0.55;
  }
  .node.streaming {
    background: color-mix(in srgb, var(--live) 8%, transparent);
  }
  .node-head,
  .identity,
  .head-meta,
  .node-meta {
    display: flex;
    align-items: center;
  }
  .node-tools {
    display: flex;
    align-items: center;
    gap: var(--space-1);
    flex: none;
    margin-top: auto;
    max-width: 100%;
    overflow: hidden;
    opacity: 0;
    transform: translateY(2px);
    transition:
      opacity var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .node:hover .node-tools,
  .node:focus-within .node-tools,
  .node.focused .node-tools,
  .node[aria-current="true"] .node-tools {
    opacity: 1;
    transform: translateY(0);
  }
  .node-tools button {
    min-height: var(--control-target);
    min-width: var(--control-target);
    padding: 0 var(--space-2);
    border: 0;
    border-radius: var(--radius-sm);
    background: var(--glass);
    color: var(--fg-dim);
    cursor: pointer;
    font-family: var(--font-structure);
    font-size: var(--text-2xs);
    font-weight: var(--weight-medium);
    transition: background var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), transform var(--dur-fast) var(--ease-out);
  }
  .node-tools button:hover,
  .node-tools button:focus-visible {
    background: var(--glass-bright);
    color: var(--fg);
    outline: none;
  }
  .node-tools button:active {
    transform: scale(var(--press-scale));
  }
  .node-head {
    flex: none;
    justify-content: space-between;
    gap: var(--space-3);
    min-width: 0;
    max-width: 100%;
  }
  .identity,
  .head-meta,
  .node-meta {
    gap: var(--space-2);
  }
  .role-name,
  .current-label,
  .fork {
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .role-name {
    max-width: 12ch;
    overflow: hidden;
    color: var(--fg-dim);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .current-label {
    padding: var(--space-xs) var(--space-xs);
    border-radius: var(--radius-pill);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }
  .node-actions {
    display: inline-grid;
    width: var(--control-target);
    height: var(--control-target);
    padding: 0;
    place-items: center;
    border: 0;
    border-radius: var(--radius-pill);
    background: transparent;
    color: var(--fg-muted);
    cursor: pointer;
    font: inherit;
    font-size: var(--text-xs);
    letter-spacing: -0.12em;
    transition: background var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out);
  }
  .node-actions:active { scale: var(--press-scale); }
  .node-actions:hover,
  .node-actions:focus-visible {
    background: var(--glass-bright);
    color: var(--fg);
    outline: none;
  }
  /* No stripes at all (cast model: roles carry no hue — identity is the
     glyph letter alone; the active path reads from the glass fill +
     weight, and streaming state from its green fill). */
  .glyph {
    font-weight: var(--weight-bold);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--glass-bright);
    color: var(--fg);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: var(--text-glyph-sm);
    flex: none;
    text-transform: uppercase;
    box-shadow:
      inset 0 1px 0 color-mix(in srgb, var(--fg) 10%, transparent),
      var(--shadow-loom-glyph);
  }
  .preview {
    display: -webkit-box;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 6;
    line-clamp: 6;
    min-width: 0;
    color: inherit;
    font-size: var(--text-xs);
    line-height: 1.5;
  }
  .token-field {
    display: flex;
    flex-direction: column;
    flex: none;
    min-width: 0;
    min-height: 0;
    padding: var(--space-2);
    align-items: stretch;
    gap: var(--space-2);
    overflow: visible;
  }
  .sentence-node {
    position: relative;
    display: flex;
    flex: none;
    flex-direction: column;
    gap: var(--space-xs);
    min-width: 0;
    padding: var(--space-xs) var(--space-xs);
    border: 1px solid color-mix(in srgb, var(--accent) 34%, var(--glass-line));
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--glass) 72%, transparent);
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sentence-node:hover,
  .sentence-node:focus-within {
    background: color-mix(in srgb, var(--accent) 8%, var(--glass));
    border-color: color-mix(in srgb, var(--accent) 78%, var(--glass-line));
  }
  .sentence-tokens {
    display: flex;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: var(--space-xs);
    min-width: 0;
  }
  .sentence-branch {
    display: inline-flex;
    align-items: center;
    align-self: flex-end;
    gap: var(--space-xs);
    min-height: var(--control-target);
    min-width: var(--control-target);
    padding: var(--space-xs) var(--space-xs);
    border: 0;
    border-radius: var(--radius-sm);
    color: var(--fg-dim);
    background: transparent;
    cursor: pointer;
    font-family: var(--font-structure);
    font-size: var(--text-2xs);
    font-weight: var(--weight-medium);
    opacity: 0.72;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      opacity var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .sentence-node:hover .sentence-branch,
  .sentence-branch:focus-visible {
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    opacity: 1;
    outline: none;
  }
  .sentence-branch:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }
  .sentence-branch:disabled {
    display: none;
  }
  .branch-dot {
    width: 6px;
    height: 6px;
    border: 1px solid currentColor;
    border-radius: 50%;
    background: color-mix(in srgb, currentColor 24%, transparent);
  }
  .token-node {
    box-sizing: border-box;
    min-width: 8px;
    min-height: 23px;
    max-width: 100%;
    padding: var(--space-xs) var(--space-xs);
    border: 1px solid color-mix(in srgb, var(--accent) 18%, var(--glass-line));
    border-radius: var(--radius-sm);
    color: inherit;
    background: var(--bg-alt);
    cursor: pointer;
    font: inherit;
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    line-height: 1.35;
    text-align: start;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
    transition:
      border-color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .token-node:hover,
  .token-node:focus-visible {
    border-color: color-mix(in srgb, var(--accent) 68%, var(--glass-line));
    background: color-mix(in srgb, var(--accent) 24%, var(--bg-elev));
    outline: none;
    transform: translateY(-1px);
  }
  .token-node:focus-visible {
    box-shadow: 0 0 0 2px var(--focus-ring);
  }
  .token-node:active {
    transform: scale(var(--press-scale));
  }
  .star {
    color: var(--fg-dim);
    font-size: var(--text-xs);
  }
  /* Steering-delta chip — trailing, truncated so a long delta can't
   * blow out the row or collide with the preview text. LoomSidebar fetches
   * and caches the edge label; this node renders it as a compact data chip. */
  .steer {
    color: var(--fg-dim);
    font-family: var(--font-data);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
    padding: var(--space-xs) var(--space-xs);
    border-radius: var(--radius-sm);
    background: var(--glass-strong);
    max-width: 11ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .node-meta {
    flex: none;
    min-width: 0;
    overflow: hidden;
  }
  .fork {
    max-width: 17ch;
    overflow: hidden;
    color: var(--fg-dim);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .note-mark {
    color: var(--fg-dim);
    font-size: var(--text-2xs);
    line-height: 1;
  }
  /* Phase-5 hook: a thin colored ring around the role glyph keyed off
   * the highlight-probe's per-node aggregate reading.  Renders only
   * when ``ring`` prop is non-null. */
  .ring {
    width: 0.9em;
    height: 0.9em;
    border-radius: 50%;
    border: 1px solid transparent;
    display: inline-block;
  }
  /* Logit-pass: numeric ``mean_logprob`` badge.  Tabular-nums so the
     digits line up across siblings even when sort:surprise reorders
     them; subdued color so the badge reads as metadata, not content.
     Like .steer, no own background — inherits the row's highlight. */
  .weight {
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    padding: 0 var(--space-2);
  }
  @media (prefers-reduced-motion: reduce) {
    .node::after {
      transform: none;
    }
    .node,
    .node-tools,
    .token-node,
    .sentence-node,
    .sentence-branch {
      transition: none;
    }
  }
  @media (pointer: coarse) {
    .node-actions {
      width: var(--control-target);
      height: var(--control-target);
    }
    .node-tools {
      overflow: visible;
      opacity: 1;
      transform: none;
    }
    .node-tools button,
    .sentence-branch,
    .token-node {
      min-height: var(--control-target);
    }
    .node-tools button {
      flex: 1 1 auto;
      padding-inline: var(--space-3);
    }
    .sentence-branch {
      padding-inline: var(--space-3);
    }
    .token-node {
      min-width: 28px;
      padding: var(--space-2);
    }
  }
</style>
