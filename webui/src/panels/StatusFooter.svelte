<script lang="ts">
  import MorphText from "../lib/ui/MorphText.svelte";
  // Single-line generation footer:
  //   ● gen 47/512 [████░░░░░░] · 23 t/s · 2.1s · ppl 8.3
  //
  // Idle state collapses to "○ idle" before the first generation lands,
  // hiding stats until they have something real to report.

  import { savedConversationState } from "../lib/stores/savedConversations.svelte";
  import Bar from "../lib/charts/Bar.svelte";
  import InfoTip from "../lib/ui/InfoTip.svelte";
  import {
    genStatus,
    geometricMeanPpl,
    pendingActions,
  } from "../lib/stores.svelte";

  let { savedEdit = false }: { savedEdit?: boolean } = $props();

  // Pending-queue badge — counts the items waiting in the FIFO queue.
  // Under the v2.x queue semantics, drain is automatic on every WS
  // ``done`` event; the per-bubble ``×`` in the chat-side
  // PendingBubbles strip handles cancellation, so there's no "apply
  // now" button here anymore.  The badge stays as a status readout.
  const pendingCount = $derived(pendingActions.queue.length);
  const pendingTitle = $derived(
    pendingCount === 1
      ? "1 queued"
      : `${pendingCount} queued`,
  );

  // Live elapsed counter — ticks while gen is active, freezes on done so
  // the user can still read the final timing after the generation lands.
  let speedView = $state(0);
  let nowMs = $state(performance.now());
  $effect(() => {
    if (!genStatus.active) return;
    const id = setInterval(() => {
      nowMs = performance.now();
      speedView = genStatus.tokPerSec;
    }, 200);
    return () => clearInterval(id);
  });

  const elapsedSec = $derived.by(() => {
    if (!genStatus.startedAt) return 0;
    const end = genStatus.active
      ? nowMs
      : genStatus.finishedAt ?? genStatus.startedAt;
    return Math.max(0, (end - genStatus.startedAt) / 1000);
  });

  const tokPerSec = $derived(genStatus.active ? speedView : genStatus.tokPerSec);
  const preparingContinuation = $derived(genStatus.active && genStatus.replay != null &&
    genStatus.replay.completed < genStatus.replay.total);

  const ppl = $derived(geometricMeanPpl(genStatus));

  // Have-anything: only render the full strip once a generation has at
  // least started.  "Active" is the obvious signal, but a finished gen
  // with startedAt set should also keep its trailing stats visible.
  const hasRun = $derived(genStatus.startedAt !== null);
  const finishLabel = $derived.by(() => {
    if (!genStatus.finishReason || genStatus.finishReason === "stop") return null;
    if (genStatus.finishReason === "length") return "Token limit";
    if (genStatus.finishReason === "cancelled") return "Stopped";
    return genStatus.finishReason;
  });
</script>

<footer class="status-footer" aria-label="Generation status">
  <div class="status-details">
    <span class="text lifecycle" class:done-label={!genStatus.active} role="status"><MorphText text={savedEdit ? "Edit saved" : !hasRun && !genStatus.active ? "Ready" : preparingContinuation ? "Preparing continuation…" : genStatus.active ? genStatus.replay ? "Continuing" : "Writing" : finishLabel ?? (genStatus.finishReason === "stop" ? "Complete" : "Ended")} numbers={false} /></span>
    {#if !savedEdit && (hasRun || genStatus.active)}
      <span class="text token-count"><MorphText text={genStatus.tokensSoFar} />{#if genStatus.active} / {genStatus.maxTokens || "?"}{/if} tokens</span>
      {#if genStatus.active}
        <span class="bar-wrap" aria-label={preparingContinuation ? "Preparing saved context" : "progress"}>
          <Bar value={preparingContinuation ? genStatus.replay!.completed : genStatus.tokensSoFar} max={preparingContinuation ? genStatus.replay!.total : genStatus.maxTokens || Math.max(genStatus.tokensSoFar, 1)} width={120} height={6} color="var(--accent)" />
        </span>
      {/if}
      {#if !preparingContinuation}<span class="text speed"><MorphText text={tokPerSec.toFixed(1)} /> tokens/s</span>{/if}
      <span class="text elapsed"><MorphText text={elapsedSec.toFixed(1)} />s</span>
      {#if ppl !== null && Number.isFinite(ppl)}<span class="text uncertainty" {...{ "aria-description": "entropy perplexity" }}>uncertainty <MorphText text={ppl.toFixed(2)} /></span>{/if}
    {/if}

    {#if pendingCount > 0}
      <span class="pending-badge" {...{ "aria-description": (pendingTitle) }}>
        <MorphText text={pendingTitle} />
      </span>
    {/if}
    {#if savedConversationState.status === "saving" || savedConversationState.status === "saved"}
      <span class="autosave-status" role="status"><MorphText text={savedConversationState.status === "saving" ? "Saving…" : "Saved"} numbers={false} /></span>
    {/if}
  </div>

  <InfoTip
    label="About generation status"
    text="Tokens are pieces of text, and speed is shown in tokens per second. Lower uncertainty means the model was more sure of its wording."
  />
</footer>

<style>
  .autosave-status { margin-inline-start: auto; color: var(--fg-muted); font: var(--text-2xs)/1.5 var(--font-ui); min-width: 6ch; }
  /* Embedded in the chat column, directly above the input row — a thin
   * status line.  Horizontal padding is zero so it aligns with the log
   * and input box; borderless — the gap above already separates it from
   * the log. */
  .status-footer {
    display: flex;
    flex: 0 0 auto;
    align-items: center;
    gap: var(--space-2);
    padding: var(--workspace-status-padding, var(--space-2)) 0;
    color: var(--fg-dim);
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    min-height: var(--workspace-status-height, calc(var(--control-target) + var(--space-2) * 2));
  }
  .status-details {
    display: flex;
    flex: 1 1 auto;
    flex-wrap: wrap;
    min-width: 0;
    align-items: center;
    gap: var(--space-2) var(--space-4);
  }
  .lifecycle { min-width: 8ch; }
  .token-count { min-width: 13ch; }
  .speed { min-width: 13ch; }
  .elapsed { min-width: 6ch; }
  .text { white-space: nowrap; }
  .done-label {
    color: var(--fg-strong);
  }
  .bar-wrap {
    display: inline-flex;
    align-items: center;
  }

  /* Pending-queue badge — status readout pushed to the right edge.
   * Display-only; per-item cancel lives on the PendingBubbles strip
   * above the composer. */
  .pending-badge {
    margin-inline-start: auto;
    background: var(--glass-strong);
    color: var(--fg-dim);
    border: 1px solid transparent;
    padding: var(--space-1) var(--space-4);
    border-radius: var(--radius-pill);
    font-size: var(--text-sm);
    font-family: var(--font-ui);
  }

  .status-footer :global(.info-tip) {
    flex: 0 0 auto;
  }

  @media (max-width: 480px) {
    .status-details {
      gap: var(--space-2);
    }
    .bar-wrap,
    .elapsed,
    .uncertainty {
      display: none;
    }
  }
</style>
