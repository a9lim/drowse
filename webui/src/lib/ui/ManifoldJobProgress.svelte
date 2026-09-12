<script lang="ts">
  import { elapsedLabel, etaLabel, remainingWorkMs } from "../manifoldProgress";
  import { manifoldJobs, dismissManifoldJob, cancelManifoldJob } from "../stores/manifoldJobs.svelte";
  import { openDrawer } from "../stores/drawers.svelte";
  import { pushToast } from "../stores/toasts.svelte";
  import { describeError } from "../runtime/errors";
  import { steerRack } from "../stores/steering.svelte";
  import Button from "./Button.svelte";

  let { compact = false }: { compact?: boolean } = $props();
  let now = $state(Date.now());
  const job = $derived(manifoldJobs.current);
  const busy = $derived(job !== null && ["running", "waiting", "cancelling"].includes(job.status));
  const fraction = $derived(job?.progress.completed !== null && job?.progress.total
    ? job.progress.completed! / job.progress.total : null);
  const remaining = $derived(job && busy ? remainingWorkMs(job.samples, job.progress.total, now) : null);
  const title = $derived(job?.status === "failed" ? "Creation needs attention"
    : job?.status === "cancelled" ? "Creation cancelled"
    : job?.status === "cancelling" ? "Cancelling…" : job?.progress.label ?? "");
  const timing = $derived(job?.status === "waiting" ? "Fitting starts next"
    : job?.status === "cancelling" ? "Waiting for the current step to stop"
    : !busy ? "" : remaining === null ? "Estimating this step…" : `${etaLabel(remaining)} in this step`);
  const announcement = $derived(job ? `${job.namespace}/${job.name}: ${title}${busy && fraction !== null ? `, ${Math.floor(fraction * 10) * 10}% of this step` : ""}` : "");

  $effect(() => {
    if (!busy) return;
    now = Date.now();
    const timer = setInterval(() => { now = Date.now(); }, 1000);
    return () => clearInterval(timer);
  });

  async function cancel(): Promise<void> {
    try { await cancelManifoldJob(); }
    catch (error) {
      pushToast(`Could not cancel: ${describeError(error)}`, { kind: "error" });
    }
  }

  function openResult(): void {
    if (!job) return;
    if (job.phase === "scoring") { openDrawer("template_lab"); return; }
    const item = steerRack.catalog.find(row => row.namespace === job.namespace && row.name === job.name);
    const fit = item?.resolved_fit_mode ?? item?.fit_mode;
    openDrawer(fit === "spectral" || fit === "authored" ? "manifolds" : "subspace");
  }
</script>

<p class="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcement}</p>
{#if job}
  <section class="job-progress" class:compact aria-label={`Creation progress for ${job.namespace}/${job.name}`}>
    <div class="job-heading">
      <div class="job-copy"><strong>{title}</strong><span class="identifier">{job.namespace}/{job.name}</span></div>
      <span class="elapsed">{elapsedLabel((job.finishedAt ?? now) - job.startedAt)} elapsed</span>
    </div>
    {#if busy || job.status === "complete"}
      <div class="meter" role="progressbar" aria-label={job.phase === "generating" ? "Examples generated" : job.progress.label}
        aria-valuemin="0" aria-valuemax="100" aria-valuenow={fraction === null ? undefined : Math.floor(fraction * 100)}
        aria-valuetext={fraction === null ? title : `${job.progress.completed} of ${job.progress.total} completed in this step`}>
        <span class="meter-fill" class:unknown={fraction === null} style:transform={`scaleX(${fraction ?? 0})`}></span>
      </div>
    {/if}
    <div class="job-meta">
      <span>{job.progress.completed !== null && job.progress.total !== null ? `${job.progress.completed} / ${job.progress.total} ${job.phase === "generating" ? "examples" : "completed"}` : job.progress.detail}</span>
      <span>{timing}</span>
    </div>
    {#if job.error}<p class="job-error">{job.error}</p>{/if}
    <div class="job-footer">
      <p>{busy ? job.phase === "generating" && job.fitAfterwards ? "Step 1 of 2 · Fitting follows. Keep this tab open; you can use other tools." : job.fitAfterwards ? "Step 2 of 2 · Keep this tab open. You can use other tools." : "Keep this tab open. You can move between tools while this runs."
        : job.status === "complete" ? job.phase === "generating" ? "Examples saved. Fit them from the concept library when ready." : job.phase === "scoring" ? "Choice scoring is complete. Inspect the returned sum and mean probabilities." : job.phase === "installing" ? "Artifact installed. Inspect its model fits before steering or probing." : "Saved and ready to steer or probe."
        : "Saved concept groups are kept. Reopen the creator or library to try again."}</p>
      <div class="job-actions">
        {#if busy && job.cancellable}
          <Button size="sm" disabled={job.status !== "running"} onclick={() => void cancel()}>Cancel</Button>
        {:else if !busy}
          <Button size="sm" onclick={openResult}>Open library</Button>
          <Button size="sm" variant="flat" onclick={dismissManifoldJob}>Dismiss</Button>
        {/if}
      </div>
    </div>
  </section>
{/if}

<style>
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .job-progress { display: grid; gap: var(--space-sm); padding: var(--surface-padding); background: var(--workspace-field-bg); color: var(--fg); font-size: var(--text-sm); border-radius: var(--popup-radius); min-width: 0; }
  .job-heading, .job-meta, .job-footer { display: flex; align-items: center; justify-content: space-between; gap: var(--space-sm); flex-wrap: wrap; }
  .job-copy { display: grid; gap: var(--space-xs); min-width: 0; }
  .job-copy strong { text-wrap: balance; }
  .identifier { font-family: var(--font-mono); overflow-wrap: anywhere; color: var(--fg-dim); }
  .elapsed, .job-meta { font-variant-numeric: tabular-nums; color: var(--fg-dim); }
  .elapsed { white-space: nowrap; }
  .job-meta { min-height: 1.4em; }
  .meter { overflow: hidden; block-size: 6px; border-radius: var(--radius); background: var(--bg-hover); }
  .meter-fill { display: block; block-size: 100%; background: var(--fg); transform-origin: left; }
  .meter-fill.unknown { transform: translateX(40%) scaleX(0.2) !important; opacity: 0.5; }
  .job-footer p { margin: 0; flex: 1 1 16rem; color: var(--fg-dim); line-height: 1.5; text-wrap: pretty; }
  .job-actions { display: flex; gap: var(--space-sm); flex-wrap: wrap; }
  .job-error { margin: 0; color: var(--accent-red); overflow-wrap: anywhere; line-height: 1.5; text-wrap: pretty; }
  .compact { border-radius: 0; }
  @media (prefers-reduced-motion: no-preference) { .meter-fill { transition: transform 600ms var(--ease-out); } }
  @media (forced-colors: active) { .meter { border: 1px solid CanvasText; } .meter-fill { background: Highlight; } }
</style>
