<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { closeDrawer, openDrawer } from "../lib/stores.svelte";
  import { apiManifolds } from "../lib/runtime/services";
  import { runtimeClient } from "../lib/runtime/client";
  import { describeError } from "../lib/runtime/errors";
  import { exampleSphere, parseSurfacePoints, surfaceLabel } from "../lib/manifolds/surfaceGeometry";
  import type { SurfaceEvidence } from "../lib/types.gen";

  let { params: _params }: { params?: unknown } = $props();
  const available = runtimeClient.mode === "http" && Boolean(apiManifolds.inspectSurface);
  let input = $state("");
  let busy = $state(false);
  let error = $state("");
  let result = $state<SurfaceEvidence | null>(null);
  let field = $state<HTMLTextAreaElement>();
  let resultHeading = $state<HTMLHeadingElement>();
  let revision = 0;
  onDestroy(() => { revision++; });

  function changed(): void { revision++; result = null; error = ""; }
  async function inspect(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    if (busy || !available || !apiManifolds.inspectSurface) return;
    result = null;
    error = "";
    let points: number[][];
    try { points = parseSurfacePoints(input); }
    catch (cause) { error = (cause as Error).message; await tick(); field?.focus(); return; }
    const submittedRevision = revision;
    busy = true;
    try {
      const evidence = await apiManifolds.inspectSurface(points);
      if (revision !== submittedRevision) return;
      result = evidence;
      await tick();
      resultHeading?.focus();
      resultHeading?.scrollIntoView({ block: "start" });
    } catch (cause) { if (revision === submittedRevision) error = describeError(cause); }
    finally { busy = false; }
  }
</script>

<section class="drawer-shell" aria-label="Surface geometry">
  <header class="header">
    <h2>Surface geometry</h2>
    <DrawerCloseButton onclick={closeDrawer} />
  </header>
  <div class="body">
    <p>Inspect a sampled surface, then author coordinates separately to steer on it.</p>
    <div class="surface-card">
      <h3>What can be identified?</h3>
      <p>Spheres, tori, Klein bottles, and real projective planes. Sparse, noisy, or very thin surfaces may remain unresolved.</p>
      <p>A match is evidence of topology, not proof that the samples form a closed surface. It does not recover a steering chart.</p>
    </div>
    {#if available}
      <form onsubmit={inspect}>
        <label for="surface-points">Point coordinates</label>
        <p id="surface-input-help">Paste a JSON array with 32–384 distinct points in 2–1,024 dimensions. Use synthetic Euclidean coordinates; model activations require whitening first.</p>
        <textarea id="surface-points" bind:this={field} bind:value={input} oninput={changed}
          readonly={busy} spellcheck="false" rows="6" aria-invalid={Boolean(error)}
          aria-describedby={`surface-input-help${error ? " surface-error" : ""}`}></textarea>
        <div class="actions">
          <Button disabled={busy} onclick={() => { input = JSON.stringify(exampleSphere()); changed(); }}>Use sphere example</Button>
          <Button variant="solid" type="submit" disabled={busy}>{busy ? "Inspecting…" : "Inspect surface"}</Button>
        </div>
        <p role="status">{busy ? "Computing surface evidence locally. This can take a minute; generation is unchanged." : ""}</p>
        {#if error}<p id="surface-error" role="alert">{error}</p>{/if}
      </form>
      {#if result}
        <section class="surface-card" aria-labelledby="surface-result">
          <p class="eyebrow">{result.candidate ? "Topology suggestion" : "No reliable match"}</p>
          <h3 id="surface-result" tabindex="-1" bind:this={resultHeading}>{surfaceLabel(result.candidate)}</h3>
          <p>{result.reason}</p>
          <p>{result.sample_count} points inspected · No steering coordinates recovered</p>
          <details>
            <summary>Evidence details</summary>
            <p>Homology over two coefficient fields helps distinguish orientable surfaces from non-orientable ones.</p>
            <dl>
              <dt>Betti numbers (mod 2)</dt><dd>{result.betti_mod2?.join(", ") ?? "Not resolved"}</dd>
              <dt>Betti numbers (mod 3)</dt><dd>{result.betti_mod3?.join(", ") ?? "Not resolved"}</dd>
              <dt>Relative persistence</dt><dd>{result.relative_persistence?.toFixed(3) ?? "Not available"} · not a confidence probability</dd>
            </dl>
          </details>
        </section>
      {/if}
      <Button onclick={() => openDrawer("manifold_builder")}>Create a manifold with coordinates</Button>
    {:else}
      <div class="surface-card">
        <h3>Available in the native Python app</h3>
        <p>This browser release cannot inspect topology or steer on Klein bottles and RP² yet. Open Drowse’s native dashboard to use these tools.</p>
        <p>Surface inspection also needs the <code>drowse[topology]</code> Python extra. No language model is needed for the inspection itself.</p>
      </div>
    {/if}
  </div>
</section>

<style>
  .drawer-shell { min-width: 0; }
  .header { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); padding: var(--surface-padding); }
  h2, h3, p { margin: 0; }
  h2 { font-size: var(--text-lg); }
  h3 { font-size: var(--text-md); color: var(--fg-strong); }
  .body { display: grid; gap: var(--space-4); padding: var(--surface-padding); color: var(--fg-muted); line-height: 1.6; }
  .surface-card { display: grid; gap: var(--space-4); padding: var(--surface-padding); border-radius: var(--radius); background: var(--bg-elev); }
  form { display: grid; gap: var(--space-4); }
  label { color: var(--fg-strong); font-weight: var(--weight-medium); }
  textarea { width: 100%; box-sizing: border-box; resize: vertical; min-height: 9rem; padding: var(--space-3); border-radius: var(--radius); background: var(--bg-deep); color: var(--fg); font: inherit; font-family: var(--font-mono); }
  .actions { display: flex; flex-wrap: wrap; gap: var(--space-4); }
  [role="alert"] { color: var(--accent-red); }
  .eyebrow { color: var(--accent); font-size: var(--text-xs); font-weight: var(--weight-medium); }
  summary { cursor: pointer; color: var(--fg-strong); }
  dl { display: grid; gap: var(--space-2); }
  dd { margin: 0 0 var(--space-2); overflow-wrap: anywhere; }
</style>
