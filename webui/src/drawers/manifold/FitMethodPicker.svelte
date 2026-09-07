<script lang="ts">
  // The flat / curved / auto choice, shared by every tab that routes
  // through a discover fit.  ``tuning`` is a $state object owned by the
  // form. The owner supplies the mutation callback so this child never
  // writes through an unbound state prop.

  import Radio from "../../lib/Radio.svelte";
  import type { DiscoverFitMode, DiscoverTuning } from "./shared";

  let {
    tuning,
    /** ``spectral``'s note differs where node counts are the practical
     *  constraint (the generated tab) from where they aren't. */
    spectralNote = "curved",
    linearOnly = false,
    onchange,
  }: {
    tuning: DiscoverTuning;
    spectralNote?: string;
    linearOnly?: boolean;
    onchange: (value: DiscoverFitMode) => void;
  } = $props();

  $effect(() => {
    if (linearOnly && tuning.fitMode !== "pca") onchange("pca");
  });
</script>

<section class="step">
  <h2 class="step-title">fit method</h2>
  {#if linearOnly}
    <p class="dim-note">linear PCA · automatic curved-shape detection is not included in the hosted release</p>
  {:else}
    <div class="radio-row" role="radiogroup" aria-label="Fit method">
      <Radio group={tuning.fitMode} value="auto" label="auto" {onchange} />
      <Radio group={tuning.fitMode} value="pca" label="pca" {onchange} />
      <Radio group={tuning.fitMode} value="spectral" label="spectral" {onchange} />
    </div>
  {/if}
  <p class="dim-note">
    {#if tuning.fitMode === "auto"}
      Suggest a flat or periodic layout per model. This does not recover Klein-bottle or RP² coordinates.
    {:else if tuning.fitMode === "pca"}
      flat
    {:else}
      {spectralNote}
    {/if}
  </p>
</section>
