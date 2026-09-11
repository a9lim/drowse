<script lang="ts">
  let example = $state(import("./RecordedDemo.svelte"));
</script>

{#await example}
  <p role="status">Loading the recorded example…</p>
{:then { default: Example }}
  <Example />
{:catch}
  <p>The recorded example could not load.</p>
  <button type="button" onclick={() => example = import("./RecordedDemo.svelte")}>Try again</button>
{/await}
