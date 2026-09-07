<script lang="ts">
  // Builder-drawer validation summary.  Callers reveal it after a submit
  // attempt so editing a field does not repeatedly interrupt screen-reader
  // users with a changing global alert.
  //
  // Renders nothing when ``messages`` is empty so callers can drop it in
  // unconditionally.

  interface Props {
    /** What action would run if validation passed — completes the
     *  "not ready to ____:" header sentence.  Examples: "extract",
     *  "build", "discover". */
    verb: string;
    messages: string[];
  }

  let { verb, messages }: Props = $props();
  const uid = $props.id();
</script>

{#if messages.length > 0}
  <section class="sk-validation" aria-labelledby={`${uid}-heading`}>
    <p id={`${uid}-heading`} class="sk-validation-head">
      Check these details before you {verb}
    </p>
    <ul>
      {#each messages as m (m)}
        <li>{m}</li>
      {/each}
    </ul>
  </section>
{/if}

<style>
  .sk-validation {
    background: var(--surface-sheen), var(--warn-bg);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    color: var(--fg-strong);
    font-size: var(--text-sm);
  }
  .sk-validation-head {
    margin: 0 0 var(--space-2);
    color: var(--accent-yellow);
    font-weight: var(--weight-medium);
  }
  .sk-validation ul {
    margin: 0;
    padding-inline-start: var(--space-6);
    color: var(--fg-dim);
  }
  .sk-validation li {
    line-height: 1.4;
  }
</style>
