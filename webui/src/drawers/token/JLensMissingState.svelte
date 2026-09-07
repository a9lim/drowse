<script lang="ts">
  import EmptyState from "./EmptyState.svelte";

  let { modelId }: { modelId: string | null } = $props();
</script>

<EmptyState title="no J-LENS fit">
  <code>drowse lens fit {modelId ?? "<model>"}</code>
</EmptyState>
