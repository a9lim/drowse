<script lang="ts">
  import EmptyState from "../../drawers/token/EmptyState.svelte";
  import InstrumentPackRecovery from "./InstrumentPackRecovery.svelte";

  let { modelId: _modelId }: { modelId: string | null } = $props();
</script>

<EmptyState title="Word insights are unavailable">
  <InstrumentPackRecovery kind="jlens" />
</EmptyState>
