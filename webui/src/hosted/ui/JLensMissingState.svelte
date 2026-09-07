<script lang="ts">
  import EmptyState from "../../drawers/token/EmptyState.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";

  let { modelId: _modelId }: { modelId: string | null } = $props();
</script>

<EmptyState
  title="Word insights are unavailable"
  detail={sessionState.info?.is_base_model
    ? "Word insights are optional for base models. This view needs a compatible J-lens pack; text completion works without it."
    : "Word insights are missing. Switch models, then download this one again."}
/>
