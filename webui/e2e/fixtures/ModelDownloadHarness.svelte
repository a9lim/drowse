<script lang="ts">
  import { onMount } from "svelte";
  import HostedHome from "../../src/hosted/ui/HostedHome.svelte";
  import type { HostedShellController } from "../../src/hosted/ui/types";

  let { controller, onChooseModels }: { controller: HostedShellController; onChooseModels: () => void } = $props();
  let snapshot = $state(controller.current());
  onMount(() => controller.subscribe(next => { snapshot = next; }));
</script>

<HostedHome {controller} {snapshot} {onChooseModels} />
