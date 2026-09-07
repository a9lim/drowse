<script lang="ts">
  let { side = "left" }: { side?: "left" | "right" } = $props();
</script>

<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
  <rect x="3" y="4" width="18" height="16" rx="3" />
  <path d={side === "left" ? "M9 4v16" : "M15 4v16"} />
</svg>
