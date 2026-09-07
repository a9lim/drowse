<script lang="ts">
  let { plain = false }: { plain?: boolean } = $props();
</script>

<span class="base-tag" class:plain title="Base model · continues text without chat instructions">[BASE]</span>

<style>
  .base-tag {
    display: inline-block;
    margin-inline-start: var(--space-3);
    padding: var(--space-1) var(--space-2);
    border-radius: var(--radius-sm);
    background: var(--accent-subtle);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    line-height: 1.5;
    white-space: nowrap;
    vertical-align: middle;
  }
  .base-tag.plain {
    flex-shrink: 0;
    margin: 0;
    padding: 0;
    border-radius: 0;
    background: none;
  }
</style>
