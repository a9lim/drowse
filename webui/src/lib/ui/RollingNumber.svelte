<script lang="ts">
  import MorphText from "./MorphText.svelte";

  let { value, digits = 0, format, signed = false }: {
    value: number;
    digits?: number;
    format?: Intl.NumberFormatOptions;
    signed?: boolean;
  } = $props();

  const numberFormat: Intl.NumberFormatOptions = $derived(format ?? {
    useGrouping: false,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
    signDisplay: signed ? "always" : "auto",
  });
  const accessibleValue = $derived(Number.isFinite(value) ? new Intl.NumberFormat("en-US", numberFormat).format(value) : "-");
</script>

<span class="rolling-number" data-value={value}><MorphText text={accessibleValue} /></span>

<style>
  .rolling-number {
    font-variant-numeric: tabular-nums;
    display: inline-block;
  }
</style>
