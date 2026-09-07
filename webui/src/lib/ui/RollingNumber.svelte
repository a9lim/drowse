<script lang="ts">
  import { createRollingNumber, type RollingNumberOptions } from "@kitlangton/rolling-number";
  import "@kitlangton/rolling-number/styles.css";

  let { value, digits = 0, format, signed = false }: {
    value: number;
    digits?: number;
    format?: Intl.NumberFormatOptions;
    signed?: boolean;
  } = $props();

  const options = $derived<RollingNumberOptions>({
    value,
    locales: "en-US",
    format: format ?? {
      useGrouping: false,
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      signDisplay: signed ? "always" : "auto",
    },
    duration: 280,
    stagger: "none",
    motionBlur: false,
    pauseOffscreen: true,
  });
  const accessibleValue = $derived(new Intl.NumberFormat("en-US", options.format).format(value));

  function rolling(node: HTMLElement, initial: RollingNumberOptions) {
    const counter = createRollingNumber(node, initial);
    return {
      update: (next: RollingNumberOptions) => counter.update(next),
      destroy: () => counter.destroy(),
    };
  }
</script>

<span class="sr-only">{accessibleValue}</span><span aria-hidden="true" class="rolling-number" data-value={value} use:rolling={options}></span>

<style>
  .rolling-number {
    font-variant-numeric: tabular-nums;
    --rn-edge-fade: 0.08em;
  }
</style>
