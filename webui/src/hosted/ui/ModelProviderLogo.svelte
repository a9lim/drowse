<script lang="ts">
  import eleutherLogo from "../../assets/model-providers/eleutherai.png";
  import openaiLogo from "../../assets/model-providers/openai.png";

  let { modelId, size = 24 }: { modelId: string; size?: number } = $props();

  const modelName = $derived(modelId.toLowerCase().split("/").at(-1) ?? "");
  const provider = $derived(
    modelName.startsWith("qwen")
      ? "qwen"
      : modelName.startsWith("gemma")
        ? "gemma"
        : modelName.startsWith("gpt2") || modelName.startsWith("gpt-2")
          ? "openai"
          : modelName.startsWith("pythia") ? "eleutherai" : null,
  );
</script>

{#if provider === "qwen"}
  <span class="model-provider-logo" style:--provider-logo-size={`${size}px`} data-provider="qwen" aria-hidden="true">
    <svg viewBox="0 0 233 236" focusable="false">
      <path d="M187.25 36.08h-52.9c0-2.25-.58-4.5-1.75-6.52l-7.08-12.26a13.04 13.04 0 0 0-11.28-6.52h-14.16a13.04 13.04 0 0 0-11.29 6.52L67.11 54.86a13.02 13.02 0 0 0 0 13.02l7.08 12.27a13.04 13.04 0 0 0 11.28 6.51h101.78a13.04 13.04 0 0 0 11.29-6.51l7.08-12.27a13.02 13.02 0 0 0 0-13.02l-7.08-12.27a13.04 13.04 0 0 0-11.29-6.51Z" />
      <path d="m12.35 99.5 26.45 45.82a13.02 13.02 0 0 0-4.77 4.77l-7.08 12.26a13.04 13.04 0 0 0 0 13.03l7.08 12.27a13.04 13.04 0 0 0 11.28 6.51h43.37a13.04 13.04 0 0 0 11.29-6.51l7.08-12.27a13.04 13.04 0 0 0 0-13.03L56.16 74.21a13.04 13.04 0 0 0-11.29-6.52H30.71a13.04 13.04 0 0 0-11.28 6.52l-7.08 12.26a13.04 13.04 0 0 0 0 13.03Z" />
      <path d="m154.73 219.26 26.45-45.81a13.04 13.04 0 0 0 6.51 1.74h14.17a13.04 13.04 0 0 0 11.28-6.51l7.08-12.27a13.04 13.04 0 0 0 0-13.03l-21.68-37.56a13.04 13.04 0 0 0-11.29-6.51h-14.16a13.04 13.04 0 0 0-11.28 6.51l-50.89 88.15a13.04 13.04 0 0 0 0 13.03l7.08 12.26a13.04 13.04 0 0 0 11.28 6.52h14.16a13.04 13.04 0 0 0 11.29-6.52Z" />
    </svg>
  </span>
{:else if provider === "gemma"}
  <span class="model-provider-logo" style:--provider-logo-size={`${size}px`} data-provider="gemma" aria-hidden="true">
    <svg viewBox="0 0 24 24" focusable="false">
      <path fill-rule="evenodd" d="M12.34 5.953a8.233 8.233 0 0 1-.247-1.125V3.72a8.25 8.25 0 0 1 5.562 2.232H12.34Zm-.69 0c.113-.373.199-.755.257-1.145V3.72a8.25 8.25 0 0 0-5.562 2.232h5.304Zm-5.433.187h5.373a7.98 7.98 0 0 1-.267.696 8.41 8.41 0 0 1-1.76 2.65L6.216 6.14Zm-.264-.187H2.977v.187h2.915a8.436 8.436 0 0 0-2.357 5.767H0v.186h3.535a8.436 8.436 0 0 0 2.357 5.767H2.977v.186h2.976v2.977h.187v-2.915a8.436 8.436 0 0 0 5.767 2.357V24h.186v-3.535a8.436 8.436 0 0 0 5.767-2.357v2.915h.186v-2.977h2.977v-.186h-2.915a8.436 8.436 0 0 0 2.357-5.767H24v-.186h-3.535a8.436 8.436 0 0 0-2.357-5.767h2.915v-.187h-2.977V2.977h-.186v2.915a8.436 8.436 0 0 0-5.767-2.357V0h-.186v3.535A8.436 8.436 0 0 0 6.14 5.892V2.977h-.187v2.976Zm6.14 14.326a8.25 8.25 0 0 0 5.562-2.233H12.34c-.108.367-.19.743-.247 1.126v1.107Zm-.186-1.087a8.015 8.015 0 0 0-.258-1.146H6.345a8.25 8.25 0 0 0 5.562 2.233v-1.087Zm-8.186-7.285h1.107a8.23 8.23 0 0 0 1.125-.247V6.345a8.25 8.25 0 0 0-2.232 5.562Zm1.087.186H3.72a8.25 8.25 0 0 0 2.232 5.562v-5.304a8.012 8.012 0 0 0-1.145-.258Zm15.47-.186a8.25 8.25 0 0 0-2.232-5.562v5.315c.367.108.743.19 1.126.247h1.107Zm-1.086.186c-.39.058-.772.144-1.146.258v5.304a8.25 8.25 0 0 0 2.233-5.562h-1.087Zm-1.332 5.69V12.41a7.97 7.97 0 0 0-.696.267 8.409 8.409 0 0 0-2.65 1.76l3.346 3.346Zm0-6.18v-5.45l-.012-.013h-5.451c.076.235.162.468.26.696a8.698 8.698 0 0 0 1.819 2.688 8.698 8.698 0 0 0 2.688 1.82c.228.097.46.183.696.259ZM6.14 17.848V12.41c.235.078.468.167.696.267a8.403 8.403 0 0 1 2.688 1.799 8.404 8.404 0 0 1 1.799 2.688c.1.228.19.46.267.696H6.152l-.012-.012Zm0-6.245V6.326l3.29 3.29a8.716 8.716 0 0 1-2.594 1.728 8.14 8.14 0 0 1-.696.259Zm6.257 6.257h5.277l-3.29-3.29a8.716 8.716 0 0 0-1.728 2.594 8.135 8.135 0 0 0-.259.696Zm-2.347-7.81a9.435 9.435 0 0 1-2.88 1.96 9.14 9.14 0 0 1 2.88 1.94 9.14 9.14 0 0 1 1.94 2.88 9.435 9.435 0 0 1 1.96-2.88 9.14 9.14 0 0 1 2.88-1.94 9.435 9.435 0 0 1-2.88-1.96 9.434 9.434 0 0 1-1.96-2.88 9.14 9.14 0 0 1-1.94 2.88Z" />
    </svg>
  </span>
{:else if provider}
  <span class="model-provider-logo" style:--provider-logo-size={`${size}px`} data-provider={provider} aria-hidden="true">
    <span class="raster-mark" style:mask-image={`url("${provider === "openai" ? openaiLogo : eleutherLogo}")`}></span>
  </span>
{/if}

<style>
  .model-provider-logo {
    display: inline-grid;
    flex: 0 0 auto;
    width: var(--provider-logo-size);
    height: var(--provider-logo-size);
    place-items: center;
    color: var(--fg);
  }

  svg {
    display: block;
    width: 100%;
    height: 100%;
    fill: currentColor;
  }

  .raster-mark {
    width: 100%;
    height: 100%;
    background: currentColor;
    mask-size: contain;
    mask-position: center;
    mask-repeat: no-repeat;
    mask-mode: alpha;
  }

  [data-provider="eleutherai"] .raster-mark { mask-mode: luminance; }
</style>
