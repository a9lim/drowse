<script lang="ts">
  import { onMount } from "svelte";
  import { appearanceState, loadAppearance } from "../stores/appearance.svelte";
  import { backgroundRenderer } from "../backgroundShader";

  let canvas: HTMLCanvasElement | undefined = $state();
  let rendered = $state(false);
  let revision = $state(0);
  let redraw = () => {};
  onMount(() => { void loadAppearance(); });

  $effect(() => {
    const target = canvas;
    const url = appearanceState.url;
    revision;
    rendered = false;
    if (!target || !url) return;
    const renderer = backgroundRenderer(target);
    if (!renderer) return;
    let disposed = false;
    let ready = false;
    const lost = (event: Event) => { event.preventDefault(); ready = false; rendered = false; };
    const restored = () => revision++;
    target.addEventListener("webglcontextlost", lost);
    target.addEventListener("webglcontextrestored", restored);
    const image = new Image();
    const draw = () => { if (ready && !disposed) renderer.draw(appearanceState); };
    redraw = draw;
    image.onload = () => {
      if (disposed) return;
      renderer.image(image);
      ready = true;
      draw();
      rendered = true;
    };
    image.src = url;
    const resize = new ResizeObserver(draw);
    resize.observe(target);
    return () => { disposed = true; image.onload = null; resize.disconnect(); target.removeEventListener("webglcontextlost", lost); target.removeEventListener("webglcontextrestored", restored); renderer.dispose(); redraw = () => {}; };
  });
  $effect(() => {
    appearanceState.effect; appearanceState.pixelSize;
    redraw();
  });
</script>

{#if appearanceState.url}
  <div class="workspace-background" aria-hidden="true" style:opacity={`calc(${appearanceState.visibility} * var(--workspace-background-strength, 1))`} style:background-image={rendered ? "none" : `url("${appearanceState.url}")`}>
    <canvas bind:this={canvas} class:rendered></canvas>
  </div>
{/if}

<style>
  .workspace-background { position: absolute; inset: 0; z-index: -1; pointer-events: none; background-size: cover; background-position: center; }
  canvas { display: block; width: 100%; height: 100%; opacity: 0; }
  canvas.rendered { opacity: 1; }
  @media (forced-colors: active) { .workspace-background { display: none; } }
</style>
