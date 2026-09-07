<script lang="ts">
  import { onMount } from "svelte";
  let { paused = false, contained = false }: { paused?: boolean; contained?: boolean } = $props();
  let updateMotion = $state<(() => void) | null>(null);
  $effect(() => { void paused; updateMotion?.(); });
  let host: HTMLElement;
  let canvas: HTMLCanvasElement;
  let status = $state<"fallback" | "loading" | "ready">("loading");
  let orbBoost = $state(1);
  const highlightTable = $derived(Array.from({ length: 101 }, (_, i) => {
    const value = i / 100;
    const highlight = Math.min(1, Math.max(0, (value - 0.12) / 0.18));
    return Math.min(1, value * (1 + (orbBoost - 1) * highlight * highlight * (3 - 2 * highlight))).toFixed(5);
  }).join(" "));

  onMount(() => {
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const saveData = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData;
    const video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "none";
    const poster = new Image();
    let source: ReturnType<typeof import("./heroShaderRuntime").createHeroShaderSource> | null = null;
    let disposed = false, initialized = false, inView = false, lost = false;
    let frame = 0, lastFrame = 0, progress = 0;
    if (motion.matches || saveData) status = "fallback";
    const moving = () => !disposed && !paused && !motion.matches && !saveData && !document.hidden && inView && !lost;
    const stop = () => {
      cancelAnimationFrame(frame);
      frame = 0; lastFrame = 0;
      video.pause();
    };
    const fail = () => { stop(); source?.dispose(); source = null; status = "fallback"; };
    const render = (now: number) => {
      frame = 0;
      if (!moving() || !source) return;
      if (!lastFrame || now - lastFrame >= 1000 / 30) {
        const delta = lastFrame ? Math.min((now - lastFrame) / 1000, 0.1) : 1 / 30;
        lastFrame = now;
        try { if (source.update(progress, delta)) status = "ready"; }
        catch { fail(); return; }
      }
      frame = requestAnimationFrame(render);
    };
    const start = () => {
      if (!source || !moving() || frame) return;
      // Video adds moving light; the renderer also works with the cached artwork.
      void video.play().then(() => { if (!moving()) video.pause(); }).catch(() => undefined);
      frame = requestAnimationFrame(render);
    };
    const onScroll = () => {
      const blend = Math.min(1, Math.max(0, window.scrollY / Math.max(innerHeight * 2.5, 1)));
      const dimming = blend * blend * (3 - 2 * blend);
      orbBoost = host.clientWidth > 760 ? 2.2 - 0.8 * dimming : 1 - 0.1 * dimming;
      if (paused || motion.matches || saveData) return;
      progress = Math.min(1, Math.max(0, window.scrollY / Math.max(innerHeight * 1.8, 1)));
      host.dataset.travel = progress.toFixed(3);
    };
    const resize = () => {
      if (source) {
        try {
          source.resize(host.clientWidth, host.clientHeight);
          if (!moving()) source.update(progress, 0);
        } catch { fail(); }
      }
      onScroll();
    };
    const initialize = async () => {
      if (initialized || !moving()) return;
      initialized = true;
      status = "loading";
      try {
        const { createHeroShaderSource } = await import("./heroShaderRuntime");
        if (disposed || lost) return;
        if (!moving()) { initialized = false; status = "fallback"; return; }
        poster.src = "/images/ethereal-orb.jpg";
        source = createHeroShaderSource(canvas, video, poster);
        resize();
        if (!source) return;
        video.src = host.clientWidth <= 760 ? "/video/ethereal-orb-720.mp4" : "/video/ethereal-orb.mp4";
        video.load();
        start();
      } catch { fail(); }
    };
    updateMotion = () => {
      if (!moving()) {
        stop();
        if (!source && (motion.matches || saveData)) status = "fallback";
      } else {
        onScroll();
        void initialize();
        start();
      }
    };
    const update = () => updateMotion?.();
    const contextLost = (event: Event) => {
      event.preventDefault(); lost = true; fail();
    };
    const contextRestored = () => {
      lost = false; initialized = false; void initialize();
    };
    const intersection = new IntersectionObserver(([entry]) => {
      inView = Boolean(entry?.isIntersecting); update();
    });
    const observer = new ResizeObserver(resize);
    intersection.observe(host); observer.observe(host);
    motion.addEventListener("change", update);
    document.addEventListener("visibilitychange", update);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("pagehide", stop);
    window.addEventListener("pageshow", update);
    canvas.addEventListener("webglcontextlost", contextLost);
    canvas.addEventListener("webglcontextrestored", contextRestored);
    onScroll();
    return () => {
      disposed = true; stop(); source?.dispose(); source = null;
      intersection.disconnect(); observer.disconnect();
      motion.removeEventListener("change", update);
      document.removeEventListener("visibilitychange", update);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("pagehide", stop);
      window.removeEventListener("pageshow", update);
      canvas.removeEventListener("webglcontextlost", contextLost);
      canvas.removeEventListener("webglcontextrestored", contextRestored);
      video.removeAttribute("src"); video.load();
      poster.removeAttribute("src");
      updateMotion = null;
    };
  });
</script>

<div class="hero-visual" class:contained bind:this={host} data-shader-status={status} data-orb-boost={orbBoost.toFixed(3)} aria-hidden="true">
  <svg class="palette-definitions" width="0" height="0" focusable="false">
    <defs>
      <filter id="hero-dark-palette" color-interpolation-filters="sRGB">
        <feComponentTransfer>
          <feFuncR type="gamma" amplitude="0.43" exponent="0.55" offset="0" />
          <feFuncG type="gamma" amplitude="0.43" exponent="0.55" offset="0" />
          <feFuncB type="gamma" amplitude="0.43" exponent="0.55" offset="0" />
        </feComponentTransfer>
        <feComponentTransfer>
          <feFuncR type="table" tableValues={highlightTable} />
          <feFuncG type="table" tableValues={highlightTable} />
          <feFuncB type="table" tableValues={highlightTable} />
        </feComponentTransfer>
      </filter>
      <filter id="hero-light-palette" color-interpolation-filters="sRGB">
        <!-- Map scene luminance to paper and lavender, without hue inversion. -->
        <feColorMatrix type="matrix" values="
          -0.07087 -0.23840 -0.02407 0 0.94902
          -0.10255 -0.34497 -0.03483 0 0.95686
          -0.02418 -0.08134 -0.00821 0 0.97255
           0        0        0       1 0
        " />
      </filter>
    </defs>
  </svg>
  <div class="visual-layer">
    <div class:visible={status === "fallback"} class="fallback"></div>
    <canvas
      bind:this={canvas}
      class:visible={status === "ready"}
    ></canvas>
  </div>
</div>

<style>
  .palette-definitions { position: absolute; overflow: hidden; }

  .hero-visual {
    position: fixed;
    z-index: 0;
    inset: 0;
    background: var(--hero-bg);
    overflow: hidden;
    isolation: isolate;
    pointer-events: none;
  }

  .visual-layer {
    position: absolute;
    inset: 0;
    filter: var(--hero-media-filter);
    transition: filter var(--dur) var(--ease-out);
  }

  .hero-visual.contained { position: absolute; }

  .fallback,
  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  .fallback {
    background: url("/images/ethereal-orb.jpg") center / cover no-repeat;
    opacity: 0;
    transition: opacity 600ms var(--ease-out);
  }

  .fallback.visible { opacity: 1; }

  canvas {
    display: block;
    opacity: 0;
    transition: opacity 1400ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  canvas.visible { opacity: 1; }

  @media (prefers-reduced-motion: reduce) {
    .fallback,
    canvas { transition: none; }
    .visual-layer, .fallback {
      transform: none;
      transition: none;
      will-change: auto;
    }
  }
</style>
