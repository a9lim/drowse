<script module lang="ts">
  /** Bump when `public-hosted/video/ethereal-orb*.mp4` is re-encoded; `/video/*` is cached as immutable. */
  export const HERO_VIDEO_VERSION = "2026-10-29";
  export const HERO_MOTION_STORAGE_KEY = "drowse.hero-motion";
  const READY_TIMEOUT_MS = 10_000;

  type NetworkHints = { saveData?: boolean; effectiveType?: string };

  /** Slow links and low-memory devices keep the cached poster instead of streaming video. */
  export function heroVideoAllowed(connection: NetworkHints | undefined, deviceMemory: number | undefined): boolean {
    if (/^(?:slow-2g|2g|3g)$/.test(connection?.effectiveType ?? "")) return false;
    return !(typeof deviceMemory === "number" && deviceMemory <= 2);
  }

  export function heroVideoUrl(width: number): string {
    return `${width <= 760 ? "/video/ethereal-orb-720.mp4" : "/video/ethereal-orb.mp4"}?v=${HERO_VIDEO_VERSION}`;
  }

  function readMotionPreference(): boolean {
    try { return localStorage.getItem(HERO_MOTION_STORAGE_KEY) === "paused"; } catch { return false; }
  }

  function writeMotionPreference(paused: boolean) {
    try {
      if (paused) localStorage.setItem(HERO_MOTION_STORAGE_KEY, "paused");
      else localStorage.removeItem(HERO_MOTION_STORAGE_KEY);
    } catch { /* The choice still applies to this page without storage. */ }
  }
</script>

<script lang="ts">
  import { onMount } from "svelte";
  import { subscribeTheme } from "../../lib/theme";
  let { paused = false, contained = false, closeup = false, motionControl = false }: {
    paused?: boolean;
    contained?: boolean;
    closeup?: boolean;
    /** Show a pause/play control. Only enable where the control is not inside an aria-hidden ancestor. */
    motionControl?: boolean;
  } = $props();
  let updateMotion = $state<(() => void) | null>(null);
  let userPaused = $state(false);
  let controlAvailable = $state(false);
  $effect(() => { void paused; void userPaused; updateMotion?.(); });
  let host: HTMLElement;
  let canvas: HTMLCanvasElement;
  let status = $state<"fallback" | "loading" | "ready">("loading");
  let orbBoost = $state(1);
  const highlightTable = $derived(Array.from({ length: 101 }, (_, i) => {
    const value = i / 100;
    const highlight = Math.min(1, Math.max(0, (value - 0.12) / 0.18));
    return Math.min(1, value * (1 + (orbBoost - 1) * highlight * highlight * (3 - 2 * highlight))).toFixed(5);
  }).join(" "));

  function toggleMotion() {
    userPaused = !userPaused;
    writeMotionPreference(userPaused);
  }

  onMount(() => {
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const hints = navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string }; deviceMemory?: number };
    const saveData = hints.connection?.saveData;
    const videoAllowed = heroVideoAllowed(hints.connection, hints.deviceMemory);
    const video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "none";
    const poster = new Image();
    let source: ReturnType<typeof import("./heroShaderRuntime").createHeroShaderSource> | null = null;
    let disposed = false, initialized = false, inView = false, lost = false, failed = false, videoRequested = false;
    let frame = 0, lastFrame = 0, progress = 0, scrollFrame = 0;
    let readyTimer: ReturnType<typeof setTimeout> | undefined;
    let cancelVideoSchedule: (() => void) | null = null;
    let lightTheme = document.documentElement.dataset.theme === "light";
    userPaused = readMotionPreference();
    const refreshControl = () => { controlAvailable = motionControl && !motion.matches && !saveData && !failed; };
    refreshControl();
    if (motion.matches || saveData || userPaused) status = "fallback";
    const moving = () => !disposed && !paused && !userPaused && !motion.matches && !saveData && !document.hidden && inView && !lost;
    const stop = () => {
      cancelAnimationFrame(frame);
      frame = 0; lastFrame = 0;
      video.pause();
    };
    const fail = () => {
      clearTimeout(readyTimer);
      cancelVideoSchedule?.();
      stop(); source?.dispose(); source = null; status = "fallback";
      failed = true; refreshControl();
    };
    const playVideo = () => {
      if (!videoRequested) return;
      void video.play().then(() => { if (!moving()) video.pause(); }).catch(() => undefined);
    };
    const render = (now: number) => {
      frame = 0;
      if (!moving() || !source) return;
      if (!lastFrame || now - lastFrame >= source.frameInterval) {
        const delta = lastFrame ? Math.min((now - lastFrame) / 1000, 0.1) : 1 / 30;
        lastFrame = now;
        try {
          if (source.update(progress, delta, lightTheme, orbBoost)) { status = "ready"; clearTimeout(readyTimer); }
        } catch { fail(); return; }
      }
      frame = requestAnimationFrame(render);
    };
    const start = () => {
      if (!source || !moving() || frame) return;
      // Video adds moving light; the renderer also works with the cached artwork.
      playVideo();
      frame = requestAnimationFrame(render);
    };
    const boost = (viewportHeight: number) => {
      const blend = Math.min(1, Math.max(0, window.scrollY / Math.max(viewportHeight * 2.5, 1)));
      const dimming = blend * blend * (3 - 2 * blend);
      return host.clientWidth > 760 ? 2.2 - 0.8 * dimming : 1 - 0.1 * dimming;
    };
    const setBoost = (value: number) => {
      // Each change rewrites a full-viewport SVG filter table, so skip imperceptible steps.
      if (Math.abs(value - orbBoost) >= 0.001) orbBoost = value;
    };
    const onScroll = () => {
      if (closeup) {
        progress = 0.38;
        setBoost(1.25);
        host.dataset.travel = progress.toFixed(3);
        return;
      }
      // The still artwork keeps the boost it had when motion stopped.
      if (paused || userPaused || motion.matches || saveData || status === "fallback") return;
      const viewportHeight = contained ? innerHeight : host.clientHeight;
      setBoost(boost(viewportHeight));
      progress = Math.min(1, Math.max(0, window.scrollY / Math.max(viewportHeight * 1.8, 1)));
      host.dataset.travel = progress.toFixed(3);
    };
    const queueScroll = () => {
      if (scrollFrame) return;
      scrollFrame = requestAnimationFrame(() => { scrollFrame = 0; onScroll(); });
    };
    const resize = () => {
      onScroll();
      if (source) {
        try {
          source.resize(host.clientWidth, host.clientHeight);
          // Resizing clears the canvas; redraw before Safari can composite it.
          source.update(progress, 0, lightTheme, orbBoost);
        } catch { fail(); }
      }
    };
    const unsubscribeTheme = subscribeTheme((theme) => {
      lightTheme = theme === "light";
      try { source?.update(progress, 0, lightTheme, orbBoost); }
      catch { fail(); }
    });
    const loadVideo = () => {
      cancelVideoSchedule = null;
      if (disposed || !source || videoRequested) return;
      videoRequested = true;
      video.src = heroVideoUrl(host.clientWidth);
      video.load();
      if (moving()) playVideo();
    };
    const scheduleVideo = () => {
      if (!videoAllowed || videoRequested || cancelVideoSchedule) return;
      // The poster renders first; video streams only after the page and its idle work settle.
      let idle = 0, timer: ReturnType<typeof setTimeout> | undefined;
      const whenIdle = () => {
        window.removeEventListener("load", whenIdle);
        if (typeof requestIdleCallback === "function") idle = requestIdleCallback(loadVideo, { timeout: 3000 });
        else timer = setTimeout(loadVideo, 1500);
      };
      cancelVideoSchedule = () => {
        window.removeEventListener("load", whenIdle);
        if (idle) cancelIdleCallback(idle);
        clearTimeout(timer);
        cancelVideoSchedule = null;
      };
      if (document.readyState === "complete") whenIdle();
      else window.addEventListener("load", whenIdle);
    };
    const initialize = async () => {
      if (initialized || failed || !moving()) return;
      initialized = true;
      status = "loading";
      try {
        const { createHeroShaderSource } = await import("./heroShaderRuntime");
        if (disposed || lost) return;
        if (!moving()) { initialized = false; status = "fallback"; return; }
        poster.onerror = fail;
        poster.src = "/images/ethereal-orb.jpg";
        source = createHeroShaderSource(canvas, video, poster);
        resize();
        if (!source) return;
        // Without a decodable poster or video frame the renderer can never draw.
        readyTimer = setTimeout(() => { if (status === "loading") fail(); }, READY_TIMEOUT_MS);
        scheduleVideo();
        start();
      } catch { fail(); }
    };
    updateMotion = () => {
      if (!moving()) {
        stop();
        if ((!source || status === "loading") && (motion.matches || saveData || userPaused)) status = "fallback";
      } else {
        onScroll();
        void initialize();
        start();
      }
    };
    const update = () => { refreshControl(); updateMotion?.(); };
    const contextLost = (event: Event) => {
      event.preventDefault(); lost = true; fail();
    };
    const contextRestored = () => {
      lost = false; failed = false; initialized = false; refreshControl(); void initialize();
    };
    const intersection = new IntersectionObserver(([entry]) => {
      inView = Boolean(entry?.isIntersecting); update();
    });
    const observer = new ResizeObserver(resize);
    intersection.observe(host); observer.observe(host);
    motion.addEventListener("change", update);
    document.addEventListener("visibilitychange", update);
    window.addEventListener("scroll", queueScroll, { passive: true });
    window.addEventListener("pagehide", stop);
    window.addEventListener("pageshow", update);
    canvas.addEventListener("webglcontextlost", contextLost);
    canvas.addEventListener("webglcontextrestored", contextRestored);
    // Set the still artwork's brightness once; scrolling only changes it while the scene moves.
    if (closeup) onScroll();
    else orbBoost = boost(contained ? innerHeight : host.clientHeight);
    return () => {
      disposed = true; stop(); source?.dispose(); source = null;
      clearTimeout(readyTimer);
      cancelVideoSchedule?.();
      cancelAnimationFrame(scrollFrame);
      unsubscribeTheme();
      intersection.disconnect(); observer.disconnect();
      motion.removeEventListener("change", update);
      document.removeEventListener("visibilitychange", update);
      window.removeEventListener("scroll", queueScroll);
      window.removeEventListener("pagehide", stop);
      window.removeEventListener("pageshow", update);
      canvas.removeEventListener("webglcontextlost", contextLost);
      canvas.removeEventListener("webglcontextrestored", contextRestored);
      if (videoRequested) { video.removeAttribute("src"); video.load(); }
      poster.onerror = null;
      poster.removeAttribute("src");
      updateMotion = null;
    };
  });
</script>

<div class="hero-visual" class:contained class:closeup bind:this={host} data-shader-status={status} data-orb-boost={orbBoost.toFixed(3)} aria-hidden="true">
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

{#if controlAvailable}
  <button type="button" class="motion-toggle" onclick={toggleMotion}
    aria-label={userPaused ? "Play background animation" : "Pause background animation"}>
    <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">
      {#if userPaused}<path d="M4.5 2.75v10.5L13 8z" />{:else}<path d="M4 2.5h2.75v11H4zM9.25 2.5H12v11H9.25z" />{/if}
    </svg>
  </button>
{/if}

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
  }

  @media (hover: none) and (pointer: coarse) {
    .hero-visual:not(.contained) { bottom: auto; height: 100lvh; }
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
    filter: var(--hero-media-filter);
    opacity: 0;
    transition: opacity 600ms var(--ease-out);
  }

  .fallback.visible { opacity: 1; }

  .hero-visual.closeup .fallback { background-size: auto 180%; background-position: 60% 48%; }

  canvas {
    display: block;
    opacity: 0;
    transition: opacity 1400ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  canvas.visible { opacity: 1; }

  /* Sits at the bottom of the first screen and scrolls away with the hero. */
  .motion-toggle {
    position: absolute;
    z-index: 2;
    top: calc(100svh - 44px - max(var(--space-md), env(safe-area-inset-bottom)));
    inset-inline-end: max(var(--surface-padding), env(safe-area-inset-right));
    display: inline-grid;
    place-items: center;
    width: 44px;
    height: 44px;
    padding: 0;
    border: 1px solid var(--glass-line);
    border-radius: var(--radius-pill);
    background: var(--landing-panel-bg);
    color: var(--fg);
    cursor: pointer;
    pointer-events: auto;
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
    transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out);
  }
  .motion-toggle:hover { background: var(--bg-hover); }
  .motion-toggle:active { scale: 0.96; }
  .motion-toggle:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .motion-toggle svg { fill: currentColor; }

  @media (prefers-reduced-motion: reduce) {
    .fallback,
    canvas, .motion-toggle { transition: none; }
    .motion-toggle:active { scale: 1; }
    .visual-layer, .fallback {
      transform: none;
      transition: none;
      will-change: auto;
    }
  }
</style>
