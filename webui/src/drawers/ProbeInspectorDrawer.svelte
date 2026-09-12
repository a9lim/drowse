<script lang="ts">
  import { geometryViewSchema } from "../lib/interfaceSchemas";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { ToolError } from "../lib/webmcp/types";
  import { onMount as onInterfaceMount } from "svelte";

  import MorphText from "../lib/ui/MorphText.svelte";
  import FluentIcon from "../lib/ui/FluentIcon.svelte";
  import { animatedDetails } from "../lib/animatedDetails";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Per-probe inspector — subsumes the layer-norms view for probes and adds a
  // rank-aware geometry plot in the whitened (Mahalanobis) frame:
  //
  //   rank 1   -> a line: poles + neutral + live point
  //   rank 2   -> a 2D scatter of node centroids (+ curve overlay if 1-D)
  //   rank 3+  -> a drag-orbit 3D scatter on the top-3 subspace PCs
  //              (+ curve / wireframe-surface overlay)
  //
  // A layer scrubber drives which fitted layer is shown; geometry is fetched
  // once (all layers) and reprojected client-side on scrub.  The live hidden-
  // state point + a fading trajectory trail ride the probe's per-token
  // ``subspace_coords_per_layer`` (gated on by ``persist_subspace_coords`` while
  // this drawer is open), stored across all layers so scrubbing is a pure read.
  //
  // v2 sheet interior: the drawer host paints the sheet (glass hairline,
  // radius, --bg-alt fill) so the root here is transparent.  The probe's
  // FAMILY hue (flat = subspace white, curved = manifold violet — the same
  // is_affine split the racks use) accents the header dot, the active layer
  // row, the share bars, and the plot's node centroids via ``--geom-node``.

  import { closeDrawer, drawerState, probeRack } from "../lib/stores.svelte";
  import { apiProbes } from "../lib/runtime/services";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import Bar from "../lib/charts/Bar.svelte";
  import { chartValue } from "../lib/charts/chartValues";
  import {
    renderProbeGeometry,
    orbitDrag,
    DEFAULT_ORBIT_QUAT,
    type OrbitState,
    type GeometryHitPoint,
  } from "../lib/charts/probeGeometry";
  import {
    pinchMetrics,
    scaleFromPinch,
    type GesturePoint,
  } from "../lib/pointerGesture";
  import type { ProbeGeometryResponse, ProbeLayerGeometry } from "../lib/types";

  let _drawerProps: { params?: unknown } = $props();
  $effect(() => {
    void _drawerProps.params;
  });

  const params = $derived(drawerState.params as { name?: string } | null);
  const probeName = $derived(params?.name ?? "");
  const displayName = $derived(probeName.split("/").pop() ?? probeName);
  const entry = $derived(probeRack.entries.get(probeName) ?? null);
  const liveTrailAvailability = runtimeOperationAvailability(
    "probe_subspace_trails",
  );

  let geom = $state<ProbeGeometryResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let selectedLayer = $state<number | null>(null);
  const orbit = $state<OrbitState>({ q: DEFAULT_ORBIT_QUAT, zoom: 1.6 });

  let canvasEl = $state<HTMLCanvasElement | null>(null);
  let rafId = 0;
  let hitPoints: GeometryHitPoint[] = [];
  let plotTitle = $state("Hover a node to see its whitened coordinates");
  let loadEpoch = 0;

  /** Family hue — the rack's flat/curved split (hue = which space). */
  const familyHue = $derived(
    geom?.is_affine === false
      ? "var(--pillar-manifold)"
      : "var(--pillar-subspace)",
  );

  // --- load geometry on probe change ---
  async function load(name: string, epoch: number): Promise<void> {
    if (!name) {
      if (epoch === loadEpoch) {
        geom = null;
        loading = false;
        error = null;
      }
      return;
    }
    loading = true;
    error = null;
    geom = null;
    try {
      const g = await apiProbes.geometry(name);
      if (epoch !== loadEpoch) return;
      geom = g;
      // default to the highest-share layer (the one carrying the most
      // steering budget — also the most concept-bearing to read from)
      let best: number | null = null;
      let bestShare = -Infinity;
      for (const l of Object.values(g.layers)) {
        const sh = Math.abs(l.mahalanobis_share);
        if (sh > bestShare) {
          bestShare = sh;
          best = l.layer;
        }
      }
      selectedLayer = best;
    } catch (e) {
      if (epoch !== loadEpoch) return;
      error = userFacingError(
        e,
        "Unable to load this direction map. Reattach the direction and try again.",
      );
    } finally {
      if (epoch === loadEpoch) loading = false;
    }
  }

  $effect(() => {
    const epoch = ++loadEpoch;
    void load(probeName, epoch);
    return () => {
      if (loadEpoch === epoch) loadEpoch += 1;
    };
  });

  // sorted layer list (ascending) for the share strip
  const layerList = $derived.by<ProbeLayerGeometry[]>(() =>
    geom ? Object.values(geom.layers).sort((a, b) => a.layer - b.layer) : [],
  );
  const activeGeom = $derived(
    geom && selectedLayer !== null
      ? (geom.layers[String(selectedLayer)] ?? null)
      : null,
  );
  const maxShare = $derived(
    layerList.reduce((m, l) => Math.max(m, Math.abs(l.mahalanobis_share)), 0),
  );

  // --- live point + trail for the selected layer (reprojected client-side) ---
  const layerKey = $derived(selectedLayer !== null ? String(selectedLayer) : "");
  const livePoint = $derived.by<number[] | null>(() => {
    const trail = entry?.subspaceTrail;
    if (!trail || trail.length === 0) return null;
    return trail[trail.length - 1]?.perLayer[layerKey] ?? null;
  });
  const trailPoints = $derived.by<number[][]>(() => {
    const trail = entry?.subspaceTrail ?? [];
    const out: number[][] = [];
    for (const s of trail) {
      const p = s.perLayer[layerKey];
      if (Array.isArray(p)) out.push(p);
    }
    return out;
  });

  // --- canvas render (rAF-coalesced; re-runs on any dependency change) ---
  $effect(() => {
    const canvas = canvasEl;
    const g = activeGeom;
    // touch the reactive deps so the effect re-subscribes
    const live = livePoint;
    const trail = trailPoints;
    const q = orbit.q;
    const zoom = orbit.zoom;
    const labels = geom?.node_labels ?? [];
    if (!canvas || !g) {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      return;
    }
    const redraw = () => {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = 0;
        hitPoints = renderProbeGeometry(canvas, {
          geom: g,
          nodeLabels: labels,
          live,
          trail,
          orbit: { q, zoom },
        });
      });
    };
    const observer = new ResizeObserver(redraw);
    observer.observe(canvas);
    redraw();
    return () => {
      observer.disconnect();
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
    };
  });

  // --- orbit interaction (rank >= 3 only) ---
  let dragPointerId: number | null = null;
  let lastX = 0;
  let lastY = 0;
  const touchPointers = new Map<number, GesturePoint>();
  let pinchGesture: { distance: number; zoom: number } | null = null;
  const canOrbit = $derived((activeGeom?.rank ?? 0) >= 3);

  function capturePlotPointer(target: HTMLElement, pointerId: number): void {
    try {
      target.setPointerCapture(pointerId);
    } catch {
      // The pointer may already have ended between events.
    }
  }

  function releasePlotPointer(target: HTMLElement, pointerId: number): void {
    if (!target.hasPointerCapture(pointerId)) return;
    try {
      target.releasePointerCapture(pointerId);
    } catch {
      // A cancelled pointer releases capture automatically.
    }
  }

  function touchPair(): [GesturePoint, GesturePoint] | null {
    const points = [...touchPointers.values()];
    return points.length >= 2 ? [points[0], points[1]] : null;
  }

  function beginPlotPinch(target: HTMLElement): void {
    const pair = touchPair();
    if (!pair) return;
    const { distance } = pinchMetrics(pair[0], pair[1]);
    if (distance <= 0) return;
    dragPointerId = null;
    pinchGesture = { distance, zoom: orbit.zoom };
    for (const pointerId of touchPointers.keys()) {
      capturePlotPointer(target, pointerId);
    }
  }

  function onPointerDown(ev: PointerEvent): void {
    if (!canOrbit) return;
    if (ev.pointerType === "touch") {
      touchPointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      if (touchPointers.size >= 2) {
        beginPlotPinch(ev.currentTarget as HTMLElement);
        ev.preventDefault();
        return;
      }
    }
    dragPointerId = ev.pointerId;
    lastX = ev.clientX;
    lastY = ev.clientY;
    capturePlotPointer(ev.currentTarget as HTMLElement, ev.pointerId);
    ev.preventDefault();
  }
  function onPointerMove(ev: PointerEvent): void {
    if (canvasEl && ev.buttons === 0) {
      const rect = canvasEl.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      let closest: GeometryHitPoint | undefined;
      let distance = 16;
      for (const hit of hitPoints) {
        const d = Math.hypot(x - hit.screen[0], y - hit.screen[1]);
        if (d <= distance) { closest = hit; distance = d; }
      }
      plotTitle = closest
        ? `${closest.label} · whitened coordinates [${closest.point.map((v) => chartValue(v)).join(", ")}]`
        : "Hover a node to see its whitened coordinates";
    }
    if (!canOrbit) return;
    if (ev.pointerType === "touch" && touchPointers.has(ev.pointerId)) {
      touchPointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      const pair = touchPair();
      if (pair && pinchGesture) {
        orbit.zoom = scaleFromPinch(
          pinchGesture.zoom,
          pinchGesture.distance,
          pinchMetrics(pair[0], pair[1]).distance,
          0.3,
          6,
        );
        ev.preventDefault();
        return;
      }
    }
    if (dragPointerId !== ev.pointerId) return;
    const dx = ev.clientX - lastX;
    const dy = ev.clientY - lastY;
    lastX = ev.clientX;
    lastY = ev.clientY;
    // Trackball: compose a screen-axis rotation onto the accumulated
    // orientation.  No Euler angles → no gimbal lock, no elevation clamp.
    orbit.q = orbitDrag(orbit.q, dx, dy);
  }
  function onPointerUp(ev: PointerEvent): void {
    const target = ev.currentTarget as HTMLElement;
    releasePlotPointer(target, ev.pointerId);
    if (ev.pointerType === "touch") {
      const wasPinching = pinchGesture !== null;
      touchPointers.delete(ev.pointerId);
      if (wasPinching) {
        if (touchPointers.size >= 2) beginPlotPinch(target);
        else pinchGesture = null;
      }
    }
    if (dragPointerId === ev.pointerId) dragPointerId = null;
  }
  // Scroll wheel = intentional zoom (the rotation-driven zoom artifact is
  // gone; this is the only zoom path now).  Multiplicative so each notch is a
  // constant ratio; clamped to a sane window.
  function onWheel(ev: WheelEvent): void {
    if (!canOrbit) return;
    ev.preventDefault();
    const factor = Math.exp(-ev.deltaY * 0.0015);
    orbit.zoom = Math.max(0.3, Math.min(6, orbit.zoom * factor));
  }

  function rotateBy(dx: number, dy: number): void {
    if (!canOrbit) return;
    orbit.q = orbitDrag(orbit.q, dx, dy);
  }

  function zoomBy(factor: number): void {
    orbit.zoom = Math.max(0.3, Math.min(6, orbit.zoom * factor));
  }

  function resetView(): void {
    orbit.q = DEFAULT_ORBIT_QUAT;
    orbit.zoom = 1.6;
  }

  function formatPoint(point: number[] | null): string {
    if (!point || point.length === 0) return "not available";
    return point.slice(0, 3).map((value) => value.toFixed(3)).join(", ");
  }

  function onClose(): void {
    closeDrawer();
  }
  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  const rankLabel = $derived.by(() => {
    const r = activeGeom?.rank ?? 0;
    if (r <= 1) return "line · rank 1";
    if (r === 2) return "2D scatter · rank 2";
    return `3D PCA scatter · rank ${r}`;
  });
  const intrinsicLabel = $derived(
    activeGeom ? `intrinsic dim ${activeGeom.intrinsic_dim}` : "",
  );

  onInterfaceMount(() => registerInterfaceController("probe_geometry", {
    schema: geometryViewSchema,
    read: () => ({ values: { layer: selectedLayer, zoom: orbit.zoom, quaternion: orbit.q }, available_layers: layerList.map(row => row.layer), can_rotate: canOrbit, loading }),
    update: (change) => {
      const layer = change.layer as number | undefined;
      if (layer !== undefined && !layerList.some(row => row.layer === layer)) throw new ToolError("NOT_FOUND", "Choose a layer from available_layers.");
      const target = layer === undefined ? activeGeom : geom?.layers[String(layer)];
      if (!target) throw new ToolError("NOT_READY", "Wait for this probe's geometry to load.");
      if ((change.rotate || change.zoom !== undefined) && target.rank < 3) throw new ToolError("UNAVAILABLE", "Orbit and zoom controls require geometry of rank 3 or greater.");
      if (layer !== undefined) selectedLayer = layer;
      if (change.reset) resetView();
      if (change.zoom !== undefined) orbit.zoom = change.zoom as number;
      if (change.rotate) { const delta = change.rotate as { dx: number; dy: number }; rotateBy(delta.dx, delta.dy); }
    },
  }));
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" style:--family={familyHue} aria-label="Probe inspector">
  <header class="drawer-header">
    <div class="title">
      <h2 class="eyebrow">Reading details</h2>
      <div class="name-row">
        {#if probeName}
          <span class="family-dot" aria-hidden="true"></span>
          <code class="name" {...{ "aria-description": (probeName) }}>{displayName}</code>
          {#if geom}
            <span class="meta">{rankLabel} · {intrinsicLabel}</span>
            {#if !geom.rank_uniform}
              <span class="warn" {...{ "aria-description": "rank varies by layer" }}>
                rank varies by layer
              </span>
            {/if}
          {/if}
        {:else}
          <span class="meta">No reading selected</span>
        {/if}
      </div>
    </div>
    <DrawerCloseButton onclick={onClose} />
  </header>

  {#if !probeName}
    <div class="body"><div class="empty">Select a saved reading to see how it is measured.</div></div>
  {:else if loading}
    <div class="body" aria-busy="true"><div class="empty loading-pulse loading-placeholder" role="status">Loading probe geometry…</div></div>
  {:else if error}
    <div class="body"><div class="empty err" role="alert">Probe geometry failed: {error}</div></div>
  {:else if !geom || !activeGeom || layerList.length === 0}
    <div class="body"><div class="empty">This probe has no fitted geometry.</div></div>
  {:else}
    <div class="body">
      <div class="bars-col">
        <div class="section-label">layers · ‖share‖</div>
        <div class="bars">
          {#each layerList as l (l.layer)}
            <button
              type="button"
              class="row"
              class:active={l.layer === selectedLayer}
              aria-pressed={l.layer === selectedLayer}
              {...{ "aria-description": (`L${l.layer} · share ${chartValue(l.mahalanobis_share)} · ${chartValue(maxShare > 0 ? Math.abs(l.mahalanobis_share) / maxShare : 0, true)} of largest layer`) }}
              onclick={() => (selectedLayer = l.layer)}
            >
              <span class="lyr">L{l.layer}</span>
              <Bar
                value={l.mahalanobis_share}
                max={maxShare || 1}
                width={200}
                height={8}
                color="var(--family)"
                title={`Share ${chartValue(l.mahalanobis_share)} · ${chartValue(maxShare > 0 ? Math.abs(l.mahalanobis_share) / maxShare : 0, true)} of largest layer`}
              />
              <span class="val">{l.mahalanobis_share.toFixed(3)}</span>
            </button>
          {/each}
        </div>
      </div>

      <div class="plot-col">
        <div class="plot-wrap" class:orbit={canOrbit}>
          <canvas
            bind:this={canvasEl}
            class="plot"
            {...{ "aria-description": (plotTitle) }}
            aria-label={`Whitened geometry for ${displayName}, layer ${selectedLayer}`}
            aria-describedby="probe-geometry-summary"
            data-orbit-zoom={orbit.zoom.toFixed(2)}
            onpointerdown={onPointerDown}
            onpointermove={onPointerMove}
            onpointerup={onPointerUp}
            onpointercancel={onPointerUp}
            onlostpointercapture={onPointerUp}
            onwheel={onWheel}
          >Whitened geometry for {displayName}, layer {selectedLayer}.</canvas>
          <span class="layer-chip"><MorphText text={`L${selectedLayer} · share ${activeGeom.mahalanobis_share.toFixed(3)}`} /></span>
          {#if canOrbit}
            <span class="orbit-hint">drag · scroll or pinch</span>
            <div class="plot-controls" aria-label="Geometry view controls">
              <button type="button" aria-label="Rotate geometry left" onclick={() => rotateBy(-18, 0)}>left</button>
              <button type="button" aria-label="Rotate geometry right" onclick={() => rotateBy(18, 0)}>right</button>
              <button type="button" aria-label="Rotate geometry up" onclick={() => rotateBy(0, -18)}>up</button>
              <button type="button" aria-label="Rotate geometry down" onclick={() => rotateBy(0, 18)}>down</button>
              <button type="button" aria-label="Zoom geometry out" onclick={() => zoomBy(0.8)}><FluentIcon name="subtract" /></button>
              <button type="button" aria-label="Zoom geometry in" onclick={() => zoomBy(1.25)}><FluentIcon name="add" /></button>
              <button type="button" onclick={resetView}>reset</button>
            </div>
          {/if}
          {#if trailPoints.length > 0}
            <span class="trail-hint"><MorphText text={`${trailPoints.length} trail pts`} /></span>
          {:else if !liveTrailAvailability.available}
            <span class="live-hint" {...{ "aria-description": (liveTrailAvailability.reason ?? undefined) }}>
              live trail unavailable here
            </span>
          {:else}
            <span class="live-hint">run for live trail</span>
          {/if}
        </div>
        <p class="coordinate-summary" id="probe-geometry-summary">
          Neutral [{formatPoint(activeGeom.neutral_white)}]; live point [<MorphText text={formatPoint(livePoint)} identity={selectedLayer} />];
          {activeGeom.node_white.length} node {activeGeom.node_white.length === 1 ? "centroid" : "centroids"}.
        </p>
        <details use:animatedDetails class="geometry-summary">
          <summary>Geometry coordinates</summary>
          <p>
            Layer {selectedLayer}; rank {activeGeom.rank}; neutral [{formatPoint(activeGeom.neutral_white)}];
            live point [<MorphText text={formatPoint(livePoint)} identity={selectedLayer} />]. Coordinates show the first three whitened dimensions.
          </p>
          <ul>
            {#each activeGeom.node_white as point, index (index)}
              <li>{geom.node_labels[index] ?? `node ${index + 1}`}: [{formatPoint(point)}]</li>
            {/each}
          </ul>
        </details>
        {#if !liveTrailAvailability.available}
          <p class="capability-note" role="note">
            {liveTrailAvailability.reason}. The fitted geometry remains available.
          </p>
        {/if}
      </div>
    </div>
  {/if}

</aside>

<style>
  /* v2 sheet interior — the host paints the sheet surface, so the root
   * stays transparent and chrome speaks sans (data stays mono). */
  .drawer {
    container: probe-inspector / inline-size;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }
  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .name-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
    flex-wrap: wrap;
  }
  .family-dot {
    align-self: center;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--family);
    flex: none;
  }
  .name {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }
  .warn {
    color: var(--accent-yellow);
    font-size: var(--text-xs);
  }

  .body {
    flex: 1 1 auto;
    overflow: hidden;
    min-height: 0;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
    display: flex;
    flex-direction: row;
    gap: var(--space-6);
  }
  .bars-col {
    flex: 0 0 15rem;
    min-height: 0;
    overflow-y: auto;
    scrollbar-gutter: stable;
  }
  .plot-col {
    flex: 1 1 auto;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .geometry-summary {
    margin-top: var(--space-3);
    color: var(--fg-muted);
    font-size: var(--text-xs);
  }
  .capability-note {
    margin: var(--space-3) 0 0;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.45;
  }
  .coordinate-summary {
    margin: var(--space-3) 0 0;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    line-height: 1.45;
  }
  .geometry-summary summary {
    cursor: pointer;
    color: var(--fg);
    min-height: 44px;
    display: flex;
    align-items: center;
  }
  .geometry-summary p,
  .geometry-summary ul {
    margin: var(--space-2) 0 0;
  }
  .geometry-summary ul {
    max-height: 9rem;
    overflow: auto;
    padding-inline-start: var(--space-5);
    font-family: var(--font-mono);
  }
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
  }
  .empty.err {
    color: var(--accent-red);
  }

  /* The plot well stays quiet so its geometry carries the information. */
  .plot-wrap {
    position: relative;
    flex: 0 0 auto;
    width: 100%;
    aspect-ratio: 1;
    border-radius: var(--radius-lg);
    background: var(--bg-deep);
    box-shadow: var(--shadow-rack);
    overflow: hidden;
    /* Palette hooks read by the canvas renderer (hue ontology). */
    --geom-node: var(--family);
    --geom-neutral: var(--fg-subtle);
  }
  .plot-wrap.orbit .plot {
    cursor: grab;
    touch-action: none;
  }
  .plot-wrap.orbit .plot:active {
    cursor: grabbing;
  }
  .plot {
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
  }
  .layer-chip {
    position: absolute;
    top: var(--space-3);
    left: var(--space-4);
    color: color-mix(in srgb, var(--family) 80%, var(--fg));
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    padding: var(--space-xs) var(--space-4);
    pointer-events: none;
  }
  .orbit-hint,
  .live-hint,
  .trail-hint {
    position: absolute;
    color: var(--fg-muted);
    font-size: var(--text-2xs);
    pointer-events: none;
  }
  .orbit-hint {
    top: var(--space-3);
    inset-inline-end: var(--space-4);
  }
  .live-hint {
    bottom: var(--space-3);
    left: 0;
    right: 0;
    text-align: center;
    font-style: italic;
  }
  .trail-hint {
    bottom: var(--space-3);
    left: var(--space-4);
    color: var(--accent-green);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  .plot-controls {
    position: absolute;
    inset-inline-end: var(--space-3);
    bottom: var(--space-3);
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: var(--space-1);
    max-width: 20rem;
  }
  .plot-controls button {
    min-width: var(--control-target);
    min-height: var(--control-target);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
    background: var(--glass-strong);
    color: var(--fg);
    cursor: pointer;
    font: inherit;
    padding: var(--space-1) var(--space-3);
  }
  .plot-controls button:active {
    transform: scale(var(--press-scale));
  }

  .section-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: var(--space-3);
    padding-inline: var(--space-3);
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: var(--data-mark-gap);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .row {
    display: grid;
    grid-template-columns: 3ch minmax(0, 1fr) 6ch;
    align-items: center;
    gap: var(--space-3);
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius);
    padding: var(--space-2) var(--space-3);
    cursor: pointer;
    text-align: start;
    color: inherit;
    font: inherit;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .row :global(.bar) {
    width: 100%;
    min-width: 0;
    max-width: 100%;
  }
  .row:hover {
    background: var(--bg-hover);
  }
  .row.active {
    background: var(--glass-strong);
    border-color: color-mix(in srgb, var(--family) 25%, var(--glass-line));
  }
  .row.active .lyr {
    color: color-mix(in srgb, var(--family) 85%, var(--fg));
  }
  .lyr {
    color: var(--fg-muted);
    text-align: end;
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }
  .val {
    color: var(--fg-dim);
    text-align: end;
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }

  @container probe-inspector (max-width: 44rem) {
    .drawer-header {
      gap: var(--space-3);
    }
    .name-row {
      flex-wrap: wrap;
      gap: var(--space-2) var(--space-3);
    }
    .body {
      flex-direction: column;
      gap: var(--space-4);
      overflow-y: auto;
    }
    .bars-col {
      flex: 0 0 auto;
      max-height: 11rem;
      padding-inline-end: 0;
    }
    .row {
      min-height: var(--control-target);
      gap: var(--space-2);
    }
    .plot-col {
      flex: 0 0 auto;
      overflow: visible;
    }
    .plot-controls {
      inset-inline-start: var(--space-3);
      max-width: none;
    }
  }

</style>
