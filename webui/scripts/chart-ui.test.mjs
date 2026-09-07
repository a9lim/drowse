import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { barExtent, barTooltip, chartValue } = await server.ssrLoadModule("/src/lib/charts/chartValues.ts");
  assert.equal(barExtent(-0.5, 1), 0.5);
  assert.equal(barExtent(2, 1), 1);
  for (const [value, max] of [[NaN, 1], [Infinity, 1], [1, 0], [1, -1], [1, Infinity]]) {
    assert.equal(barExtent(value, max), 0);
  }
  assert.equal(barTooltip(-0.5, 1), "-0.5 · 50% of scale (1)");
  assert.equal(barTooltip(2, 1), "2 · 200% of scale (1)", "tooltip preserves overflow despite clipped fill");
  assert.equal(barTooltip(0.125, 1, true), "12.5%");
  assert.equal(barTooltip(NaN, 1), "Not available");
  assert.equal(barTooltip(1, 0), "1 · scale unavailable");
  assert.equal(chartValue(0.000001, true), "<0.01%");

  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: Bar } = await server.ssrLoadModule("/src/lib/charts/Bar.svelte");
  const negative = render(Bar, { props: { value: -0.5, max: 1, bipolar: true } }).body;
  assert.match(negative, /left: 25%; width: 25%/);
  assert.match(negative, /title="-0.5 · 50% of scale \(1\)"/);
  const invalid = render(Bar, { props: { value: NaN, max: 1 } }).body;
  assert.doesNotMatch(invalid, /NaN/);
  const { default: Sparkline } = await server.ssrLoadModule("/src/lib/charts/Sparkline.svelte");
  const spark = render(Sparkline, { props: { points: [0.1, null, NaN, 0.25], percentage: true } }).body;
  assert.match(spark, /Latest 25% · range 10% to 25% · 2 readings/);
  assert.match(spark, /stroke-linecap="round"/);
  assert.match(render(Sparkline, { props: { points: [] } }).body, /No readings yet/);
  const { default: MiniMap } = await server.ssrLoadModule("/src/panels/manifold/ManifoldMiniMap.svelte");
  const mapProps = {
    info: { name: "test", domain: { type: "box", axes: [{ name: "x", lo: -1, hi: 1 }, { name: "y", lo: -1, hi: 1 }] },
      node_labels: ["node"], node_coords: [[0.25, -0.5]] },
    trajectory: [[0.1, 0.2]], settled: null,
  };
  assert.match(render(MiniMap, { props: mapProps }).body, /node · x 0.25 · y -0.5/);
  assert.match(render(MiniMap, { props: mapProps }).body, /Live coordinates \[0.1, 0.2\]/);
  assert.match(render(MiniMap, { props: { ...mapProps, settled: [0.3, 0.4] } }).body, /Final coordinates \[0.3, 0.4\]/);

  const source = await readFile(new URL("../src/drawers/ProbeInspectorDrawer.svelte", import.meta.url), "utf8");
  assert.match(source, /new ResizeObserver\(redraw\)/);
  assert.match(source, /observer\.observe\(canvas\)/);
  assert.match(source, /observer\.disconnect\(\)/);
  assert.match(source, /@container probe-inspector/);
  assert.match(source, /aspect-ratio: 1;/);
  assert.match(source, /grid-template-columns: 3ch minmax\(0, 1fr\) 6ch;/);
  assert.match(source, /aria-pressed=\{l.layer === selectedLayer\}/);
  const barSource = await readFile(new URL("../src/lib/charts/Bar.svelte", import.meta.url), "utf8");
  assert.match(barSource, /border-radius: var\(--radius-pill\);/);
  assert.match(barSource, /border-radius: inherit;/);
  assert.doesNotMatch(barSource, /preserveAspectRatio/);

  const { renderProbeGeometry, DEFAULT_ORBIT_QUAT } = await server.ssrLoadModule("/src/lib/charts/probeGeometry.ts");
  const textCalls = [];
  const ctx = new Proxy({
    measureText: (text) => ({ width: text.length * 6 }),
    fillText: (text, x, y) => textCalls.push({ text, x, y }),
  }, { get: (target, key) => target[key] ?? (() => {}) });
  let width = 240, height = 600;
  const canvas = { getContext: () => ctx, getBoundingClientRect: () => ({ width, height }) };
  const oldWindow = globalThis.window, oldStyle = globalThis.getComputedStyle;
  globalThis.window = { devicePixelRatio: 2 };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "#888" });
  try {
    const input = {
      geom: { rank: 2, node_white: [[1, 0], [0, 1]], neutral_white: [0, 0], overlay: null },
      nodeLabels: ["right_edge_name", "y"], live: null, trail: [], orbit: { q: DEFAULT_ORBIT_QUAT, zoom: 1 },
    };
    for (const size of [[240, 600], [600, 240], [400, 400]]) {
      [width, height] = size;
      const hits = renderProbeGeometry(canvas, input);
      const label = textCalls.findLast((call) => call.text === "right_edge_name");
      assert.ok(label.x >= 8 && label.x + label.text.length * 6 <= width - 8,
        "node labels move inward before they clip against the chart edge");
      assert.equal(canvas.width, width * 2);
      assert.equal(canvas.height, height * 2);
      const [neutral, x, y] = hits;
      assert.ok(Math.abs((x.screen[0] - neutral.screen[0]) - (neutral.screen[1] - y.screen[1])) < 1e-9,
        "equal data distances have equal pixel lengths at every aspect ratio");
      assert.deepEqual(x.point, [1, 0]);
    }
    for (const rank of [1, 3]) {
      input.geom = { ...input.geom, rank, pca_rotation: [[1, 0, 0], [0, 1, 0]] };
      assert.equal(renderProbeGeometry(canvas, input).length, 3, `rank ${rank} exposes neutral and both nodes`);
    }
  } finally {
    if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow;
    if (oldStyle === undefined) delete globalThis.getComputedStyle; else globalThis.getComputedStyle = oldStyle;
  }
  console.log("Chart values, rounded bars, responsive geometry, and projected hover coordinates passed");
} finally {
  await server.close();
}
