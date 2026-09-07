import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile("src/drawers/RackDrawer.svelte", "utf8");
const fit = source.slice(source.indexOf("  function onFit("), source.indexOf("  function onDeleteClick("));
const inspect = source.slice(source.indexOf("  async function toggleInspect("), source.indexOf("</script>"));
const { outputText } = ts.transpileModule(`${fit}\n${inspect}\nreturn { onFit, toggleInspect };`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
});
const requests = [];
const details = [];
const toasts = [];
const refreshed = [];
const state = {
  fittingAvailability: { available: true },
  busyKeys: new Set(), errorMsg: null, mounted: true,
  detailCache: new Map(), detailErrors: new Map(), detailRevisions: new Map(),
  detailLoading: new Set(), inspectKeys: new Set(),
  rowKey: m => `${m.namespace}/${m.name}`,
  pushToast: (message, options) => { toasts.push({ message, ...options }); return toasts.length; },
  dismissToast: () => {}, updateToast: () => {},
  describeError: error => error.message,
  isFittingCancellation: error => error.code === "FITTING_CANCELLED",
  apiManifoldFitStream: () => new Promise((resolve, reject) => requests.push({ resolve, reject })),
  apiManifolds: { get: () => new Promise((resolve, reject) => details.push({ resolve, reject })) },
  refreshManifoldList: async () => refreshed.push("manifold"),
  refreshProbeList: async () => refreshed.push("probe"),
  refreshVectorList: async () => refreshed.push("vector"),
};
const { onFit, toggleInspect } = new Function(...Object.keys(state), outputText)(...Object.values(state));
const flush = () => new Promise(resolve => setImmediate(resolve));
const model = { namespace: "local", name: "test" };
onFit(model);
onFit(model);
assert.equal(requests.length, 1, "duplicate fit clicks are ignored");
requests[0].reject({ code: "FITTING_CANCELLED" });
await flush();
assert.equal(state.busyKeys.size, 0);
assert.match(toasts.at(-1).message, /cancelled/);
assert.equal(toasts.at(-1).kind, "info");

const stale = toggleInspect(model);
onFit(model);
requests[1].resolve({});
await flush();
assert.deepEqual(refreshed, ["manifold", "probe", "vector"]);
assert.equal(details.length, 2, "an open inspector fetches the new fit diagnostics");
details[1].resolve({ revision: "new" });
await flush();
details[0].resolve({ revision: "old" });
await stale;
assert.deepEqual(state.detailCache.get("local/test"), { revision: "new" },
  "late pre-fit inspector responses cannot overwrite new diagnostics");
assert.equal(state.busyKeys.size, 0);
onFit(model);
requests[2].reject(new Error("fit failed"));
await flush();
assert.equal(toasts.at(-1).kind, "error");
assert.equal(state.busyKeys.size, 0);
assert.match(source, /mounted && drawerState\.params === origin/);
console.log("Rack fitting: duplicate prevention, cancellation, metadata refresh, stale inspector responses, and failure cleanup passed");
