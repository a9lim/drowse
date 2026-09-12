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
assert.match(source, /mounted && drawerState\.open === originDrawer && drawerState\.params === origin/);

const attach = source.slice(source.indexOf("  async function onCustomAttachSubmit("), source.indexOf("  // ----- fit / delete"));
const probe = source.slice(source.indexOf("  async function onProbe("), source.indexOf("  // ----- custom-attach form"));
const attachCode = ts.transpileModule(`${attach}\n${probe}\nreturn { onCustomAttachSubmit, onProbe };`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
}).outputText;
for (const surface of ["custom", "catalog"]) {
  for (const outcome of ["closed", "closing", "replaced", "switched", "token", "rack"]) {
    const origin = null;
    const returnTarget = outcome === "rack" ? null : { node_id: "original-token" };
    let resolveAttach;
    let finishes = 0;
    const attachState = {
      customAttaching: false,
      customSelector: "local/test",
      customAlias: "",
      customTopN: 3,
      browserMode: false,
      mounted: true,
      drawerState: { open: "subspace", params: origin },
      busyKeys: new Set(),
      rowKey: () => "local/test",
      isProbed: () => false,
      selectorChoice: () => ({ selector: "local/test", available: true }),
      attachProbe: () => new Promise(resolve => { resolveAttach = resolve; }),
      pushToast: () => {},
      describeError: error => { throw error; },
      finishProbeSetup: target => {
        assert.equal(target, returnTarget, "attachment returns to the original token");
        finishes += 1;
      },
      get returnToToken() {
        assert.ok(this.mounted, "closed drawers must not read destroyed derived state");
        assert.equal(this.drawerState.open, "subspace", "closing drawers must not read inert derived state");
        assert.equal(this.drawerState.params, origin, "replaced drawers must not read stale derived state");
        return returnTarget;
      },
    };
    const handlers = new Function("state", `with (state) { ${attachCode} }`)(attachState);
    const pending = surface === "custom"
      ? handlers.onCustomAttachSubmit({ preventDefault() {} })
      : handlers.onProbe(model);
    if (outcome === "closed") attachState.mounted = false;
    if (outcome === "closing") attachState.drawerState.open = null;
    if (outcome === "replaced") attachState.drawerState.params = {};
    if (outcome === "switched") attachState.drawerState.open = "manifolds";
    resolveAttach({ name: "local/test" });
    await pending;
    assert.equal(finishes, outcome === "token" || surface === "catalog" && outcome === "rack" ? 1 : 0);
    assert.equal(attachState.customAttaching, false);
    assert.equal(attachState.busyKeys.size, 0);
  }
}
console.log("Rack fitting: duplicate prevention, cancellation, metadata refresh, stale inspector responses, failure cleanup, and 12 delayed-attachment cases passed");
