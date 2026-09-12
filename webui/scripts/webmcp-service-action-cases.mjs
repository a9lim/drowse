import assert from "node:assert/strict";

export async function runServiceActionCases({ server, tools, validateInput, JobRegistry, installRuntimeClient, sessionState, instrument, context }) {
  const address = { namespace: "local", name: "example" };
  const node = { label: "pirate", statements: ["Ahoy"] };
  const template = { ...address, slot: "SLOT", values: ["Mon", "Tue"], contexts: [{ turns: [{ role: "user", content: "Day?" }], assistant: "It is SLOT" }], description: "Days", tags: ["calendar"] };
  const { addTransferredFile, getTransferredFile } = await server.ssrLoadModule("/src/lib/webmcp/files.ts");
  const transfer = await addTransferredFile(new Blob(["archive bytes"]), "example.drowse");
  const cases = {
    drowse_list_sessions: [{}, "sessions.list", []],
    drowse_list_profiles: [{}, "profiles.list", []],
    drowse_get_profile: [{ name: "example" }, "profiles.get", ["example"]],
    drowse_correlate_profiles: [{ names: ["example", "other"] }, "profiles.correlation", [["example", "other"]]],
    drowse_compare_profiles: [{ a: "example", b: "other" }, "profiles.pairwise", ["example", "other"]],
    drowse_extract_profile: [{ concept: "pirate", baseline: null }, "profiles.extract", [{ concept: "pirate", baseline: null }, undefined]],
    drowse_list_manifolds: [{ query: "pirate" }, "manifolds.list", []],
    drowse_get_manifold: [address, "manifolds.get", ["local", "example"]],
    drowse_search_manifolds: [{ query: "pirate", limit: 5 }, "manifolds.search", ["pirate", 5]],
    drowse_install_manifold: [{ target: "owner/repo", force: false }, "manifolds.install", [{ target: "owner/repo", force: false }]],
    drowse_create_manifold: [{ ...address, description: "Persona", domain: { type: "sphere", dim: 1 }, nodes: [{ ...node, coords: [0] }] }, "manifolds.create"],
    drowse_discover_manifold: [{ ...address, fit_mode: "pca", nodes: [node] }, "manifolds.createDiscover"],
    drowse_generate_manifold: [{ ...address, concepts: ["pirate"], samples_per_prompt: 2 }, "manifolds.generate"],
    drowse_fit_manifold: [{ ...address, layers: "workspace", force: false }, "manifolds.fit", ["local", "example", { layers: "workspace", force: false }]],
    drowse_merge_manifolds: [{ ...address, sources: [address, { namespace: "default", name: "other" }] }, "manifolds.merge"],
    drowse_delete_manifold: [address, "manifolds.delete", ["local", "example"]],
    drowse_inspect_surface: [{ points: [[0], [1]] }, "manifolds.inspectSurface", [[[0], [1]]]],
    drowse_list_manifold_packs: [{}, "manifolds.drowseArchiveList", [], "browser"],
    drowse_delete_manifold_pack: [{ primary: "local/example" }, "manifolds.drowseArchiveDelete", ["local/example"], "browser"],
    drowse_import_manifold_pack: [{ file_id: transfer.file_id, force: true }, "manifolds.drowseArchiveInstall", [getTransferredFile(transfer.file_id), { force: true }], "browser"],
    drowse_export_manifold_pack: [{ primary: "local/example" }, "manifolds.drowseArchiveExport", ["local/example"], "browser"],
    drowse_list_templates: [{}, "templates.list", []],
    drowse_get_template: [address, "templates.get", ["local", "example"]],
    drowse_create_template: [template, "templates.create"],
    drowse_update_template: [{ ...address, description: "New purpose" }, "templates.create", [{ ...template, description: "New purpose", force: true }]],
    drowse_delete_template: [address, "templates.delete", ["local", "example"]],
    drowse_score_template: [{ ...address, steering: "" }, "templates.score", ["local", "example", ""]],
    drowse_template_manifold: [{ ...address, template_ref: "local/days", fit_mode: "auto" }, "manifolds.createFromTemplate"],
    drowse_read_tree: [{ node_id: "a", include_tokens: true }, "tree.get", []],
    drowse_navigate_tree: [{ node_id: "a" }, "tree.navigate", ["a"]],
    drowse_edit_node: [{ node_id: "a", text: "Edited" }, "tree.edit", ["a", "Edited"]],
    drowse_branch_node: [{ node_id: "a", text: "Sibling", role: "user" }, "tree.branch", ["a", "Sibling", undefined, "user"]],
    drowse_delete_node: [{ node_id: "a" }, "tree.delete", ["a"]],
    drowse_star_node: [{ node_id: "a", starred: true }, "tree.star", ["a", true]],
    drowse_annotate_node: [{ node_id: "a", text: "Research note" }, "tree.note", ["a", "Research note"]],
    drowse_filter_tree: [{ expression: "starred" }, "tree.filter", ["starred"]],
    drowse_compare_nodes: [{ a_id: "a", b_id: "b" }, "tree.diff", ["a", "b"]],
    drowse_joint_logprobs: [{ a_id: "a", b_id: "b" }, "tree.jointLogprobs", ["a", "b"]],
    drowse_export_transcript: [{ node_id: "a" }, "tree.transcriptExport", ["a"]],
    drowse_load_transcript: [{ yaml: "turns: []", mode: "here", strict: false }, "tree.transcriptLoad", ["turns: []", "here", false]],
    drowse_edge_label: [{ parent_id: "root", child_id: "a" }, "tree.edgeLabel", ["root", "a"]],
    drowse_probe_geometry: [{ name: "example" }, "probes.geometry", ["example"]],
    drowse_instrument_sources: [{ family: "lens" }, "instruments.sources", ["lens"]],
    drowse_set_instrument_live: [{ family: "lens", enabled: true, layers: null }, "instruments.setLive", ["lens", { enabled: true, layers: null }]],
    drowse_use_lens_source: [{ source: "local:relp" }, "instruments.setLensSource", ["local:relp"]],
    drowse_activate_instrument: [{ family: "sae", source: "resident", layer: 0 }, "instruments.activateInstalledPack", ["sae", { source: "resident", layer: 0 }], "browser"],
    drowse_prepare_instrument: [{ family: "lens", operation: "fit", seq_len: null, prompts: 5 }, "instruments.startPreparation", ["lens", { seq_len: null, prompts: 5, operation: "fit" }]],
    drowse_preparation_status: [{ family: "lens" }, "instruments.preparationStatus", ["lens"]],
    drowse_cancel_preparation: [{ family: "lens", operation: "fit", started_at: 100 }, "instruments.cancelPreparation", ["lens"]],
    drowse_token_readout: [{ family: "lens", node_id: "a", raw_index: 0, steered: false }, "instruments.tokenReadout", ["lens", "a", 0, { steered: false }, undefined]],
    drowse_validate_lens_token: [{ word: "sail" }, "instruments.validateLensToken", ["sail"]],
    drowse_validate_sae_feature: [{ feature_id: 3 }, "instruments.validateSaeFeature", [3]],
    drowse_sae_feature_metadata: [{ ids: [0, 3] }, "instruments.saeFeaturesMetadata", [[0, 3]]],
  };
  assert.deepEqual(Object.keys(cases).sort(), tools.map(tool => tool.name).sort(), "every callable service action requires dispatch/completion and rejection coverage");
  const info = { ...sessionState.info, instruments: [instrument("geometry", "geometry"), instrument("lens", "local:relp"), instrument("sae", "resident")] };
  const tree = { rev: 1, model_id: "test-model", root_id: "root", active_node_id: "a", children_of: { root: ["a"], a: [] }, nodes: [{ id: "root", parent_id: null, role: "root", text: "" }, { id: "a", parent_id: "root", role: "assistant", text: "Hello", tokens: [{ measurements: { provenance: "captured", instruments: {} } }], recipe: { steering: "" } }] };
  const done = { state: "done", operation: "fit", started_at: 100, finished_at: 101, message: "complete", progress: { current: 5, total: 5 }, error: null, cancellable: true };
  const resultFor = (method) => ({
    "sessions.get": info, "sessions.list": { sessions: [] },
    "profiles.list": { profiles: [] }, "manifolds.list": { manifolds: [{ ...address, node_labels: ["pirate"] }] },
    "templates.list": { templates: [template] }, "templates.get": template, "tree.get": tree,
    "tree.replayCapabilities": { jointLogprobs: { available: true } },
    "instruments.sources": { sources: [] }, "instruments.startPreparation": done, "instruments.cancelPreparation": { ...done, message: "cancelled" },
    "probes.list": { probes: [] }, "manifolds.drowseArchiveList": { packs: [{ primary: "local/example" }] },
    "manifolds.drowseArchiveExport": new Blob(["exported archive"]),
  })[method] ?? { dispatched: method, provenance: "fixture-evidence" };
  const calls = [];
  const originalRuntime = Object.fromEntries(Object.entries(context.runtime).map(([key, value]) => [key, typeof value === "object" ? { ...value } : value]));
  const runtime = context.runtime;
  runtime.mode = "http";
  for (const tool of tools) for (const method of tool.services) {
    const [service, member] = method.split(".");
    runtime[service] ??= {};
    runtime[service][member] = async (...args) => { calls.push({ method, args: args.filter(value => typeof value !== "function") }); return resultFor(method); };
  }
  runtime.sessions.get = async () => info;
  runtime.probes.list = async () => ({ probes: [] });
  runtime.instruments.preparationStatus = async (...args) => { calls.push({ method: "instruments.preparationStatus", args }); return { ...done, state: "running", finished_at: null }; };
  installRuntimeClient(runtime);
  sessionState.info = info;
  try {
    for (const tool of tools) {
      const [input, method, expected = [input], mode = "http"] = cases[tool.name];
      runtime.mode = mode;
      const ctx = { ...context, session: info, runtime, jobs: new JobRegistry(), hosted: null };
      assert.equal(tool.available(ctx), null, `${tool.name} must be available with its prerequisites`);
      assert.ok(tool.available({ ...ctx, runtime: null, session: null }), `${tool.name} must be absent without a workspace`);
      calls.length = 0;
      assert.throws(() => validateInput(tool.inputSchema, { ...input, unsupported_parameter: true }), undefined, `${tool.name} rejects unknown parameters`);
      assert.equal(calls.length, 0, "rejected inputs must cause no service calls");
      validateInput(tool.inputSchema, input);
      let result = await tool.execute(input, ctx);
      if (result?.id && ctx.jobs.list().some(job => job.id === result.id)) {
        for (let attempt = 0; attempt < 30 && ["queued", "running"].includes(ctx.jobs.get(result.id).state); attempt++) await new Promise(resolve => setTimeout(resolve, 2));
        const job = ctx.jobs.get(result.id);
        assert.equal(job.state, "completed", `${tool.name} completion: ${JSON.stringify(job.error)}`);
        result = job.result;
      }
      assert.ok(result !== undefined, `${tool.name} returns a reviewable result`);
      const dispatched = calls.find(call => call.method === method);
      assert.ok(dispatched, `${tool.name} must dispatch ${method}`);
      const correlated = ["profiles.extract", "manifolds.install", "manifolds.generate", "manifolds.fit", "templates.score"].includes(method) && mode === "http";
      if (correlated) assert.ok(ctx.jobs.list().some(job => job.id === dispatched.args.at(-1)), `${tool.name} uses its own retained receipt identifier`);
      assert.deepEqual(correlated ? dispatched.args.slice(0, -1) : dispatched.args, expected, `${tool.name} preserves exact service arguments`);
      if (tool.name === "drowse_delete_node") assert.ok(calls.findIndex(call => call.method === "tree.navigate") < calls.findIndex(call => call.method === "tree.delete"), "active subtree moves before deletion");
      if (tool.name === "drowse_read_tree") assert.equal(result.node.tokens[0].measurements.provenance, "captured");
      if (tool.name === "drowse_export_manifold_pack") assert.equal(await getTransferredFile(result.file.file_id).text(), "exported archive");
      if (tool.name === "drowse_prepare_instrument") assert.equal(result.status.progress.current, 5, "job completion contains the authoritative backend status");

      const [service, member] = method.split(".");
      const original = runtime[service][member];
      const failure = Object.assign(new Error("Backend rejected this action"), { code: "BACKEND_REJECTED", status: 409, body: { reason: "fixture prerequisite" } });
      runtime[service][member] = async () => { throw failure; };
      try {
        const failingContext = { ...ctx, jobs: new JobRegistry() };
        try {
          const failed = await tool.execute(input, failingContext);
          assert.ok(failed?.id, `${tool.name} must reject a backend failure or return its failed job`);
          for (let attempt = 0; attempt < 30 && ["queued", "running"].includes(failingContext.jobs.get(failed.id).state); attempt++) await new Promise(resolve => setTimeout(resolve, 2));
          assert.equal(failingContext.jobs.get(failed.id).state, "failed", `${tool.name} never claims success after backend rejection`);
          assert.equal(failingContext.jobs.get(failed.id).error.code, "BACKEND_REJECTED");
        } catch (error) { assert.equal(error, failure, `${tool.name} preserves the original backend error and details`); }
      } finally { runtime[service][member] = original; }
      if (tool.name === "drowse_cancel_preparation") {
        const { lensFit } = await server.ssrLoadModule("/src/lib/stores/instrumentAuthoring.svelte.ts");
        await lensFit.watchTracked(done);
      }
    }

    runtime.mode = "browser";
    let rejectFit;
    let fitProgress;
    let cancelCalls = 0;
    runtime.manifolds.fit = async (_namespace, _name, _options, progress) => {
      fitProgress = progress;
      return new Promise((_resolve, reject) => { rejectFit = reject; });
    };
    const hosted = { snapshot: { modelVariantId: "fixture", lifecycle: "ready", fitting: { phase: "idle" }, generation: { phase: "idle" }, download: { phase: "idle" } }, cancelFitting: async () => { cancelCalls++; rejectFit(Object.assign(new Error("Cancelled fitting"), { code: "FITTING_CANCELLED" })); } };
    const jobs = new JobRegistry();
    const fit = await tools.find(tool => tool.name === "drowse_fit_manifold").execute(address, { ...context, runtime, hosted, session: info, jobs });
    await new Promise(resolve => setTimeout(resolve, 0));
    fitProgress({ event: "progress", data: { stage: "fit", current: 1, total: 3 } });
    const { manifoldJobs, cancelManifoldJob } = await server.ssrLoadModule("/src/lib/stores/manifoldJobs.svelte.ts");
    assert.equal(manifoldJobs.current.status, "running");
    assert.equal(manifoldJobs.current.cancellable, true);
    assert.equal(jobs.get(fit.id).cancellable, true);
    await cancelManifoldJob();
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(cancelCalls, 1, "visible cancel uses exactly this application's fitting job");
    assert.equal(jobs.get(fit.id).state, "cancelled");
    assert.equal(manifoldJobs.current.status, "cancelled");

    runtime.mode = "http";
    const { createPreparationSlice } = await server.ssrLoadModule("/src/lib/stores/preparations.svelte.ts");
    let backend = { ...done, state: "running", finished_at: null, progress: { current: 0, total: 2 } };
    let settles = 0;
    let reads = 0;
    runtime.instruments.startPreparation = async () => backend;
    runtime.instruments.preparationStatus = async () => { reads++; backend = { ...backend, progress: { current: reads, total: 2 }, ...(reads === 2 ? { state: "done", finished_at: 102 } : {}) }; return backend; };
    const slice = createPreparationSlice("lens", "fit", { label: "Test fit", intervalMs: 1, successMessage: "Done", onSettled: async () => { settles++; } });
    const observed = [];
    const tracked = slice.startTracked({}, { onProgress: status => observed.push(status.progress.current) });
    await assert.rejects(() => slice.startTracked({}), { code: "BUSY" });
    const completed = await tracked;
    assert.equal(completed.state, "done");
    assert.deepEqual(observed, [0, 1, 2]);
    assert.equal(settles, 1, "the shared UI reconciles once before tracked completion returns");
    assert.equal(slice.state.running, false);
    assert.equal(slice.state.polling, false);
    runtime.instruments.preparationStatus = async () => ({ ...backend, state: "running", started_at: 999 });
    await assert.rejects(() => slice.cancelOwned(100), { code: "PREPARATION_REPLACED" });
    const failedSlice = createPreparationSlice("lens", "fit", { label: "Test fit", intervalMs: 1, successMessage: "Done" });
    runtime.instruments.startPreparation = async () => { throw Object.assign(new Error("Preparation rejected"), { code: "PREPARATION_REJECTED" }); };
    await assert.rejects(() => failedSlice.startTracked({}), { code: "PREPARATION_REJECTED" });
    assert.equal(failedSlice.state.running, false);

    const { reconcileServiceJob } = await server.ssrLoadModule("/src/lib/webmcp/serviceTools.ts");
    const recoveryJobs = new JobRegistry();
    const abandoned = recoveryJobs.start("fit-manifold", async () => new Promise(() => {}), { cancellable: false });
    recoveryJobs.setRecovery(abandoned.id, { type: "http_operation", operation_id: abandoned.id, kind: "fit-manifold", model_id: "test-model" });
    recoveryJobs.interrupt("Page reloaded");
    let receiptReads = 0;
    runtime.sessions.operationStatus = async operationId => { receiptReads++; return { request_id: operationId, path: "/drowse/v1/manifolds/local/example/fit", model_id: "test-model", state: "completed", progress: ["done"], result: { fitted: true }, result_truncated: false }; };
    const callsBeforeRecovery = calls.length;
    const reconciled = await reconcileServiceJob(abandoned.id, { ...context, runtime, session: info, jobs: recoveryJobs });
    assert.equal(reconciled.state, "completed");
    assert.equal(reconciled.result.fitted, true);
    assert.equal(receiptReads, 1);
    assert.ok(calls.slice(callsBeforeRecovery).every(call => ["manifolds.list", "profiles.list"].includes(call.method)), "receipt recovery only refreshes authoritative reads, never restarts fitting");
    const missing = recoveryJobs.start("fit-manifold", async () => new Promise(() => {}), { cancellable: false });
    recoveryJobs.setRecovery(missing.id, { type: "http_operation", operation_id: missing.id, kind: "fit-manifold", model_id: "test-model" });
    recoveryJobs.interrupt("Server disconnected");
    runtime.sessions.operationStatus = async () => { throw Object.assign(new Error("Receipt unavailable after server restart"), { status: 404 }); };
    await assert.rejects(() => reconcileServiceJob(missing.id, { ...context, runtime, jobs: recoveryJobs }), { status: 404 });
    assert.equal(recoveryJobs.get(missing.id).state, "interrupted", "missing receipts never imply successful completion or permit automatic replay");

    const { HttpRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/http-client.ts");
    const http = new HttpRuntimeClient();
    const previousFetch = globalThis.fetch;
    const requests = [];
    globalThis.fetch = async (path, init = {}) => {
      const headers = new Headers(init.headers);
      requests.push({ path, headers, body: init.body ? JSON.parse(init.body) : null });
      const result = { canonical: "pirate", profile: {}, fitted: true };
      return headers.get("accept") === "text/event-stream"
        ? new Response(`event: done\ndata: ${JSON.stringify(result)}\n\n`)
        : new Response(JSON.stringify(result), { headers: { "Content-Type": "application/json" } });
    };
    try {
      await http.manifolds.fit("local", "example", {}, () => {}, "fit-receipt");
      await http.manifolds.generate({ name: "example", concepts: ["pirate"] }, () => {}, "generate-receipt");
      await http.manifolds.install({ target: "owner/repo" }, () => {}, "install-receipt");
      await http.profiles.extract({ concept: "pirate" }, () => {}, undefined, "extract-receipt");
      await http.templates.score("local", "days", "", "score-receipt");
      await http.sessions.operationStatus("fit-receipt");
      assert.deepEqual(requests.slice(0, 5).map(request => request.headers.get("x-drowse-request-id")), ["fit-receipt", "generate-receipt", "install-receipt", "extract-receipt", "score-receipt"]);
      assert.ok(requests.slice(0, 5).every(request => request.headers.get("content-type") === "application/json"));
      assert.equal(requests[4].body.steering, "");
      assert.equal(requests[5].path, "/drowse/v1/operations/fit-receipt");
    } finally { globalThis.fetch = previousFetch; }
  } finally { for (const [key, value] of Object.entries(originalRuntime)) { if (typeof value === "object") Object.assign(runtime[key], value); else runtime[key] = value; } }
}
