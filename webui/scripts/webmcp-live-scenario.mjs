import assert from "node:assert/strict";
import { createServer } from "node:net";
import puppeteer from "puppeteer";

export async function nativeEvidenceBrowserOptions() {
  const socket = createServer();
  await new Promise(resolve => socket.listen(0, "127.0.0.1", resolve));
  const port = socket.address().port;
  await new Promise(resolve => socket.close(resolve));
  return { port, executablePath: await puppeteer.executablePath(), args: ["--enable-features=WebMCP", `--remote-debugging-port=${port}`] };
}

export async function runConnectedNativeEvidence(port, url, options) {
  const browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}` });
  try {
    const page = (await browser.pages()).find(page => page.url() === url);
    assert.ok(page, "The real workbench must have an existing native browser target");
    return await runNativeEvidence(page, options);
  } finally { await browser.disconnect(); }
}

export async function readConnectedNativeState(port, url) {
  const browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}` });
  try {
    const page = (await browser.pages()).find(page => page.url() === url);
    assert.ok(page, "The real workbench must have an existing native browser target");
    const { workspace } = await invoke(page, "drowse_get_state");
    if (!workspace) return { workspace: null };
    await group(page, "conversation");
    const tree = await invoke(page, "drowse_read_tree", { node_id: workspace.active_node_id });
    return { model_id: workspace.model_id, sampling: workspace.sampling, steering: workspace.steering,
      probes: workspace.probes, generation: workspace.generation, node: tree.node };
  } finally { await browser.disconnect(); }
}

export async function runNativeEvidence(page, { runtime, control = null, disconnect = false }) {
  const evidence = { runtime, native: true, real_model: true, browser: await page.browser().version(),
    scenarios: [], observations: [], started_at: new Date().toISOString() };
  const initial = await until(async () => {
    const state = await invoke(page, "drowse_get_state");
    return state.workspace ? state : null;
  }, "A real model workspace did not become ready", 120000);
  assert.equal(initial.runtime, runtime);
  assert.ok(initial.workspace, "A real model workspace is required");
  evidence.model_id = initial.workspace.model_id;
  evidence.model_capabilities = initial.workspace.role_capabilities;
  evidence.source = initial.onboarding?.runtime ?? null;
  const before = await invoke(page, "drowse_read_workspace");
  const promptReference = await invoke(page, "drowse_explain_control", { control: "system_prompt" });
  const defaultSystemPrompt = promptReference.session_defaults.system_prompt;
  await group(page, "chat");
  await invoke(page, "drowse_set_sampling", { temperature: 0, max_tokens: 32, seed: 1729, return_top_k: 3 });

  if (initial.workspace.is_base_model) {
    await invoke(page, "drowse_set_system_prompt", { text: "Speak like a pirate" }, false);
    await group(page, "raw");
    await invoke(page, "drowse_set_raw_buffer", { action: "edit", text: "Question: Why do sailors use maps?\nA pirate answers: Ahoy," });
    await invoke(page, "drowse_set_raw_buffer", { action: "save" });
    const job = await invoke(page, "drowse_raw_continue", { sampling: { seed: 1729, max_tokens: 32 }, request_id: "live-base-continuation" });
    const result = await terminal(page, job.id);
    assert.equal(result.state, "completed");
    assert.ok(result.result.effective.every(item => item.recipe.system_prompt === null), "Raw branches must record that chat instructions were bypassed");
    evidence.scenarios.push({ intent: "base model pirate continuation", tools: ["drowse_set_raw_buffer", "drowse_raw_continue"], result });
  } else {
    const prompt = "Explain why sailors use maps in one sentence.";
    const conditions = [{ label: "baseline", steering: "", system_prompt: before.sampling.system_prompt || null },
      { label: "pirate instructions", steering: "", system_prompt: [before.sampling.system_prompt, "Speak like a friendly pirate. Keep the answer useful and clear."].filter(Boolean).join("\n\n") }];
    if (before.role_capabilities.assistant) conditions.push({ label: "pirate role header", steering: "", sampling: { assistant_role: "pirate" } });
    if (control) conditions.push({ label: "activation steering", steering: `0.01 ${control}` });
    await group(page, "analysis");
    const admission = await invoke(page, "drowse_compare_generations", { text: prompt, conditions,
      sampling: { temperature: 0, max_tokens: 32, seed: 1729 }, request_id: "live-matched-conditions" });
    const result = await terminal(page, admission.id, 300000);
    assert.equal(result.state, "completed", JSON.stringify(result));
    assert.equal(result.result.comparisons.length, conditions.length);
    assert.equal((await invoke(page, "drowse_read_workspace")).sampling.system_prompt, before.sampling.system_prompt);
    evidence.scenarios.push({ intent: "compare prompting, role substitution and steering", tools: ["drowse_compare_generations"], conditions, result });
    await group(page, "conversation");
    for (const comparison of result.result.comparisons) {
      for (const id of comparison.node_ids ?? []) {
        const row = await invoke(page, "drowse_read_tree", { node_id: id });
        const condition = conditions.find(item => item.label === comparison.label);
        const expectedPrompt = Object.hasOwn(condition, "system_prompt") ? condition.system_prompt : defaultSystemPrompt;
        assert.equal(row.node.recipe.system_prompt, expectedPrompt, `${comparison.label}: the branch must retain its effective prompt for replay`);
        evidence.observations.push({ condition: comparison.label, node: row.node });
      }
    }
    await group(page, "chat");
    await invoke(page, "drowse_set_sampling", { temperature: 0.9, frequency_penalty: 0.3 });
    const adjusted = await invoke(page, "drowse_read_workspace");
    assert.equal(adjusted.sampling.temperature, 0.9);
    assert.equal(adjusted.sampling.frequency_penalty, 0.3);
    evidence.scenarios.push({ intent: "more random and less repetitive", tools: ["drowse_set_sampling"], settings: adjusted.sampling });
    await group(page, "saved");
    const saved = await invoke(page, "drowse_save_chat", { name: "WebMCP live setup" });
    await invoke(page, "drowse_update_chat", { id: saved.chat.id, name: "Captain" });
    assert.equal((await invoke(page, "drowse_read_workspace")).sampling.system_prompt, before.sampling.system_prompt);
    evidence.scenarios.push({ intent: "save setup and cosmetic naming", tools: ["drowse_save_chat", "drowse_update_chat"], chat_id: saved.chat.id });
    await group(page, "artifacts");
    const missing = await invoke(page, "drowse_list_manifolds", { query: "not-installed-webmcp-control" });
    assert.equal(missing.items.length, 0);
    evidence.scenarios.push({ intent: "missing control", tools: ["drowse_list_manifolds"], creates_artifact: false });

    await group(page, "chat");
    const cancellation = await invoke(page, "drowse_start_generation", { text: "Count from one to five hundred, spelling out every number.", n: 2, sampling: { max_tokens: 512, seed: 99 }, request_id: "live-cancellation" });
    await until(async () => {
      const job = await invoke(page, "drowse_get_job", { job_id: cancellation.id });
      return ["started", "generation_progress"].includes(job.progress?.type) || job.progress?.phase === "generation";
    }, "Real generation never started before cancellation");
    await invoke(page, "drowse_cancel_job", { job_id: cancellation.id });
    const cancelled = await terminal(page, cancellation.id);
    assert.equal(cancelled.state, "cancelled", JSON.stringify(cancelled));
    evidence.scenarios.push({ intent: "cancel owned real generation", tools: ["drowse_start_generation", "drowse_cancel_job"], result: cancelled });

    if (disconnect) {
      const interrupted = await invoke(page, "drowse_start_generation", { text: "List all the numbers from one to five hundred.", sampling: { max_tokens: 512, seed: 101 }, request_id: "live-network-recovery" });
      await until(async () => {
        const job = await invoke(page, "drowse_reconcile_job", { job_id: interrupted.id });
        return ["started", "generation_progress"].includes(job.progress?.type) || job.progress?.phase === "generation";
      }, "Real generation never started before disconnection");
      await page.setOfflineMode(true);
      await new Promise(resolve => setTimeout(resolve, 250));
      await page.setOfflineMode(false);
      await page.reload({ waitUntil: "domcontentloaded" });
      const reconciled = await until(async () => {
        const state = await invoke(page, "drowse_get_state");
        if (!state.workspace) return null;
        const job = await invoke(page, "drowse_reconcile_job", { job_id: interrupted.id });
        return ["completed", "cancelled", "failed"].includes(job.state) ? job : null;
      }, "Reconnect reconciliation did not settle", 120000);
      evidence.scenarios.push({ intent: "network loss, reload and reconciliation", tools: ["drowse_reconcile_job"], result: reconciled });
    } else {
      await page.reload({ waitUntil: "domcontentloaded" });
      const recovered = await invoke(page, "drowse_get_job", { job_id: admission.id });
      assert.equal(recovered.state, "completed");
      evidence.scenarios.push({ intent: "recover completed operation after reload", tools: ["drowse_get_job"], job_id: recovered.id, state: recovered.state });
    }
  }
  evidence.finished_at = new Date().toISOString();
  return evidence;
}

export async function until(predicate, message, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const result = await predicate();
    if (result) return result;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw Error(message);
}

export async function invoke(page, name, input = {}, success = true) {
  const tool = await until(() => page.webmcp.tools().find(tool => tool.name === name), `Missing native tool ${name}`);
  const execution = await tool.execute(input);
  assert.equal(execution.status, "Completed", `${name}: ${execution.errorText}`);
  let output = execution.output;
  if (output?.result_id) {
    let text = "", offset = 0;
    do {
      const piece = await invoke(page, "drowse_read_result", { result_id: output.result_id, offset, limit: 8000 });
      text += piece.text; offset = piece.next_offset;
    } while (offset !== null);
    output = JSON.parse(text);
  }
  assert.equal(output?.ok, success, `${name}: ${JSON.stringify(output)}`);
  return success ? output.data : output.error;
}

async function group(page, name) { await invoke(page, "drowse_select_tool_group", { group: name }); }
async function terminal(page, id, timeout = 120000) {
  return until(async () => {
    const job = await invoke(page, "drowse_get_job", { job_id: id });
    return ["completed", "cancelled", "failed", "interrupted"].includes(job.state) ? job : null;
  }, `Job ${id} never finalized`, timeout);
}
