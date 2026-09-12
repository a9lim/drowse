import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import puppeteer from "puppeteer";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ configFile: `${root}/vite.hosted.config.ts`, logLevel: "error", server: { watch: null, host: "127.0.0.1", port: 0 } });
await server.listen();
const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
const browser = await puppeteer.launch({ headless: true, args: ["--enable-features=WebMCP"] });
const pause = () => new Promise(resolve => setTimeout(resolve, 30));
async function until(predicate, message, timeout = 20000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { const value = await predicate(); if (value) return value; await pause(); }
  throw Error(message);
}
async function tool(page, name) {
  try { return await until(() => page.webmcp.tools().find(tool => tool.name === name), `Tool not registered: ${name}`); }
  catch (error) {
    console.error({ url: page.url(), tools: page.webmcp.tools().map(tool => tool.name), body: (await page.$eval("body", element => element.innerText)).slice(0,2000) });
    const state = page.webmcp.tools().find(tool => tool.name === "drowse_get_state");
    if (state) console.error(JSON.stringify((await state.execute({})).output));
    throw error;
  }
}
async function invoke(page, name, input = {}, success = true) {
  const response = await (await tool(page, name)).execute(input);
  assert.equal(response.status, "Completed", `${name}: ${response.errorText}`);
  let output = response.output;
  if (output?.result_id) {
    let text = "", offset = 0;
    do {
      const chunk = await invoke(page, "drowse_read_result", { result_id: output.result_id, offset, limit: 8000 });
      text += chunk.data.text; offset = chunk.data.next_offset;
    } while (offset !== null);
    output = JSON.parse(text);
  }
  assert.equal(output?.ok, success, `${name}: ${JSON.stringify(output)}`);
  return output;
}
async function group(page, name) { await invoke(page, "drowse_select_tool_group", { group: name }); }
async function finished(page, id) {
  return until(async () => {
    const { data } = await invoke(page, "drowse_get_job", { job_id: id });
    return ["completed", "cancelled", "failed", "interrupted"].includes(data.state) ? data : null;
  }, `Job did not finalize: ${id}`, 30000);
}
try {
  const version = await browser.version();
  assert.equal(version, "Chrome/152.0.7977.75", "Use the browser bundled with pinned Puppeteer 25.10.0 for the native lane");
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.goto(`${origin}/app?layoutFixture=1`);
  await tool(page, "drowse_start_generation");
  const initial = await invoke(page, "drowse_get_state");
  assert.equal(initial.data.workspace.is_base_model, false);
  assert.deepEqual(initial.data.webmcp.registration_errors, []);
  const guide = await invoke(page, "drowse_explain_control", { control: "pirate" });
  assert.match(guide.data.when, /start with prompting/i);
  const instructions = "Answer accurately. Speak in a playful pirate voice.";
  const instructionResult = await invoke(page, "drowse_set_system_prompt", { text: instructions, request_id: "pirate-instructions" });
  assert.notEqual(instructionResult.revision, "/app:public", "Editing settings must not remount the tool registry");
  assert.deepEqual(await invoke(page, "drowse_set_system_prompt", { text: instructions, request_id: "pirate-instructions" }), instructionResult);
  const current = await invoke(page, "drowse_read_workspace");
  assert.equal(current.data.sampling.system_prompt, instructions);
  assert.equal(current.data.steering.expression, "");
  await invoke(page, "drowse_set_sampling", { temperature: 0.8, frequency_penalty: 0.3 });
  await invoke(page, "drowse_set_sampling", { temperature: 0.7, expected_revision: initial.revision }, false);
  await invoke(page, "drowse_set_role_labels", { assistant: "pirate" }, false);
  await invoke(page, "drowse_set_composer", { text: "Explain why the sky is blue." });
  assert.equal(await page.$eval("textarea", element => element.value), "Explain why the sky is blue.");
  const draftRevision = (await invoke(page, "drowse_get_state")).revision;
  await page.$eval("textarea", element => { element.value = "A newer human draft"; element.dispatchEvent(new Event("input", { bubbles: true })); });
  const rejected = await invoke(page, "drowse_set_composer", { text: "Outdated agent draft", expected_revision: draftRevision }, false);
  assert.equal(rejected.error.code, "stale_state");
  assert.equal(await page.$eval("textarea", element => element.value), "A newer human draft");
  await invoke(page, "drowse_set_composer", { text: "Explain why the sky is blue." });
  const receipt = await invoke(page, "drowse_start_generation", { use_composer: true, n: 2, sampling: { seed: 42 }, request_id: "two-siblings" });
  const completed = await finished(page, receipt.data.id);
  assert.equal(completed.state, "completed", JSON.stringify(completed));
  assert.deepEqual(await invoke(page, "drowse_start_generation", { use_composer: true, n: 2, sampling: { seed: 42 }, request_id: "two-siblings" }), receipt);
  await group(page, "saved");
  const saved = await invoke(page, "drowse_save_chat", { name: "Pirate setup" });
  await invoke(page, "drowse_update_chat", { id: saved.data.chat.id, name: "Captain" });
  assert.equal((await invoke(page, "drowse_read_workspace")).data.sampling.system_prompt, instructions);
  await group(page, "steering");
  await invoke(page, "drowse_set_steering", { expression: "0.5 ???" }, false);
  await group(page, "artifacts");
  await invoke(page, "drowse_get_manifold", { namespace: "absent", name: "pirate" }, false);
  const catalogue = await invoke(page, "drowse_list_actions");
  for (const name of catalogue.data.groups) {
    await group(page, name);
    assert.ok(page.webmcp.tools().length <= 32, `${name}: tool count exceeds bounded discovery budget`);
    const state = await invoke(page, "drowse_get_state");
    assert.deepEqual(state.data.webmcp.registration_errors, [], `${name}: native schema registration must succeed`);
  }
  await group(page, "interface");
  await invoke(page, "drowse_set_loom_view", { filter: { expression: "pirate", mode: "text" }, sort: "surprise", comparison_visible: false });
  await invoke(page, "drowse_set_analysis_view", { target: "lens", fields: { sort: "depth" } });
  await group(page, "files");
  const file = await invoke(page, "drowse_begin_file", { name: "native.txt", size: 12, mime_type: "text/plain" });
  await invoke(page, "drowse_write_file", { file_id: file.data.file_id, offset: 0, base64: Buffer.from("native bytes").toString("base64") });
  await invoke(page, "drowse_finish_file", { file_id: file.data.file_id });
  assert.equal(Buffer.from((await invoke(page, "drowse_read_file", { file_id: file.data.file_id, offset: 0 })).data.base64, "base64").toString(), "native bytes");
  await invoke(page, "drowse_delete_file", { file_id: file.data.file_id });
  await page.reload();
  assert.equal((await invoke(page, "drowse_get_job", { job_id: receipt.data.id })).data.state, "completed", "Completed jobs survive a real native-browser reload");
  await page.goto(`${origin}/credits`);
  await tool(page, "drowse_about");
  assert.equal(page.webmcp.tools().some(tool => tool.name === "drowse_start_generation"), false);
  assert.equal((await invoke(page, "drowse_get_state")).data.workspace, null);

  const base = await browser.newPage();
  await base.goto(`${origin}/app?layoutFixture=base`);
  await tool(base, "drowse_start_generation");
  assert.equal((await invoke(base, "drowse_get_state")).data.workspace.is_base_model, true);
  await invoke(base, "drowse_set_system_prompt", { text: "Speak like a pirate" }, false);
  await invoke(base, "drowse_start_generation", { text: "Ahoy" }, false);
  await group(base, "raw");
  await invoke(base, "drowse_set_raw_buffer", { action: "edit", text: "A pirate explains the sea: Ahoy," });
  await invoke(base, "drowse_set_raw_buffer", { action: "save" });
  const raw = await invoke(base, "drowse_raw_continue", { sampling: { seed: 42, max_tokens: 16 } });
  assert.equal((await finished(base, raw.data.id)).state, "completed");
  const onboardingContext = await browser.createBrowserContext();
  const onboarding = await onboardingContext.newPage();
  onboarding.on("dialog", dialog => dialog.accept());
  await onboarding.goto(`${origin}/app?layoutFixture=setup&fixtureInstruments=1`);
  await tool(onboarding, "drowse_select_tool_group");
  await group(onboarding, "models");
  const candidates = await until(async () => {
    const { data } = await invoke(onboarding, "drowse_list_models");
    return data.phase === "supported" && data.models.length ? data.models : null;
  }, "Fixture model catalogue did not load");
  const selected = candidates.find(model => model.fit !== "blocked");
  assert.ok(selected, JSON.stringify(candidates));
  assert.equal((await finished(onboarding, (await invoke(onboarding, "drowse_download_model", { model_id: selected.id })).data.id)).state, "completed");
  assert.equal((await finished(onboarding, (await invoke(onboarding, "drowse_load_model", { model_id: selected.id })).data.id)).state, "completed");
  await tool(onboarding, "drowse_read_workspace");
  await invoke(onboarding, "drowse_open_page", { page: "contact" });
  await until(() => new URL(onboarding.url()).pathname === "/contact", "Leaving a loaded model must reach contact, not trigger a competing onboarding reload");
  await tool(onboarding, "drowse_read_contact");
  assert.equal(page.webmcp.tools().some(tool => tool.name === "drowse_start_generation"), false);
  assert.deepEqual(pageErrors, []);
  console.log(`${version}: native discovery, all groups/schemas, lifetime cleanup, pirate instruction, cosmetic naming, sampling, stale-state/idempotent calls, sibling completion and base/raw workflow passed (deterministic runtime fixtures, not model-behavior evidence)`);
} finally { await browser.close(); await server.close(); }

const unsupportedServer = await createServer({ configFile: `${root}/vite.hosted.config.ts`, logLevel: "error", server: { watch: null, host: "127.0.0.1", port: 0 } });
await unsupportedServer.listen();
const unsupportedBrowser = await puppeteer.launch({ headless: true, args: ["--disable-features=WebMCP"] });
try {
  const page = await unsupportedBrowser.newPage();
  await page.goto(`http://127.0.0.1:${unsupportedServer.httpServer.address().port}/app?layoutFixture=1`);
  await page.waitForSelector("textarea");
  assert.equal(page.webmcp.tools().length, 0);
  console.log("Unsupported native browser: normal workspace startup passed");
} finally { await unsupportedBrowser.close(); await unsupportedServer.close(); }
