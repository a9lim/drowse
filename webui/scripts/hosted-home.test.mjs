import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: HostedHome } = await server.ssrLoadModule("/src/hosted/ui/HostedHome.svelte");
  const model = {
    id: "gemma", modelId: "google/gemma", tier: "fastest", name: "Gemma 3 1B", sourceUrl: "", license: "", size: "1 GB",
    context: "2K", contextTokens: 2048, fit: "recommended", installed: true, reason: "", language: "", modelDownloadBytes: 1,
    firstRunBytes: 1, remainingDownloadBytes: 0, firstRunPacks: [], setupComplete: true,
  };
  const snapshot = (phase, models) => ({
    phase, headline: "", detail: "", checks: [], models,
    download: { available: true, phase: "idle", reason: "" },
    runtime: { available: phase === "supported", phase: "unloaded", reason: "" },
  });
  const controller = { current: () => snapshot("supported", [model]) };
  const props = (state) => ({ controller, snapshot: state, onChooseModels() {}, onChats() {} });

  const checking = render(HostedHome, { props: props(snapshot("checking", [{ ...model, setupComplete: false, installed: false }])) }).body;
  assert.match(checking, /Checking this device…/, "the chats page renders before the device check finishes");
  assert.doesNotMatch(checking, /needs to be downloaded again|no longer available/, "no recovery notice while install state is unknown");

  const ready = render(HostedHome, { props: props(snapshot("supported", [model])) }).body;
  assert.doesNotMatch(ready, /Checking this device/);
  assert.match(ready, />New chat</, "the primary action says New chat");
  assert.doesNotMatch(ready, /New Instance/);
  assert.match(ready, /Continue with Gemma 3 1B/, "Continue names the model it opens");

  const empty = render(HostedHome, { props: props(snapshot("supported", [{ ...model, setupComplete: false, installed: false }])) }).body;
  assert.match(empty, /Download a model/);

  const [home, root, prompt] = await Promise.all([
    readFile("src/hosted/ui/HostedHome.svelte", "utf8"),
    readFile("src/hosted/ui/HostedRoot.svelte", "utf8"),
    readFile("src/hosted/ui/PwaUpdatePrompt.svelte", "utf8"),
  ]);
  assert.match(home, /No saved chats yet\. Start a chat with a model and it autosaves here\./);
  assert.match(home, /No chats yet\. Choose a model to start\./);
  assert.match(home, /<input[^>]*type="file"[^>]*multiple/, "several backups can be imported at once");
  assert.match(home, /UNDO_MS = 10_000/, "deleted chats can be restored for 10 seconds");
  assert.match(home, /setInterval\(\(\) => \{ now = Date\.now\(\); \}, 30_000\)/, "relative times refresh");
  assert.match(root, /window\.history\.pushState/, "Chats and Models moves add history entries");
  assert.match(root, /addEventListener\("popstate", onPopState\)/);
  assert.match(root, /removeEventListener\("popstate", onPopState\)/);
  assert.match(root, /addEventListener\("beforeunload", onBeforeUnload\)/);
  assert.match(root, /onBack=\{\(\) => showHome\(\)\}/, "first run can reach the Chats page without reloading");
  assert.match(prompt, /Updating will pause your download/);
  console.log("hosted home: early render, checking state, copy, undo, history, and update guards passed");
} finally { await server.close(); }
