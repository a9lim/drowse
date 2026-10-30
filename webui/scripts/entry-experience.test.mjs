import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    defaultModelSelection,
    downloadSizeSummary,
    downloadUnavailableReason,
    modelsBySize,
    modelsForChooser,
  } = await server.ssrLoadModule("/src/hosted/ui/modelSelection.ts");
  const sizeOptions = [
    { id: "gemma", firstRunBytes: 639_000_000 },
    { id: "gpt2", firstRunBytes: 673_000_000 },
    { id: "pythia", firstRunBytes: 295_000_000 },
    { id: "qwen", firstRunBytes: 1_100_000_000 },
  ];
  assert.deepEqual(modelsBySize(sizeOptions).map(model => model.id), ["pythia", "gemma", "gpt2", "qwen"]);
  assert.equal(sizeOptions[0].id, "gemma", "sorting never mutates the controller snapshot");
  assert.deepEqual(modelsBySize([]), []);
  assert.deepEqual(modelsBySize([{ id: "a", firstRunBytes: 0 }, { id: "b", firstRunBytes: 0 }]).map(model => model.id), ["a", "b"]);
  const lockedDownload = { available: false, reason: "The signed catalog could not be verified." };
  const availableDownload = { available: true, reason: "Downloads are ready." };
  const unavailableModel = { fit: "blocked", setupIssue: "Insufficient graphics memory.", reason: "A compact completion model." };
  assert.equal(downloadUnavailableReason(unavailableModel, lockedDownload), lockedDownload.reason);
  assert.equal(downloadUnavailableReason({ reason: unavailableModel.reason }, lockedDownload), lockedDownload.reason);
  assert.equal(downloadUnavailableReason(null, lockedDownload), lockedDownload.reason);
  assert.equal(downloadUnavailableReason(unavailableModel, availableDownload), unavailableModel.setupIssue);
  assert.equal(downloadUnavailableReason({ reason: unavailableModel.reason }, availableDownload), unavailableModel.reason);
  assert.equal(downloadUnavailableReason(null, availableDownload), availableDownload.reason);
  const baseModel = { id: "base", modelType: "base", fit: "recommended" };
  const chatModel = { id: "chat", fit: "eligible" };
  assert.equal(defaultModelSelection([], null), null);
  assert.equal(defaultModelSelection([baseModel], null), null);
  assert.equal(defaultModelSelection([baseModel, chatModel], null), "chat");
  assert.equal(defaultModelSelection([baseModel, chatModel], "base"), "base");
  assert.equal(defaultModelSelection([baseModel, chatModel], null, "base"), "base");
  assert.equal(defaultModelSelection([baseModel, chatModel], "removed", "removed"), "chat");
  assert.deepEqual(modelsForChooser([
    { id: "base", modelType: "base", fit: "eligible", firstRunBytes: 1 },
    { id: "small", fit: "eligible", firstRunBytes: 10 },
    { id: "best", fit: "recommended", firstRunBytes: 50 },
  ]).map(model => model.id), ["best", "small", "base"], "chat models first, the recommendation first among them");
  const pack = (bytes, installed = false) => ({ selected: true, installed, bytes });
  assert.equal(downloadSizeSummary({ installed: false, modelDownloadBytes: 300_000_000, firstRunPacks: [pack(500_000_000), pack(400_000_000)] }),
    "Model 300 MB + tools 900 MB");
  assert.equal(downloadSizeSummary({ installed: true, modelDownloadBytes: 300_000_000, firstRunPacks: [pack(900_000_000), pack(1, true)] }),
    "Tools 900 MB");
  assert.equal(downloadSizeSummary({ installed: false, modelDownloadBytes: 1_249_000_000, firstRunPacks: [] }), "1.2 GB");
  const {
    clearConversationOpen,
    peekConversationOpen,
    queueConversationOpen,
    forgetEntryModel,
    readEntryMemory,
    rememberCompletedOnboarding,
  } = await server.ssrLoadModule("/src/hosted/runtime/entryExperience.ts");

  const local = memoryStorage();
  assert.equal(readEntryMemory(local), null);
  local.setItem("polythetic.entry.v1", JSON.stringify({ version: 1, completedAt: 42, lastModelVariantId: "previous" }));
  assert.equal(readEntryMemory(local)?.lastModelVariantId, "previous");
  rememberCompletedOnboarding("gemma3-1b", local, 1234);
  assert.deepEqual(readEntryMemory(local), {
    version: 1,
    completedAt: 1234,
    lastModelVariantId: "gemma3-1b",
  });

  forgetEntryModel("other-model", null, local);
  assert.equal(readEntryMemory(local)?.lastModelVariantId, "gemma3-1b", "removing a different model keeps entry memory");
  forgetEntryModel("gemma3-1b", "qwen3-1b", local);
  assert.deepEqual(readEntryMemory(local), { version: 1, completedAt: 1234, lastModelVariantId: "qwen3-1b" }, "a deleted model hands entry memory to another installed model");
  forgetEntryModel("qwen3-1b", null, local);
  assert.equal(readEntryMemory(local), null, "deleting the last installed model clears entry memory");
  rememberCompletedOnboarding("gemma3-1b", local, 1234);

  local.setItem("drowse.entry.v1", "{broken");
  assert.equal(readEntryMemory(local), null);
  local.setItem("drowse.entry.v1", JSON.stringify({ version: 2 }));
  assert.equal(readEntryMemory(local), null);

  const session = memoryStorage();
  session.setItem("saklas.pending-conversation.v1", JSON.stringify({ version: 1, id: "legacy-chat", modelId: "previous" }));
  assert.equal(peekConversationOpen(session)?.id, "legacy-chat");
  clearConversationOpen(session);
  assert.equal(peekConversationOpen(session), null);
  assert.equal(queueConversationOpen("chat-a", "gemma3-1b", session), true);
  assert.deepEqual(peekConversationOpen(session), {
    version: 1,
    id: "chat-a",
    modelId: "gemma3-1b",
  });
  clearConversationOpen(session);
  assert.equal(peekConversationOpen(session), null);

  const unavailable = {
    getItem() { throw new Error("unavailable"); },
    setItem() { throw new Error("unavailable"); },
    removeItem() { throw new Error("unavailable"); },
  };
  assert.equal(readEntryMemory(unavailable), null);
  rememberCompletedOnboarding("gemma3-1b", unavailable);
  forgetEntryModel("gemma3-1b", null, unavailable);
  assert.equal(queueConversationOpen("chat-a", "gemma3-1b", unavailable), false);
  assert.equal(peekConversationOpen(unavailable), null);
  clearConversationOpen(unavailable);

  console.log("entry experience tests passed");
} finally {
  await server.close();
}

function memoryStorage() {
  const values = new Map();
  return {
    getItem(key) { return values.get(key) ?? null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
  };
}
