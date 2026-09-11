import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const { BrowserLoomRuntime, deriveSeedSchedule } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserLoom.ts",
  );
  const { streamWebLlmGeneration } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmGeneration.ts",
  );
  const { BrowserInstrumentRuntime } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserInstrumentRuntime.ts",
  );
  const freshSession = {
    id: "default", model_id: "fixture/model", created: 1,
    config: { max_tokens: 64 },
  };
  const freshA = new BrowserLoomRuntime({ session: freshSession, generation: {} });
  const freshB = new BrowserLoomRuntime({ session: freshSession, generation: {} });
  const treeA = await freshA.request({ service: "tree", method: "get", args: [] });
  const treeB = await freshB.request({ service: "tree", method: "get", args: [] });
  assert.notEqual(treeA.root_id, treeB.root_id, "fresh chats must not collide in the saved-conversation library");
  assert.deepEqual(deriveSeedSchedule(123, 2), [90123737, 1496822196]);
  assert.deepEqual(deriveSeedSchedule(-5, 3), [1367731838, 485166549, 659370988]);
  const ids = (prefix) => {
    let value = 0;
    return (kind) => `${prefix}-${kind}-${++value}`;
  };
  const generated = (overrides = {}) => ({
    text: "Local answer",
    tokens: 2,
    finishReason: "stop",
    usage: { promptTokens: 4, completionTokens: 2, totalTokens: 6 },
    meanLogprob: -0.15,
    meanSurprise: 0.15,
    prefillTokensPerSecond: 10,
    decodeTokensPerSecond: 5,
    ...overrides,
  });
  const generation = (options = {}) => ({
    plans: [],
    stopped: 0,
    runtimeCapabilities() {
      return {
        topK: true,
        forcedReplay: options.forcedReplay ?? true,
        replayScoring: false,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
        sceneStitching: true,
      };
    },
    async tokenizeText(text) {
      return Array.from(text, (character) => character.codePointAt(0));
    },
    async streamGeneration(plan, onToken) {
      this.plans.push(plan);
      if (options.wait) await options.wait;
      if (options.error) throw options.error;
      const forcedPrefix = plan.replay?.forcedPrefixTokenIds ?? [];
      let rawIndex = 0;
      for (const tokenId of forcedPrefix) {
        const text = String.fromCodePoint(tokenId);
        await plan.onRawToken?.({
          text,
          tokenId,
          logprob: -0.05,
          rawIndex,
          topAlts: null,
        });
        await onToken({
          text,
          thinking: false,
          tokenId,
          logprob: -0.05,
          perplexity: Math.exp(0.05),
          rawIndex,
        });
        rawIndex += 1;
      }
      await onToken({
        text: "Local ",
        thinking: false,
        tokenId: null,
        logprob: -0.1,
        perplexity: Math.exp(0.1),
        rawIndex: null,
      });
      await onToken({
        text: "answer",
        thinking: false,
        tokenId: null,
        logprob: -0.2,
        perplexity: Math.exp(0.2),
        rawIndex: null,
      });
      const prefixText = forcedPrefix.map((tokenId) => String.fromCodePoint(tokenId)).join("");
      return generated({
        text: `${prefixText}Local answer`,
        tokens: forcedPrefix.length + 2,
        usage: {
          promptTokens: 4,
          completionTokens: forcedPrefix.length + 2,
          totalTokens: forcedPrefix.length + 6,
        },
      });
    },
    async stop() { this.stopped += 1; },
  });
  const cancellableGeneration = (mode) => {
    let signalStarted;
    let settle;
    const started = new Promise((resolve) => { signalStarted = resolve; });
    return {
      started,
      stops: 0,
      async streamGeneration(_plan, onToken) {
        await onToken({
          text: "Partial",
          thinking: false,
          tokenId: 7,
          logprob: -0.4,
          perplexity: Math.exp(0.4),
          rawIndex: 0,
        });
        const interrupted = new Promise((resolve, reject) => {
          settle = mode === "resolve"
            ? resolve
            : () => reject(new DOMException("interrupted", "AbortError"));
        });
        signalStarted();
        await interrupted;
        return generated({
          text: "Partial",
          tokens: 1,
          finishReason: "stop",
          usage: { promptTokens: 3, completionTokens: 1, totalTokens: 4 },
          meanLogprob: -0.4,
          meanSurprise: 0.4,
        });
      },
      async stop() {
        this.stops += 1;
        settle?.();
      },
    };
  };

  const baseEngine = generation();
  const externallyStoppedEngine = generation();
  const streamBeforeStop = externallyStoppedEngine.streamGeneration.bind(externallyStoppedEngine);
  externallyStoppedEngine.streamGeneration = async (...args) => ({
    ...await streamBeforeStop(...args), terminalReason: "external_stop",
  });
  const externallyStoppedRuntime = new BrowserLoomRuntime({
    session: { ...session(), is_base_model: true },
    generation: externallyStoppedEngine,
    createId: ids("external-stop"),
  });
  const externallyStoppedEvents = [];
  await externallyStoppedRuntime.generate({
    type: "submit", text: "Keep this passage", authored_role: "user", generated_role: "assistant",
  }, event => externallyStoppedEvents.push(event));
  assert.equal(externallyStoppedEvents.find(event => event.type === "done").result.finish_reason, "cancelled");
  assert.equal(externallyStoppedRuntime.snapshot().tree.nodes.find(node => node.role === "assistant").finish_reason, "cancelled");
  const baseRuntime = new BrowserLoomRuntime({
    session: { ...session({ system_prompt: "Do not insert this system prompt" }), is_base_model: true },
    generation: baseEngine,
    createId: ids("base"),
  });
  await baseRuntime.generate({
    type: "submit", text: "A passage\nwith two lines", authored_role: "user",
    generated_role: "assistant", raw: false,
  }, () => {});
  assert.deepEqual(baseEngine.plans[0].input, { kind: "raw", prompt: "A passage\nwith two lines" });
  await baseRuntime.generate({ type: "generate", input: null, raw: false, stateless: false }, () => {});
  assert.equal(baseEngine.plans[1].input.kind, "raw");
  assert.equal(baseEngine.plans[1].input.prompt, "A passage\nwith two lines");
  assert.equal(String.fromCodePoint(...baseEngine.plans[1].replay.forcedPrefixTokenIds), "Local answer",
    "continuation replays the existing generated suffix without chat delimiters");
  await assert.rejects(baseRuntime.generate({
    type: "generate", input: [{ role: "user", content: "Do not template me" }],
  }, () => {}), /plain text prompt/);
  assert.equal(baseEngine.plans.length, 2, "a chat-message array must never reach a base model");
  const originalBaseTree = baseRuntime.snapshot().tree;
  const promptNode = originalBaseTree.nodes.find(node => node.role === "user");
  await baseRuntime.generate({
    type: "submit", text: "An edited ending", authored_role: "user", generated_role: null,
    parent_node_id: promptNode.id, raw: true,
  }, () => {});
  const editedBaseTree = baseRuntime.snapshot().tree;
  for (const originalNode of originalBaseTree.nodes) {
    assert.deepEqual(editedBaseTree.nodes.find(node => node.id === originalNode.id), originalNode,
      "editing a base completion must leave every original branch node unchanged");
  }
  assert.equal(editedBaseTree.nodes.length, originalBaseTree.nodes.length + 1);

  const commitEngine = generation();
  const commitRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: commitEngine,
    now: () => 10,
    createId: ids("commit"),
  });
  assert.deepEqual(await commitRuntime.request({
    service: "tree", method: "replayCapabilities", args: [],
  }), {
    jointLogprobs: {
      available: false,
      reason: "This model runtime cannot replay exact token likelihoods for branch comparison.",
    },
  });
  const commitEvents = [];
  await commitRuntime.generate(
    {
      type: "submit",
      text: "Saved only",
      authored_role: "assistant",
      generated_role: null,
      raw: false,
    },
    (event) => commitEvents.push(event),
  );
  assert.deepEqual(commitEvents.map((event) => event.type), [
    "started", "tree_mutated", "done",
  ]);
  assert.equal(commitEvents[0].node_id, null);
  const autoThinkingEngine = generation();
  const autoThinkingRuntime = new BrowserLoomRuntime({
    session: {
      ...session({ thinking: null }),
      supports_thinking: true,
      thinking_is_optional: true,
      thinking_input_supported: true,
      strips_history_thinking: true,
    },
    generation: autoThinkingEngine,
    createId: ids("auto-thinking"),
  });
  await autoThinkingRuntime.generate({
    type: "generate",
    input: "Use the model default",
    stateless: true,
  }, () => {});
  assert.equal(autoThinkingEngine.plans.at(-1).thinking, true);
  await autoThinkingRuntime.generate({
    type: "generate",
    input: "Disable thinking explicitly",
    stateless: true,
    thinking: false,
  }, () => {});
  assert.equal(autoThinkingEngine.plans.at(-1).thinking, false);
  assert.equal(commitEvents[1].op, "add_user");
  assert.equal(commitEvents[1].rev, 1);
  assert.equal(commitEvents[2].node_id, commitEvents[1].added[0].id);
  assert.equal(commitEngine.plans.length, 0);
  const committed = commitRuntime.snapshot();
  assert.equal(committed.tree.nodes[1].role, "assistant");
  assert.equal(committed.tree.nodes[1].role_label, null);
  assert.equal(committed.session.history_length, 1);

  const committedNodeId = committed.tree.nodes[1].id;
  const appendEvents = [];
  await commitRuntime.generate({
    type: "submit",
    text: " plus more",
    authored_role: "assistant",
    generated_role: null,
    raw: false,
  }, (event) => appendEvents.push(event));
  assert.deepEqual(appendEvents.map((event) => event.type), [
    "started", "tree_mutated", "done",
  ]);
  assert.equal(appendEvents[1].op, "edit");
  assert.deepEqual(appendEvents[1].added, []);
  assert.deepEqual(appendEvents[1].updated.map((node) => node.id), [committedNodeId]);
  assert.equal(appendEvents[1].active_node_id, committedNodeId);
  const appendedCommit = commitRuntime.snapshot();
  assert.equal(appendedCommit.tree.nodes.length, 2);
  assert.equal(appendedCommit.tree.active_node_id, committedNodeId);
  assert.equal(appendedCommit.tree.nodes[1].text, "Saved only plus more");
  assert.equal(appendedCommit.tree.nodes[1].recipe, null);
  assert.equal(appendedCommit.tree.nodes[1].tokens, null);
  assert.equal(appendedCommit.tree.nodes[1].edit_count, 1);
  assert.deepEqual(
    appendedCommit.tree.nodes[1].raw_token_ids,
    Array.from("Saved only plus more", (character) => character.codePointAt(0)),
  );
  assert.equal(appendEvents[2].node_id, committedNodeId);
  assert.equal(appendEvents[2].result.text, " plus more");

  const continuationEngine = generation();
  const continuationRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: continuationEngine,
    createId: ids("continuation"),
  });
  await continuationRuntime.generate({
    type: "submit",
    text: "Prompt",
    authored_role: "user",
    generated_role: "assistant",
  }, () => {});
  const beforeContinuation = continuationRuntime.snapshot();
  const branchOnlyRuntime = new BrowserLoomRuntime({
    session: beforeContinuation.session,
    initialTree: beforeContinuation.tree,
    generation: generation(),
    createId: ids("branch-only"),
  });
  await branchOnlyRuntime.generate({
    type: "generate", input: null, stateless: false, n: 1,
    parent_node_id: beforeContinuation.tree.active_node_id,
    append_same_role: false,
  }, () => {});
  const branchOnly = branchOnlyRuntime.snapshot().tree;
  assert.equal(branchOnly.nodes.length, beforeContinuation.tree.nodes.length + 1);
  assert.equal(branchOnly.nodes.at(-1).parent_id, beforeContinuation.tree.active_node_id);
  for (const original of beforeContinuation.tree.nodes) {
    assert.deepEqual(branchOnly.nodes.find(node => node.id === original.id), original,
      "one explicit Weave alternative preserves all original nodes");
  }
  const continuedId = beforeContinuation.tree.active_node_id;
  const continuedPrefix = beforeContinuation.tree.nodes.at(-1).text;
  const continuationEvents = [];
  await continuationRuntime.generate({
    type: "generate",
    input: null,
    stateless: false,
    thinking: true,
  }, (event) => continuationEvents.push(event));
  const continuationPlan = continuationEngine.plans[1];
  assert.deepEqual(
    continuationPlan.replay.forcedPrefixTokenIds,
    Array.from(continuedPrefix, (character) => character.codePointAt(0)),
  );
  assert.deepEqual(continuationPlan.input, {
    kind: "chat",
    messages: [{ role: "user", content: "Prompt" }],
  });
  assert.equal(continuationPlan.sampling.max_tokens, 64 + continuedPrefix.length);
  assert.equal(continuationPlan.thinking, false);
  assert.equal(continuationEvents[1].op, "begin_assistant");
  assert.deepEqual(continuationEvents[1].added, []);
  assert.deepEqual(continuationEvents[1].updated.map((node) => node.id), [continuedId]);
  assert.equal(continuationEvents[1].updated[0].text, continuedPrefix);
  assert.deepEqual(continuationEvents.filter((event) => event.type === "token")
    .map((event) => event.text), ["Local ", "answer"]);
  assert.equal(continuationEvents[1].active_node_id, continuedId);
  assert.equal(continuationEvents.at(-1).node_id, continuedId);
  const continued = continuationRuntime.snapshot();
  assert.equal(continued.tree.nodes.length, beforeContinuation.tree.nodes.length);
  assert.equal(continued.tree.active_node_id, continuedId);
  assert.equal(continued.tree.nodes.at(-1).text, `${continuedPrefix}Local answer`);
  assert.equal(continued.tree.nodes.at(-1).recipe.thinking, false);
  assert.equal(continued.session.history_length, beforeContinuation.session.history_length);

  const failedContinuationEngine = generation({ error: new Error("continuation failed") });
  const failedContinuationRuntime = new BrowserLoomRuntime({
    session: continued.session,
    generation: failedContinuationEngine,
    initialTree: continued.tree,
    createId: ids("failed-continuation"),
  });
  const failedContinuationBefore = failedContinuationRuntime.snapshot();
  const failedContinuationEvents = [];
  await assert.rejects(
    failedContinuationRuntime.generate({
      type: "generate",
      input: null,
      stateless: false,
    }, (event) => failedContinuationEvents.push(event)),
    /continuation failed/,
  );
  const failedContinuationAfter = failedContinuationRuntime.snapshot();
  assert.deepEqual(failedContinuationAfter.tree.nodes, failedContinuationBefore.tree.nodes);
  assert.equal(failedContinuationAfter.tree.active_node_id, continuedId);
  assert.equal(failedContinuationAfter.tree.rev, failedContinuationBefore.tree.rev + 2);
  assert.deepEqual(
    failedContinuationEvents.filter((event) => event.type === "tree_mutated")
      .map((event) => event.op),
    ["begin_assistant", "finalize_assistant"],
  );
  assert.equal(
    failedContinuationEvents.some((event) => event.type === "tree_mutated" && event.op === "delete"),
    false,
  );

  const chatEngine = generation();
  const chatRuntime = new BrowserLoomRuntime({
    session: session({ system_prompt: "Stay precise." }),
    generation: chatEngine,
    now: () => 20,
    createId: ids("chat"),
  });
  const chatEvents = [];
  await chatRuntime.generate(
    {
      type: "submit",
      text: "Question",
      authored_role: "user",
      generated_role: "assistant",
      steering: null,
      sampling: { temperature: 0.25, return_top_k: 5 },
      thinking: false,
      raw: false,
    },
    (event) => chatEvents.push(event),
  );
  assert.deepEqual(chatEvents.map((event) => event.type), [
    "started",
    "tree_mutated",
    "tree_mutated",
    "token",
    "token",
    "tree_mutated",
    "done",
  ]);
  assert.equal(chatEvents[0].type, "started");
  assert.ok(
    chatEvents.findIndex((event) =>
      event.type === "tree_mutated" && event.op === "finalize_assistant"
    ) <
      chatEvents.findIndex((event) => event.type === "done"),
  );
  assert.equal(
    chatEvents.slice(1, -1).every((event) =>
      event.type === "tree_mutated" || event.type === "token"
    ),
    true,
  );
  assert.deepEqual(
    chatEvents.filter((event) => event.type === "tree_mutated").map((event) => event.op),
    ["add_user", "begin_assistant", "finalize_assistant"],
  );
  for (const mutation of chatEvents.filter((event) => event.type === "tree_mutated")) {
    assert.deepEqual(Object.keys(mutation.cast).sort(), ["assistant", "user"]);
    assert.ok(Array.isArray(mutation.added));
    assert.ok(Array.isArray(mutation.removed));
    assert.ok(Array.isArray(mutation.updated));
  }
  assert.equal(chatEvents.at(-2).active_node_id, null);
  assert.equal(chatEvents[3].token_id, null);
  assert.equal(chatEvents[3].raw_index, null);
  assert.equal(chatEvents[3].node_id, chatEvents[2].added[0].id);
  assert.deepEqual(chatEngine.plans[0].input, {
    kind: "chat",
    messages: [
      { role: "system", content: "Stay precise." },
      { role: "user", content: "Question" },
    ],
  });
  assert.equal(chatEngine.plans[0].sampling.temperature, 0.25);
  assert.equal(chatEngine.plans[0].sampling.return_top_k, 5);
  const chat = chatRuntime.snapshot();
  assert.equal(chat.tree.rev, 3);
  assert.equal(chat.tree.nodes[1].role_label, null);
  assert.equal(chat.tree.nodes[2].role_label, null);
  assert.equal(chat.tree.nodes[2].text, "Local answer");
  assert.equal(chat.tree.nodes[2].tokens.length, 2);
  assert.equal("token_id" in chat.tree.nodes[2].tokens[0], false);
  assert.equal(chat.tree.nodes[2].raw_token_ids, null);
  assert.equal(chat.tree.nodes[2].mean_logprob, -0.15);
  assert.equal(chat.tree.nodes[2].recipe.sampling.return_top_k, 5);
  assert.equal(chat.session.history_length, 2);

  const configuredSystemEngine = generation();
  const configuredSystemRuntime = new BrowserLoomRuntime({
    session: session({ system_prompt: "Configured system" }),
    generation: configuredSystemEngine,
    createId: ids("configured-system"),
  });
  await configuredSystemRuntime.generate({
    type: "generate",
    input: [
      { role: "system", content: "Explicit system" },
      { role: "user", content: "Question" },
    ],
    stateless: true,
  }, () => {});
  assert.deepEqual(configuredSystemEngine.plans[0].input, {
    kind: "chat",
    messages: [
      { role: "system", content: "Configured system" },
      { role: "system", content: "Explicit system" },
      { role: "user", content: "Question" },
    ],
  });

  const namedEngine = generation();
  const renamedEngine = generation();
  const renamedRuntime = new BrowserLoomRuntime({
    session: session(), generation: renamedEngine, createId: ids("renamed"),
  });
  for (const name of [null, "pirate", "forest_guide", null]) {
    await renamedRuntime.generate({
      type: "submit", text: "Arrgh", authored_role: "user", generated_role: "assistant",
      sampling: name === null ? {} : { assistant_role: name },
    }, () => {});
    assert.equal(renamedEngine.plans.at(-1).generationRoleName, name);
    const node = renamedRuntime.snapshot().tree.nodes.at(-1);
    assert.equal(node.role, "assistant");
    assert.equal(node.role_label, name);
    assert.equal(node.recipe.sampling.assistant_role, name);
  }
  assert.deepEqual(
    renamedEngine.plans.at(-1).input.messages.filter(message => message.role === "assistant")
      .map(message => message.name ?? null),
    [null, "pirate", "forest_guide"],
    "Changing the next speaker must preserve the actual names of historical turns",
  );
  const renamedTranscript = await renamedRuntime.request({ service: "tree", method: "transcriptExport", args: [null] });
  assert.match(renamedTranscript.yaml, /speaker: pirate/);
  assert.match(renamedTranscript.yaml, /speaker: forest_guide/);
  const namedRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: namedEngine,
    createId: ids("named"),
  });
  const beforeInvalidNamedRole = namedRuntime.snapshot().tree;
  await assert.rejects(
    namedRuntime.generate({
      type: "submit",
      text: "Review this",
      authored_role: "user",
      generated_role: "assistant",
      sampling: { user_role: "Analyst", assistant_role: "guide" },
    }, () => {}),
    (error) => error.code === "INVALID_GENERATION_ROLE",
  );
  assert.deepEqual(namedRuntime.snapshot().tree, beforeInvalidNamedRole);
  assert.equal(namedEngine.plans.length, 0);
  await namedRuntime.generate({
    type: "submit",
    text: "Review this",
    authored_role: "user",
    generated_role: "assistant",
    sampling: { user_role: "analyst", assistant_role: "guide" },
  }, () => {});
  assert.deepEqual(namedEngine.plans[0].input, {
    kind: "chat",
    messages: [{ role: "user", content: "Review this", name: "analyst" }],
  });
  assert.equal(namedEngine.plans[0].generationRoleName, "guide");
  assert.deepEqual(
    namedRuntime.snapshot().tree.nodes.slice(1).map((node) => node.role_label),
    ["analyst", "guide"],
  );
  await namedRuntime.generate({ type: "generate", input: null, stateless: false }, () => {});
  assert.deepEqual(namedEngine.plans[1].input, {
    kind: "chat",
    messages: [
      { role: "user", content: "Review this", name: "analyst" },
      { role: "assistant", content: "Local answer", name: "guide" },
    ],
  });
  await namedRuntime.generate({
    type: "generate",
    input: [{ role: "user", content: "Explicit replay", label: "researcher" }],
    stateless: true,
  }, () => {});
  assert.deepEqual(namedEngine.plans[2].input, {
    kind: "chat",
    messages: [{ role: "user", content: "Explicit replay", name: "researcher" }],
  });
  await assert.rejects(
    namedRuntime.generate({
      type: "generate",
      input: [{ role: "system", content: "Named system", label: "director" }],
    }, () => {}),
    (error) => error.code === "ROLE_SUBSTITUTION_UNAVAILABLE",
  );
  const beforeUserSeatPlans = namedEngine.plans.length;
  const userSeatEvents = [];
  await namedRuntime.generate({
    type: "generate",
    generate_seat: "user",
    stateless: false,
  }, (event) => {
    userSeatEvents.push(event);
  });
  assert.equal(namedEngine.plans.at(-1).generationSeat, "user");
  assert.equal(namedRuntime.snapshot().tree.nodes.at(-1).role, "user");
  await namedRuntime.generate({
    type: "generate",
    raw: true,
    input: [{ role: "user", content: "Raw replay", label: "researcher" }],
    sampling: { assistant_role: "guide" },
  }, () => {});
  assert.deepEqual(namedEngine.plans.at(-1).input, {
    kind: "chat",
    messages: [{ role: "user", content: "Raw replay", name: "researcher" }],
  });
  assert.equal(namedEngine.plans.at(-1).generationSeat, "assistant");
  assert.equal(namedEngine.plans.at(-1).generationRoleName, "guide");
  await namedRuntime.generate({
    type: "submit",
    text: "Question",
    authored_role: "assistant",
    generated_role: "user",
  }, (event) => userSeatEvents.push(event));
  assert.equal(namedEngine.plans.at(-1).generationSeat, "user");
  assert.equal(namedEngine.plans.length, beforeUserSeatPlans + 3);
  assert.equal(userSeatEvents.some((event) => event.type === "done"), true);

  const exportedTranscript = await chatRuntime.request({
    service: "tree", method: "transcriptExport", args: [null],
  });
  assert.equal(exportedTranscript.node_id, chat.tree.active_node_id);
  assert.match(exportedTranscript.yaml, /^drowse_transcript: 2/m);
  assert.match(exportedTranscript.yaml, /model_id: fixture\/drowse-tiny/);
  assert.match(exportedTranscript.yaml, /text: Question/);
  const transcriptRuntime = new BrowserLoomRuntime({
    session: session({ system_prompt: "Stay precise." }),
    generation: generation(),
    createId: ids("transcript"),
  });
  const importedTranscript = await transcriptRuntime.request({
    service: "tree",
    method: "transcriptLoad",
    args: [exportedTranscript.yaml, "default", false],
  });
  assert.deepEqual(importedTranscript.guards, []);
  assert.equal(transcriptRuntime.snapshot().tree.nodes.length, 3);
  assert.equal(transcriptRuntime.snapshot().tree.nodes[1].text, "Question");
  assert.equal(transcriptRuntime.snapshot().tree.nodes[2].text, "Local answer");
  const beforeInvalidTranscript = transcriptRuntime.snapshot().tree;
  await assert.rejects(
    transcriptRuntime.request({
      service: "tree",
      method: "transcriptLoad",
      args: [exportedTranscript.yaml.replace("temperature: 0.25", "temperature: fast"), "default", false],
    }),
  );
  assert.deepEqual(transcriptRuntime.snapshot().tree, beforeInvalidTranscript);
  const mismatchedTranscript = exportedTranscript.yaml.replace(
    "model_id: fixture/drowse-tiny",
    "model_id: fixture/other-model",
  );
  const guardedImport = await transcriptRuntime.request({
    service: "tree",
    method: "transcriptLoad",
    args: [mismatchedTranscript, "default", false],
  });
  assert.match(guardedImport.guards[0], /^model_mismatch:/);
  await assert.rejects(
    transcriptRuntime.request({
      service: "tree",
      method: "transcriptLoad",
      args: [mismatchedTranscript, "merge", false],
    }),
    (error) => error.code === "TRANSCRIPT_MODEL_MISMATCH",
  );

  const probeSession = session();
  probeSession.probes = ["calm"];
  let resolvedProbeHash = "abc123";
  const probeRuntime = new BrowserLoomRuntime({
    session: probeSession,
    generation: generation(),
    createId: ids("probe-hash-source"),
    probeHashes: async () => ({ calm: resolvedProbeHash }),
  });
  await probeRuntime.generate({
    type: "submit",
    text: "Hash this probe",
    authored_role: "user",
    generated_role: "assistant",
  }, () => {});
  const probeRecipe = probeRuntime.snapshot().tree.nodes.at(-1).recipe;
  assert.deepEqual(probeRecipe.probes, ["calm"]);
  assert.deepEqual(probeRecipe.probe_hashes, { calm: "abc123" });
  resolvedProbeHash = "current456";
  const probeExport = await probeRuntime.request({
    service: "tree", method: "transcriptExport", args: [null],
  });
  assert.match(probeExport.yaml, /sha256: current456/);
  assert.doesNotMatch(probeExport.yaml, /sha256: ["']{2}/);
  const hashedTranscript = probeExport.yaml;

  const exactProbeSession = session();
  exactProbeSession.probes = ["calm"];
  const exactProbeRuntime = new BrowserLoomRuntime({
    session: exactProbeSession,
    generation: generation(),
    createId: ids("probe-hash-exact"),
    probeHashes: () => ({ calm: "current456" }),
  });
  const exactProbeImport = await exactProbeRuntime.request({
    service: "tree", method: "transcriptLoad",
    args: [hashedTranscript, "default", true],
  });
  assert.deepEqual(exactProbeImport.guards, []);

  const driftProbeSession = session();
  driftProbeSession.probes = ["calm"];
  const driftProbeRuntime = new BrowserLoomRuntime({
    session: driftProbeSession,
    generation: generation(),
    createId: ids("probe-hash-drift"),
    probeHashes: () => ({ calm: "different" }),
  });
  await assert.rejects(
    driftProbeRuntime.request({
      service: "tree", method: "transcriptLoad",
      args: [hashedTranscript, "default", true],
    }),
    (error) => error.code === "TRANSCRIPT_PROBE_DRIFT",
  );
  assert.equal(driftProbeRuntime.snapshot().tree.nodes.length, 1);
  const guardedProbeImport = await driftProbeRuntime.request({
    service: "tree", method: "transcriptLoad",
    args: [hashedTranscript, "default", false],
  });
  assert.deepEqual(guardedProbeImport.guards, ['probe_drift: ["calm"]']);

  const unknownProbeSession = session();
  unknownProbeSession.probes = ["calm"];
  const unknownProbeRuntime = new BrowserLoomRuntime({
    session: unknownProbeSession,
    generation: generation(),
    createId: ids("probe-hash-unknown"),
  });
  await assert.rejects(
    unknownProbeRuntime.request({
      service: "tree", method: "transcriptLoad",
      args: [hashedTranscript, "default", true],
    }),
    (error) => error.code === "TRANSCRIPT_PROBE_HASH_UNAVAILABLE",
  );
  const missingProbeRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: generation(),
    createId: ids("probe-hash-missing"),
  });
  const missingProbeImport = await missingProbeRuntime.request({
    service: "tree", method: "transcriptLoad",
    args: [hashedTranscript, "default", true],
  });
  assert.deepEqual(missingProbeImport.guards, ['probes_missing: ["calm"]']);

  const fanEngine = generation();
  const fanRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: fanEngine,
    createId: ids("fan"),
  });
  const fanEvents = [];
  await fanRuntime.generate(
    {
      type: "submit",
      text: "Question",
      authored_role: "user",
      generated_role: "assistant",
      sampling: { seed: 123 },
      n: 2,
    },
    (event) => fanEvents.push(event),
  );
  assert.deepEqual(
    fanEvents.filter((event) => event.type === "started").map((event) => [
      event.sibling_index,
      event.sibling_count,
    ]),
    [[0, 2], [1, 2]],
  );
  assert.deepEqual(
    fanEvents.filter((event) => event.type === "done").map((event) => [
      event.sibling_index,
      event.sibling_count,
    ]),
    [[0, 2], [1, 2]],
  );
  assert.deepEqual(fanEngine.plans.map((plan) => plan.sampling.seed), [90123737, 1496822196]);
  const fanTree = fanRuntime.snapshot().tree;
  assert.equal(fanTree.nodes.length, 4);
  assert.equal(fanTree.children_of[fanTree.nodes[1].id].length, 2);
  assert.deepEqual(
    fanTree.nodes.slice(2).map((node) => node.recipe.seed),
    [90123737, 1496822196],
  );

  const rawEngine = generation();
  const rawRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: rawEngine,
    createId: ids("raw"),
  });
  await rawRuntime.generate(
    {
      type: "submit",
      text: "prefix",
      authored_role: "user",
      generated_role: "assistant",
      raw: true,
    },
    () => {},
  );
  assert.deepEqual(rawEngine.plans[0].input, { kind: "raw", prompt: "prefix" });

  const exactRawEngine = {
    plans: [],
    async streamGeneration(plan, onToken) {
      this.plans.push(plan);
      for (const [rawIndex, token] of [
        { text: "<think>", tokenId: 10 },
        { text: "hidden", tokenId: 11 },
        { text: "</think>", tokenId: 12 },
        { text: "answer", tokenId: 13 },
      ].entries()) {
        await plan.onRawToken({
          ...token,
          rawIndex,
          logprob: -0.1,
          topAlts: null,
        });
      }
      await onToken({
        text: "answer",
        thinking: false,
        tokenId: 13,
        logprob: -0.1,
        perplexity: Math.exp(0.1),
        rawIndex: 3,
      });
      return generated({ text: "answer", tokens: 4 });
    },
    async stop() {},
  };
  const exactRawRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: exactRawEngine,
    createId: ids("exact-raw"),
  });
  await exactRawRuntime.generate(
    {
      type: "submit",
      text: "Question",
      authored_role: "user",
      generated_role: "assistant",
      raw: false,
    },
    () => {},
  );
  const exactRawNode = exactRawRuntime.snapshot().tree.nodes.at(-1);
  assert.deepEqual(exactRawNode.raw_token_ids, [10, 11, 12, 13]);
  assert.deepEqual(exactRawNode.tokens.map((token) => token.raw_index), [3]);

  const replayTree = structuredClone(exactRawRuntime.snapshot().tree);
  const replayParent = replayTree.nodes[1];
  const branchA = replayTree.nodes[2];
  branchA.text = "ab";
  branchA.role_label = "pirate";
  branchA.applied_steering = "left:role-pirate";
  branchA.raw_token_ids = [10, 11];
  branchA.tokens = [
    { text: "a", token_id: 10, raw_index: 0, logprob: -0.1, perplexity: Math.exp(0.1) },
    {
      text: "b",
      token_id: 11,
      raw_index: 1,
      logprob: -0.7,
      perplexity: Math.exp(0.7),
      top_alts: [
        { id: 11, text: "b", logprob: -0.7 },
        { id: 42, text: "d", logprob: -2.4 },
      ],
    },
  ];
  branchA.thinking_tokens = [];
  branchA.recipe = {
    ...branchA.recipe,
    steering: "left:role-pirate",
    thinking: false,
    seed: 7,
    sampling: {
      ...branchA.recipe.sampling,
      max_tokens: 4,
      seed: 7,
      user_role: "reader",
      assistant_role: "guide",
      persist_per_layer_scores: true,
      persist_subspace_coords: false,
      return_top_k: 5,
    },
  };
  const branchB = structuredClone(branchA);
  branchB.id = "replay-branch-b";
  branchB.text = "ac";
  branchB.applied_steering = "right";
  branchB.raw_token_ids = [10, 12];
  branchB.tokens = [
    { text: "a", token_id: 10, raw_index: 0, logprob: -1.1, perplexity: Math.exp(1.1) },
    { text: "c", token_id: 12, raw_index: 1, logprob: -0.6, perplexity: Math.exp(0.6) },
  ];
  branchB.recipe = {
    ...branchB.recipe,
    steering: "right",
    seed: 11,
    sampling: { ...branchB.recipe.sampling, max_tokens: 4, seed: 11 },
  };
  replayTree.nodes.push(branchB);
  replayTree.children_of[replayParent.id] = [branchA.id, branchB.id];
  replayTree.children_of[branchB.id] = [];
  replayTree.active_node_id = branchB.id;
  replayTree.rev += 1;

  const replayLive = { geometry: false, lens: false, sae: false };
  const replayCompiler = {
    listProfiles: () => ({ profiles: [] }),
    instrumentDescriptor: () => ({
      jlens: {
        source: "fixture-jlens",
        displayName: "Fixture J-lens",
        layers: [0, 1],
        words: [{ word: "yes", tokenId: 10 }],
      },
      sae: {
        source: "fixture-sae",
        displayName: "Fixture SAE",
        layer: 1,
        features: 2,
      },
    }),
    instrumentLiveState: () => ({ ...replayLive }),
    setInstrumentLive(family, enabled) { replayLive[family] = enabled; },
    listProbes: () => [],
  };
  const replayInstruments = new BrowserInstrumentRuntime(replayCompiler);
  const tokenText = new Map([[10, "a"], [11, "b"], [12, "c"], [20, "x"], [30, "y"], [40, "z"], [42, "d"], [99, "!"]]);
  const exactReplayEngine = {
    plans: [],
    runtimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
        sceneStitching: true,
      };
    },
    async tokenizeText(text) { return [...text].map((piece) => piece.codePointAt(0)); },
    async decodeTokens(ids) { return ids.map((id) => tokenText.get(id) ?? `:${id}:`).join(""); },
    async streamGeneration(plan, onToken) {
      this.plans.push(plan);
      const forced = [...(plan.replay?.forcedPrefixTokenIds ?? [])];
      const emitted = plan.replay?.scoreTokenIds !== undefined ||
          plan.sampling.max_tokens === forced.length
        ? forced
        : [...forced, 99];
      const isRight = plan.steeringExpression === "right";
      const logprob = (tokenId, rawIndex) => {
        const rows = isRight
          ? new Map([[20, -0.2], [10, -1.1], [40, -1.5], [11, -2.1], [12, -0.6], [42, -2.5], [99, -3]])
          : new Map([[10, -0.1], [20, -1], [30, -2], [11, -0.7], [12, -2.2], [42, -2.4], [99, -3.1]]);
        return (rows.get(tokenId) ?? -4) - rawIndex * 0.01;
      };
      const readoutWidth = plan.readoutTopK ?? 2;
      const measurements = () => ({
        version: 1,
        scope: "token",
        provenance: "captured",
        instruments: {
          ...(replayLive.geometry ? { geometry: { readings: {} } } : {}),
          ...(replayLive.lens ? {
            lens: {
              binding: { source: "fixture-jlens", steering: plan.steeringExpression ?? null },
              readings: {
                "jlens/yes": {
                  value: 0.8,
                  unit: "mean_token_probability",
                  per_layer: { "0": 0.8 },
                  depth: null,
                },
              },
              readout: {
                layers: [0, 1].map((layer) => ({
                  layer,
                  tokens: [
                    { token: "yes", id: 10, logprob: -0.2 - layer * 0.1 },
                    { token: "no", id: 20, logprob: -1.2 - layer * 0.1 },
                  ].slice(0, readoutWidth),
                })),
                aggregate: [
                  { token: "aggregate-only", strength: 0.9, com: 0.5, spread: 0.2 },
                  { token: "yes", strength: 0.8, com: 0.45, spread: 0.15 },
                ].slice(0, readoutWidth),
              },
            },
          } : {}),
          ...(replayLive.sae ? {
            sae: {
              binding: { source: "fixture-sae", steering: plan.steeringExpression ?? null, layer: 1 },
              readings: {
                "sae/1": {
                  value: 2,
                  unit: "raw_activation",
                  per_layer: { "1": 2 },
                  depth: null,
                },
              },
              readout: { features: [
                { id: 0, activation: 3, label: null, max_act: null },
                { id: 1, activation: 2, label: null, max_act: null },
              ] },
            },
          } : {}),
        },
        scores: {
          ...(replayLive.lens ? { "jlens/yes": 0.8 } : {}),
          ...(replayLive.sae ? { "sae/1": 2 } : {}),
        },
        per_layer_scores: {
          ...(replayLive.lens ? { "0": { "jlens/yes": 0.8 } } : {}),
          ...(replayLive.sae ? { "1": { "sae/1": 2 } } : {}),
        },
      });
      for (const [rawIndex, tokenId] of emitted.entries()) {
        const top = (isRight
          ? [20, 10, 40]
          : [10, 20, 30]
        ).map((id) => ({ tokenId: id, logprob: logprob(id, rawIndex) }));
        const replayScore = plan.replay?.scoreTokenIds === undefined ? undefined : {
          emittedTokenId: tokenId,
          sampledTokenId: top[0].tokenId,
          forcedTokenId: tokenId,
          requestedLogprobs: plan.replay.scoreTokenIds.map((id) => ({
            tokenId: id,
            logprob: logprob(id, rawIndex),
          })),
          argmax: top[0],
          topLogprobs: top,
        };
        const raw = {
          text: tokenText.get(tokenId) ?? `:${tokenId}:`,
          tokenId,
          logprob: logprob(tokenId, rawIndex),
          rawIndex,
          topAlts: null,
          ...(replayScore ? { replayScore } : {}),
          measurements: measurements(),
        };
        await plan.onRawTokenStart?.(rawIndex);
        await plan.onRawToken?.(raw);
        await onToken({
          ...raw,
          thinking: false,
          perplexity: Math.exp(-raw.logprob),
        });
      }
      const text = emitted.map((id) => tokenText.get(id) ?? `:${id}:`).join("");
      return generated({ text, tokens: emitted.length });
    },
    async stop() {},
  };
  const replayCompiled = [];
  const exactReplayRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: exactReplayEngine,
    initialTree: replayTree,
    instruments: replayInstruments,
    createId: ids("replay"),
    compileSteering(expression) {
      replayCompiled.push(expression);
      return {
        hookAbi: "post-block-residual-v4",
        activeRole: expression.includes(":role-pirate") ? "pirate" : null,
      };
    },
  });
  const replaySession = await exactReplayRuntime.request({
    service: "sessions", method: "get", args: [],
  });
  assert.equal(replaySession.instruments.every((row) => row.capabilities.token_readout), true);
  assert.deepEqual(await exactReplayRuntime.request({
    service: "tree", method: "replayCapabilities", args: [],
  }), {
    jointLogprobs: { available: true, reason: null },
  });
  replayCompiler.setInstrumentLive("sae", true);
  const lensReplayProgress = [];
  const lensReplay = await exactReplayRuntime.request({
    service: "instruments",
    method: "tokenReadout",
    args: ["lens", branchA.id, 0, { topK: 1, steered: false, raw: false, layers: "all" }],
  }, (event) => lensReplayProgress.push(event));
  assert.equal(lensReplay.measurements.scope, "replay");
  assert.equal(lensReplay.measurements.provenance, "replayed");
  assert.equal(lensReplay.measurements.instruments.lens.binding.steering, null);
  assert.deepEqual(Object.keys(lensReplay.measurements.instruments), ["lens"]);
  assert.deepEqual(lensReplay.measurements.scores, { "jlens/yes": 0.8 });
  assert.deepEqual(lensReplay.measurements.per_layer_scores, {
    "0": { "jlens/yes": 0.8 },
  });
  const unsteeredReplayPlan = exactReplayEngine.plans[0];
  assert.deepEqual(unsteeredReplayPlan.input, {
    kind: "chat",
    messages: [{ role: "user", content: "Question" }],
  });
  assert.deepEqual(unsteeredReplayPlan.replay.forcedPrefixTokenIds, [10]);
  assert.equal(unsteeredReplayPlan.sampling.seed, 7);
  assert.equal(unsteeredReplayPlan.sampling.max_tokens, 1);
  assert.equal(unsteeredReplayPlan.sampling.user_role, "reader");
  assert.equal(unsteeredReplayPlan.sampling.assistant_role, "pirate");
  assert.equal(unsteeredReplayPlan.sampling.persist_per_layer_scores, true);
  assert.equal(unsteeredReplayPlan.sampling.persist_subspace_coords, false);
  assert.equal(unsteeredReplayPlan.sampling.return_top_k, 5);
  assert.equal(unsteeredReplayPlan.readoutTopK, 1);
  assert.equal(unsteeredReplayPlan.measurementTargetRawIndex, 0);
  assert.equal(unsteeredReplayPlan.thinking, false);
  assert.equal(unsteeredReplayPlan.steeringExpression, null);
  assert.equal(unsteeredReplayPlan.generationSeat, "assistant");
  assert.equal(unsteeredReplayPlan.generationRoleName, "pirate");
  assert.deepEqual(
    lensReplayProgress.map((event) => event.data.phase),
    ["context", "readout", "complete"],
  );
  assert.deepEqual(
    lensReplayProgress.map((event) => event.data.progress),
    [0, 0.9, 1],
  );
  assert.deepEqual(
    lensReplay.measurements.instruments.lens.readout.layers.map((row) => row.tokens.length),
    [1, 1],
  );
  assert.equal(lensReplay.measurements.instruments.lens.readout.aggregate.length, 1);
  assert.equal(
    lensReplay.measurements.instruments.lens.readout.aggregate[0].token,
    "aggregate-only",
  );
  assert.equal(replayLive.lens, false);
  assert.equal(replayLive.sae, true);
  replayCompiler.setInstrumentLive("sae", false);
  assert.equal((await exactReplayRuntime.request({
    service: "sessions", method: "get", args: [],
  })).instruments.find((row) => row.family === "lens").live.enabled, false);

  const rawReplay = await exactReplayRuntime.request({
    service: "instruments",
    method: "tokenReadout",
    args: ["lens", branchA.id, 1, { topK: 1, steered: true, raw: true, layers: "all" }],
  });
  assert.equal(rawReplay.measurements.instruments.lens.binding.steering, "left:role-pirate");
  const rawReplayPlan = exactReplayEngine.plans[1];
  assert.deepEqual(rawReplayPlan.input, { kind: "raw", prompt: "Question" });
  assert.deepEqual(rawReplayPlan.replay.forcedPrefixTokenIds, [10, 11]);
  assert.equal(rawReplayPlan.readoutTopK, 1);
  assert.equal(rawReplayPlan.generationSeat, null);
  assert.equal(rawReplayPlan.generationRoleName, null);

  const joint = await exactReplayRuntime.request({
    service: "tree", method: "jointLogprobs", args: [branchA.id, branchB.id],
  });
  assert.equal(joint.parent_id, replayParent.id);
  assert.equal(joint.rows.length, 3);
  assert.deepEqual(joint.rows[0], {
    a_index: 0,
    b_index: 0,
    a_text: "a",
    b_text: "a",
    aligned: true,
    lp_a_in_a: -0.1,
    lp_b_in_b: -1.1,
    lp_a_in_b: -1.1,
    lp_b_in_a: -0.1,
    rank_changed: true,
    approx_kl: joint.rows[0].approx_kl,
  });
  assert.ok(Number.isFinite(joint.rows[0].approx_kl));
  assert.equal(joint.rows[1].aligned, false);
  assert.equal(joint.rows[1].lp_a_in_b, null);
  assert.equal(joint.rows[1].approx_kl, null);
  assert.equal(joint.rows[2].aligned, false);
  assert.equal(joint.n_rank1_changed, 1);
  const jointPlans = exactReplayEngine.plans.slice(2);
  assert.equal(jointPlans.length, 4);
  assert.deepEqual(jointPlans[0].replay.scoreTokenIds, [10, 11, 12]);
  assert.deepEqual(jointPlans[2].replay.scoreTokenIds, [10, 11, 12, 20, 40]);
  assert.deepEqual(jointPlans[3].replay.scoreTokenIds, [10, 11, 12, 20, 30]);
  assert.equal(jointPlans[0].generationRoleName, "pirate");
  assert.equal(jointPlans[0].hookProgram.activeRole, "pirate");

  const beforeForkPlan = exactReplayEngine.plans.length;
  assert.equal(branchA.tokens[1].top_alts[1].id, 42);
  const forkEvents = [];
  await exactReplayRuntime.generate({
    type: "generate",
    fork_node_id: branchA.id,
    fork_raw_index: 1,
    fork_alt_token_id: 42,
  }, (event) => forkEvents.push(event));
  const forkPlan = exactReplayEngine.plans[beforeForkPlan];
  assert.deepEqual(forkPlan.replay.forcedPrefixTokenIds, [10, 42]);
  assert.equal(forkPlan.sampling.max_tokens, 6);
  assert.equal(forkPlan.sampling.seed, 7);
  assert.equal(forkPlan.sampling.user_role, "reader");
  assert.equal(forkPlan.sampling.assistant_role, "pirate");
  assert.equal(forkPlan.sampling.persist_per_layer_scores, true);
  assert.equal(forkPlan.sampling.persist_subspace_coords, false);
  assert.equal(forkPlan.sampling.return_top_k, 5);
  assert.equal(forkPlan.steeringExpression, "left:role-pirate");
  assert.equal(forkPlan.generationSeat, "assistant");
  assert.equal(forkPlan.generationRoleName, "pirate");
  const forked = exactReplayRuntime.snapshot().tree.nodes.at(-1);
  assert.equal(forked.parent_id, replayParent.id);
  assert.equal(forked.role, "assistant");
  assert.equal(forked.role_label, "pirate");
  assert.deepEqual(forked.raw_token_ids, [10, 42, 99]);
  assert.equal(forkEvents.some((event) => event.type === "done"), true);
  const forkBeginning = forkEvents.find((event) => event.type === "tree_mutated" &&
    event.op === "begin_assistant").added[0];
  assert.equal(forkBeginning.text, "a");
  assert.deepEqual(forkBeginning.tokens, branchA.tokens.slice(0, 1));
  assert.equal(forkPlan.measurementStartRawIndex, 1);
  assert.deepEqual(forkEvents.filter((event) => event.type === "token")
    .map((event) => event.text), ["d", "!"]);
  assert.deepEqual(forkEvents.filter((event) => event.type === "generation_progress")
    .map((event) => [event.completed, event.total]), [[0, 2], [2, 2]]);
  assert.equal(forked.text, "ad!");

  const resampledEvents = [];
  await exactReplayRuntime.generate({
    type: "generate", fork_node_id: branchA.id, fork_raw_index: 1,
    fork_alt_token_id: 11, fork_seed: 123,
  }, (event) => resampledEvents.push(event));
  assert.equal(exactReplayEngine.plans.at(-1).sampling.seed, 123);
  assert.deepEqual(exactReplayEngine.plans.at(-1).replay.forcedPrefixTokenIds, [10, 11]);
  assert.equal(exactReplayRuntime.snapshot().tree.nodes.find((node) => node.id === branchA.id)
    .recipe.sampling.seed, 7);
  assert.equal(resampledEvents.find((event) => event.type === "tree_mutated" &&
    event.op === "begin_assistant").added[0].text, "ab");
  assert.deepEqual(resampledEvents.filter((event) => event.type === "token")
    .map((event) => event.text), ["!"]);

  const stoppedPrefixEvents = [];
  const stoppedPrefixRuntime = new BrowserLoomRuntime({
    session: exactReplayRuntime.snapshot().session,
    initialTree: replayTree,
    generation: {
      ...exactReplayEngine,
      async stop() {},
      async streamGeneration(plan) {
        await plan.onRawToken({ tokenId: 10, rawIndex: 0 });
        await stoppedPrefixRuntime.stop();
        return generated({ text: "a", tokens: 1, terminalReason: "external_stop",
          usage: { promptTokens: 4, completionTokens: 1, totalTokens: 5 } });
      },
    },
    createId: ids("stop-prefix"),
    compileSteering() { return { hookAbi: "post-block-residual-v4", activeRole: "pirate" }; },
  });
  await stoppedPrefixRuntime.generate({
    type: "generate", fork_node_id: branchA.id, fork_raw_index: 1, fork_alt_token_id: 11,
  }, (event) => stoppedPrefixEvents.push(event));
  const stoppedPrefix = stoppedPrefixRuntime.snapshot().tree.nodes.at(-1);
  assert.equal(stoppedPrefix.text, "ab");
  assert.deepEqual(stoppedPrefix.raw_token_ids, [10, 11]);
  assert.equal(stoppedPrefix.finish_reason, "cancelled");
  assert.equal(stoppedPrefixEvents.at(-1).result.text, "ab");
  assert.deepEqual(stoppedPrefixRuntime.snapshot().tree.nodes.find((node) => node.id === branchA.id), branchA);

  const beforeTextForkPlan = exactReplayEngine.plans.length;
  await exactReplayRuntime.generate({
    type: "generate",
    fork_node_id: branchA.id,
    fork_raw_index: 1,
    fork_replacement_text: "XY",
  }, () => {});
  const textForkPlan = exactReplayEngine.plans[beforeTextForkPlan];
  assert.deepEqual(textForkPlan.replay.forcedPrefixTokenIds, [10, 88, 89]);
  assert.equal(textForkPlan.sampling.max_tokens, 7);
  assert.equal(textForkPlan.sampling.seed, 7);
  assert.equal(textForkPlan.steeringExpression, "left:role-pirate");
  const textForked = exactReplayRuntime.snapshot().tree.nodes.at(-1);
  assert.equal(textForked.parent_id, replayParent.id);
  assert.deepEqual(textForked.raw_token_ids, [10, 88, 89, 99]);

  const statelessTree = chatRuntime.snapshot().tree;
  const statelessEngine = generation();
  const statelessRuntime = new BrowserLoomRuntime({
    session: session({ system_prompt: "Stay precise." }),
    generation: statelessEngine,
    initialTree: statelessTree,
    createId: ids("stateless"),
  });
  const beforeStateless = statelessRuntime.snapshot().tree;
  const statelessEvents = [];
  await statelessRuntime.generate(
    {
      type: "generate",
      input: [{ role: "user", content: "Shadow" }],
      stateless: true,
      raw: false,
    },
    (event) => statelessEvents.push(event),
  );
  assert.equal(statelessEvents.some((event) => event.type === "tree_mutated"), false);
  assert.deepEqual(statelessRuntime.snapshot().tree, beforeStateless);
  assert.deepEqual(statelessEngine.plans[0].input, {
    kind: "chat",
    messages: [
      { role: "system", content: "Stay precise." },
      { role: "user", content: "Shadow" },
    ],
  });

  const stringStatelessEngine = generation();
  const stringStatelessRuntime = new BrowserLoomRuntime({
    session: session({ system_prompt: "Stay precise." }),
    generation: stringStatelessEngine,
    initialTree: statelessTree,
    createId: ids("string-stateless"),
  });
  const stringStatelessBefore = stringStatelessRuntime.snapshot().tree;
  const stringStatelessEvents = [];
  await stringStatelessRuntime.generate({
    type: "generate",
    input: "Fresh prompt",
    sampling: { user_role: "researcher" },
  }, (event) => stringStatelessEvents.push(event));
  assert.equal(stringStatelessEvents.some((event) => event.type === "tree_mutated"), false);
  assert.deepEqual(stringStatelessRuntime.snapshot().tree, stringStatelessBefore);
  assert.deepEqual(stringStatelessEngine.plans[0].input, {
    kind: "chat",
    messages: [
      { role: "system", content: "Stay precise." },
      { role: "user", content: "Fresh prompt", name: "researcher" },
    ],
  });

  const stringStatefulEngine = generation();
  const stringStatefulRuntime = new BrowserLoomRuntime({
    session: session({ system_prompt: "Stay precise." }),
    generation: stringStatefulEngine,
    initialTree: statelessTree,
    createId: ids("string-stateful"),
  });
  const stringStatefulEvents = [];
  await stringStatefulRuntime.generate({
    type: "generate",
    input: "Fresh prompt",
    stateless: false,
    sampling: { user_role: "researcher" },
  }, (event) => stringStatefulEvents.push(event));
  assert.deepEqual(
    stringStatefulEvents.filter((event) => event.type === "tree_mutated")
      .map((event) => event.op),
    ["add_user", "begin_assistant", "finalize_assistant"],
  );
  assert.deepEqual(stringStatefulEngine.plans[0].input, {
    kind: "chat",
    messages: [
      { role: "system", content: "Stay precise." },
      { role: "user", content: "Question" },
      { role: "assistant", content: "Local answer" },
      { role: "user", content: "Fresh prompt", name: "researcher" },
    ],
  });
  assert.deepEqual(
    stringStatefulRuntime.snapshot().tree.nodes.slice(-2).map((node) => node.role),
    ["user", "assistant"],
  );
  assert.equal(stringStatefulRuntime.snapshot().tree.nodes.at(-2).text, "Fresh prompt");

  const rawStatelessEngine = generation();
  const rawStatelessRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: rawStatelessEngine,
    initialTree: statelessTree,
    createId: ids("raw-stateless"),
  });
  const rawStatelessBefore = rawStatelessRuntime.snapshot().tree;
  await rawStatelessRuntime.generate({
    type: "generate",
    input: " flat tail",
    raw: true,
  }, () => {});
  assert.deepEqual(rawStatelessEngine.plans[0].input, {
    kind: "raw",
    prompt: " flat tail",
  });
  assert.deepEqual(rawStatelessRuntime.snapshot().tree, rawStatelessBefore);

  const rawStatefulEngine = generation();
  const rawStatefulRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: rawStatefulEngine,
    initialTree: statelessTree,
    createId: ids("raw-stateful"),
  });
  await rawStatefulRuntime.generate({
    type: "generate",
    input: " flat tail",
    raw: true,
    stateless: false,
  }, () => {});
  assert.deepEqual(rawStatefulEngine.plans[0].input, {
    kind: "raw",
    prompt: "QuestionLocal answer flat tail",
  });
  assert.equal(rawStatefulRuntime.snapshot().tree.nodes.at(-2).text, " flat tail");

  const hookProgram = { hookAbi: "post-block-residual-v4" };
  const compiled = [];
  const steeringEvents = [];
  const steeringEngine = generation();
  const steeringRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: steeringEngine,
    createId: ids("steering"),
    compileSteering(expression) {
      compiled.push(expression);
      return hookProgram;
    },
  });
  await steeringRuntime.generate(
    {
      type: "submit",
      text: "Prompt",
      authored_role: "user",
      generated_role: "assistant",
      steering: "0.5 honest",
    },
    (event) => steeringEvents.push(event),
  );
  assert.deepEqual(compiled, ["0.5 honest"]);
  assert.equal(steeringEngine.plans[0].hookProgram, hookProgram);
  assert.equal(steeringRuntime.snapshot().tree.nodes[2].applied_steering, "0.5 honest");
  assert.equal(steeringEvents.at(-1).result.applied_steering, "0.5 honest");
  const comparisonEvents = [];
  await steeringRuntime.generate({
    type: "generate",
    input: [{ role: "user", content: "Prompt" }],
    parent_node_id: steeringRuntime.snapshot().tree.nodes[2].id,
    stateless: true,
    recipe_override: "unsteered",
  }, event => comparisonEvents.push(event));
  assert.equal(comparisonEvents.at(-1).result.applied_steering, null);
  assert.equal(steeringEngine.plans.at(-1).steeringExpression, null);
  assert.deepEqual(steeringEngine.plans.at(-1).sampling, steeringEngine.plans[0].sampling);

  const validationCompiles = [];
  const validationEngine = generation();
  const validationRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: validationEngine,
    createId: ids("validation"),
    compileSteering(expression, probeRequests) {
      validationCompiles.push({ expression, probeRequests });
      if (expression.includes("wrong-variant")) throw new Error("variant is unavailable");
      if (expression.includes("missing-instrument")) throw new Error("instrument is unavailable");
      if (expression.includes("over-capacity")) throw new Error("structured capacity exceeded");
      if (expression.includes("missing")) throw new Error("selector is missing");
      return hookProgram;
    },
  });
  const validationTreeBefore = validationRuntime.snapshot().tree;
  const prospectiveProbes = [{
    selector: "local/emotions:sae-release",
    name: "mood alias",
    top_n: 4,
  }];
  assert.deepEqual(await validationRuntime.request({
    service: "sessions",
    method: "validateSteering",
    args: ["0.5 calm", undefined, { probeRequests: prospectiveProbes, tree: validationTreeBefore }],
  }), { valid: true, expression: "0.5 calm", error: null });
  assert.deepEqual(await validationRuntime.request({
    service: "sessions",
    method: "validateSteering",
    args: ["  calm  "],
  }), { valid: true, expression: "0.5 calm", error: null });
  assert.deepEqual(validationCompiles, [
    {
      expression: "0.5 calm",
      probeRequests: prospectiveProbes,
    },
    {
      expression: "  calm  ",
      probeRequests: undefined,
    },
  ]);
  for (const [expression, message] of [
    ["missing", /selector is missing/],
    ["wrong-variant", /variant is unavailable/],
    ["missing-instrument", /instrument is unavailable/],
    ["over-capacity", /structured capacity exceeded/],
  ]) {
    const result = await validationRuntime.request({
      service: "sessions", method: "validateSteering", args: [expression],
    });
    assert.equal(result.valid, false);
    assert.match(result.error, message);
  }
  const invalidImportTree = structuredClone(validationTreeBefore);
  invalidImportTree.active_node_id = "missing-node";
  const invalidTreeValidation = await validationRuntime.request({
    service: "sessions",
    method: "validateSteering",
    args: ["0.5 calm", undefined, { tree: invalidImportTree }],
  });
  assert.equal(invalidTreeValidation.valid, false);
  assert.deepEqual(validationRuntime.snapshot().tree, validationTreeBefore);
  assert.equal(validationEngine.plans.length, 0);

  const activeRoleEngine = generation();
  const activeRoleRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: activeRoleEngine,
    createId: ids("active-role"),
    compileSteering(expression) {
      return {
        hookAbi: "post-block-residual-v4",
        activeRole: expression.includes(":role-pirate") ? "pirate" : null,
      };
    },
  });
  const roleSteering = "0.5 honest.deceptive:role-pirate";
  await activeRoleRuntime.generate({
    type: "submit",
    text: "Tell me a story",
    authored_role: "user",
    generated_role: "assistant",
    steering: roleSteering,
    sampling: { user_role: "reader", assistant_role: "guide" },
  }, () => {});
  assert.equal(activeRoleEngine.plans[0].generationSeat, "assistant");
  assert.equal(activeRoleEngine.plans[0].generationRoleName, "pirate");
  assert.equal(activeRoleRuntime.snapshot().tree.nodes.at(-1).role_label, "pirate");
  await activeRoleRuntime.generate({
    type: "generate",
    input: null,
    stateless: false,
    generate_seat: "user",
    steering: roleSteering,
    sampling: { user_role: "reader" },
  }, () => {});
  assert.equal(activeRoleEngine.plans[1].generationSeat, "user");
  assert.equal(activeRoleEngine.plans[1].generationRoleName, "reader");
  assert.equal(activeRoleRuntime.snapshot().tree.nodes.at(-1).role_label, "reader");
  await activeRoleRuntime.generate({
    type: "generate",
    input: null,
    stateless: false,
    raw: true,
    steering: roleSteering,
  }, () => {});
  assert.equal(activeRoleEngine.plans[2].generationSeat, null);
  assert.equal(activeRoleEngine.plans[2].generationRoleName, null);
  assert.equal(activeRoleRuntime.snapshot().tree.nodes.at(-1).role_label, null);

  const recipeExpressions = [];
  const recipeEngine = generation();
  const recipeRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: recipeEngine,
    createId: ids("recipe"),
    compileSteering(expression) {
      recipeExpressions.push(expression);
      return hookProgram;
    },
  });
  await recipeRuntime.generate({
    type: "submit",
    text: "Prompt",
    authored_role: "user",
    generated_role: "assistant",
    steering: "0.5 calm",
    sampling: { temperature: 0.8, seed: 7 },
  }, () => {});
  let recipeParent = recipeRuntime.snapshot().tree.active_node_id;
  await recipeRuntime.generate({
    type: "generate",
    parent_node_id: recipeParent,
    stateless: false,
    recipe_override: "inverted",
  }, () => {});
  recipeParent = recipeRuntime.snapshot().tree.active_node_id;
  assert.equal(recipeEngine.plans.at(-1).steeringExpression, "-0.5 calm");
  assert.equal(recipeEngine.plans.at(-1).sampling.temperature, 0.8);
  assert.equal(recipeEngine.plans.at(-1).sampling.seed, 7);
  await recipeRuntime.generate({
    type: "generate",
    parent_node_id: recipeParent,
    stateless: false,
    recipe_override: "cool",
  }, () => {});
  recipeParent = recipeRuntime.snapshot().tree.active_node_id;
  assert.equal(recipeEngine.plans.at(-1).sampling.temperature, 0.3);
  assert.equal(recipeEngine.plans.at(-1).sampling.seed, 7);
  await recipeRuntime.generate({
    type: "generate",
    parent_node_id: recipeParent,
    stateless: false,
    recipe_override: "seed=42, temperature=1.5",
  }, () => {});
  recipeParent = recipeRuntime.snapshot().tree.active_node_id;
  assert.equal(recipeEngine.plans.at(-1).sampling.temperature, 1.5);
  assert.equal(recipeEngine.plans.at(-1).sampling.seed, 42);
  await recipeRuntime.generate({
    type: "generate",
    parent_node_id: recipeParent,
    stateless: false,
    recipe_override: "steering=0.6,0.3 emotions%0.2,0.8, temperature=0.4",
  }, () => {});
  recipeParent = recipeRuntime.snapshot().tree.active_node_id;
  assert.equal(recipeEngine.plans.at(-1).steeringExpression, "0.6,0.3 emotions%0.2,0.8");
  assert.equal(recipeEngine.plans.at(-1).sampling.temperature, 0.4);
  await recipeRuntime.generate({
    type: "generate",
    parent_node_id: recipeParent,
    stateless: false,
    recipe_override: "inverted",
  }, () => {});
  recipeParent = recipeRuntime.snapshot().tree.active_node_id;
  assert.equal(recipeEngine.plans.at(-1).steeringExpression, "-0.6,-0.3 emotions%0.2,0.8");
  await recipeRuntime.generate({
    type: "generate",
    parent_node_id: recipeParent,
    stateless: false,
    recipe_override: "unsteered",
  }, () => {});
  assert.equal(recipeEngine.plans.at(-1).steeringExpression, null);
  assert.equal(recipeEngine.plans.at(-1).hookProgram, null);
  assert.equal(recipeRuntime.snapshot().tree.nodes.at(-1).recipe.steering, null);
  assert.deepEqual(recipeExpressions, [
    "0.5 calm",
    "-0.5 calm",
    "-0.5 calm",
    "-0.5 calm",
    "0.6,0.3 emotions%0.2,0.8",
    "-0.6,-0.3 emotions%0.2,0.8",
  ]);
  const invalidRecipeEvents = [];
  await assert.rejects(
    recipeRuntime.generate({
      type: "generate",
      recipe_override: "temperature=fast",
    }, (event) => invalidRecipeEvents.push(event)),
    (error) => error.code === "INVALID_RECIPE_OVERRIDE",
  );
  assert.deepEqual(invalidRecipeEvents, []);
  const plansBeforeExactCoords = recipeEngine.plans.length;
  await recipeRuntime.generate({
    type: "generate",
    input: [{ role: "user", content: "Exact trails" }],
    stateless: true,
    recipe_override: {
      sampling: { persist_subspace_coords: true },
    },
  }, () => {});
  assert.equal(recipeEngine.plans.length, plansBeforeExactCoords + 1);
  assert.equal(recipeEngine.plans.at(-1).sampling.persist_subspace_coords, true);
  await recipeRuntime.generate({
    type: "generate",
    stateless: false,
    recipe_override: {
      sampling: {
        user_role: "critic",
        assistant_role: "guide",
        persist_per_layer_scores: true,
        persist_subspace_coords: false,
      },
    },
  }, () => {});
  const replayedRecipePlan = recipeEngine.plans.at(-1);
  assert.equal(replayedRecipePlan.sampling.user_role, "critic");
  assert.equal(replayedRecipePlan.sampling.assistant_role, "guide");
  assert.equal(replayedRecipePlan.sampling.persist_per_layer_scores, true);
  assert.equal(replayedRecipePlan.sampling.persist_subspace_coords, false);
  assert.equal(replayedRecipePlan.generationRoleName, "guide");
  assert.equal(recipeRuntime.snapshot().tree.nodes.at(-1).role_label, "guide");
  assert.equal(
    recipeRuntime.snapshot().tree.nodes.at(-1).recipe.sampling.persist_per_layer_scores,
    true,
  );
  assert.equal(
    recipeRuntime.snapshot().tree.nodes.at(-1).recipe.sampling.persist_subspace_coords,
    false,
  );

  const unsupportedRoles = session();
  unsupportedRoles.role_substitution_supported = false;
  unsupportedRoles.user_role_supported = false;
  const unsupportedRoleRuntime = new BrowserLoomRuntime({
    session: unsupportedRoles,
    generation: generation(),
    createId: ids("unsupported-role"),
  });
  await assert.rejects(
    unsupportedRoleRuntime.generate({
      type: "generate",
      input: [{ role: "user", content: "Named replay", label: "critic" }],
    }, () => {}),
    (error) => error.code === "ROLE_SUBSTITUTION_UNAVAILABLE",
  );
  await assert.rejects(
    unsupportedRoleRuntime.generate({
      type: "generate",
      recipe_override: { sampling: { assistant_role: "guide" } },
    }, () => {}),
    (error) => error.code === "ROLE_SUBSTITUTION_UNAVAILABLE",
  );

  const castExpressions = [];
  const castEngine = generation();
  const castRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: castEngine,
    createId: ids("cast-recipe"),
    compileSteering(expression) {
      castExpressions.push(expression);
      return hookProgram;
    },
  });
  await castRuntime.request({
    service: "tree",
    method: "castPut",
    args: ["assistant", { steering: "0.25 calm", seed: 9, notes: "standing" }],
  });
  await castRuntime.generate({
    type: "submit",
    text: "Use the standing recipe",
    authored_role: "user",
    generated_role: "assistant",
  }, () => {});
  assert.equal(castEngine.plans.at(-1).steeringExpression, "0.25 calm");
  assert.equal(castEngine.plans.at(-1).sampling.seed, 9);
  let castParent = castRuntime.snapshot().tree.active_node_id;
  await castRuntime.generate({
    type: "generate",
    parent_node_id: castParent,
    stateless: false,
    steering: "0.5 direct",
    sampling: { seed: 3 },
  }, () => {});
  assert.equal(castEngine.plans.at(-1).steeringExpression, "0.5 direct");
  assert.equal(castEngine.plans.at(-1).sampling.seed, 3);
  castParent = castRuntime.snapshot().tree.active_node_id;
  await castRuntime.generate({
    type: "generate",
    parent_node_id: castParent,
    stateless: false,
    recipe_override: "unsteered",
  }, () => {});
  assert.equal(castEngine.plans.at(-1).steeringExpression, null);
  assert.deepEqual(castExpressions, ["0.25 calm", "0.5 direct"]);
  const clearedEvents = [];
  await castRuntime.generate({
    type: "submit",
    text: "The controls have been cleared",
    authored_role: "user",
    generated_role: "assistant",
    steering: "",
  }, event => clearedEvents.push(event));
  assert.equal(castEngine.plans.at(-1).steeringExpression, null);
  assert.equal(clearedEvents.at(-1).result.applied_steering, null);

  const unavailableSteering = new BrowserLoomRuntime({
    session: session(),
    generation: generation(),
    createId: ids("unavailable"),
  });
  await assert.rejects(
    unavailableSteering.generate(
      {
        type: "submit",
        text: "Prompt",
        authored_role: "user",
        generated_role: "assistant",
        steering: "honest",
      },
      () => {},
    ),
    (error) => error.code === "STEERING_COMPILER_UNAVAILABLE",
  );
  assert.equal(unavailableSteering.snapshot().tree.nodes.length, 1);

  for (const rejectedRequest of [{
    request: {
      type: "submit",
      text: "Prompt",
      authored_role: "user",
      generated_role: "assistant",
      thinking: true,
    },
    code: "THINKING_STREAM_UNAVAILABLE",
  }]) {
    const preflightRuntime = new BrowserLoomRuntime({
      session: session(),
      generation: generation(),
      createId: ids(`preflight-${rejectedRequest.code}`),
    });
    const preflightEvents = [];
    await assert.rejects(
      preflightRuntime.generate(
        rejectedRequest.request,
        (event) => preflightEvents.push(event),
      ),
      (error) => error.code === rejectedRequest.code,
    );
    assert.deepEqual(preflightEvents, []);
    assert.equal(preflightRuntime.snapshot().tree.nodes.length, 1);
  }

  const failedRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: generation({ error: new Error("decode failed") }),
    createId: ids("failed"),
  });
  const failedEvents = [];
  await assert.rejects(
    failedRuntime.generate(
      {
        type: "submit",
        text: "Prompt",
        authored_role: "user",
        generated_role: "assistant",
      },
      (event) => failedEvents.push(event),
    ),
    /decode failed/,
  );
  assert.deepEqual(failedEvents.map((event) => event.type), [
    "started", "tree_mutated", "tree_mutated", "tree_mutated",
  ]);
  assert.equal(failedEvents.at(-1).op, "delete");
  const failedTree = failedRuntime.snapshot().tree;
  assert.equal(failedTree.nodes.length, 2);
  assert.equal(failedTree.active_node_id, failedTree.nodes[1].id);

  for (const mode of ["resolve", "reject"]) {
    const engine = cancellableGeneration(mode);
    const runtime = new BrowserLoomRuntime({
      session: session(),
      generation: engine,
      createId: ids(`cancel-${mode}`),
    });
    const events = [];
    const pending = runtime.generate({
      type: "submit",
      text: "Keep the partial reply",
      authored_role: "user",
      generated_role: "assistant",
    }, (event) => events.push(event));
    await engine.started;
    await runtime.stop();
    await pending;

    assert.equal(engine.stops, 1);
    assert.deepEqual(
      events.filter((event) => event.type === "tree_mutated").map((event) => event.op),
      ["add_user", "begin_assistant", "finalize_assistant"],
    );
    const finalized = events.find(
      (event) => event.type === "tree_mutated" && event.op === "finalize_assistant",
    );
    const done = events.find((event) => event.type === "done");
    assert.equal(finalized.updated[0].text, "Partial");
    assert.equal(finalized.updated[0].finish_reason, "cancelled");
    assert.equal(done.result.text, "Partial");
    assert.equal(done.result.finish_reason, "cancelled");
    assert.ok(events.indexOf(finalized) < events.indexOf(done));
    assert.equal(events.some(
      (event) => event.type === "tree_mutated" && event.op === "delete",
    ), false);
    const tree = runtime.snapshot().tree;
    assert.equal(tree.nodes.length, 3);
    assert.equal(tree.nodes[2].text, "Partial");
    assert.equal(tree.nodes[2].finish_reason, "cancelled");
    assert.equal(tree.nodes[2].mean_logprob, -0.4);
  }

  {
    let markPreparing;
    let finishPreparing;
    const preparing = new Promise(resolve => { markPreparing = resolve; });
    const prepared = new Promise(resolve => { finishPreparing = resolve; });
    const engine = generation();
    const generate = engine.streamGeneration.bind(engine);
    const signals = [];
    engine.streamGeneration = async (plan, onToken) => {
      signals.push(plan.signal);
      markPreparing();
      await prepared;
      plan.signal?.throwIfAborted();
      return generate(plan, onToken);
    };
    const runtime = new BrowserLoomRuntime({
      session: session(), generation: engine, createId: ids("cancel-setup"),
    });
    const events = [];
    const request = { type: "submit", text: "Stop before decoding", authored_role: "user", generated_role: "assistant" };
    const pending = runtime.generate(request, event => events.push(event));
    await preparing;
    await runtime.stop();
    finishPreparing();
    await pending;
    assert.equal(events.filter(event => event.type === "token").length, 0);
    assert.equal(events.find(event => event.type === "done").result.finish_reason, "cancelled");
    assert.equal(engine.plans.length, 0, "a stop during setup must prevent decoding from starting");
    const nextEvents = [];
    await runtime.generate(request, event => nextEvents.push(event));
    assert.equal(engine.plans.length, 1);
    assert.equal(signals[0].aborted, true);
    assert.equal(signals[1].aborted, false);
    assert.equal(nextEvents.find(event => event.type === "done").result.text, "Local answer");
  }

  const usageAwareCancellation = webLlmCancellableGeneration(streamWebLlmGeneration);
  const usageAwareRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: usageAwareCancellation,
    createId: ids("cancel-usage"),
  });
  const usageAwareEvents = [];
  const usageAwarePending = usageAwareRuntime.generate({
    type: "submit",
    text: "Keep prompt usage",
    authored_role: "user",
    generated_role: "assistant",
  }, (event) => usageAwareEvents.push(event));
  await usageAwareCancellation.started;
  await usageAwareRuntime.stop();
  await usageAwarePending;
  assert.deepEqual(
    usageAwareEvents.find((event) => event.type === "done").result.usage,
    { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
  );

  const internalAbortEngine = cancellableGeneration("reject");
  const internalAbortRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: internalAbortEngine,
    createId: ids("cancel-internal"),
  });
  const internalAbortEvents = [];
  const internalAbort = internalAbortRuntime.generate({
    type: "submit",
    text: "This backend interruption is not a user stop",
    authored_role: "user",
    generated_role: "assistant",
  }, (event) => internalAbortEvents.push(event));
  await internalAbortEngine.started;
  await internalAbortRuntime.stop(false);
  await assert.rejects(internalAbort, (error) => error.name === "AbortError");
  assert.deepEqual(
    internalAbortEvents
      .filter((event) => event.type === "tree_mutated")
      .map((event) => event.op),
    ["add_user", "begin_assistant", "delete"],
  );
  assert.equal(internalAbortRuntime.snapshot().tree.nodes.length, 2);

  const concurrencyEngine = cancellableGeneration("resolve");
  const concurrencyRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: concurrencyEngine,
    initialTree: chatRuntime.snapshot().tree,
    now: () => 77,
    createId: ids("concurrent"),
  });
  const startingTree = concurrencyRuntime.snapshot().tree;
  const originalUser = startingTree.nodes.find((node) => node.role === "user");
  const originalAssistant = startingTree.nodes.find((node) =>
    node.role === "assistant" && node.parent_id === originalUser.id
  );
  const outsideEdit = await concurrencyRuntime.request({
    service: "tree",
    method: "branch",
    args: [originalUser.id, "outside edit", undefined, "user"],
  });
  const outsideDelete = await concurrencyRuntime.request({
    service: "tree",
    method: "branch",
    args: [originalUser.id, "outside delete", undefined, "user"],
  });
  await concurrencyRuntime.request({
    service: "tree",
    method: "navigate",
    args: [originalAssistant.id],
  });
  const concurrentMutations = [];
  const concurrentGeneration = concurrencyRuntime.generate({
    type: "submit",
    text: "reserved prompt",
    authored_role: "user",
    generated_role: "assistant",
  }, (event) => {
    if (event.type === "tree_mutated") concurrentMutations.push(event);
  });
  await concurrencyEngine.started;
  const reservedUser = concurrentMutations.find((event) => event.op === "add_user").added[0];
  const liveAssistant = concurrentMutations.find(
    (event) => event.op === "begin_assistant",
  ).added[0];
  const captureConcurrentMutation = (event) => concurrentMutations.push(event.data);

  await concurrencyRuntime.request({
    service: "tree",
    method: "navigate",
    args: [outsideEdit.node_id],
  }, captureConcurrentMutation);
  const liveSibling = await concurrencyRuntime.request({
    service: "tree",
    method: "branch",
    args: [liveAssistant.id, "safe sibling", undefined, "assistant"],
  }, captureConcurrentMutation);
  await concurrencyRuntime.request({
    service: "tree",
    method: "star",
    args: [liveAssistant.id, true],
  }, captureConcurrentMutation);
  await concurrencyRuntime.request({
    service: "tree",
    method: "note",
    args: [liveAssistant.id, "keep the partial"],
  }, captureConcurrentMutation);
  await concurrencyRuntime.request({
    service: "tree",
    method: "edit",
    args: [outsideEdit.node_id, "edited outside"],
  }, captureConcurrentMutation);
  await concurrencyRuntime.request({
    service: "tree",
    method: "navigate",
    args: [outsideDelete.node_id],
  }, captureConcurrentMutation);
  await concurrencyRuntime.request({
    service: "tree",
    method: "delete",
    args: [outsideDelete.node_id],
  }, captureConcurrentMutation);
  await concurrencyRuntime.request({
    service: "tree",
    method: "navigate",
    args: [outsideEdit.node_id],
  }, captureConcurrentMutation);

  for (const [method, args] of [
    ["edit", [reservedUser.id, "blocked root"]],
    ["edit", [liveAssistant.id, "blocked descendant"]],
    ["edit", [liveSibling.node_id, "blocked sibling"]],
    ["delete", [originalAssistant.id]],
    ["delete", [reservedUser.id]],
    ["restore", [concurrencyRuntime.snapshot().tree]],
    ["reset", []],
    ["transcriptLoad", ["", "default", true]],
  ]) {
    await assert.rejects(
      concurrencyRuntime.request({ service: "tree", method, args }, captureConcurrentMutation),
      (error) => error.code === "MUTATION_DURING_GENERATION" && error.status === 409,
    );
  }
  await concurrencyRuntime.stop();
  await concurrentGeneration;

  const concurrentTree = concurrencyRuntime.snapshot().tree;
  const finalizedConcurrentNode = concurrentTree.nodes.find(
    (node) => node.id === liveAssistant.id,
  );
  assert.equal(finalizedConcurrentNode.text, "Partial");
  assert.equal(finalizedConcurrentNode.starred, true);
  assert.equal(finalizedConcurrentNode.notes, "keep the partial");
  assert.equal(finalizedConcurrentNode.finish_reason, "cancelled");
  assert.equal(concurrentTree.active_node_id, outsideEdit.node_id);
  assert.equal(concurrentTree.nodes.find((node) => node.id === outsideEdit.node_id).text, "edited outside");
  assert.equal(concurrentTree.nodes.some((node) => node.id === outsideDelete.node_id), false);
  assert.equal(concurrentTree.nodes.some((node) => node.id === liveSibling.node_id), true);
  assert.deepEqual(
    concurrentMutations.map((event) => event.rev),
    concurrentMutations.map((event) => event.rev).toSorted((left, right) => left - right),
  );
  assert.equal(
    new Set(concurrentMutations.map((event) => event.rev)).size,
    concurrentMutations.length,
  );
  assert.equal(concurrentMutations.at(-1).op, "finalize_assistant");
  assert.equal(concurrentMutations.at(-1).active_node_id, null);

  const cappedEngine = generation();
  const cappedRuntime = new BrowserLoomRuntime({
    session: session({ max_tokens: 8_192 }),
    generation: cappedEngine,
    maxOutputTokens: 256,
    now: () => 122,
    createId: ids("capped"),
  });
  assert.equal(cappedRuntime.snapshot().session.config.max_tokens, 256);
  const cappedPatch = await cappedRuntime.request({
    service: "sessions",
    method: "patch",
    args: [{ max_tokens: 8_192 }],
  });
  assert.equal(cappedPatch.config.max_tokens, 256);
  await cappedRuntime.generate({
    type: "submit",
    text: "Keep this short",
    authored_role: "user",
    generated_role: "assistant",
    raw: false,
    sampling: { max_tokens: 8_192 },
  }, () => {});
  assert.equal(cappedEngine.plans[0].sampling.max_tokens, 256);
  assert.equal(cappedEngine.plans[0].maxOutputTokens, 256);

  const serviceRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: generation({ forcedReplay: false }),
    now: () => 123,
    createId: ids("service"),
  });
  const serviceMutationEvents = [];
  const captureServiceMutation = (event) => serviceMutationEvents.push(event);
  const patched = await serviceRuntime.request({
    service: "sessions",
    method: "patch",
    args: [{ temperature: 3, max_tokens: 32768 }],
  });
  assert.equal(patched.config.temperature, 3);
  assert.equal(patched.config.max_tokens, 32768);
  assert.deepEqual(await serviceRuntime.request({
    service: "profiles", method: "list", args: [],
  }), { profiles: [] });
  assert.deepEqual(await serviceRuntime.request({
    service: "profiles", method: "correlation", args: [[]],
  }), { names: [], matrix: {}, layers_shared: {} });
  assert.deepEqual(await serviceRuntime.request({
    service: "probes", method: "list", args: [],
  }), { probes: [] });
  assert.deepEqual(await serviceRuntime.request({
    service: "instruments", method: "sources", args: ["lens"],
  }), { sources: [] });
  assert.equal((await serviceRuntime.request({
    service: "sessions", method: "validateSteering", args: ["honest +"],
  })).valid, false);
  const initial = await serviceRuntime.request({ service: "tree", method: "get", args: [] });
  const restored = structuredClone(chatRuntime.snapshot().tree);
  restored.model_id = initial.model_id;
  restored.session_id = "portable-session";
  restored.nodes[2].aggregate_readings = { confidence: 0.8 };
  restored.nodes[2].tokens[0].probes = { confidence: 0.2 };
  restored.nodes[2].tokens[1].probes = { confidence: 0.9 };
  const restoreResult = await serviceRuntime.request({
    service: "tree", method: "restore", args: [restored],
  }, captureServiceMutation);
  assert.equal(restoreResult.rev, restored.rev + 1);
  assert.equal(serviceRuntime.snapshot().tree.session_id, initial.session_id);
  const restoredActive = await serviceRuntime.request({
    service: "tree", method: "active", args: [],
  });
  assert.equal(restoredActive.messages.length, restoredActive.node_ids.length);
  assert.equal(restoredActive.node_ids.includes(restored.root_id), false);
  assert.equal((await serviceRuntime.request({
    service: "sessions", method: "get", args: [],
  })).history_length, restoredActive.node_ids.length);
  const navigated = await serviceRuntime.request({
    service: "tree", method: "navigate", args: [restored.root_id],
  }, captureServiceMutation);
  assert.equal(navigated.active_node_id, restored.root_id);
  assert.equal(navigated.rev, restored.rev + 2);
  assert.deepEqual(navigated.messages, []);
  assert.deepEqual(navigated.node_ids, []);
  assert.equal((await serviceRuntime.request({
    service: "sessions", method: "get", args: [],
  })).history_length, 0);
  const staleRevisionTree = structuredClone(serviceRuntime.snapshot().tree);
  staleRevisionTree.rev = 0;
  const monotonicRestoreEvents = [];
  const monotonicRestore = await serviceRuntime.request({
    service: "tree", method: "restore", args: [staleRevisionTree],
  }, (event) => monotonicRestoreEvents.push(event));
  assert.equal(monotonicRestore.rev, navigated.rev + 1);
  assert.equal(monotonicRestoreEvents[0].data.rev, navigated.rev + 1);
  assert.equal(monotonicRestoreEvents[0].data.op, "restore");
  assert.deepEqual(await serviceRuntime.request({
    service: "tree", method: "filter", args: ["confidence >= 0.8"],
  }), {
    expr: "confidence >= 0.8",
    matching_node_ids: [restored.nodes[2].id],
  });
  assert.deepEqual(await serviceRuntime.request({
    service: "tree", method: "filter", args: ["any:confidence > 0.8"],
  }), {
    expr: "any:confidence > 0.8",
    matching_node_ids: [restored.nodes[2].id],
  });
  assert.deepEqual(await serviceRuntime.request({
    service: "tree", method: "filter", args: [""],
  }), { expr: "", matching_node_ids: [] });
  await assert.rejects(
    serviceRuntime.request({
      service: "tree", method: "filter", args: ["sometimes:confidence > 0"],
    }),
    (error) => error.code === "INVALID_TREE_FILTER" && error.status === 400,
  );
  assert.deepEqual(await serviceRuntime.request({
    service: "tree", method: "edgeLabel", args: [restored.root_id, restored.nodes[1].id],
  }), { label: "" });
  const compareBranch = await serviceRuntime.request({
    service: "tree",
    method: "branch",
    args: [restored.nodes[2].id, "A different local answer", undefined, "assistant"],
  });
  const compareTree = await serviceRuntime.request({
    service: "tree", method: "get", args: [],
  });
  const compareIndex = compareTree.nodes.findIndex((node) => node.id === compareBranch.node_id);
  compareTree.nodes[compareIndex].aggregate_readings = { confidence: 0.35 };
  await serviceRuntime.request({ service: "tree", method: "restore", args: [compareTree] });
  const compared = await serviceRuntime.request({
    service: "tree", method: "diff", args: [restored.nodes[2].id, compareBranch.node_id],
  });
  assert.equal(compared.parent_id, restored.nodes[1].id);
  assert.equal(compared.steering_delta, "");
  assert.deepEqual(compared.readings, [{
    name: "confidence", delta: -0.45, a_value: 0.8, b_value: 0.35,
  }]);
  assert.deepEqual(compared.text, [
    { state: "delete", text: "Local" },
    { state: "insert", text: "A different local" },
    { state: "equal", text: "answer" },
  ]);
  const steeredTree = structuredClone(compareTree);
  steeredTree.nodes[compareIndex].applied_steering = "0.5 calm";
  const unresolvedDeltaRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: generation(),
    initialTree: steeredTree,
    createId: ids("unresolved-delta"),
  });
  await assert.rejects(
    unresolvedDeltaRuntime.request({
      service: "tree", method: "diff", args: [restored.nodes[2].id, compareBranch.node_id],
    }),
    (error) => error.code === "STEERING_DELTA_UNAVAILABLE",
  );
  const resolvedDeltaRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: generation(),
    initialTree: steeredTree,
    createId: ids("resolved-delta"),
    steeringDelta(parent, child) {
      return `${parent ?? "none"}->${child ?? "none"}`;
    },
  });
  const resolvedComparison = await resolvedDeltaRuntime.request({
    service: "tree", method: "diff", args: [restored.nodes[2].id, compareBranch.node_id],
  });
  assert.equal(resolvedComparison.steering_delta, "none->0.5 calm");
  assert.equal(resolvedComparison.parent_to_b_delta, "none->0.5 calm");

  const edited = await serviceRuntime.request({
    service: "tree",
    method: "edit",
    args: [restored.nodes[1].id, "Edited question"],
  }, captureServiceMutation);
  assert.equal(edited.text, "Edited question");
  assert.equal(edited.edit_count, 1);
  assert.equal(edited.edited_at, 123);
  const starred = await serviceRuntime.request({
    service: "tree",
    method: "star",
    args: [restored.nodes[1].id, true],
  }, captureServiceMutation);
  assert.equal(starred.starred, true);
  const starredRevision = (await serviceRuntime.request({
    service: "tree", method: "get", args: [],
  })).rev;
  await serviceRuntime.request({
    service: "tree",
    method: "star",
    args: [restored.nodes[1].id, true],
  }, captureServiceMutation);
  assert.equal((await serviceRuntime.request({
    service: "tree", method: "get", args: [],
  })).rev, starredRevision);
  const noted = await serviceRuntime.request({
    service: "tree",
    method: "note",
    args: [restored.nodes[1].id, "Keep this branch"],
  }, captureServiceMutation);
  assert.equal(noted.notes, "Keep this branch");
  const branch = await serviceRuntime.request({
    service: "tree",
    method: "branch",
    args: [restored.nodes[1].id, "Alternate", undefined, "assistant"],
  }, captureServiceMutation);
  assert.equal(branch.node.role, "assistant");
  assert.equal(branch.node.parent_id, restored.root_id);
  assert.equal(branch.active_path.active_node_id, branch.node_id);
  assert.deepEqual(await serviceRuntime.request({
    service: "tree", method: "delete", args: [restored.nodes[1].id],
  }, captureServiceMutation), { removed: 3 });
  const mutatedTree = await serviceRuntime.request({
    service: "tree", method: "get", args: [],
  });
  assert.equal(mutatedTree.nodes.length, 2);
  assert.equal(mutatedTree.nodes[1].id, branch.node_id);
  assert.deepEqual(await serviceRuntime.request({
    service: "tree", method: "cast", args: [],
  }), {
    cast: {
      user: { recipe: null, notes: "", origin: "structural" },
      assistant: { recipe: null, notes: "", origin: "structural" },
    },
  });
  assert.deepEqual(await serviceRuntime.request({
    service: "tree",
    method: "castPut",
    args: ["guide", { steering: "0.25 calm", thinking: false, seed: 7, notes: "Local guide" }],
  }, captureServiceMutation), {
    label: "guide",
    member: {
      recipe: {
        steering: "0.25 calm",
        sampling: null,
        thinking: false,
        seed: 7,
        probes: [],
        probe_hashes: {},
      },
      notes: "Local guide",
      origin: "configured",
    },
  });
  const castAfterPut = await serviceRuntime.request({
    service: "tree", method: "cast", args: [],
  });
  assert.equal(castAfterPut.cast.guide.origin, "configured");
  const castRevision = (await serviceRuntime.request({
    service: "tree", method: "get", args: [],
  })).rev;
  await serviceRuntime.request({
    service: "tree", method: "castPut", args: ["guide", {
      steering: "0.25 calm", thinking: false, seed: 7, notes: "Local guide",
    }],
  }, captureServiceMutation);
  assert.equal((await serviceRuntime.request({
    service: "tree", method: "get", args: [],
  })).rev, castRevision);
  await assert.rejects(
    serviceRuntime.request({
      service: "tree", method: "castPut", args: ["Bad Label", {}],
    }),
    (error) => error.code === "INVALID_CAST_LABEL",
  );
  await assert.rejects(
    serviceRuntime.request({
      service: "tree", method: "castPut", args: ["guide", { steering: "calm +" }],
    }),
    (error) => error.code === "INVALID_STEERING",
  );
  await serviceRuntime.request({
    service: "tree", method: "castDelete", args: ["guide"],
  }, captureServiceMutation);
  assert.equal((await serviceRuntime.request({
    service: "tree", method: "cast", args: [],
  })).cast.guide, undefined);
  assert.deepEqual(serviceMutationEvents.map((event) => event.event), [
    "tree_mutated", "tree_mutated", "tree_mutated", "tree_mutated",
    "tree_mutated", "tree_mutated", "tree_mutated", "tree_mutated",
    "tree_mutated",
  ]);
  assert.deepEqual(serviceMutationEvents.map((event) => event.data.op), [
    "restore", "navigate", "edit", "star", "note", "branch", "delete", "cast", "cast",
  ]);
  const deleteMutation = serviceMutationEvents.find((event) => event.data.op === "delete").data;
  assert.equal(deleteMutation.removed.length, 3);
  assert.equal(deleteMutation.active_node_id, null);
  const castMutations = serviceMutationEvents.filter((event) => event.data.op === "cast");
  assert.equal(castMutations[0].data.cast.guide.origin, "configured");
  assert.equal(castMutations[1].data.cast.guide, undefined);

  await assert.rejects(
    serviceRuntime.generate(
      { type: "generate", input: null, n: 33 },
      () => {},
    ),
    (error) => error.code === "INVALID_GENERATION_REQUEST",
  );
  await assert.rejects(
    serviceRuntime.generate(
      {
        type: "generate",
        fork_node_id: "node",
        fork_raw_index: 0,
        fork_alt_token_id: 1,
      },
      () => {},
    ),
    (error) => error.code === "TOKEN_FORK_UNAVAILABLE",
  );
  await assert.rejects(
    serviceRuntime.generate(
      {
        type: "generate",
        fork_node_id: "node",
        fork_raw_index: 0,
        fork_alt_token_id: 1,
        fork_replacement_text: "replacement",
      },
      () => {},
    ),
    (error) => error.code === "INVALID_GENERATION_REQUEST",
  );
  await serviceRuntime.stop();
  assert.equal(serviceRuntime.snapshot().tree.nodes.length, 2);
  const beforeReset = serviceRuntime.snapshot().tree;
  await serviceRuntime.request({
    service: "tree", method: "reset", args: [],
  }, captureServiceMutation);
  const resetTree = serviceRuntime.snapshot().tree;
  const resetMutation = serviceMutationEvents.at(-1).data;
  assert.equal(resetMutation.op, "reset");
  assert.equal(resetMutation.rev, beforeReset.rev + 1);
  assert.deepEqual(resetMutation.removed.toSorted(), beforeReset.nodes.map((node) => node.id).toSorted());
  assert.deepEqual(resetMutation.added.map((node) => node.id), [resetTree.root_id]);
  assert.equal(resetMutation.active_node_id, resetTree.root_id);
  assert.equal(resetTree.nodes.length, 1);
  assert.notEqual(resetTree.root_id, beforeReset.root_id);

  const sharedLoomFixture = JSON.parse(await readFile(
    new URL("../../browser-runtime/fixtures/loom-operation-transcript-v1.json", import.meta.url),
    "utf8",
  ));
  const sharedIds = sharedLoomFixture.operations
    .filter((operation) => operation.createdId !== undefined)
    .map((operation) => operation.createdId);
  const sharedLoomRuntime = new BrowserLoomRuntime({
    session: session(),
    generation: generation(),
    initialTree: sharedLoomFixture.initial,
    now: () => 200,
    createId() {
      const id = sharedIds.shift();
      assert.notEqual(id, undefined, "the shared loom fixture exhausted its node IDs");
      return id;
    },
  });
  const sharedMutations = [];
  let sharedPreservedMeasurements;
  for (const operation of sharedLoomFixture.operations) {
    let method = operation.op;
    let args;
    if (method === "edit") args = [operation.node, operation.text];
    else if (method === "branch") {
      args = [operation.node, operation.text, undefined, operation.role];
    } else if (method === "star") args = [operation.node, operation.value];
    else if (method === "note") args = [operation.node, operation.text];
    else if (method === "navigate" || method === "delete") args = [operation.node];
    else if (method === "castPut") {
      args = [operation.label, {
        steering: operation.steering,
        thinking: operation.thinking,
        seed: operation.seed,
        notes: operation.notes,
      }];
    } else if (method === "reset") {
      const assistant = sharedLoomRuntime.snapshot().tree.nodes.find(
        (node) => node.id === "assistant-1",
      );
      sharedPreservedMeasurements = assistant.tokens[0].measurements;
      args = [];
    } else {
      assert.fail(`unknown shared loom operation ${JSON.stringify(method)}`);
    }
    await sharedLoomRuntime.request(
      { service: "tree", method, args },
      (event) => sharedMutations.push(event.data),
    );
  }
  assert.deepEqual(
    sharedMutations.map((event) => ({
      op: event.op,
      rev: event.rev,
      added: event.added.map((node) => node.id),
      removed: event.removed,
      updated: event.updated.map((node) => node.id),
      active_node_id: event.active_node_id,
    })),
    sharedLoomFixture.expected.events,
  );
  assert.deepEqual(
    sharedPreservedMeasurements,
    sharedLoomFixture.expected.preservedMeasurementsBeforeReset,
  );
  assert.deepEqual(
    canonicalBrowserTree(sharedLoomRuntime.snapshot().tree),
    sharedLoomFixture.expected.final,
  );

  console.log("Browser authoritative loom checks passed");
} finally {
  await server.close();
}

function session(config = {}) {
  return {
    id: "default",
    model_id: "fixture/drowse-tiny",
    device: "webgpu fixture",
    dtype: "float32",
    created: 0,
    config: {
      temperature: 0,
      top_p: 1,
      top_k: null,
      max_tokens: 64,
      system_prompt: null,
      thinking: false,
      ...config,
    },
    profiles: [],
    probes: [],
    history_length: 0,
    supports_thinking: false,
    thinking_is_optional: false,
    is_base_model: false,
    jlens_fitted: false,
    instruments: [],
    default_steering: null,
    role_substitution_supported: true,
    user_role_supported: true,
    default_assistant_role: "assistant",
    default_user_role: "user",
    scene_mode: true,
    thinking_input_supported: false,
    strips_history_thinking: false,
  };
}

function canonicalBrowserTree(tree) {
  return {
    rev: tree.rev,
    root_id: tree.root_id,
    active_node_id: tree.active_node_id,
    nodes: tree.nodes.map((node) => ({
      id: node.id,
      parent_id: node.parent_id,
      role: node.role,
      text: node.text,
      starred: node.starred,
      notes: node.notes,
      edit_count: node.edit_count,
    })),
    children_of: tree.children_of,
    cast: tree.cast,
  };
}

function webLlmCancellableGeneration(streamWebLlmGeneration) {
  let rejectStream;
  let markStarted;
  const started = new Promise((resolve) => { markStarted = resolve; });
  const engine = {
    chat: {
      completions: {
        async create() {
          return {
            async *[Symbol.asyncIterator]() {
              yield {
                choices: [{
                  index: 0,
                  delta: { content: "Partial" },
                  finish_reason: null,
                  logprobs: {
                    content: [{
                      token: "Partial",
                      token_id: 7,
                      logprob: -0.4,
                      drowse_sampler: {
                        entropy_nats: 0.3,
                        perplexity: Math.exp(0.3),
                      },
                    }],
                  },
                }],
              };
              yield {
                choices: [],
                usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
              };
              const interrupted = new Promise((_resolve, reject) => {
                rejectStream = reject;
              });
              markStarted();
              await interrupted;
            },
          };
        },
      },
    },
    completions: { async create() { throw new Error("unexpected raw generation"); } },
    async getDrowseRuntimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
        sceneStitching: true,
      };
    },
  };
  const hooks = {
    async clear() {},
    async interrupt() {},
    async install() {},
    async read() { return undefined; },
    assertSteeringSupported() {},
  };
  return {
    started,
    runtimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
        sceneStitching: true,
      };
    },
    streamGeneration(plan, onToken) {
      return streamWebLlmGeneration(engine, hooks, plan, onToken);
    },
    async stop() {
      rejectStream?.(new DOMException("interrupted", "AbortError"));
    },
  };
}
