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
  const {
    measurementEnvelope,
    measurementGateScores,
    readWebLlmRuntimeCapabilities,
    streamWebLlmGeneration,
    webLlmGenerationErrorUsage,
    validateWebLlmGenerationSettings,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmGeneration.ts",
  );
  const { DrowseWebLlmRuntime } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmEngine.ts",
  );
  const { compileStructuredHookProgram } = await server.ssrLoadModule(
    "/src/hosted/runtime/structuredHookProgram.ts",
  );
  const {
    BROWSER_RETURN_TOP_K_MAX,
    BROWSER_SAMPLING_TOP_K_MAX,
    HTTP_RETURN_TOP_K_DEFAULT,
    HTTP_RETURN_TOP_K_MAX,
    clampTokenAlternativeCount,
    tokenAlternativeDefault,
    tokenAlternativeLimit,
  } = await server.ssrLoadModule(
    "/src/lib/runtime/samplingCapabilities.ts",
  );
  const {
    STANDARD_STRUCTURED_HOOK_PROFILE,
    STANDARD_STRUCTURED_HOOK_PROFILE_V2,
    STANDARD_STRUCTURED_HOOK_PROFILE_V3,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/structuredHookProfile.ts",
  );

  const hookProgram = {
    hookAbi: "post-block-residual-v4",
    hiddenSize: 1,
    layerCount: 1,
    enabled: Uint32Array.of(1),
    basis: Float32Array.of(1),
    neutral: Float32Array.of(0),
    target: Float32Array.of(1),
    along: Float32Array.of(0.5),
    collapse: Float32Array.of(1),
    probeBasis: Float32Array.of(1),
    probeNeutral: Float32Array.of(0),
  };
  assert.equal(tokenAlternativeLimit("http"), HTTP_RETURN_TOP_K_MAX);
  assert.equal(tokenAlternativeLimit("browser"), BROWSER_RETURN_TOP_K_MAX);
  assert.equal(tokenAlternativeLimit("fake"), 0);
  assert.equal(tokenAlternativeDefault("http"), HTTP_RETURN_TOP_K_DEFAULT);
  assert.equal(tokenAlternativeDefault("browser"), 5);
  assert.equal(tokenAlternativeDefault("fake"), 0);
  assert.equal(clampTokenAlternativeCount(262144, "browser"), 262144);
  assert.equal(clampTokenAlternativeCount(8, "http"), 8);
  for (const top_k of [1025, 4097, 262144, BROWSER_SAMPLING_TOP_K_MAX]) {
    assert.equal(validateWebLlmGenerationSettings({ top_k }, false).top_k, top_k);
  }
  assert.throws(() => validateWebLlmGenerationSettings({ top_k: Number.MAX_SAFE_INTEGER + 1 }, false), /top_k/);
  assert.throws(() => validateWebLlmGenerationSettings({ seed: -1 }, false), /seed/);
  assert.equal(validateWebLlmGenerationSettings({ seed: Number.MAX_SAFE_INTEGER }, false).seed, Number.MAX_SAFE_INTEGER);
  const drowseEngineMethods = {
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
    async tokenizeDrowseText(text) {
      return [...text].map((character) => character.codePointAt(0));
    },
    async decodeDrowseTokens(tokenIds) {
      return String.fromCodePoint(...tokenIds);
    },
  };
  assert.deepEqual(await readWebLlmRuntimeCapabilities({
    async getDrowseRuntimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
      };
    },
  }), {
    topK: true,
    forcedReplay: true,
    replayScoring: true,
    tokenizer: true,
    namedRoles: true,
    userSeatGeneration: true,
    sceneStitching: false,
  });
  await assert.rejects(
    readWebLlmRuntimeCapabilities({
      async getDrowseRuntimeCapabilities() {
        return {
          topK: true,
          forcedReplay: true,
          replayScoring: true,
          tokenizer: true,
          namedRoles: true,
          userSeatGeneration: true,
          sceneStitching: "yes",
        };
      },
    }),
    (error) => error.code === "INVALID_DROWSE_CAPABILITIES",
  );
  const chatChunks = [
    {
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null, logprobs: null }],
    },
    {
      choices: [{
        index: 0,
        delta: { content: "Hello" },
        finish_reason: null,
        logprobs: { content: [{
          token: "Hello",
          token_id: 11,
          logprob: -0.2,
          top_logprobs: [{ token: "Hello", token_id: 11, logprob: -0.2 }],
        }] },
      }],
    },
    {
      choices: [{
        index: 0,
        delta: { content: " world" },
        finish_reason: null,
        logprobs: { content: [{
          token: " world",
          token_id: 12,
          logprob: -0.4,
          top_logprobs: [{ token: " world", token_id: 12, logprob: -0.4 }],
        }] },
      }],
    },
    {
      choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }],
    },
    {
      choices: [],
      usage: {
        prompt_tokens: 3,
        completion_tokens: 2,
        total_tokens: 5,
        extra: { prefill_tokens_per_s: 8.5, decode_tokens_per_s: 4.25 },
      },
    },
  ];
  const calls = [];
  const engine = {
    ...drowseEngineMethods,
    chat: {
      completions: {
        async create(request) {
          calls.push(["chat", request]);
          return from(chatChunks);
        },
      },
    },
    completions: {
      async create(request) {
        calls.push(["raw", request]);
        return from([
          {
            choices: [{
              index: 0,
              text: "raw text",
              finish_reason: "length",
              logprobs: {
                content: [{
                  token: "raw text",
                  token_id: 13,
                  logprob: -0.3,
                }],
              },
            }],
          },
        ]);
      },
    },
  };
  const hookCalls = [];
  const hooks = {
    async clear() { hookCalls.push("clear"); },
    async interrupt() { hookCalls.push("interrupt"); },
    async install(program) { hookCalls.push(["install", program]); },
    async read() {
      hookCalls.push("read");
      return Float32Array.of(0.25, -0.5);
    },
    assertSteeringSupported(expression) { hookCalls.push(["assert", expression]); },
  };

  const tokens = [];
  const result = await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      sampling: { temperature: 0.4, max_tokens: 32 },
    },
    (token) => tokens.push(token),
  );
  assert.equal(calls[0][0], "chat");
  assert.deepEqual(calls[0][1], {
    stream: true,
    stream_options: { include_usage: true },
    logprobs: true,
    top_logprobs: 1,
    temperature: 0.4,
    max_tokens: 32,
    messages: [{ role: "user", content: "Hi" }],
  });
  assert.deepEqual(tokens.map((token) => token.text), ["Hello", " world"]);
  assert.equal(tokens[0].tokenId, 11);
  assert.equal(tokens[0].rawIndex, 0);
  assert.equal(tokens[1].tokenId, 12);
  assert.equal(tokens[1].rawIndex, 1);
  assert.equal(tokens[0].topAlts, undefined);
  assert.equal(tokens[0].thinking, false);
  assert.equal(tokens[0].logprob, -0.2);
  assert.equal(tokens[0].samplerEntropy, 0.75);
  assert.ok(Math.abs(tokens[0].perplexity - Math.exp(0.75)) < 1e-12);
  assert.deepEqual(result, {
    text: "Hello world",
    thinkingText: null,
    tokens: 2,
    finishReason: "stop",
    terminalReason: "eos",
    usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
    meanLogprob: -0.30000000000000004,
    meanSurprise: 0.30000000000000004,
    prefillTokensPerSecond: 8.5,
    decodeTokensPerSecond: 4.25,
  });
  assert.deepEqual(hookCalls, ["clear", "clear"]);

  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      sampling: { max_tokens: 8_192 },
      maxOutputTokens: 256,
    },
    () => {},
  );
  assert.equal(calls.at(-1)[1].max_tokens, 256);

  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      sampling: { max_tokens: 8_192 },
      maxOutputTokens: 256,
      replay: { forcedPrefixTokenIds: [1, 2, 3] },
    },
    () => {},
  );
  assert.equal(calls.at(-1)[1].max_tokens, 259);

  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      sampling: {
        temperature: -2,
        top_p: 0,
        top_k: 0,
        stop: ["never appears"],
      },
    },
    () => {},
  );
  assert.equal(calls.at(-1)[1].temperature, -2);
  assert.equal(calls.at(-1)[1].top_p, 0);
  assert.equal(calls.at(-1)[1].top_k, 0);
  assert.equal(Object.hasOwn(calls.at(-1)[1], "stop"), false);

  const namedRoleTokens = [];
  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: {
        kind: "chat",
        messages: [
          { role: "user", content: "Hi", name: "curious_user" },
          { role: "assistant", content: "Hello", name: "prior_guide" },
          { role: "user", content: "Continue" },
        ],
      },
      thinking: false,
      generationRoleName: "forest_guide",
    },
    (token) => namedRoleTokens.push(token),
  );
  assert.deepEqual(calls.at(-1)[1].messages, [
    { role: "user", content: "Hi", name: "curious_user" },
    { role: "assistant", content: "Hello", name: "prior_guide" },
    { role: "user", content: "Continue" },
  ]);
  assert.deepEqual(calls.at(-1)[1].extra_body, {
    drowse_generation_role: "forest_guide",
  });
  assert.deepEqual(namedRoleTokens.map((token) => token.text), ["Hello", " world"]);

  for (const thinkingProfile of [null, {
    start: "<think>", end: "</think>", startTokenIds: [100], endTokenIds: [101], startsInThinking: false,
  }]) {
    await streamWebLlmGeneration(engine, hooks, {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      thinking: false,
      thinkingProfile,
    }, () => {});
    assert.deepEqual(calls.at(-1)[1].extra_body, thinkingProfile ? { enable_thinking: false } : undefined,
      "models without thinking must not receive an empty thinking-block prompt override");
  }

  await assert.rejects(
    streamWebLlmGeneration(
      engine,
      hooks,
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        generationRoleName: "Forest Guide",
      },
      () => {},
    ),
    (error) => error.code === "INVALID_GENERATION_ROLE",
  );
  await assert.rejects(
    streamWebLlmGeneration(
      engine,
      hooks,
      {
        input: { kind: "raw", prompt: "Hi" },
        generationRoleName: "forest_guide",
      },
      () => {},
    ),
    (error) => error.code === "GENERATION_ROLE_UNAVAILABLE",
  );
  await assert.rejects(
    streamWebLlmGeneration(
      engine,
      hooks,
      {
        input: {
          kind: "chat",
          messages: [{ role: "system", content: "system", name: "named_system" }],
        },
      },
      () => {},
    ),
    (error) => error.code === "INVALID_GENERATION_INPUT",
  );
  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: {
        kind: "chat",
        messages: [{ role: "assistant", content: "Where next?" }],
      },
      sampling: { top_k: 20 },
      generationSeat: "user",
      generationRoleName: "curious_user",
      replay: {
        forcedPrefixTokenIds: [91, 92],
        scoreTokenIds: [91, 17],
      },
    },
    () => {},
  );
  assert.equal(calls.at(-1)[1].top_k, 20);
  assert.deepEqual(calls.at(-1)[1].extra_body, {
    drowse_forced_prefix_token_ids: [91, 92],
    drowse_score_token_ids: [91, 17],
    drowse_generation_seat: "user",
    drowse_generation_role: "curious_user",
  });

  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      generationSeat: "user",
    },
    () => {},
  );
  assert.equal(calls.at(-1)[1].extra_body.drowse_generation_seat, "user");

  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: {
        kind: "chat",
        messages: [
          { role: "user", content: "Question" },
          { role: "assistant", content: "First answer" },
          { role: "assistant", content: "A/B shadow" },
        ],
      },
      generationSeat: "assistant",
    },
    () => {},
  );
  assert.deepEqual(calls.at(-1)[1].messages.map((message) => message.role), [
    "user", "assistant", "assistant",
  ]);
  await assert.rejects(
    streamWebLlmGeneration(
      {
        ...engine,
        async getDrowseRuntimeCapabilities() {
          return {
            ...await drowseEngineMethods.getDrowseRuntimeCapabilities(),
            sceneStitching: false,
          };
        },
      },
      hooks,
      {
        input: {
          kind: "chat",
          messages: [
            { role: "user", content: "Question" },
            { role: "assistant", content: "First answer" },
            { role: "assistant", content: "A/B shadow" },
          ],
        },
      },
      () => {},
    ),
    (error) => error.code === "SCENE_STITCHING_UNAVAILABLE",
  );

  const rawTokens = [];
  const rawResult = await streamWebLlmGeneration(
    engine,
    hooks,
    { input: { kind: "raw", prompt: "plain" } },
    (token) => rawTokens.push(token),
  );
  assert.equal(calls.at(-1)[0], "raw");
  assert.equal(calls.at(-1)[1].prompt, "plain");
  assert.deepEqual(rawTokens.map((token) => token.text), ["raw text"]);
  assert.equal(rawResult.finishReason, "length");
  assert.deepEqual(rawResult.usage, { promptTokens: 0, completionTokens: 1, totalTokens: 1 });

  for (const [chunks, code] of [
    [[
      {
        choices: [{
          index: 0,
          delta: { content: "x" },
          finish_reason: null,
          logprobs: { content: [{ token: "x", token_id: 1, logprob: -0.1 }] },
        }],
      },
      { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
    ], "MISSING_GENERATION_TERMINATION"],
    [[
      {
        choices: [{
          index: 0,
          delta: { content: "x" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "x", token_id: 1, logprob: -0.1 }] },
        }],
      },
    ], "MISSING_GENERATION_USAGE"],
    [[
      {
        choices: [{
          index: 0,
          delta: { content: "x" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "x", token_id: 1, logprob: -0.1 }] },
        }],
      },
      { choices: [], usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 } },
    ], "INCONSISTENT_GENERATION_USAGE"],
    [[
      {
        choices: [{
          index: 0,
          delta: { content: "x" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "x", token_id: 1, logprob: -0.1 }] },
        }],
      },
      { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 3 } },
    ], "INCONSISTENT_GENERATION_USAGE"],
    [[
      {
        choices: [{
          index: 0,
          delta: { content: "x" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "x", token_id: 1, logprob: -0.1 }] },
        }],
      },
      { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
      { choices: [], usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 } },
    ], "INCONSISTENT_GENERATION_USAGE"],
    [[
      {
        choices: [{
          index: 0,
          delta: { content: "x" },
          finish_reason: "mystery",
          logprobs: { content: [{ token: "x", token_id: 1, logprob: -0.1 }] },
        }],
      },
      { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
    ], "INVALID_GENERATION_TERMINATION"],
  ]) {
    await assert.rejects(
      streamWebLlmGeneration(
        {
          ...engine,
          chat: { completions: { async create() { return exactFrom(chunks); } } },
        },
        hooks,
        { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
        () => {},
      ),
      (error) => error.code === code,
    );
  }

  const exactTokenChunk = {
    choices: [{
      index: 0,
      delta: { content: "x" },
      finish_reason: null,
      logprobs: {
        content: [{
          token: "x",
          token_id: 1,
          logprob: -0.1,
          drowse_sampler: {
            entropy_nats: 0.25,
            perplexity: Math.exp(0.25),
          },
        }],
      },
    }],
  };
  await assert.rejects(
    streamWebLlmGeneration(
      {
        ...engine,
        chat: { completions: { async create() {
          return rawFrom([
            exactTokenChunk,
            { choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }] },
            { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
          ]);
        } } },
      },
      hooks,
      { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
      () => {},
    ),
    (error) => error.code === "MISSING_GENERATION_TERMINATION",
  );
  await assert.rejects(
    streamWebLlmGeneration(
      {
        ...engine,
        chat: { completions: { async create() {
          return rawFrom([{
            choices: [{
              index: 0,
              delta: {},
              finish_reason: "length",
              drowse_finish_reason: "eos",
              logprobs: null,
            }],
          }]);
        } } },
      },
      hooks,
      { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
      () => {},
    ),
    (error) => error.code === "INCONSISTENT_GENERATION_TERMINATION",
  );
  await assert.rejects(
    streamWebLlmGeneration(
      {
        ...engine,
        chat: { completions: { async create() {
          return rawFrom([
            exactTokenChunk,
            {
              choices: [{
                index: 0,
                delta: {},
                finish_reason: "stop",
                drowse_finish_reason: "stop_sequence",
                logprobs: null,
              }],
            },
            { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
          ]);
        } } },
      },
      hooks,
      { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
      () => {},
    ),
    (error) => error.code === "STOP_SEQUENCE_METADATA_MISMATCH",
  );

  const alternativeTokens = [];
  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "x" }] },
      sampling: { return_top_k: 1 },
    },
    (token) => alternativeTokens.push(token),
  );
  assert.deepEqual(alternativeTokens[0].topAlts, [
    { id: 11, text: "Hello", logprob: -0.2 },
  ]);
  assert.equal(calls.at(-1)[1].top_logprobs, 1);

  await streamWebLlmGeneration(
    engine,
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "x" }] },
      sampling: { return_top_k: BROWSER_RETURN_TOP_K_MAX },
    },
    () => {},
  );
  assert.equal(calls.at(-1)[1].top_logprobs, BROWSER_RETURN_TOP_K_MAX);
  await assert.rejects(
    streamWebLlmGeneration(
      engine,
      hooks,
      {
        input: { kind: "chat", messages: [{ role: "user", content: "x" }] },
        sampling: { return_top_k: BROWSER_RETURN_TOP_K_MAX + 1 },
      },
      () => {},
    ),
    (error) => error.code === "INVALID_SAMPLING",
  );

  let replayRequest;
  const replayRawTokens = [];
  const replayVisibleTokens = [];
  await streamWebLlmGeneration(
    {
      ...drowseEngineMethods,
      chat: { completions: { async create(request) {
        replayRequest = request;
        return from([{
          choices: [{
            index: 0,
            delta: { content: "forced" },
            finish_reason: "stop",
            logprobs: { content: [{
              token: "forced",
              token_id: 91,
              logprob: -4,
              top_logprobs: [{ token: "normal", token_id: 7, logprob: -0.1 }],
              drowse_replay: {
                emitted_token_id: 91,
                sampled_token_id: 7,
                forced_token_id: 91,
                selected_logprobs: [
                  { token_id: 91, logprob: -3 },
                  { token_id: 17, logprob: Number.NEGATIVE_INFINITY },
                ],
                argmax: { token_id: 7, logprob: -0.1 },
                top_logprobs: [
                  { token_id: 7, logprob: -0.1 },
                  { token_id: 8, logprob: -2.4 },
                ],
              },
            }] },
          }],
        }]);
      } } },
      completions: { async create() { throw new Error("unexpected raw generation"); } },
    },
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "score" }] },
      sampling: { top_k: 32, top_p: 0.8 },
      replay: { forcedPrefixTokenIds: [91], scoreTokenIds: [91, 17] },
      async onRawToken(token) { replayRawTokens.push(token); },
    },
    (token) => replayVisibleTokens.push(token),
  );
  assert.equal(replayRequest.top_k, 32);
  assert.equal(replayRequest.top_p, 0.8);
  assert.deepEqual(replayRequest.extra_body, {
    drowse_forced_prefix_token_ids: [91],
    drowse_score_token_ids: [91, 17],
  });
  const expectedReplayScore = {
    emittedTokenId: 91,
    sampledTokenId: 7,
    forcedTokenId: 91,
    requestedLogprobs: [
      { tokenId: 91, logprob: -3 },
      { tokenId: 17, logprob: Number.NEGATIVE_INFINITY },
    ],
    argmax: { tokenId: 7, logprob: -0.1 },
    topLogprobs: [
      { tokenId: 7, logprob: -0.1 },
      { tokenId: 8, logprob: -2.4 },
    ],
  };
  assert.deepEqual(replayRawTokens[0].replayScore, expectedReplayScore);
  assert.deepEqual(replayVisibleTokens[0].replayScore, expectedReplayScore);
  assert.equal(replayRawTokens[0].rawIndex, 0);

  const zeroProbabilityRawTokens = [];
  const zeroProbabilityVisibleTokens = [];
  const zeroProbabilityResult = await streamWebLlmGeneration(
    {
      ...drowseEngineMethods,
      chat: { completions: { async create() {
        return from([{
          choices: [{
            index: 0,
            delta: { content: "forced" },
            finish_reason: "stop",
            logprobs: { content: [{
              token: "forced",
              token_id: 91,
              logprob: Number.NEGATIVE_INFINITY,
              top_logprobs: [
                { token: "normal", token_id: 7, logprob: 0 },
                { token: "forced", token_id: 91, logprob: Number.NEGATIVE_INFINITY },
              ],
            }] },
          }],
        }]);
      } } },
    },
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "score" }] },
      sampling: { temperature: 0, return_top_k: 2 },
      replay: { forcedPrefixTokenIds: [91] },
      onRawToken(token) { zeroProbabilityRawTokens.push(token); },
    },
    (token) => zeroProbabilityVisibleTokens.push(token),
  );
  assert.equal(zeroProbabilityResult.text, "forced");
  assert.equal(zeroProbabilityRawTokens[0].logprob, Number.NEGATIVE_INFINITY);
  assert.equal(zeroProbabilityRawTokens[0].topAlts[1].logprob, Number.NEGATIVE_INFINITY);
  assert.equal(zeroProbabilityVisibleTokens[0].logprob, null);
  assert.deepEqual(zeroProbabilityVisibleTokens[0].topAlts, [
    { id: 7, text: "normal", logprob: 0 },
  ]);
  assert.equal(zeroProbabilityResult.meanLogprob, null);
  assert.equal(zeroProbabilityResult.meanSurprise, null);

  for (const invalidLogprob of [0.01, Number.POSITIVE_INFINITY, Number.NaN]) {
    for (const field of ["token", "alternative", "replay"]) {
      const row = { token: "x", token_id: 1, logprob: -1 };
      if (field === "token") row.logprob = invalidLogprob;
      if (field === "alternative") row.top_logprobs = [{ token: "y", token_id: 2, logprob: invalidLogprob }];
      if (field === "replay") row.drowse_replay = {
        emitted_token_id: 1, sampled_token_id: 1, forced_token_id: null,
        selected_logprobs: [{ token_id: 1, logprob: invalidLogprob }],
        argmax: { token_id: 1, logprob: -1 },
        top_logprobs: [{ token_id: 1, logprob: -1 }],
      };
      await assert.rejects(streamWebLlmGeneration({
        ...drowseEngineMethods,
        chat: { completions: { async create() {
          return from([{ choices: [{ index: 0, delta: { content: "x" }, finish_reason: "stop", logprobs: { content: [row] } }] }]);
        } } },
      }, hooks, {
        input: { kind: "chat", messages: [{ role: "user", content: "test" }] },
        sampling: { temperature: 1, return_top_k: 2 },
      }, () => undefined), (error) => error.code === "INVALID_GENERATION_CHUNK", `${field} must reject invalid log-probabilities`);
    }
  }

  let unsupportedCreateCalled = false;
  let unsupportedClearCalled = false;
  await assert.rejects(
    streamWebLlmGeneration(
      {
        ...engine,
        async getDrowseRuntimeCapabilities() {
          return {
            topK: false,
            forcedReplay: false,
            replayScoring: false,
            tokenizer: false,
            namedRoles: false,
            userSeatGeneration: false,
            sceneStitching: false,
          };
        },
        chat: { completions: { async create() {
          unsupportedCreateCalled = true;
          return from([]);
        } } },
      },
      {
        ...hooks,
        async clear() { unsupportedClearCalled = true; },
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "x" }] },
        sampling: { top_k: 4 },
      },
      () => {},
    ),
    (error) => error.code === "TOP_K_SAMPLING_UNAVAILABLE",
  );
  assert.equal(unsupportedCreateCalled, false);
  assert.equal(unsupportedClearCalled, false);
  await assert.rejects(
    streamWebLlmGeneration(
      engine,
      hooks,
      { input: { kind: "raw", prompt: "x" }, thinking: true },
      () => {},
    ),
    (error) => error.code === "THINKING_STREAM_UNAVAILABLE",
  );
  await assert.rejects(
    streamWebLlmGeneration(
      engine,
      hooks,
      { input: { kind: "raw", prompt: "x" }, steeringExpression: "honest" },
      () => {},
    ),
    (error) => error.code === "STEERING_PROGRAM_REQUIRED",
  );

  const steeredCalls = [];
  const steeredTokens = [];
  await streamWebLlmGeneration(
    engine,
    {
      async clear() { steeredCalls.push("clear"); },
      async interrupt() { steeredCalls.push("interrupt"); },
      async install(program) { steeredCalls.push(["install", program]); },
      async read() {
        steeredCalls.push("read");
        return Float32Array.of(0.75);
      },
      assertSteeringSupported(expression) { steeredCalls.push(["assert", expression]); },
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      steeringExpression: "0.5 honest",
      hookProgram,
    },
    (token) => steeredTokens.push(token),
  );
  assert.deepEqual(steeredCalls, [
    ["assert", "0.5 honest"],
    "clear",
    ["install", hookProgram],
    "clear",
  ]);
  assert.equal(steeredTokens[0].hookMeasurements, undefined);

  const gatedProgram = compileStructuredHookProgram(1, [{
    affineGroups: [{
      active: true,
      basis: [[1]],
      neutral: [0],
      target: [1],
      along: 0.5,
      kappa: [0],
    }],
    probes: [{ kind: "linear", direction: [1], bias: 0 }],
  }], STANDARD_STRUCTURED_HOOK_PROFILE);
  gatedProgram.controls = {
    affine: [{
      enabled: true,
      phase: { kind: "generated_only" },
      gate: { slots: [{ layer: 0, probe: 0 }], operator: ">", threshold: 0.5 },
    }, null, null, null],
    curve: [null, null, null, null],
  };
  gatedProgram.measurementSchema = {
    layerMap: [7],
    modelLayerCount: 8,
    probes: [
      { name: "sae/3", family: "sae", featureId: 3 },
      null, null, null, null, null, null, null,
    ],
    lensSource: null,
    saeSource: "fixture-sae",
  };
  const controlUpdates = [];
  const measuredTokens = [];
  const measuredResult = await streamWebLlmGeneration(
    engine,
    {
      async clear() {},
      async interrupt() {},
      async install(program) {
        assert.equal(program.affineActive[0], 0);
      },
      async updateControls(affineActive, curveActive) {
        controlUpdates.push([[...affineActive], [...curveActive]]);
      },
      async read() { return Float32Array.of(0.75, 0, 0, 0, 0, 0, 0, 0); },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      steeringExpression: "0.5 honest@when:honest>0.5",
      hookProgram: gatedProgram,
      measurementSpecialTokenIds: [99],
      sampling: { persist_per_layer_scores: true },
    },
    (token) => measuredTokens.push(token),
  );
  assert.equal(controlUpdates.length, 1);
  assert.equal(controlUpdates[0][0][0], 1);
  assert.equal(gatedProgram.affineActive[0], 1);
  assert.equal(measuredTokens[0].measurements.instruments.sae.readout, undefined);
  assert.deepEqual(measuredTokens[0].measurements.instruments.sae.readings, {
    "sae/3": {
      value: 0.75,
      unit: "raw_activation",
      per_layer: { "7": 0.75 },
      depth: { center: [1], spread: [0], basis: "single_layer" },
    },
  });
  assert.deepEqual(measuredTokens[0].measurements.per_layer_scores, {
    "7": { "sae/3": 0.75 },
  });
  assert.equal(measuredTokens[0].measurements.scope, "token");
  assert.equal(measuredResult.measurements.scope, "aggregate");
  assert.equal(measuredTokens[0].measurements.scope, "token");

  const measurementOnlyProgram = { ...gatedProgram, controls: undefined };
  const tokenOnlyMeasurements = [];
  const tokenOnlyResult = await streamWebLlmGeneration(
    engine,
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async read() { return Float32Array.of(0.5, 0, 0, 0, 0, 0, 0, 0); },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: measurementOnlyProgram,
      sampling: {
        persist_per_layer_scores: false,
        return_probe_readings: false,
      },
    },
    (token) => tokenOnlyMeasurements.push(token),
  );
  assert.equal(tokenOnlyMeasurements.length, 2);
  assert.equal(tokenOnlyMeasurements[0].measurements.scores["sae/3"], 0.5);
  assert.equal(
    Object.hasOwn(tokenOnlyMeasurements[0].measurements, "per_layer_scores"),
    false,
  );
  assert.equal(tokenOnlyResult.measurements, undefined);

  const aggregateValues = [0.25, 0.9];
  const aggregateTokens = [];
  const aggregateRawTokens = [];
  const lastContentResult = await streamWebLlmGeneration(
    engineWithChat([
      {
        choices: [{
          index: 0,
          delta: { content: "content" },
          finish_reason: null,
          logprobs: { content: [{ token: "content", token_id: 11, logprob: -0.1 }] },
        }],
      },
      {
        choices: [{
          index: 0,
          delta: { content: "" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "", token_id: 99, logprob: -0.2 }] },
        }],
      },
    ]),
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async read() {
        return Float32Array.of(aggregateValues.shift(), 0, 0, 0, 0, 0, 0, 0);
      },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: measurementOnlyProgram,
      measurementSpecialTokenIds: [99],
      onRawToken(token) { aggregateRawTokens.push(token); },
    },
    (token) => aggregateTokens.push(token),
  );
  assert.deepEqual(aggregateTokens.map((token) => token.tokenId), [11]);
  assert.deepEqual(aggregateRawTokens.map((token) => token.tokenId), [11]);
  assert.equal(lastContentResult.tokens, 1);
  assert.equal(lastContentResult.measurements.scores["sae/3"], 0.25);

  let splitStopInterrupted = 0;
  const splitStopRaw = [];
  const splitStopVisible = [];
  const splitStopMeasurements = [0.2, 0.95];
  const splitStopStream = {
    async *[Symbol.asyncIterator]() {
      for (const [token, tokenId] of [["answer<ST", 21], ["OP>ignored", 22]]) {
        yield {
          choices: [{
            index: 0,
            delta: { content: token },
            finish_reason: null,
            logprobs: {
              content: [{
                token,
                token_id: tokenId,
                logprob: -0.3,
                drowse_sampler: {
                  entropy_nats: 0.4,
                  perplexity: Math.exp(0.4),
                },
              }],
            },
          }],
        };
      }
      yield {
        choices: [{
          index: 0,
          delta: {},
          finish_reason: "abort",
          drowse_finish_reason: "external_stop",
          logprobs: null,
        }],
      };
      yield {
        choices: [],
        usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
      };
    },
  };
  const userAbort = new AbortController();
  const cancelledTokens = [];
  let abortInterrupts = 0;
  await assert.rejects(streamWebLlmGeneration(
    {
      ...drowseEngineMethods,
      chat: { completions: { async create() { return splitStopStream; } } },
      completions: { async create() { throw new Error("unexpected raw generation"); } },
    },
    {
      async clear() {},
      async interrupt() { abortInterrupts += 1; },
      async install() {},
      async read() {},
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      signal: userAbort.signal,
    },
    token => { cancelledTokens.push(token); userAbort.abort(); },
  ), error => {
    assert.equal(error.name, "AbortError");
    assert.deepEqual(webLlmGenerationErrorUsage(error), {
      promptTokens: 3, completionTokens: 2, totalTokens: 5,
    });
    return true;
  });
  assert.equal(cancelledTokens.length, 1, "tokens arriving after Stop must not reach the conversation");
  assert.equal(abortInterrupts, 1);

  for (const cancelAt of ["raw-start", "raw-token", "visible-token"]) {
    const controller = new AbortController();
    const visible = [];
    let interrupts = 0;
    const rows = ["a", "b"].map((token, index) => ({
      token, token_id: index + 10, logprob: -0.2,
      drowse_sampler: { entropy_nats: 0.4, perplexity: Math.exp(0.4) },
    }));
    const engine = {
      async getDrowseRuntimeCapabilities() {
        return { topK: true, forcedReplay: true, replayScoring: true, tokenizer: true,
          namedRoles: true, userSeatGeneration: true, sceneStitching: true };
      },
      chat: { completions: { async create() {
        return (async function* () {
          yield { choices: [{ index: 0, delta: { content: "ab" }, finish_reason: null,
            logprobs: { content: rows } }] };
          yield { choices: [{ index: 0, delta: {}, finish_reason: "abort",
            drowse_finish_reason: "external_stop", logprobs: null }] };
          yield { choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } };
        })();
      } } },
    };
    await assert.rejects(streamWebLlmGeneration(engine, {
      async clear() {}, async install() {}, async read() {},
      async interrupt() { interrupts += 1; }, assertSteeringSupported() {},
    }, {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      signal: controller.signal,
      async onRawTokenStart() { if (cancelAt === "raw-start") controller.abort(); },
      async onRawToken() { if (cancelAt === "raw-token") controller.abort(); },
    }, async token => {
      visible.push(token.text);
      if (cancelAt === "visible-token") controller.abort();
    }), error => error.name === "AbortError");
    assert.deepEqual(visible, cancelAt === "visible-token" ? ["a"] : [], cancelAt);
    assert.equal(interrupts, 1);
  }

  const splitStopResult = await streamWebLlmGeneration(
    {
      ...drowseEngineMethods,
      chat: { completions: { async create() { return splitStopStream; } } },
      completions: { async create() { throw new Error("unexpected raw generation"); } },
    },
    {
      async clear() {},
      async interrupt() { splitStopInterrupted += 1; },
      async install() {},
      async read() {
        return Float32Array.of(splitStopMeasurements.shift(), 0, 0, 0, 0, 0, 0, 0);
      },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      sampling: { stop: ["<STOP>"] },
      hookProgram: measurementOnlyProgram,
      measurementSpecialTokenIds: [99],
      onRawToken(token) { splitStopRaw.push(token); },
    },
    (token) => splitStopVisible.push(token),
  );
  assert.equal(splitStopInterrupted, 1);
  assert.deepEqual(splitStopRaw.map(({ text, tokenId }) => [text, tokenId]), [
    ["answer<ST", 21],
    ["OP>ignored", 22],
  ]);
  assert.deepEqual(
    splitStopVisible.map(({ text, tokenId }) => [text, tokenId]),
    [["answer", 21]],
  );
  assert.equal(splitStopResult.text, "answer");
  assert.equal(splitStopResult.tokens, 2);
  assert.equal(splitStopResult.finishReason, "stop");
  assert.equal(splitStopResult.terminalReason, "stop_sequence");
  assert.ok(Math.abs(splitStopResult.measurements.scores["sae/3"] - 0.2) < 1e-6);

  let preflightCreateCalls = 0;
  let preflightClearCalls = 0;
  const preflightEngine = {
    ...drowseEngineMethods,
    chat: { completions: { async create() {
      preflightCreateCalls += 1;
      return from([]);
    } } },
    completions: { async create() {
      preflightCreateCalls += 1;
      return from([]);
    } },
  };
  const preflightHooks = {
    async clear() { preflightClearCalls += 1; },
    async interrupt() {},
    async install() {},
    async read() { return undefined; },
    assertSteeringSupported() {},
  };
  await assert.rejects(
    streamWebLlmGeneration(
      preflightEngine,
      preflightHooks,
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: measurementOnlyProgram,
      },
      () => {},
    ),
    (error) => error.code === "MEASUREMENT_SPECIAL_TOKEN_IDS_UNAVAILABLE",
  );
  await streamWebLlmGeneration(
    preflightEngine,
    preflightHooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      sampling: { persist_subspace_coords: true },
    },
    () => {},
  );
  const inertMeasurementProgram = {
    ...measurementOnlyProgram,
    probeKind: new Uint32Array(measurementOnlyProgram.probeKind.length),
  };
  await assert.rejects(
    streamWebLlmGeneration(
      preflightEngine,
      preflightHooks,
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: inertMeasurementProgram,
        measurementSpecialTokenIds: [99],
      },
      () => {},
    ),
    (error) => error.code === "INERT_MEASUREMENT_PROBE",
  );
  assert.equal(preflightCreateCalls, 1);
  assert.equal(preflightClearCalls, 2);

  let missingScalarInterrupted = false;
  let missingScalarEmitted = false;
  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([chatChunks[1]]),
      {
        async clear() {},
        async interrupt() { missingScalarInterrupted = true; },
        async install() {},
        async read() { return undefined; },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: measurementOnlyProgram,
        measurementSpecialTokenIds: [99],
      },
      () => { missingScalarEmitted = true; },
    ),
    (error) => error.code === "SCALAR_MEASUREMENTS_UNAVAILABLE",
  );
  assert.equal(missingScalarInterrupted, true);
  assert.equal(missingScalarEmitted, false);

  const scopedProgram = compileStructuredHookProgram(1, [{
    probes: [
      { kind: "sae", direction: [1], bias: 0 },
      { kind: "jlens", direction: [0], bias: 0 },
    ],
  }], STANDARD_STRUCTURED_HOOK_PROFILE_V2, {
    bindingId: "a".repeat(64),
    layerIndices: Int32Array.of(0),
    tokenIds: Int32Array.from(
      { length: STANDARD_STRUCTURED_HOOK_PROFILE_V2.maxProbes },
      (_, index) => index === 1 ? 17 : 0,
    ),
  });
  scopedProgram.measurementSchema = {
    layerMap: [0],
    modelLayerCount: 1,
    probes: [
      { name: "sae/3", family: "sae", featureId: 3 },
      { name: "jlens/calm", family: "lens", tokenId: 17 },
      null, null, null, null, null, null,
    ],
    lensSource: "fixture-lens",
    saeSource: "fixture-sae",
  };
  scopedProgram.controls = {
    affine: new Array(STANDARD_STRUCTURED_HOOK_PROFILE_V2.maxAffineGroups).fill(null),
    curve: new Array(STANDARD_STRUCTURED_HOOK_PROFILE_V2.maxCurves).fill(null),
  };
  const scopedReads = [];
  const scopedTokens = [];
  let scalarReadActive = false;
  await streamWebLlmGeneration(
    engine,
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async read() {
        assert.equal(scalarReadActive, false);
        scalarReadActive = true;
        scopedReads.push("scalar:start");
        await Promise.resolve();
        scopedReads.push("scalar:end");
        scalarReadActive = false;
        return Float32Array.of(0.25, 0.5, 0, 0, 0, 0, 0, 0);
      },
      async readGeometry() {
        assert.equal(scalarReadActive, false, "TVM measurement scopes must not overlap");
        scopedReads.push("geometry");
        return new Float32Array();
      },
      async updateControls() { scopedReads.push("controls"); },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: scopedProgram,
      measurementSpecialTokenIds: [99],
    },
    (token) => scopedTokens.push(token),
  );
  assert.deepEqual(scopedReads, [
    "scalar:start", "scalar:end",
    "scalar:start", "scalar:end",
  ]);
  assert.equal(scopedTokens.length, 2);
  assert.equal(scopedTokens[0].measurements.instruments.geometry, undefined);
  assert.equal(scopedTokens[0].measurements.instruments.sae.readings["sae/3"].value, 0.25);
  assert.equal(scopedTokens[0].measurements.instruments.lens.readings["jlens/calm"].value, 0.5);

  const staticProgram = compileStructuredHookProgram(1, [{
    affineGroups: [{
      active: true,
      basis: [[1]],
      neutral: [0],
      target: [1],
      along: 0.5,
      kappa: [0],
    }],
  }], STANDARD_STRUCTURED_HOOK_PROFILE);
  staticProgram.controls = {
    affine: [{
      enabled: true,
      phase: { kind: "both" },
      gate: null,
    }, null, null, null],
    curve: [null, null, null, null],
  };
  let staticInstalls = 0;
  let staticControlUpdates = 0;
  await streamWebLlmGeneration(
    engine,
    {
      async clear() {},
      async interrupt() {},
      async install(program) {
        staticInstalls += 1;
        assert.equal(program.affineActive[0], 1);
      },
      async updateControls() { staticControlUpdates += 1; },
      async read() { return undefined; },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      steeringExpression: "0.5 honest",
      hookProgram: staticProgram,
    },
    () => undefined,
  );
  assert.equal(staticInstalls, 1);
  assert.equal(staticControlUpdates, 0);

  const oriented = measurementEnvelope({
    format: "drowse-structured-v2",
    layerCount: 2,
    probeKind: Uint32Array.of(1, 3, 2, 1, 3, 2),
    measurementSchema: {
      layerMap: [4, 9],
      modelLayerCount: 12,
      probes: [
        { name: "calm.focused", family: "geometry" },
        { name: "jlens/calm", family: "lens", tokenId: 17 },
        { name: "sae/8", family: "sae", featureId: 8 },
      ],
      lensSource: "fixture-lens",
      saeSource: "fixture-sae",
    },
  }, Float32Array.of(0.125, 0.25, 0.375, 0.5, 0.625, 0.75), null);
  assert.deepEqual(oriented.per_layer_scores, {
    "4": { "calm.focused": 0.125, "jlens/calm": 0.25, "sae/8": 0.375 },
    "9": { "calm.focused": 0.5, "jlens/calm": 0.625, "sae/8": 0.75 },
  });
  assert.deepEqual(oriented.instruments.lens.readings, {
    "jlens/calm": {
      value: 0.4375,
      unit: "mean_token_probability",
      per_layer: { "4": 0.25, "9": 0.625 },
      depth: {
        center: [0.688312],
        spread: [0.205343],
        basis: "readout_probability_mass",
      },
    },
  });
  assert.deepEqual(oriented.instruments.sae.readings, {
    "sae/8": {
      value: 0.5625,
      unit: "raw_activation",
      per_layer: { "4": 0.375, "9": 0.75 },
      depth: {
        center: [0.666667],
        spread: [0.214275],
        basis: "single_layer",
      },
    },
  });
  assert.equal(oriented.instruments.lens.readout, undefined);
  assert.equal(oriented.instruments.sae.readout, undefined);

  const preciseProgram = {
    format: "drowse-structured-v2", layerCount: 2,
    probeKind: Uint32Array.of(3, 2, 3, 2),
    measurementSchema: {
      layerMap: [0, 1], modelLayerCount: 2,
      probes: [
        { name: "jlens/rare", family: "lens", tokenId: 17 },
        { name: "sae/8", family: "sae", featureId: 8 },
      ],
      lensSource: "fixture-lens", saeSource: "fixture-sae",
    },
  };
  const preciseValues = Float32Array.of(1e-8, 1.2345678, 3e-8, 2.3456789);
  const precise = measurementEnvelope(preciseProgram, preciseValues, null);
  for (const [index, family, name] of [[0, "lens", "jlens/rare"], [1, "sae", "sae/8"]]) {
    const expected = (preciseValues[index] + preciseValues[index + 2]) / 2;
    assert.equal(precise.scores[name], expected, "stored scores retain measured precision");
    assert.equal(precise.instruments[family].readings[name].value, expected);
    assert.equal(precise.instruments[family].readings[name].per_layer["0"], preciseValues[index]);
    assert.equal(precise.per_layer_scores["0"][name], preciseValues[index]);
  }
  for (const [index, invalid] of [[0, NaN], [2, Infinity], [1, -Infinity], [0, -0.1], [2, 1.1], [3, -0.1]]) {
    const invalidValues = preciseValues.slice();
    invalidValues[index] = invalid;
    assert.throws(() => measurementEnvelope(preciseProgram, invalidValues, null),
      (error) => error.code === "INVALID_HOOK_MEASUREMENTS", "invalid layers must not silently change the average");
  }
  const inactiveProgram = { ...preciseProgram, probeKind: Uint32Array.of(0, 2, 3, 2) };
  const inactiveValues = preciseValues.slice();
  inactiveValues[0] = NaN;
  assert.equal(measurementEnvelope(inactiveProgram, inactiveValues, null).scores["jlens/rare"], preciseValues[2]);

  const exactReadoutProgram = compileStructuredHookProgram(
    1,
    [{}, {}, {}],
    STANDARD_STRUCTURED_HOOK_PROFILE_V3,
    {
      bindingId: "a".repeat(64),
      layerIndices: Int32Array.of(0, 2),
      tokenIds: new Int32Array(8),
    },
  );
  exactReadoutProgram.measurementSchema = {
    layerMap: [4, 7, 9],
    modelLayerCount: 10,
    probes: new Array(8).fill(null),
    lensSource: "fixture-lens",
    saeSource: "fixture-sae",
    saeLayer: 7,
    saeFeatureCount: 5000,
    lensReadout: true,
    saeReadout: true,
    saeFeatureMetadata: {
      "4097": { label: "high feature", maxAct: 8 },
    },
  };
  const aggregateTokenIds = Int32Array.from([42, 43, 44, 45, 46, 47, 48, 49]);
  const layerTokenIds = Int32Array.from([
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16,
  ]);
  const exactEnvelope = measurementEnvelope(
    exactReadoutProgram,
    new Float32Array(24),
    null,
    undefined,
    {
      tokenIds: aggregateTokenIds,
      tokens: [...aggregateTokenIds].map((id) => ` token-${id}`),
      strength: Float32Array.from([0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13]),
      centerOfMass: Float32Array.from([0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1]),
      spread: Float32Array.from([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
      fittedLayerCount: 2,
      layerIndices: Int32Array.from([0, 2]),
      layerTokenIds,
      layerTokens: [...layerTokenIds].map((id) => ` layer-${id}`),
      layerProbabilities: Float32Array.from([
        0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0,
        0.35, 0.25, 0.18, 0.14, 0.1, 0.07, 0.05, 0,
      ]),
    },
    {
      featureIds: Int32Array.from([4097, 3]),
      activations: Float32Array.from([4, 0.5]),
      runtimeLayerIndex: 1,
      featureCount: 5000,
    },
  );
  assert.equal(exactEnvelope.instruments.lens.readout.aggregate[0].token, " token-42");
  assert.equal(
    exactEnvelope.instruments.lens.readout.layers.flatMap((row) => row.tokens)
      .some((token) => token.id === 42),
    false,
  );
  assert.deepEqual(
    exactEnvelope.instruments.lens.readout.layers.map((row) => row.layer),
    [4, 9],
  );
  assert.deepEqual(exactEnvelope.instruments.sae.readout.features[0], {
    id: 4097,
    activation: 4,
    label: "high feature",
    max_act: 8,
  });
  assert.equal(exactEnvelope.instruments.sae.binding.layer, 7);
  assert.equal(
    exactEnvelope.instruments.lens.readout.layers[0].tokens.at(-1).logprob,
    Math.log(1e-45),
  );
  assert.equal(Object.hasOwn(exactEnvelope, "scores"), false);

  const exactLensReadoutAtWidth = (width) => {
    const probabilities = Float32Array.from([
      0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04,
      0.35, 0.25, 0.18, 0.14, 0.1, 0.07, 0.05, 0.03,
    ]);
    const selectColumns = (values) => {
      const selected = values instanceof Int32Array
        ? new Int32Array(2 * width)
        : new Float32Array(2 * width);
      for (let row = 0; row < 2; row += 1) {
        selected.set(values.subarray(row * 8, row * 8 + width), row * width);
      }
      return selected;
    };
    return {
      tokenIds: aggregateTokenIds.slice(0, width),
      tokens: [...aggregateTokenIds.slice(0, width)].map((id) => ` token-${id}`),
      strength: Float32Array.from([
        0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13,
      ]).slice(0, width),
      centerOfMass: Float32Array.from([
        0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1,
      ]).slice(0, width),
      spread: Float32Array.from([
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
      ]).slice(0, width),
      fittedLayerCount: 2,
      layerIndices: Int32Array.from([0, 2]),
      layerTokenIds: selectColumns(layerTokenIds),
      layerTokens: [...selectColumns(layerTokenIds)].map((id) => ` layer-${id}`),
      layerProbabilities: selectColumns(probabilities),
    };
  };
  for (const alternatives of [5, 8, 262144]) {
    const width = Math.min(alternatives, 8);
    const requestedWidths = [];
    let scalarReads = 0;
    const exactTokens = [];
    const exactResult = await streamWebLlmGeneration(
      engineWithChat([chatChunks[1]]),
      {
        async clear() {},
        async interrupt() {},
        async install() {},
        async read() {
          scalarReads += 1;
          throw new Error("full readout must not read an undeclared scalar buffer");
        },
        async readBundle(topK) {
          requestedWidths.push(["bundle", topK]);
          return {
            scalar: undefined,
            geometry: undefined,
            jlensTopTokens: exactLensReadoutAtWidth(topK),
            saeTopFeatures: {
              featureIds: Int32Array.from([4097, 3]),
              activations: Float32Array.from([4, 0.5]),
              runtimeLayerIndex: 1,
              featureCount: 5000,
            },
          };
        },
        async readJlensTopTokens() {
          throw new Error("bundled readout must not use the legacy lens read");
        },
        async readSaeTopFeatures() {
          throw new Error("bundled readout must not use the legacy SAE read");
        },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: exactReadoutProgram,
        measurementSpecialTokenIds: [99],
        sampling: { return_top_k: alternatives },
      },
      (token) => exactTokens.push(token),
    );
    assert.equal(scalarReads, 0);
    assert.deepEqual(requestedWidths, [["bundle", width]]);
    assert.equal(exactTokens[0].measurements.instruments.lens.readout.aggregate.length, width);
    assert.deepEqual(
      exactTokens[0].measurements.instruments.lens.readout.layers.map((row) => row.tokens.length),
      [width, width],
    );
    assert.equal(exactResult.measurements.instruments.lens.readout.aggregate.length, width);
  }

  let targetedBundleReads = 0;
  const targetedStarts = [];
  const targetedTokens = [];
  await streamWebLlmGeneration(
    engineWithChat([chatChunks[1], chatChunks[2], chatChunks[3]]),
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async read() { throw new Error("targeted readout has no scalar probes"); },
      async readBundle(topK) {
        targetedBundleReads += 1;
        return {
          scalar: undefined,
          geometry: undefined,
          jlensTopTokens: exactLensReadoutAtWidth(topK),
          saeTopFeatures: {
            featureIds: Int32Array.from([4097, 3]),
            activations: Float32Array.from([4, 0.5]),
            runtimeLayerIndex: 1,
            featureCount: 5000,
          },
        };
      },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: exactReadoutProgram,
      measurementSpecialTokenIds: [99],
      measurementTargetRawIndex: 1,
      onRawTokenStart(rawIndex) { targetedStarts.push(rawIndex); },
    },
    (token) => targetedTokens.push(token),
  );
  assert.deepEqual(targetedStarts, [0, 1]);
  assert.equal(targetedBundleReads, 1);
  assert.equal(targetedTokens[0].measurements, undefined);
  assert.equal(targetedTokens[1].measurements.instruments.lens.readout.layers.length, 2);

  let continuationReadouts = 0;
  const continuationTokens = [];
  const continuationResult = await streamWebLlmGeneration(
    engineWithChat([chatChunks[1], chatChunks[2], chatChunks[3]]),
    {
      async clear() {}, async interrupt() {}, async install() {},
      async read() { assert.fail("an unchanged prefix should not read probes"); },
      async readBundle(topK) {
        continuationReadouts += 1;
        return { jlensTopTokens: exactLensReadoutAtWidth(topK),
          saeTopFeatures: { featureIds: Int32Array.of(3), activations: Float32Array.of(0.5),
            runtimeLayerIndex: 1, featureCount: 5000 } };
      },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: exactReadoutProgram,
      replay: { forcedPrefixTokenIds: [11] },
      measurementStartRawIndex: 1,
      measurementSpecialTokenIds: [99],
    },
    (token) => continuationTokens.push(token),
  );
  assert.equal(continuationReadouts, 1);
  assert.deepEqual(continuationTokens.map((token) => token.text), targetedTokens.map((token) => token.text));
  assert.deepEqual(continuationTokens.map((token) => token.logprob), targetedTokens.map((token) => token.logprob));
  assert.equal(continuationTokens[0].measurements, undefined);
  assert.ok(continuationResult.measurements.instruments.lens.readout);

  const replayGateUpdates = [];
  let replayGateReads = 0;
  await streamWebLlmGeneration(engine, {
    async clear() {}, async interrupt() {}, async install() {},
    async updateControls(affine) { replayGateUpdates.push([...affine]); },
    async readBundle() { assert.fail("prefix gates must not trigger discovery readouts"); },
    async read() { replayGateReads += 1; return Float32Array.of(0.75, 0, 0, 0, 0, 0, 0, 0); },
    assertSteeringSupported() {},
  }, {
    input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
    hookProgram: gatedProgram, replay: { forcedPrefixTokenIds: [11, 12] },
    measurementStartRawIndex: 2, measurementSpecialTokenIds: [99],
  }, () => {});
  assert.equal(replayGateReads, 2);
  assert.equal(replayGateUpdates[0][0], 1);

  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([chatChunks[1]]),
      {
        async clear() {},
        async interrupt() {},
        async install() {},
        async read() { return undefined; },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        measurementTargetRawIndex: -1,
      },
      () => undefined,
    ),
    (error) => error.code === "INVALID_MEASUREMENT_TARGET",
  );

  let overCapacityCreateCalled = false;
  let overCapacityInstallCalled = false;
  await assert.rejects(
    streamWebLlmGeneration(
      {
        ...engineWithChat([chatChunks[1]]),
        chat: { completions: { async create() {
          overCapacityCreateCalled = true;
          return from([chatChunks[1]]);
        } } },
      },
      {
        async clear() {},
        async interrupt() {},
        async install() { overCapacityInstallCalled = true; },
        async read() { return new Float32Array(24); },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: exactReadoutProgram,
        readoutTopK: 9,
        measurementSpecialTokenIds: [99],
      },
      () => undefined,
    ),
    (error) => error.code === "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
  );
  assert.equal(overCapacityCreateCalled, false);
  assert.equal(overCapacityInstallCalled, false);

  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([chatChunks[1]]),
      {
        async clear() {},
        async interrupt() {},
        async install() {},
        async read() { return new Float32Array(24); },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: exactReadoutProgram,
        measurementSpecialTokenIds: [99],
      },
      () => undefined,
    ),
    (error) => error.code === "JLENS_FULL_READOUT_UNAVAILABLE",
  );

  const trailProgram = (layers, layerMap = layers.map((_, index) => index)) => {
    const intrinsicDim = 2;
    const candidateCount = layers[0].candidates.length;
    assert.ok(layers.every((layer) => layer.candidates.length === candidateCount));
    const program = compileStructuredHookProgram(
      2,
      layers.map(({ rank, candidates }) => ({
        geometryWhitener: {
          rank: 1,
          ridge: 1,
          basis: [[1, 0]],
          correction: [0],
        },
        geometryProbes: [{
          active: true,
          mean: [0, 0],
          inverseMean: [0, 0],
          basis: Array.from({ length: rank }, (_, axis) =>
            axis === 0 ? [1, 0] : [0, 1]
          ),
          gramInverse: rank === 1 ? [1] : [1, 0, 0, 1],
          cholesky: rank === 1 ? [1] : [1, 0, 0, 1],
          candidateWhite: candidates,
          coordinateMap: Array.from({ length: intrinsicDim }, (_, axis) =>
            Array.from({ length: rank }, (_, component) => Number(axis === component))
          ),
          coordinateBias: [0, 0],
        }],
      })),
      STANDARD_STRUCTURED_HOOK_PROFILE_V2,
    );
    program.measurementSchema = {
      layerMap,
      modelLayerCount: Math.max(...layerMap) + 1,
      probes: new Array(program.profile.maxProbes).fill(null),
      geometryProbes: [{
        name: "local/trail",
        scoreKeys: ["local/trail:fraction"],
        manifold: "local/trail",
        labels: Array.from({ length: candidateCount }, (_, index) => `anchor-${index}`),
        topN: candidateCount,
        intrinsicDim,
        rank: layers[0].rank,
        shareWeights: layers.map(() => 1),
        assignBandwidth: layers[0].candidates.map(() => 1),
        assignLogVolumeBias: layers[0].candidates.map(() => 0),
        labelScale: 1,
      }, ...new Array(program.profile.maxGeometryProbes - 1).fill(null)],
      lensSource: null,
      saeSource: null,
    };
    return program;
  };
  const trailValues = (program, coordinates) => {
    const values = new Float32Array(
      program.layerCount * program.profile.maxGeometryProbes *
        program.profile.geometryOutputStride,
    );
    coordinates.forEach((coordinate, layerIndex) => {
      const slot = layerIndex * program.profile.maxGeometryProbes;
      const rank = program.geometryRank[slot];
      const candidateCount = program.geometryCandidateCount[slot];
      const offset = slot * program.profile.geometryOutputStride;
      values.set([1, 0.5, 0.25, 0.75], offset);
      values.set(coordinate, offset + 4);
      const distanceOffset = offset + 4 + program.profile.maxIntrinsicDim;
      for (let candidate = 0; candidate < candidateCount; candidate += 1) {
        let squared = 0;
        for (let axis = 0; axis < rank; axis += 1) {
          const anchor = program.geometryNodeWhite[
            (slot * program.profile.maxGeometryCandidates + candidate) *
              program.profile.maxRank + axis
          ];
          squared += (coordinate[axis] - anchor) ** 2;
        }
        values[distanceOffset + candidate] = Math.sqrt(squared);
      }
    });
    return values;
  };

  const mixedRankTrailProgram = trailProgram([
    { rank: 2, candidates: [[0, 0], [1, 0], [0, 1]] },
    { rank: 1, candidates: [[-1], [1], [0]] },
  ], [4, 9]);
  const mixedRankTrailValues = trailValues(
    mixedRankTrailProgram,
    [[0.25, 0.75], [0.4]],
  );
  const disabledTrailEnvelope = measurementEnvelope(
    mixedRankTrailProgram,
    undefined,
    null,
    mixedRankTrailValues,
  );
  assert.equal(
    Object.hasOwn(
      disabledTrailEnvelope.instruments.geometry.readings["local/trail"],
      "subspace_coords_per_layer",
    ),
    false,
  );
  const mixedRankTrailEnvelope = measurementEnvelope(
    mixedRankTrailProgram,
    undefined,
    null,
    mixedRankTrailValues,
    undefined,
    undefined,
    8,
    true,
  );
  assert.deepEqual(
    Object.fromEntries(Object.entries(
      mixedRankTrailEnvelope.instruments.geometry.readings["local/trail"]
        .subspace_coords_per_layer,
    ).map(([layer, coords]) => [layer, coords.map((value) => Number(value.toFixed(6)))])),
    { "4": [0.25, 0.75], "9": [0.4] },
  );

  const sharedMeasurementFixture = JSON.parse(await readFile(
    new URL("../../browser-runtime/fixtures/measurement-envelope-v1.json", import.meta.url),
    "utf8",
  ));
  const sharedGeometry = sharedMeasurementFixture.geometry;
  const sharedProgram = compileStructuredHookProgram(
    1,
    sharedGeometry.rows.map(() => ({
      geometryWhitener: {
        rank: 1,
        ridge: 1,
        basis: [[1]],
        correction: [0],
      },
      geometryProbes: [{
        active: true,
        mean: [0],
        inverseMean: [0],
        basis: [[1]],
        gramInverse: [1],
        cholesky: [1],
        candidateWhite: [[0], [1]],
        coordinateMap: [[1]],
        coordinateBias: [0],
      }],
    })),
    STANDARD_STRUCTURED_HOOK_PROFILE_V2,
  );
  const sharedProbeCount = sharedProgram.profile.maxProbes;
  for (let layer = 0; layer < sharedProgram.layerCount; layer += 1) {
    sharedProgram.probeKind[layer * sharedProbeCount] = 1;
  }
  sharedProgram.probeKind[sharedProbeCount + 1] = 1;
  sharedProgram.measurementSchema = {
    layerMap: sharedMeasurementFixture.layerMap,
    modelLayerCount: sharedMeasurementFixture.modelLayerCount,
    probes: [
      {
        name: sharedMeasurementFixture.lens.name,
        family: "lens",
        tokenId: sharedMeasurementFixture.lens.tokenId,
      },
      {
        name: sharedMeasurementFixture.sae.name,
        family: "sae",
        featureId: sharedMeasurementFixture.sae.featureId,
        label: sharedMeasurementFixture.sae.label,
        maxAct: sharedMeasurementFixture.sae.maxAct,
      },
      ...new Array(sharedProbeCount - 2).fill(null),
    ],
    geometryProbes: [{
      name: sharedGeometry.name,
      scoreKeys: [
        `${sharedGeometry.name}:fraction`,
        `${sharedGeometry.name}:membership`,
        ...sharedGeometry.labels.flatMap((label) => [
          `${sharedGeometry.name}@${label}`,
          `${sharedGeometry.name}~${label}`,
        ]),
      ],
      manifold: sharedGeometry.name,
      labels: sharedGeometry.labels,
      topN: sharedGeometry.labels.length,
      intrinsicDim: 1,
      rank: 1,
      shareWeights: sharedGeometry.shareWeights,
      assignBandwidth: sharedGeometry.assignBandwidth,
      assignLogVolumeBias: sharedGeometry.assignLogVolumeBias,
      labelScale: sharedGeometry.labelScale,
    }, ...new Array(sharedProgram.profile.maxGeometryProbes - 1).fill(null)],
    lensSource: sharedMeasurementFixture.lens.source,
    saeSource: sharedMeasurementFixture.sae.source,
    saeLayer: sharedMeasurementFixture.sae.layer,
    saeFeatureCount: sharedMeasurementFixture.sae.featureCount,
    lensReadout: true,
    saeReadout: true,
    saeFeatureMetadata: {
      [sharedMeasurementFixture.sae.featureId]: {
        label: sharedMeasurementFixture.sae.label,
        maxAct: sharedMeasurementFixture.sae.maxAct,
      },
    },
  };
  const sharedScalarValues = new Float32Array(
    sharedProgram.layerCount * sharedProbeCount,
  );
  sharedMeasurementFixture.lens.probabilities.forEach((probability, layer) => {
    sharedScalarValues[layer * sharedProbeCount] = probability;
  });
  sharedScalarValues[sharedProbeCount + 1] = sharedMeasurementFixture.sae.value;
  const sharedGeometryValues = new Float32Array(
    sharedProgram.layerCount * sharedProgram.profile.maxGeometryProbes *
      sharedProgram.profile.geometryOutputStride,
  );
  sharedGeometry.rows.forEach((row, layer) => {
    const slot = layer * sharedProgram.profile.maxGeometryProbes;
    const offset = slot * sharedProgram.profile.geometryOutputStride;
    sharedGeometryValues.set(
      [1, row.fraction, row.residual, row.membership, ...row.coords],
      offset,
    );
    sharedGeometryValues.set(
      row.distances,
      offset + 4 + sharedProgram.profile.maxIntrinsicDim,
    );
  });
  const sharedEnvelope = measurementEnvelope(
    sharedProgram,
    sharedScalarValues,
    sharedMeasurementFixture.steering,
    sharedGeometryValues,
    {
      tokenIds: Int32Array.of(sharedMeasurementFixture.lens.tokenId),
      tokens: [sharedMeasurementFixture.lens.token],
      strength: Float32Array.of(0.5),
      centerOfMass: Float32Array.of(
        sharedMeasurementFixture.expected.instruments.lens.readout.aggregate[0].com,
      ),
      spread: Float32Array.of(
        sharedMeasurementFixture.expected.instruments.lens.readout.aggregate[0].spread,
      ),
      fittedLayerCount: 2,
      layerIndices: Int32Array.of(0, 1),
      layerTokenIds: Int32Array.of(
        sharedMeasurementFixture.lens.tokenId,
        sharedMeasurementFixture.lens.tokenId,
      ),
      layerTokens: [
        sharedMeasurementFixture.lens.token,
        sharedMeasurementFixture.lens.token,
      ],
      layerProbabilities: Float32Array.from(sharedMeasurementFixture.lens.probabilities),
    },
    {
      featureIds: Int32Array.of(sharedMeasurementFixture.sae.featureId),
      activations: Float32Array.of(sharedMeasurementFixture.sae.activation),
      runtimeLayerIndex: 1,
      featureCount: sharedMeasurementFixture.sae.featureCount,
    },
    1,
    true,
  );
  assertParity(
    sharedEnvelope,
    sharedMeasurementFixture.expected,
    sharedMeasurementFixture.tolerance,
  );
  const sharedGateScores = measurementGateScores(sharedEnvelope);
  assert.ok(`${sharedGeometry.name}:fraction` in sharedGateScores);
  assert.equal(`${sharedGeometry.name}:fraction` in sharedEnvelope.scores, false);

  const geometryGateProgram = trailProgram([
    { rank: 1, candidates: [[0], [1]] },
  ], [4]);
  geometryGateProgram.controls = {
    affine: [{
      enabled: true,
      phase: { kind: "generated_only" },
      gate: {
        slots: [],
        scoreKey: "local/trail:fraction",
        operator: ">",
        threshold: 0.4,
      },
    }, ...new Array(geometryGateProgram.affineActive.length - 1).fill(null)],
    curve: new Array(geometryGateProgram.curveActive.length).fill(null),
  };
  const geometryGateValues = trailValues(geometryGateProgram, [[0.25]]);
  const geometryGateUpdates = [];
  const geometryGateTokens = [];
  const geometryGateRawTokens = [];
  const geometryGateResult = await streamWebLlmGeneration(
    engineWithChat([chatChunks[1]]),
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async updateControls(affineActive) {
        geometryGateUpdates.push([...affineActive]);
      },
      async readBundle() { return { geometry: geometryGateValues }; },
      async read() { assert.fail("a geometry-only gate must not request scalar probes"); },
      async readGeometry() { assert.fail("the bundled geometry read must be reused"); },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: geometryGateProgram,
      measurementSpecialTokenIds: [99],
      sampling: { persist_per_layer_scores: true },
      onRawToken(token) { geometryGateRawTokens.push(token); },
    },
    (token) => geometryGateTokens.push(token),
  );
  assert.equal(geometryGateUpdates[0][0], 1);
  for (const persisted of [
    geometryGateRawTokens[0].measurements,
    geometryGateTokens[0].measurements,
    geometryGateResult.measurements,
  ]) {
    assert.deepEqual(Object.keys(persisted.scores), ["local/trail"]);
    assert.equal("local/trail:fraction" in persisted.scores, false);
    assert.deepEqual(Object.keys(persisted.per_layer_scores["4"]), ["local/trail"]);
  }

  const insufficientAnchorProgram = trailProgram([
    { rank: 2, candidates: [[0, 0], [1, 0]] },
  ], [12]);
  const insufficientAnchorValues = trailValues(insufficientAnchorProgram, [[0.25, 0.5]]);
  assert.equal(
    Object.hasOwn(
      measurementEnvelope(
        insufficientAnchorProgram,
        undefined,
        null,
        insufficientAnchorValues,
      ).instruments.geometry.readings["local/trail"],
      "subspace_coords_per_layer",
    ),
    false,
  );
  assert.throws(
    () => measurementEnvelope(
      insufficientAnchorProgram,
      undefined,
      null,
      insufficientAnchorValues,
      undefined,
      undefined,
      8,
      true,
    ),
    (error) => error.code === "SUBSPACE_COORD_RECONSTRUCTION_FAILED" &&
      error.message.includes("model layer 12"),
  );

  const rankDeficientAnchorProgram = trailProgram([
    { rank: 2, candidates: [[0, 0], [1, 0], [2, 0]] },
  ]);
  assert.throws(
    () => measurementEnvelope(
      rankDeficientAnchorProgram,
      undefined,
      null,
      trailValues(rankDeficientAnchorProgram, [[0.25, 0.5]]),
      undefined,
      undefined,
      8,
      true,
    ),
    (error) => error.code === "SUBSPACE_COORD_RECONSTRUCTION_FAILED" &&
      error.message.includes("affine rank"),
  );

  const nonfiniteAnchorProgram = trailProgram([
    { rank: 2, candidates: [[0, 0], [1, 0], [0, 1]] },
  ]);
  const nonfiniteAnchorValues = trailValues(nonfiniteAnchorProgram, [[0.25, 0.5]]);
  nonfiniteAnchorProgram.geometryNodeWhite[0] = Number.NaN;
  assert.throws(
    () => measurementEnvelope(
      nonfiniteAnchorProgram,
      undefined,
      null,
      nonfiniteAnchorValues,
      undefined,
      undefined,
      8,
      true,
    ),
    (error) => error.code === "SUBSPACE_COORD_RECONSTRUCTION_FAILED" &&
      error.message.includes("non-finite"),
  );

  const inconsistentDistanceProgram = trailProgram([
    { rank: 2, candidates: [[0, 0], [1, 0], [0, 1]] },
  ]);
  const inconsistentDistanceValues = trailValues(inconsistentDistanceProgram, [[0.25, 0.5]]);
  inconsistentDistanceValues[4 + inconsistentDistanceProgram.profile.maxIntrinsicDim + 2] += 0.2;
  assert.throws(
    () => measurementEnvelope(
      inconsistentDistanceProgram,
      undefined,
      null,
      inconsistentDistanceValues,
      undefined,
      undefined,
      8,
      true,
    ),
    (error) => error.code === "SUBSPACE_COORD_RECONSTRUCTION_FAILED" &&
      error.message.includes("exact whitened coordinate"),
  );

  const exactTrailTokens = [];
  let exactTrailBundleReads = 0;
  const exactTrailResult = await streamWebLlmGeneration(
    engineWithChat([chatChunks[1]]),
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async readBundle() {
        exactTrailBundleReads += 1;
        return { geometry: mixedRankTrailValues };
      },
      async read() { assert.fail("exact trails must not add a scalar read"); },
      async readGeometry() { assert.fail("exact trails must reuse the bundled geometry read"); },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      hookProgram: mixedRankTrailProgram,
      measurementSpecialTokenIds: [99],
      sampling: { persist_subspace_coords: true },
    },
    (token) => exactTrailTokens.push(token),
  );
  assert.equal(exactTrailBundleReads, 1);
  assert.deepEqual(
    exactTrailTokens[0].measurements.instruments.geometry.readings["local/trail"]
      .subspace_coords_per_layer,
    exactTrailResult.measurements.instruments.geometry.readings["local/trail"]
      .subspace_coords_per_layer,
  );
  assert.equal(exactTrailTokens[0].measurements.scope, "token");
  assert.equal(exactTrailResult.measurements.scope, "aggregate");

  const geometryValues = new Float32Array(2 * 8 * 41);
  geometryValues.set([1, 0.25, 0.2, 0.8, 0.2, 0.4, 0, 0, 1, 3], 0);
  geometryValues.set([1, 0.75, 0.4, 0.6, 0.8, 0.6, 0, 0, 3, 1], 8 * 41);
  const geometryEnvelope = measurementEnvelope({
    format: "drowse-structured-v3",
    layerCount: 2,
    profile: STANDARD_STRUCTURED_HOOK_PROFILE_V2,
    probeKind: new Uint32Array(16),
    measurementSchema: {
      layerMap: [4, 9],
      modelLayerCount: 12,
      probes: new Array(8).fill(null),
      geometryProbes: [{
        name: "local/demo",
        scoreKeys: [
          "local/demo[1]",
          "local/demo:fraction",
          "local/demo:membership",
          "local/demo@calm",
          "local/demo~alert",
        ],
        manifold: "local/demo",
        labels: ["calm", "alert"],
        topN: 2,
        intrinsicDim: 2,
        rank: 2,
        shareWeights: [0.25, 0.75],
        assignBandwidth: [1, 2],
        assignLogVolumeBias: [0, 0],
        labelScale: 2,
      }, null, null, null, null, null, null, null],
      lensSource: null,
      saeSource: null,
    },
  }, undefined, null, geometryValues);
  assert.ok(Math.abs(geometryEnvelope.scores["local/demo"] - 0.65) < 1e-6);
  assert.deepEqual(Object.keys(geometryEnvelope.scores), ["local/demo"]);
  const geometryGateScores = measurementGateScores(geometryEnvelope);
  assert.ok(Math.abs(geometryGateScores["local/demo[1]"] - 0.55) < 1e-6);
  assert.ok(Math.abs(geometryGateScores["local/demo:fraction"] - 0.625) < 1e-6);
  assert.ok(Math.abs(geometryGateScores["local/demo:membership"] - 0.65) < 1e-6);
  assert.ok(Math.abs(geometryGateScores["local/demo@calm"] + 1.25) < 1e-6);
  assert.deepEqual(geometryEnvelope.instruments.geometry.readings["local/demo"].nearest, [
    ["alert", 0.75],
    ["calm", 1.25],
  ]);
  assert.deepEqual(
    Object.fromEntries(Object.entries(
      geometryEnvelope.instruments.geometry.readings["local/demo"].coords_per_layer,
    ).map(([layer, coords]) => [layer, coords.map((value) => Number(value.toFixed(6)))])),
    { "4": [0.2, 0.4], "9": [0.8, 0.6] },
  );
  assert.ok(geometryGateScores["local/demo~alert"] > 0.5);
  assert.deepEqual(Object.keys(geometryEnvelope.per_layer_scores["4"]), ["local/demo"]);

  const thinkingProgram = compileStructuredHookProgram(1, [{
    affineGroups: [{
      active: true,
      basis: [[1]],
      neutral: [0],
      target: [1],
      along: 0.5,
      kappa: [0],
    }],
  }], STANDARD_STRUCTURED_HOOK_PROFILE);
  thinkingProgram.controls = {
    affine: [{
      enabled: true,
      phase: { kind: "thinking_only" },
      gate: null,
    }, null, null, null],
    curve: [null, null, null, null],
  };
  const thinkingUpdates = [];
  const thinkingTokens = [];
  let thinkingRequest;
  const thinkingResult = await streamWebLlmGeneration(
    {
      chat: { completions: { async create(request) {
        thinkingRequest = request;
        return from(["<thi", "nk>", "reason", "</thi", "nk>", "answer"].map((token, index) => ({
          choices: [{
            index: 0,
            delta: { content: token },
            finish_reason: null,
            logprobs: { content: [{ token, token_id: 100 + index, logprob: -0.1 }] },
          }],
        })));
      } } },
      completions: { async create() { throw new Error("unexpected raw generation"); } },
    },
    {
      async clear() {},
      async interrupt() {},
      async install(program) { assert.equal(program.affineActive[0], 0); },
      async updateControls(affineActive) { thinkingUpdates.push(affineActive[0]); },
      async read() { return new Float32Array(8); },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "think" }] },
      thinking: true,
      thinkingProfile: {
        start: "<think>",
        end: "</think>",
        startTokenIds: [100, 101],
        endTokenIds: [103, 104],
        startsInThinking: false,
      },
      steeringExpression: "0.5 honest@thinking",
      hookProgram: thinkingProgram,
    },
    (token) => thinkingTokens.push(token),
  );
  assert.deepEqual(thinkingRequest.extra_body, { enable_thinking: true });
  assert.deepEqual(thinkingUpdates, [1, 0]);
  assert.equal(thinkingProgram.affineActive[0], 1, "generation does not mutate the compiled payload");
  assert.deepEqual(thinkingTokens.map(({ text, thinking }) => [text, thinking]), [
    ["reason", true],
    ["answer", false],
  ]);
  assert.equal(thinkingResult.text, "answer");
  assert.equal(thinkingResult.thinkingText, "reason");

  const delimiterSpoofTokens = [];
  const delimiterSpoofResult = await streamWebLlmGeneration(
    engineWithChat([{
      choices: [{
        index: 0,
        delta: { content: "<think>ordinary text</think>" },
        finish_reason: null,
        logprobs: {
          content: [{
            token: "<think>ordinary text</think>",
            token_id: 999,
            logprob: -0.25,
          }],
        },
      }],
    }]),
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "quote" }] },
      thinking: true,
      thinkingProfile: {
        start: "<think>",
        end: "</think>",
        startTokenIds: [40],
        endTokenIds: [42],
        startsInThinking: false,
      },
    },
    (token) => delimiterSpoofTokens.push(token),
  );
  assert.deepEqual(
    delimiterSpoofTokens.map(({ text, thinking, tokenId }) => [text, thinking, tokenId]),
    [["<think>ordinary text</think>", false, 999]],
  );
  assert.equal(delimiterSpoofResult.text, "<think>ordinary text</think>");

  const exactRawTokens = [];
  const visibleThinkingTokens = [];
  const exactThinkingResult = await streamWebLlmGeneration(
    {
      chat: { completions: { async create() {
        return from(["<think>", "reason", "</think>", "answer"].map((token, index) => ({
          choices: [{
            index: 0,
            delta: { content: token },
            finish_reason: null,
            logprobs: { content: [{
              token,
              token_id: 40 + index,
              logprob: -0.1 * (index + 1),
              top_logprobs: [{
                token,
                token_id: 40 + index,
                logprob: -0.1 * (index + 1),
              }],
            }] },
          }],
        })));
      } } },
      completions: { async create() { throw new Error("unexpected raw generation"); } },
    },
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "think" }] },
      thinking: true,
      thinkingProfile: {
        start: "<think>",
        end: "</think>",
        startTokenIds: [40],
        endTokenIds: [42],
        startsInThinking: false,
      },
      async onRawToken(token) { exactRawTokens.push(token); },
    },
    (token) => visibleThinkingTokens.push(token),
  );
  assert.deepEqual(exactRawTokens.map(({ text, tokenId, rawIndex }) => [text, tokenId, rawIndex]), [
    ["<think>", 40, 0],
    ["reason", 41, 1],
    ["</think>", 42, 2],
    ["answer", 43, 3],
  ]);
  assert.deepEqual(visibleThinkingTokens.map(({ text, thinking, rawIndex }) => [text, thinking, rawIndex]), [
    ["reason", true, 1],
    ["answer", false, 3],
  ]);
  assert.equal(exactThinkingResult.meanLogprob, -0.4);
  assert.equal(exactThinkingResult.meanSurprise, 0.4);

  const specialSummaryResult = await streamWebLlmGeneration(
    engineWithChat([
      {
        choices: [{
          index: 0,
          delta: { content: "answer" },
          finish_reason: null,
          logprobs: { content: [{ token: "answer", token_id: 43, logprob: -0.4 }] },
        }],
      },
      {
        choices: [{
          index: 0,
          delta: { content: "<|end|>" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "<|end|>", token_id: 99, logprob: -9 }] },
        }],
      },
    ]),
    hooks,
    {
      input: { kind: "chat", messages: [{ role: "user", content: "summarize" }] },
      measurementSpecialTokenIds: [99],
    },
    () => {},
  );
  assert.equal(specialSummaryResult.meanLogprob, -0.4);
  assert.equal(specialSummaryResult.meanSurprise, 0.4);

  const unicodeRawTokens = [];
  const unicodeVisibleTokens = [];
  let unicodeMeasurementReads = 0;
  let unicodeControlUpdates = 0;
  const unicodeResult = await streamWebLlmGeneration(
    engineWithChat([
      {
        choices: [{
          index: 0,
          delta: { content: "" },
          finish_reason: null,
          logprobs: { content: [{ token: "", token_id: 100, logprob: -0.2 }] },
        }],
      },
      {
        choices: [{
          index: 0,
          delta: { content: "😀" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "😀", token_id: 101, logprob: -0.3 }] },
        }],
      },
    ]),
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async updateControls() { unicodeControlUpdates += 1; },
      async read() {
        unicodeMeasurementReads += 1;
        return new Float32Array(8);
      },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "emoji" }] },
      steeringExpression: "0.5 honest@response",
      hookProgram: gatedProgram,
      measurementSpecialTokenIds: [99],
      async onRawToken(token) { unicodeRawTokens.push(token); },
    },
    (token) => unicodeVisibleTokens.push(token),
  );
  assert.deepEqual(
    unicodeRawTokens.map(({ text, tokenId, rawIndex }) => [text, tokenId, rawIndex]),
    [["", 100, 0], ["😀", 101, 1]],
  );
  assert.deepEqual(
    unicodeVisibleTokens.map(({ text, tokenId, rawIndex }) => [text, tokenId, rawIndex]),
    [["😀", 101, 1]],
  );
  assert.equal(unicodeResult.text, "😀");
  assert.equal(unicodeResult.tokens, 2);
  assert.equal(unicodeMeasurementReads, 2);
  assert.equal(unicodeControlUpdates, 0);

  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([{
        choices: [{
          index: 0,
          delta: { content: "unframed" },
          finish_reason: null,
          logprobs: null,
        }],
      }]),
      hooks,
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        onRawToken() {},
      },
      () => assert.fail("unframed raw tokens must not be emitted"),
    ),
    (error) => error.code === "RAW_TOKEN_BOUNDARY_UNAVAILABLE",
  );

  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([{
        choices: [{
          index: 0,
          delta: { content: "ab" },
          finish_reason: null,
          logprobs: { content: [
            { token: "a", logprob: -0.1 },
            { token: "b", logprob: -0.2 },
          ] },
        }],
      }]),
      {
        async clear() {},
        async interrupt() {},
        async install() {},
        async updateControls() {},
        async read() { return Float32Array.of(1); },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        steeringExpression: "0.5 honest@response",
        hookProgram: gatedProgram,
        measurementSpecialTokenIds: [99],
      },
      () => {},
    ),
    (error) => error.code === "BATCHED_TOKEN_CONTROLS_UNAVAILABLE",
  );

  let batchedMeasurementReads = 0;
  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([{
        choices: [{
          index: 0,
          delta: { content: "ab" },
          finish_reason: null,
          logprobs: { content: [
            { token: "a", logprob: -0.1 },
            { token: "b", logprob: -0.2 },
          ] },
        }],
      }]),
      {
        async clear() {},
        async interrupt() {},
        async install() {},
        async read() {
          batchedMeasurementReads += 1;
          return Float32Array.of(1);
        },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: measurementOnlyProgram,
        measurementSpecialTokenIds: [99],
      },
      () => assert.fail("batched measured tokens must fail before emission"),
    ),
    (error) => error.code === "BATCHED_TOKEN_MEASUREMENTS_UNAVAILABLE",
  );
  assert.equal(batchedMeasurementReads, 0);

  const unmeasuredBatch = [];
  const unmeasuredBatchResult = await streamWebLlmGeneration(
    engineWithChat([{
      choices: [{
        index: 0,
        delta: { content: "ab" },
        finish_reason: "stop",
        logprobs: { content: [
          { token: "a", token_id: 1, logprob: -0.1 },
          { token: "b", token_id: 2, logprob: -0.2 },
        ] },
      }],
    }]),
    hooks,
    { input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] } },
    (token) => unmeasuredBatch.push(token),
  );
  assert.deepEqual(unmeasuredBatch.map((token) => token.tokenId), [1, 2]);
  assert.equal(unmeasuredBatchResult.text, "ab");

  let pureSteeringBatchReads = 0;
  const pureSteeringBatch = [];
  await streamWebLlmGeneration(
    engineWithChat([{
      choices: [{
        index: 0,
        delta: { content: "ab" },
        finish_reason: "stop",
        logprobs: { content: [
          { token: "a", token_id: 1, logprob: -0.1 },
          { token: "b", token_id: 2, logprob: -0.2 },
        ] },
      }],
    }]),
    {
      async clear() {},
      async interrupt() {},
      async install() {},
      async read() {
        pureSteeringBatchReads += 1;
        return Float32Array.of(1);
      },
      assertSteeringSupported() {},
    },
    {
      input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
      steeringExpression: "0.5 honest",
      hookProgram,
    },
    (token) => pureSteeringBatch.push(token),
  );
  assert.deepEqual(pureSteeringBatch.map((token) => token.tokenId), [1, 2]);
  assert.equal(pureSteeringBatchReads, 0);
  assert.equal(pureSteeringBatch.every((token) => token.measurements === undefined), true);

  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([{
        choices: [{
          index: 0,
          delta: { content: "unframed" },
          finish_reason: null,
          logprobs: null,
        }],
      }]),
      {
        async clear() {},
        async interrupt() {},
        async install() {},
        async read() { return Float32Array.of(1); },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "Hi" }] },
        hookProgram: measurementOnlyProgram,
        measurementSpecialTokenIds: [99],
      },
      () => assert.fail("unframed measured text must not be emitted"),
    ),
    (error) => error.code === "RAW_TOKEN_BOUNDARY_UNAVAILABLE",
  );

  const malformedEngine = engineWithChat([
    {
      choices: [
        { index: 0, delta: { content: "a" }, finish_reason: null },
        { index: 1, delta: { content: "b" }, finish_reason: null },
      ],
    },
  ]);
  await assert.rejects(
    streamWebLlmGeneration(
      malformedEngine,
      hooks,
      { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
      () => {},
    ),
    (error) => error.code === "MULTIPLE_GENERATION_CHOICES_UNSUPPORTED",
  );
  assert.equal(hookCalls.at(-2), "interrupt");
  assert.equal(hookCalls.at(-1), "clear");
  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([{ choices: [{ index: 1, delta: { content: "x" } }] }]),
      hooks,
      { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
      () => {},
    ),
    (error) => error.code === "INVALID_GENERATION_CHOICE",
  );

  let interrupted = false;
  let drained = false;
  let clearCount = 0;
  const drainingStream = {
    async *[Symbol.asyncIterator]() {
      yield {
        choices: [{
          index: 0,
          delta: { content: "x" },
          logprobs: { content: [] },
        }],
      };
      drained = true;
      yield { choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
    },
  };
  const drainingEngine = {
    chat: { completions: { async create() { return drainingStream; } } },
    completions: { async create() { return drainingStream; } },
  };
  await assert.rejects(
    streamWebLlmGeneration(
      drainingEngine,
      {
        async clear() {
          clearCount += 1;
          if (clearCount === 2) assert.equal(drained, true);
        },
        async interrupt() { interrupted = true; },
        async install() {},
        async read() { return undefined; },
        assertSteeringSupported() {},
      },
      { input: { kind: "chat", messages: [{ role: "user", content: "x" }] } },
      () => {},
    ),
    (error) => error.code === "GENERATION_TOKEN_TEXT_MISMATCH",
  );
  assert.equal(interrupted, true);
  assert.equal(drained, true);
  assert.equal(clearCount, 2);

  const cleanupCalls = [];
  await assert.rejects(
    streamWebLlmGeneration(
      engineWithChat([], new Error("stream failed")),
      {
        async clear() { cleanupCalls.push("clear"); },
        async interrupt() { cleanupCalls.push("interrupt"); },
        async install() { cleanupCalls.push("install"); },
        async read() { return undefined; },
        assertSteeringSupported() {},
      },
      {
        input: { kind: "chat", messages: [{ role: "user", content: "x" }] },
        hookProgram,
      },
      () => {},
    ),
    /stream failed/,
  );
  assert.deepEqual(cleanupCalls, ["clear", "install", "clear"]);

  let release;
  const blocked = new Promise((resolve) => { release = resolve; });
  const runtimeEngine = {
    ...engineWithChat([], null, blocked),
    async unload() {},
    async reload() {},
    async interruptGenerate() {},
    async resetChat() {},
    async supportsDrowseRankOneHooks() { return true; },
    async setDrowseRankOneProgram() {},
    async clearDrowseRankOneProgram() {},
    async readDrowseMeasurements() { return undefined; },
  };
  const runtime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "post-block-residual-v4",
    MLCEngine: class {},
  });
  runtime.engine = runtimeEngine;
  runtime.drowseCapabilities = await readWebLlmRuntimeCapabilities(runtimeEngine);
  const pending = runtime.streamGeneration(
    { input: { kind: "raw", prompt: "x" } },
    () => {},
  );
  await Promise.resolve();
  await assert.rejects(
    runtime.streamGeneration({ input: { kind: "raw", prompt: "y" } }, () => {}),
    (error) => error.code === "GENERATION_IN_PROGRESS",
  );
  release();
  await pending;

  console.log("WebLLM streaming generation checks passed");
} finally {
  await server.close();
}

function from(values, failure = null, wait = null) {
  return {
    async *[Symbol.asyncIterator]() {
      if (wait) await wait;
      const expanded = values.flatMap(expandFixtureChunk);
      for (const value of expanded) yield value;
      const choices = expanded.flatMap((value) =>
        Array.isArray(value?.choices) ? value.choices : []
      );
      const hasFinishReason = choices.some(
        (choice) => typeof choice?.finish_reason === "string",
      );
      const hasUsage = expanded.some((value) => value?.usage != null);
      let completionTokens = 0;
      for (const choice of choices) {
        const rows = choice?.logprobs?.content;
        if (Array.isArray(rows)) {
          if (choice.drowse_finish_reason !== "eos") {
            completionTokens += rows.length;
          }
          continue;
        }
        const content = choice?.delta?.content ?? choice?.text;
        if (typeof content === "string" && content !== "") completionTokens += 1;
      }
      if (!hasFinishReason) {
        yield {
          choices: [{
            index: 0,
            delta: {},
            finish_reason: "stop",
            drowse_finish_reason: "eos",
            logprobs: null,
          }],
        };
      }
      if (!hasUsage) {
        yield {
          choices: [],
          usage: {
            prompt_tokens: 0,
            completion_tokens: completionTokens,
            total_tokens: completionTokens,
          },
        };
      }
      if (failure) throw failure;
    },
  };
}

function exactFrom(values, failure = null) {
  return {
    async *[Symbol.asyncIterator]() {
      for (const value of values.flatMap(expandFixtureChunk)) yield value;
      if (failure) throw failure;
    },
  };
}

function rawFrom(values, failure = null) {
  return {
    async *[Symbol.asyncIterator]() {
      for (const value of values) yield value;
      if (failure) throw failure;
    },
  };
}

function expandFixtureChunk(value) {
  if (!Array.isArray(value?.choices) || value.choices.length !== 1) {
    return [value];
  }
  const choice = decorateFixtureChoice(value.choices[0]);
  const rows = choice?.logprobs?.content;
  const terminal = fixtureTerminalReason(choice.finish_reason);
  const hasVisibleRow = Array.isArray(rows) && rows.some((row) => row.token !== "");
  if (terminal !== null && hasVisibleRow) {
    const tokenChoice = {
      ...choice,
      finish_reason: null,
      drowse_finish_reason: undefined,
    };
    const terminalChoice = choice.delta !== undefined
      ? {
          index: choice.index,
          delta: {},
          finish_reason: choice.finish_reason,
          drowse_finish_reason: terminal,
          logprobs: null,
        }
      : {
          index: choice.index,
          text: "",
          finish_reason: choice.finish_reason,
          drowse_finish_reason: terminal,
          logprobs: null,
        };
    return [{ ...value, choices: [tokenChoice] }, { choices: [terminalChoice] }];
  }
  return [{
    ...value,
    choices: [{
      ...choice,
      ...(terminal === null ? {} : { drowse_finish_reason: terminal }),
    }],
  }];
}

function decorateFixtureChoice(choice) {
  if (!Array.isArray(choice?.logprobs?.content)) return choice;
  return {
    ...choice,
    logprobs: {
      ...choice.logprobs,
      content: choice.logprobs.content.map((row) => ({
        ...row,
        drowse_sampler: row.drowse_sampler ?? {
          entropy_nats: 0.75,
          perplexity: Math.exp(0.75),
        },
      })),
    },
  };
}

function fixtureTerminalReason(finishReason) {
  if (finishReason === "stop") return "eos";
  if (finishReason === "length") return "length";
  if (finishReason === "abort") return "external_stop";
  return null;
}

function assertParity(actual, expected, tolerance, path = "measurement") {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", `${path} must be numeric`);
    assert.ok(
      Math.abs(actual - expected) <= tolerance,
      `${path}: expected ${expected}, received ${actual}`,
    );
    return;
  }
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${path} must be an array`);
    assert.equal(actual.length, expected.length, `${path} length`);
    expected.forEach((value, index) => {
      assertParity(actual[index], value, tolerance, `${path}[${index}]`);
    });
    return;
  }
  if (expected !== null && typeof expected === "object") {
    assert.equal(
      actual !== null && typeof actual === "object" && !Array.isArray(actual),
      true,
      `${path} must be an object`,
    );
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${path} keys`);
    for (const [key, value] of Object.entries(expected)) {
      assertParity(actual[key], value, tolerance, `${path}.${key}`);
    }
    return;
  }
  assert.equal(actual, expected, path);
}

function engineWithChat(values, failure = null, wait = null) {
  return {
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
    async tokenizeDrowseText(text) {
      return [...text].map((character) => character.codePointAt(0));
    },
    async decodeDrowseTokens(tokenIds) {
      return String.fromCodePoint(...tokenIds);
    },
    chat: { completions: { async create() { return from(values, failure, wait); } } },
    completions: { async create() { return from(values, failure, wait); } },
  };
}
