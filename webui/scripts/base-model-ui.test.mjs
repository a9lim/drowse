import assert from "node:assert/strict";
import { createServer } from "vite";
import { readFile } from "node:fs/promises";
import postcss from "postcss";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { completionAnchor } = await server.ssrLoadModule("/src/lib/completionSelection.ts");
  const turns = [{ nodeId: "prompt", text: "Once " }, { nodeId: "reply", text: "upon a moon." }];
  assert.deepEqual(completionAnchor([], "root", 0), { parentNodeId: "root" });
  assert.deepEqual(completionAnchor(turns, "root", 0), { parentNodeId: "root" });
  assert.deepEqual(completionAnchor(turns, "root", 3), { parentNodeId: "root", branch: { nodeId: "prompt", text: "Onc" } });
  assert.deepEqual(completionAnchor(turns, "root", 5), { parentNodeId: "prompt" });
  assert.deepEqual(completionAnchor(turns, "root", 9), { parentNodeId: "prompt", branch: { nodeId: "reply", text: "upon" } });
  assert.deepEqual(completionAnchor(turns, "root", 17), { parentNodeId: "reply" });
  assert.deepEqual(completionAnchor([{ nodeId: "unicode", text: "A 🌙 moon" }], "root", 4), { parentNodeId: "root", branch: { nodeId: "unicode", text: "A 🌙" } });
  assert.throws(() => completionAnchor([{ text: "draft" }], "root", 1), /not saved/);
  const { render } = await server.ssrLoadModule("svelte/server");
  const { completionTokenViews, projectEditedTokens } = await server.ssrLoadModule("/src/lib/rawTokenInspection.ts");
  const capturedTokens = ["I ", "love ", "marmots", "."].map((text, tokenId) => ({ text, tokenId, logprob: -tokenId - 0.1 }));
  const originalViews = completionTokenViews([
    { role: "user", generated: false, text: "Prompt: ", nodeId: "prompt" },
    { role: "assistant", generated: true, text: "I love marmots.", nodeId: "original", tokens: capturedTokens },
  ]);
  const insertedViews = projectEditedTokens(originalViews, "Prompt: I really love marmots.");
  assert.equal(insertedViews.map(view => view.text).join(""), "Prompt: I really love marmots.");
  assert.equal(insertedViews.find(view => view.text === "really ").tok, null);
  assert.equal(insertedViews.find(view => view.text === "really ").source, "draft");
  assert.equal(insertedViews.find(view => view.text === "love ").tok, capturedTokens[1]);
  assert.equal(insertedViews.find(view => view.text === "love ").contextChanged, true);
  assert.equal(insertedViews.find(view => view.text === "I ").contextChanged, false);
  assert.equal(insertedViews.find(view => view.text === "love ").nodeId, "original");
  const promptEdit = projectEditedTokens(originalViews, "New prompt: I love marmots.");
  assert.ok(promptEdit.filter(view => view.tok).every(view => view.contextChanged));
  const withinPrompt = projectEditedTokens(originalViews, "Prompt: extra I love marmots.");
  assert.equal(withinPrompt[0].source, "user", "unchanged user text retains its origin");
  const twiceEdited = projectEditedTokens(insertedViews, "Prompt: I really love marmots!");
  assert.equal(twiceEdited.find(view => view.text === "love ").tok, capturedTokens[1], "unaffected middle tokens survive successive edits");
  assert.ok(twiceEdited.filter(view => view.tok).every(view => view.tok.text === view.text), "only whole unchanged tokens retain recorded scores");
  const partialEdit = projectEditedTokens(originalViews, "Prompt: I loved marmots.");
  assert.ok(!partialEdit.some(view => view.tok === capturedTokens[1]), "a changed token never inherits the old token's probability");
  const appended = projectEditedTokens(originalViews, "Prompt: I love marmots. More");
  assert.ok(appended.filter(view => view.tok).every(view => !view.contextChanged), "appending does not invalidate earlier context");
  assert.deepEqual(projectEditedTokens(originalViews, ""), []);
  assert.equal(completionTokenViews([{ role: "user", generated: false, text: "I ", tokens: [capturedTokens[0]] }])[0].source, "user", "capturing a user token does not make it model-authored");
  assert.equal(completionTokenViews([{ role: "assistant", generated: true, text: "different", tokens: capturedTokens }])[0].tok, null, "mismatched text cannot borrow old measurements");
  for (const text of ["😀", "😁", "Prompt: I love 🦫.", "\nPrompt:\tI love marmots.\n", "Prompt: I love", ""]) {
    const projected = projectEditedTokens(originalViews, text);
    assert.equal(projected.map(view => view.text).join(""), text);
    for (const view of projected) assert.equal(view.text.isWellFormed(), true);
  }
  const emojiViews = projectEditedTokens([], "😀");
  assert.equal(projectEditedTokens(emojiViews, "😁")[0].text, "😁");
  const { tokenProbabilityRows } = await server.ssrLoadModule("/src/lib/tokenProbabilities.ts");
  const candidateRows = tokenProbabilityRows({ text: " mean", tokenId: 6, logprob: Math.log(0.001), topAlts: [
    { id: 1, text: " love", logprob: Math.log(0.451) },
    { id: 2, text: " wear", logprob: Math.log(0.047) },
  ] });
  assert.deepEqual(candidateRows.map(row => row.text), [" love", " wear", " mean"]);
  assert.equal(candidateRows.at(-1).chosen, true, "the clicked token stays visible even outside the captured top K");
  assert.equal(tokenProbabilityRows({ text: " mean", tokenId: 6, logprob: -1, topAlts: [{ id: 6, text: " mean", logprob: -1 }] }).length, 1);
  assert.deepEqual(tokenProbabilityRows({ text: "unknown" }), []);
  console.log("Token inspection: draft provenance, unchanged scores, changed-context labels, Unicode, and chosen-token probabilities passed");
  const { default: BaseModelTag } = await server.ssrLoadModule("/src/lib/ui/BaseModelTag.svelte");
  assert.match(render(BaseModelTag, { props: { plain: true } }).body, /class="[^"]*plain/);
  assert.doesNotMatch(render(BaseModelTag).body, /class="[^"]*plain/);
  const tagSource = await readFile(new URL("../src/lib/ui/BaseModelTag.svelte", import.meta.url), "utf8");
  assert.match(tagSource, /\.base-tag\.plain\s*\{[^}]*padding: 0;[^}]*background: none;/);
  const pickerSource = await readFile(new URL("../src/hosted/ui/HostedApp.svelte", import.meta.url), "utf8");
  assert.match(pickerSource, /<strong><span>\{model.name\}<\/span>\{#if model.modelType === "base"\}<BaseModelTag plain \/>/);
  assert.match(pickerSource, /\.model-name strong\s*\{[^}]*display: flex;[^}]*align-items: baseline;/);
  assert.match(pickerSource, /modelsBySize\(snapshot.models.filter\(\(model\) => model.modelType === "base"\)\)/);
  assert.match(pickerSource, /\.check-mark\s*\{[^}]*margin-top: var\(--space-xs\);/,
    "checkmarks share the heading's top inset instead of sitting above it");
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser" });
  const { default: TokenLogitsPopover } = await server.ssrLoadModule("/src/lib/ui/TokenLogitsPopover.svelte");
  const probabilityTable = render(TokenLogitsPopover, { props: {
    token: { text: " mean", tokenId: 6, logprob: Math.log(0.001), topAlts: [
      { id: 1, text: " love", logprob: Math.log(0.451) },
      { id: 2, text: " wear", logprob: Math.log(0.047) },
      { id: 3, text: " like", logprob: Math.log(0.038) },
      { id: 4, text: " enjoy", logprob: Math.log(0.025) },
      { id: 5, text: " adore", logprob: Math.log(0.014) },
    ] }, anchor: {}, contextChanged: true, onclose() {},
  } }).body;
  assert.equal((probabilityTable.match(/<tr\b/g) ?? []).length, 7, "five captured alternatives plus the chosen token and header remain visible");
  for (const probability of ["0.451", "0.047", "0.038", "0.025", "0.014", "0.001"]) {
    assert.ok(probabilityTable.includes(probability), "probabilities are not renormalized over the captured subset");
  }
  assert.match(probabilityTable, /Recorded before your edit/);
  assert.match(probabilityTable, /not raw logits/);
  assert.match(probabilityTable, /this token/);
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { genUiMode, effectiveRawMode, loadGenUiMode, setGenUiMode } =
    await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  sessionState.info = { model_id: "fixture/base", is_base_model: true };
  genUiMode.mode = "chat";
  assert.equal(effectiveRawMode(), true, "stale chat state cannot override a base model");
  loadGenUiMode();
  assert.equal(genUiMode.mode, "raw");
  setGenUiMode("chat");
  assert.equal(genUiMode.mode, "raw", "base models cannot enable a chat template");
  sessionState.info = { model_id: "fixture/instruct", is_base_model: false };
  setGenUiMode("chat");
  assert.equal(effectiveRawMode(), false);
  setGenUiMode("raw");
  assert.equal(effectiveRawMode(), true, "instruct models retain optional raw mode");
  console.log("Base mode enforcement and instruct-mode compatibility passed");

  const { chatLog, genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { default: RawBuffer } = await server.ssrLoadModule("/src/panels/RawBuffer.svelte");
  const { default: StatusFooter } = await server.ssrLoadModule("/src/panels/StatusFooter.svelte");
  sessionState.info = { model_id: "fixture/base", is_base_model: true };
  chatLog.turns = [{ role: "assistant", text: "I love marmots because", generated: true,
    tokens: [{ text: "I", logprob: -0.2 }] }];
  genStatus.active = true;
  genStatus.startedAt = 1;
  genStatus.tokensSoFar = 1;
  const streaming = render(RawBuffer).body;
  assert.match(streaming, /<textarea[^>]*readonly/);
  assert.match(streaming, /<label for="completion-buffer"[^>]*>Text completion<\/label>/);
  assert.match(streaming, /<button(?=[^>]*disabled)(?=[^>]*title="Stop or finish generation before inspecting")[^>]*>/);
  genStatus.active = false;
  genStatus.finishReason = "cancelled";
  const stopped = render(StatusFooter).body;
  assert.doesNotMatch(stopped, /rolling-number/, "generation metrics render directly without animated digits");
  const statusSource = await readFile(new URL("../src/panels/StatusFooter.svelte", import.meta.url), "utf8");
  assert.doesNotMatch(statusSource, /RollingNumber/, "live, completed, and queued counts must all update instantly");
  genStatus.tokensSoFar = 24;
  genStatus.tokPerSec = 7.34;
  genStatus.finishedAt = genStatus.startedAt + 7700;
  const metrics = render(StatusFooter).body;
  assert.match(metrics, /24 tokens/);
  assert.match(metrics, /7\.3 tokens\/s/);
  assert.match(metrics, /7\.7s/);
  assert.match(stopped, /Stopped ·/);
  assert.ok(stopped.indexOf("Stopped") < stopped.indexOf("tokens/s"), "stop reason precedes secondary metrics");
  genStatus.finishReason = "length";
  assert.match(render(StatusFooter).body, /Token limit ·/);
  genStatus.finishReason = null;
  assert.match(render(StatusFooter).body, /Ended ·/, "an error or unknown finish must not be called complete");
  genStatus.finishReason = "stop";
  genStatus.tokensSoFar = 0;
  chatLog.turns = [{ role: "user", text: "I love marmots because", generated: false }];
  const saved = render(RawBuffer).body;
  assert.match(saved, /Edit saved/);
  assert.doesNotMatch(saved, /tokens\/s/);
  chatLog.turns = [{ role: "assistant", text: "", generated: true }];
  assert.doesNotMatch(render(RawBuffer).body, /Edit saved/, "zero-token model output is not an authored save");
  console.log("Base completion rendering: streaming lock, inspection guard, visible label, finish reasons, and save status passed");

  const { highlightState } = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  const { tokenRowToScore } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { highlightStyleString } = await server.ssrLoadModule("/src/lib/highlight.ts");
  const { SURPRISE_TARGET, ENTROPY_TARGET } = await server.ssrLoadModule("/src/lib/tokens.ts");
  const measured = [
    { text: " likely", logprob: -0.1, sampler_entropy: 0.2 },
    { text: " unusual", logprob: -3, sampler_entropy: 1.4 },
  ].map(tokenRowToScore);
  chatLog.turns = [
    { role: "user", text: "Prompt", generated: false },
    { role: "assistant", text: " likely unusual", generated: true, tokens: measured },
  ];
  highlightState.target = SURPRISE_TARGET;
  highlightState.compareTwo = false;
  const coloredBuffer = render(RawBuffer).body;
  assert.doesNotMatch(coloredBuffer, /class="edit-actions /, "a clean buffer has no empty action group to wrap on narrow screens");
  assert.match(coloredBuffer, /class="color-mirror [^"]*"[^>]*aria-hidden="true"/);
  assert.match(coloredBuffer, /<span(?![^>]*style=)[^>]*>Prompt<\/span>/, "unmeasured prompt text stays untinted");
  for (const token of measured) {
    assert.ok(coloredBuffer.includes(highlightStyleString(token)), "rehydrated scores use the shared highlight ramp in edit mode");
  }
  assert.notEqual(highlightStyleString(measured[0]), highlightStyleString(measured[1]), "different surprisal values have different tints");
  genStatus.active = true;
  assert.match(render(RawBuffer).body, /class="color-mirror /, "the mirror stays available while measured tokens stream");
  genStatus.active = false;
  highlightState.target = ENTROPY_TARGET;
  assert.ok(render(RawBuffer).body.includes(highlightStyleString(measured[1])), "sampler entropy colors use the same mirror");
  measured[0].probes = { "fixture/trait": -0.5, "sae/7": 0.2 };
  measured[1].probes = { "fixture/trait": 0.5, "sae/7": 0.8 };
  highlightState.target = "fixture/trait";
  const probeColors = render(RawBuffer).body;
  assert.match(probeColors, /var\(--accent-red\)/);
  assert.match(probeColors, /var\(--accent-green\)/);
  highlightState.target = "sae/7";
  assert.match(render(RawBuffer).body, /var\(--pillar-sae\)/);
  highlightState.target = SURPRISE_TARGET;
  highlightState.compareTwo = true;
  highlightState.compareTarget = ENTROPY_TARGET;
  highlightState.smoothBlend = false;
  assert.match(render(RawBuffer).body, /background-image: linear-gradient/);
  highlightState.smoothBlend = true;
  assert.ok(render(RawBuffer).body.includes(highlightStyleString(measured[1])), "smooth comparison uses the shared ramp");
  highlightState.compareTwo = false;
  chatLog.turns[1].tokens = [{ text: " likely unusual", logprob: 0, samplerEntropy: 0.2 }];
  assert.match(render(RawBuffer).body, /Recorded token surprisal is zero/);
  highlightState.compareTwo = true;
  assert.doesNotMatch(render(RawBuffer).body, /stay uncolored/, "nonzero second-channel colors are not described as uncolored");
  highlightState.compareTwo = false;
  chatLog.turns[1].tokens = [{ text: " likely unusual" }];
  assert.match(render(RawBuffer).body, /No readings were recorded for this color/);
  chatLog.turns[1].tokens = [{ text: "mismatched", logprob: -1 }];
  assert.doesNotMatch(render(RawBuffer).body, /class="color-mirror /, "token/text mismatch never paints misaligned scores");
  highlightState.target = null;
  assert.doesNotMatch(render(RawBuffer).body, /class="color-mirror |class="color-notice /, "No color removes the mirror and notices");
  highlightState.compareTarget = null;
  highlightState.smoothBlend = false;
  const { default: Chat } = await server.ssrLoadModule("/src/panels/Chat.svelte");
  assert.match(render(Chat, { props: { headersVisible: false } }).body, /aria-label="Active model" title="fixture\/base"/);
  sessionState.info = { model_id: "fixture/instruct", is_base_model: false };
  setGenUiMode("chat");
  assert.match(render(Chat, { props: { headersVisible: false } }).body, /aria-label="Active model" title="fixture\/instruct"/);
  sessionState.info = { model_id: "fixture/base", is_base_model: true };
  console.log("Base token colors: edit and live mirrors, surprise, entropy, compare, zero/missing readings, text alignment, and model identity passed");

  const { default: SteeringRack } = await server.ssrLoadModule("/src/panels/SteeringRack.svelte");
  const { default: ProbeRack } = await server.ssrLoadModule("/src/panels/ProbeRack.svelte");
  const { default: JLensMissingState } = await server.ssrLoadModule("/src/hosted/ui/JLensMissingState.svelte");
  const { default: JLensSourceSection } = await server.ssrLoadModule("/src/hosted/ui/JLensSourceSection.svelte");
  assert.match(render(SteeringRack, { props: { family: "subspace" } }).body, /Add a direction to steer the next completion/);
  assert.match(render(ProbeRack, { props: { family: "subspace" } }).body, /Readings observe the completion/);
  for (const component of [JLensMissingState, JLensSourceSection]) {
    const base = render(component, { props: { modelId: "fixture/base" } }).body;
    assert.match(base, /Word insights are optional for base models/);
    assert.doesNotMatch(base, /download this one again/);
  }
  sessionState.info = { model_id: "fixture/instruct", is_base_model: false };
  assert.match(render(SteeringRack, { props: { family: "subspace" } }).body, /Add a direction to steer the next reply/);
  assert.match(render(ProbeRack, { props: { family: "subspace" } }).body, /Readings observe the reply/);
  assert.match(render(JLensMissingState, { props: { modelId: "fixture/instruct" } }).body, /download this one again/);
  console.log("Completion-specific controls and optional J-lens copy passed; instruct wording preserved");

  const { default: Landing } = await server.ssrLoadModule("/src/hosted/ui/Landing.svelte");
  const landing = render(Landing).body;
  for (const name of ["Gemma 3 1B", "Gemma 3 4B", "Qwen3 1.7B", "Qwen3 4B", "Gemma 3 1B PT", "GPT-2 · 124M", "Pythia 70M", "Qwen 3.5 2B Base"]) {
    assert.ok(landing.includes(name), `${name} remains represented on the landing page`);
  }
  assert.doesNotMatch(landing, /Under evaluation|\bbeta\b|Polythetic|Saklas/);
  assert.doesNotMatch(landing, /Install in Drowse|Download files|huggingface\.co|[?&]model=/,
    "the informational model roster never exposes per-model install or file links");
  assert.equal((landing.match(/class="model-group /g) ?? []).length, 2, "both model types share one group layout");
  assert.equal((landing.match(/class="model-row /g) ?? []).length, 8);
  assert.equal((landing.match(/data-provider=/g) ?? []).length, 8, "every listed model has a logo");
  assert.equal((landing.match(/href="\/app"[^>]*>Open Drowse<\/a>/g) ?? []).length, 2,
    "both primary actions use the same label and app entry");
  assert.match(landing, /Run a language model on your device/);
  assert.doesNotMatch(landing, /Readouts are measurements|not a transcript of the model’s thoughts|J-lens and SAE tools require compatible packs/);
  assert.doesNotMatch(landing, /Interpretability, hands-on|section-signal|reading-section|footer-surface/,
    "the homepage has no eyebrow copy or added background wrappers");
  const landingSource = await readFile(new URL("../src/hosted/ui/Landing.svelte", import.meta.url), "utf8");
  const landingStyles = postcss.parse(landingSource.match(/<style>([\s\S]*?)<\/style>/)[1]);
  assert.match(landing, /Follow instructions\.<br\b[^>]*>Explore a conversation\./);
  assert.match(landing, /Continue raw text\.<br\b[^>]*>Study next-token predictions\./);
  assert.match(landingSource, /\.model-group \{[^}]*grid-template-rows: subgrid;[^}]*grid-row: span 3;/,
    "both model columns share heading, description, and model-list tracks");
  assert.match(landingSource, /\.model-list \{[^}]*grid-auto-rows: 1fr;/,
    "matching model rows stay aligned across columns");
  assert.match(landingSource, /\.model-group h3\s*\{[^}]*line-height: 1\.2;[^}]*text-box-edge: cap alphabetic;[^}]*text-box-trim: trim-start;/,
    "the first heading's cap edge respects the card's top inset");
  assert.match(landingSource, /\.model-row:last-child \.model-copy \{ align-self: end; \}/,
    "the last model's text stays against the bottom inset in shared grid rows");
  assert.match(landingSource, /\.model-row:last-child \.model-copy > span\s*\{[^}]*text-box-trim: trim-end;/,
    "the final provider line does not add hidden font leading below the text");
  assert.match(landingSource, /\.model-group \{ grid-template-rows: auto auto auto; grid-row: auto; \}/,
    "stacked mobile groups return to natural content height");
  let lightPanelRules = 0;
  landingStyles.walkRules(rule => {
    if (/\.primary-action|\.skip-link/.test(rule.selector)) return;
    if (rule.selector.replace(/\s+/g, " ").trim() === ".capabilities li, .model-group") {
      lightPanelRules++;
      const declarations = Object.fromEntries(rule.nodes.filter(node => node.type === "decl").map(node => [node.prop, node.value]));
      if (rule.parent.type === "atrule") {
        if (rule.parent.name === "supports") {
          assert.match(declarations["backdrop-filter"], /url\("#landing-glass"\)/);
        } else {
          assert.equal(declarations.background, "Canvas", "high-contrast and reduced-transparency cards are opaque");
          assert.equal(declarations["backdrop-filter"], "none");
          assert.equal(declarations["box-shadow"], "none");
        }
      } else {
        assert.equal(declarations.background, "var(--landing-panel-bg)");
        assert.equal(declarations["--fg"], "var(--landing-panel-ink)");
        assert.equal(declarations.padding, "var(--surface-padding)");
        assert.equal(declarations["border-radius"], "var(--radius-lg)");
      }
      return;
    }
    rule.walkDecls(declaration => {
      assert.ok(!/^(background(?:-.+)?|box-shadow|backdrop-filter|border(?:-.+)?)$/.test(declaration.prop),
        `${rule.selector}: only feature and model cards gain containers`);
    });
  });
  assert.equal(lightPanelRules, 3, "cards share base glass, progressive refraction, and accessibility fallback");
  assert.match(landingSource, /<feDisplacementMap in="SourceGraphic" in2="rim" scale="32"/,
    "refraction displaces the backdrop, never the text");
  assert.match(landingSource, /prefers-reduced-transparency: reduce/);
  assert.match(landingSource, /pointer-events: none;/, "glass decorations do not intercept interaction");
  assert.doesNotMatch(landingSource, /landing-panel-edge|landing-panel-sheen|brightness\(|inset 0 1px/,
    "landing glass has no white highlights or backdrop darkening");
  assert.doesNotMatch(landingSource, /capabilities as capability, index|0\{index \+ 1\}/,
    "feature headings do not have decorative mono labels above them");

  const { default: ModelProviderLogo } = await server.ssrLoadModule("/src/hosted/ui/ModelProviderLogo.svelte");
  for (const [modelId, provider] of [
    ["gemma3-1b-pt-q4f32_1", "gemma"], ["google/gemma-3-1b-pt", "gemma"],
    ["qwen35-2b-base-q4f32_1", "qwen"], ["Qwen/Qwen3-1.7B", "qwen"],
    ["gpt2-base-q0f32", "openai"], ["openai-community/gpt2", "openai"],
    ["pythia-70m-base-q0f32", "eleutherai"], ["EleutherAI/pythia-70m-deduped", "eleutherai"],
  ]) {
    const logo = render(ModelProviderLogo, { props: { modelId } }).body;
    assert.ok(logo.includes(`data-provider="${provider}"`), modelId);
    assert.match(logo, /aria-hidden="true"/, "logos do not duplicate adjacent accessible model names");
    assert.doesNotMatch(logo, /https?:\/\//, "provider images are bundled, not remotely loaded");
  }
  assert.doesNotMatch(render(ModelProviderLogo, { props: { modelId: "unrecognized/model" } }).body, /data-provider/);
  const logoSource = await readFile(new URL("../src/hosted/ui/ModelProviderLogo.svelte", import.meta.url), "utf8");
  assert.match(logoSource, /\[data-provider="eleutherai"\] \.raster-mark \{ mask-mode: luminance;/,
    "the white-on-black EleutherAI asset must not become a solid alpha-mask square");
  const setupSource = await readFile(new URL("../src/hosted/ui/HostedApp.svelte", import.meta.url), "utf8");
  assert.match(setupSource, /Compatible J-lens and SAE packs are optional/);
  assert.doesNotMatch(setupSource, /Each download includes a precomputed J-lens and SAE/);
  console.log("Unified informational home, shared app entry, all eight logos, qualified model identities, and research copy passed");
} finally {
  await server.close();
}
