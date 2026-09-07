import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";
import { states, animatedStates, frameCount, icon } from "./tab-icon-artwork.mjs";

const server = await createServer({ configFile: false, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { tabLabels, faviconPath, TAB_FRAME_COUNT, workbenchTabState, setupTabState } = await server.ssrLoadModule("/src/lib/tabIdentity.ts");
  assert.deepEqual([...states].sort(), Object.keys(tabLabels).sort());
  assert.equal(TAB_FRAME_COUNT, frameCount);
  const { CHAT_ACCENTS } = await server.ssrLoadModule("/src/lib/chatAccent.ts");
  const luminance = hex => {
    const channels = hex.match(/[a-f\d]{2}/gi).map(value => parseInt(value, 16) / 255)
      .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  for (const accent of CHAT_ACCENTS) for (const theme of ["light", "dark"]) {
    const tile = luminance(accent[theme]);
    const ink = luminance(theme === "dark" ? "#141822" : "#ffffff");
    assert.ok((Math.max(tile, ink) + 0.05) / (Math.min(tile, ink) + 0.05) >= 4.5, `${accent.id}/${theme} glyph contrast`);
  }
  const idle = { boot: "ready", runtime: null, active: false, replay: false, thinking: false, view: "conversation", section: "response", drawer: null, saveStatus: "saved" };
  for (const [changes, expected] of [
    [{}, "conversation"], [{ view: "branches" }, "loom"], [{ view: "controls" }, "controls"],
    [{ view: "controls", section: "model" }, "models"], [{ view: "controls", section: "chat" }, "chat-settings"],
    [{ active: true }, "working"], [{ active: true, view: "branches" }, "loom-working"],
    [{ active: true, thinking: true }, "thinking"], [{ active: true, replay: true }, "analyzing"],
    [{ drawer: "token_drilldown" }, "tokens"], [{ drawer: "node_compare" }, "comparison"],
    [{ drawer: "load_conversation" }, "chats"], [{ drawer: "save_conversation", saveStatus: "saving" }, "saving"],
    [{ runtime: "training" }, "training"], [{ runtime: "error", active: true }, "error"],
    [{ boot: "failed", active: true }, "error"], [{ saveStatus: "error" }, "error"],
  ]) assert.equal(workbenchTabState({ ...idle, ...changes }), expected);
  const setup = { phase: "supported", runtime: { phase: "ready" }, download: { phase: "idle" } };
  assert.equal(setupTabState(setup), "models");
  for (const [download, expected] of [
    [{ phase: "downloading" }, "downloading"], [{ phase: "downloading", progress: { offline: true } }, "offline"],
    [{ phase: "downloading", progress: { stalled: true } }, "paused"], [{ phase: "failed" }, "error"],
  ]) assert.equal(setupTabState({ ...setup, download }), expected);
  assert.match(icon("home"), /data-fluent="home"/);
  assert.match(icon("home"), /data-squircle="true"/);
  const fingerprints = new Set();
  for (const state of states) {
    assert.equal(icon(state, frameCount), icon(state, 0), `${state} loops without a discontinuous reset`);
    if (!animatedStates.has(state)) assert.match(icon(state), /data-fluent=/);
    const frames = animatedStates.has(state) ? frameCount : 1;
    for (const theme of ["light", "dark"]) {
      assert.ok(icon(state, 0, theme).includes(`style="--tile:${theme === "dark" ? "#c5b3ff" : "#5b3fbf"};`), `${state}/${theme} keeps the purple brand color`);
      assert.match(icon(state, 0, theme), /data-squircle="true"/);
      assert.doesNotMatch(icon(state, 0, theme), /rotate\(/);
      assert.match(icon(state, 0, theme, false, "#f2b8d4"), /--tile:#f2b8d4/);
      const rendered = new Set();
      for (let frame = 0; frame < frames; frame++) {
        const path = faviconPath(state, frame, theme);
        for (const root of ["public", "public-hosted"]) {
          const bytes = await readFile(`${root}${path}`);
          assert.equal(bytes.subarray(1, 4).toString(), "PNG");
          assert.equal(bytes.readUInt32BE(16), 96);
          assert.equal(bytes.readUInt32BE(20), 96);
          if (root === "public") rendered.add(bytes.toString("base64"));
        }
      }
      assert.ok(frames === 1 ? rendered.size === 1 : rendered.size > 1, `${state}/${theme} must move, not reuse identical frames`);
      if (theme === "light") fingerprints.add([...rendered][0]);
    }
    assert.notEqual(icon(state, 0, "light"), icon(state, 0, "dark"));
  }
  assert.equal(fingerprints.size, states.length, "every state changes the whole silhouette or internal glyph");
  console.log("tab identity: all state priorities, both themes, distinct glyphs, animation frames, and asset dimensions passed");
} finally { await server.close(); }
