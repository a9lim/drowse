import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { backgroundDimensions, backgroundSettings, DEFAULT_BACKGROUND, validateBackgroundFile, BACKGROUND_MAX_BYTES } from "../src/lib/appearance.ts";

assert.deepEqual(backgroundSettings({}), DEFAULT_BACKGROUND);
assert.deepEqual(backgroundSettings({ effect: "invalid", pixelSize: -2, visibility: 9 }), { effect: "dither", pixelSize: 1, visibility: 0.14 });
assert.deepEqual(backgroundSettings({ effect: "original", pixelSize: Infinity, visibility: NaN }), { effect: "original", pixelSize: 3, visibility: 0.1 });
assert.equal(backgroundSettings({ pixelSize: 2.7 }).pixelSize, 3);
assert.deepEqual(backgroundDimensions(4096, 2048), { width: 2048, height: 1024 });
assert.deepEqual(backgroundDimensions(300, 400), { width: 300, height: 400 });
assert.deepEqual(backgroundDimensions(100, 4000), { width: 51, height: 2048 });
for (const type of ["image/jpeg", "image/png", "image/webp"]) validateBackgroundFile({ type, size: BACKGROUND_MAX_BYTES });
for (const file of [{ type: "image/svg+xml", size: 2 }, { type: "image/png", size: 0 }, { type: "image/jpeg", size: BACKGROUND_MAX_BYTES + 1 }]) assert.throws(() => validateBackgroundFile(file), /Choose /);

const source = async path => readFile(new URL(`../src/${path}`, import.meta.url), "utf8");
const [density, app, background, store, shader] = await Promise.all([source("lib/style/workspace.css"), source("App.svelte"), source("lib/ui/WorkspaceBackground.svelte"), source("lib/stores/appearance.svelte.ts"), source("lib/backgroundShader.ts")]);
assert.match(density, /min-width: 761px\) and \(pointer: fine\)/);
assert.match(density, /--control-target: 40px/);
assert.match(density, /@media \(max-width: 760px\), \(pointer: coarse\)[\s\S]*--control-target: 44px/);
assert.match(density, /--workspace-message-bg: var\(--workspace-field-bg\)/, "each message has its own visible surface");
assert.match(density, /--workspace-buffer-bg: transparent/);
assert.match(density, /--popup-radius: calc\(var\(--radius\) \+ var\(--surface-padding\)\)/);
assert.match(density, /:root\[data-theme="light"\] .shell\s*\{\s*--workspace-background-strength: 0.55;/);
assert.match(background, /var\(--workspace-background-strength, 1\)/);
for (const label of ["Your chats", "Models", "All tools", "Appearance", "Help and shortcuts"]) assert.ok(app.includes(label));
assert.match(background, /aria-hidden="true"/);
assert.match(background, /forced-colors: active/);
assert.match(background, /resize.disconnect\(\)/);
assert.match(background, /renderer.dispose\(\)/);
assert.doesNotMatch(background + shader, /requestAnimationFrame|setInterval/);
assert.match(store, /URL.revokeObjectURL/);
assert.match(store, /bitmap.close\(\)/);
assert.doesNotMatch(store, /fetch\(|XMLHttpRequest|localStorage/);
const [palette, transcript, loom] = await Promise.all([source("panels/CommandPalette.svelte"), source("drawers/TranscriptDrawer.svelte"), source("panels/loom/LoomSidebar.svelte")]);
const [chat, chatHome, autosave] = await Promise.all([source("panels/Chat.svelte"), source("hosted/ui/HostedHome.svelte"), source("lib/ui/ConversationAutosave.svelte")]);
assert.match(density, /:root:has\(\.workspace-material\),\s*\.workspace-material/,
  "portaled menus and tooltips inherit the same highlight-free material");
for (const material of ["surface-sheen", "control-sheen", "action-sheen", "shadow-control", "shadow-card"]) {
  assert.ok(density.includes(`--${material}: none;`));
}
assert.match(app, /class="shell workspace-material"/);
assert.match(chatHome, /class="home-shell workspace-material workspace-library"/);
const models = await source("hosted/ui/HostedApp.svelte");
assert.match(models, /class="app-shell"/);
assert.doesNotMatch(models, /workspace-material|workspace-library|current="models" compact/);
assert.match(models, /\.model-grid > button\s*\{[^}]*min-height: 100px;/);
assert.match(chatHome, /<PageHeader current="chats" compact/);
assert.match(chatHome, /--workspace-surface-padding: 16px;/);
assert.match(chatHome, /--text-onboarding-heading: var\(--text-page-title\);/);
assert.match(chatHome, /"Download a model" : "New Instance"/);
assert.match(density, /\.workspace-library\s*\{[^}]*--surface-padding: var\(--workspace-surface-padding\);/);
assert.match(density, /\.workspace-material \[data-chat-accent\]\s*\{[^}]*--surface-card: var\(--workspace-panel-bg\);/,
  "per-chat accent scopes retain the shared clear material");
assert.match(chatHome, /@media \(max-width: 430px\)[\s\S]*\.chat-title-row\s*\{[^}]*grid-template-columns: minmax\(0, 1fr\);/,
  "phone-width timestamps do not squeeze chat titles out of view");
assert.match(chatHome, /\.chat-footer\s*\{[^}]*flex-wrap: wrap;/,
  "chat actions wrap instead of squeezing into fixed columns");
assert.doesNotMatch(chatHome, /backdrop-filter: blur/, "Saved chats use flat, unblurred surfaces");
assert.match(chat, /\.chat\s*\{[^}]*border-radius: var\(--chat-radius\);[^}]*background: var\(--workspace-panel-bg\);/,
  "Chat and completion share a faint rounded container");
assert.ok(app.indexOf('<ConversationAutosave />') > app.indexOf('class="workspace-notices"'),
  "autosave warnings participate in the viewport layout instead of pushing the workspace offscreen");
assert.match(app, /grid-template-rows: auto minmax\(0, 1fr\)/);
assert.match(app, /\.workspace-notices\s*\{[^}]*max-height: 25dvh;[^}]*overflow-y: auto;/);
assert.match(autosave, /<Button variant="solid"/);
assert.doesNotMatch(autosave, /<button/);
assert.match(chat, /\.input-row\s*\{[^}]*padding: var\(--surface-padding\);[^}]*background: var\(--workspace-field-bg\);/);
assert.match(chat, /\.input\s*\{[^}]*--shadow-field-keyboard: none;[^}]*background: transparent;/);
assert.match(chat, /\.input-actions\s*\{[^}]*flex-wrap: nowrap;/);
assert.match(chat, /\.chat > \.log\s*\{[^}]*flex: 1 1 0;/,
  "mobile messages yield space to keep composer actions reachable");
assert.match(chat, /let rolePlanOpen = \$state\(false\)/,
  "advanced writer options start behind their visible role summary");
assert.match(chat, /class="summary-action">Edit roles/);
assert.match(chat, /if \(!rolesValid\) rolePlanOpen = true/,
  "invalid role fields remain visible for correction");
assert.match(density, /prefers-reduced-transparency: reduce/);
assert.match(palette, /\.row\s*\{[^}]*flex-shrink: 0/);
assert.match(palette, /\.label\s*\{[^}]*grid-column: 2;[^}]*white-space: normal/);
assert.match(palette, /\.group\s*\{[^}]*grid-column: 2; grid-row: 2/);
assert.match(transcript, /\.tabs\s*\{[^}]*grid-column: 1 \/ -1; grid-row: 2/);
assert.match(loom, /ariaLabel="Loom view"/);
for (const view of ["weave", "map", "path", "options", "saved"]) assert.ok(loom.includes(`{ value: "${view}", label:`));
console.log("workspace density: responsive controls, flat surfaces, local image limits, settings normalization, and resource cleanup passed");
