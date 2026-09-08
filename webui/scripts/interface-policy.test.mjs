import assert from "node:assert/strict";
import { parse } from "svelte/compiler";
import { readdir, readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const sourceRoots = ["src", "hosted", "public-hosted"];
const textExtensions = new Set([".css", ".html", ".svelte", ".ts"]);

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await sourceFiles(path));
    else if (textExtensions.has(extname(entry.name))) files.push(path);
  }
  return files;
}

const files = (await Promise.all(sourceRoots.map(sourceFiles))).flat();
const violations = [];
for (const file of files) {
  const source = await readFile(file, "utf8");
  if (extname(file) === ".svelte") {
    const visit = node => {
      if (!node || typeof node !== "object") return;
      if (node.type === "RegularElement" && node.attributes?.some(attribute => attribute.name === "title")) {
        violations.push(`${file}: native hover title; use accessible descriptions or explicit InfoTip help`);
      }
      if (node.type === "RegularElement" && node.name === "svg" && /<title[\s>]/.test(source.slice(node.start, node.end))) {
        violations.push(`${file}: SVG hover title; use desc or aria-label`);
      }
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) value.forEach(visit);
        else if (value && typeof value === "object") visit(value);
      }
    };
    visit(parse(source, { modern: true }).fragment);
  }

  if (/transition\s*:\s*all\b/i.test(source)) {
    violations.push(`${file}: transition: all`);
  }
  if (/tabindex\s*=\s*["']?[1-9]\d*/i.test(source)) {
    violations.push(`${file}: positive tabindex`);
  }
  if (/user-scalable\s*=\s*no|maximum-scale\s*=\s*1(?:\.0+)?\b/i.test(source)) {
    violations.push(`${file}: zoom-blocking viewport directive`);
  }
  if (/(?:margin|padding|border)-(?:left|right)\s*:/i.test(source)) {
    violations.push(`${file}: physical inline spacing or border`);
  }
  if (/text-align\s*:\s*(?:left|right)\b/i.test(source)) {
    violations.push(`${file}: physical text alignment`);
  }
  if (
    extname(file) === ".svelte" &&
    /(?:instanceof\s+Error\s*\?\s*\w+\.message|\w+\.error\?\.message)/u.test(source)
  ) {
    violations.push(`${file}: raw runtime error can reach the interface`);
  }
  if (
    file !== "src/lib/style/tokens.css" &&
    (extname(file) === ".css" || extname(file) === ".svelte") &&
    /(?:#[0-9a-f]{3,8}\b|rgba?\()/i.test(source)
  ) {
    violations.push(`${file}: raw color outside the theme token contract`);
  }
  if (
    file !== "src/lib/style/tokens.css" &&
    file !== "src/lib/style/fonts.css" &&
    (extname(file) === ".css" || extname(file) === ".svelte") &&
    /font-size\s*:\s*(?:(?:[1-9]\d*(?:\.\d+)?|0?\.\d+)(?:px|rem|em)\b|clamp\s*\()/i.test(source)
  ) {
    violations.push(`${file}: raw font size outside the type-token contract`);
  }
  if (
    file !== "src/lib/style/tokens.css" &&
    file !== "src/lib/style/fonts.css" &&
    (extname(file) === ".css" || extname(file) === ".svelte") &&
    /font-weight\s*:\s*\d/i.test(source)
  ) {
    violations.push(`${file}: raw font weight outside the type-token contract`);
  }
  if (
    file !== "src/lib/style/tokens.css" &&
    (extname(file) === ".css" || extname(file) === ".svelte")
  ) {
    const css = extname(file) === ".svelte" ? source.match(/<style[^>]*>([\s\S]*?)<\/style>/)?.[1] ?? "" : source;
    for (const match of css.matchAll(/(?:^|[;{])\s*((?:margin|padding|gap|row-gap|column-gap)(?:-[a-z-]+)?)\s*:\s*([^;}]+)/g)) {
      if (match[1] === "margin" && /^-1px(?:\s*!important)?$/.test(match[2].trim())) continue;
      for (const size of match[2].matchAll(/(-?[\d.]+)(px|rem|em)\b/g)) {
        if (size[2] !== "px" || Number(size[1]) % 8 !== 0) {
          violations.push(`${file}: spacing outside the 8px grid (${match[1]}: ${match[2].trim()})`);
        }
      }
    }
    const allowedRadiusParts = new Set([
      "var(--radius-sm)",
      "var(--radius-inset)",
      "var(--radius-group)",
      "var(--radius)",
      "var(--radius-lg)",
      "var(--popup-radius)",
      "var(--chat-radius)",
      "var(--radius-pill)",
      "var(--data-mark-radius)",
      "50%",
      "inherit",
      "0",
    ]);
    for (const match of source.matchAll(/border-radius\s*:\s*([^;\n}]+)/gi)) {
      const value = match[1].trim();
      if (!value.split(/\s+/).every((part) => allowedRadiusParts.has(part))) {
        violations.push(`${file}: radius outside the shared geometry contract (${value})`);
      }
    }
  }
}
assert.deepEqual(violations, []);

for (const file of [
  "src/hosted/ui/HostedApp.svelte",
  "src/drawers/LoadConversationDrawer.svelte",
  "src/drawers/ManifoldPacksDrawer.svelte",
  "src/panels/rack/SteerCard.svelte",
]) {
  const source = await readFile(file, "utf8");

  assert.match(source, /background:\s*var\(--warning-bg\)/, `${file}: use the shared yellow warning surface`);
  assert.match(source, /color:\s*var\(--warning-ink\)/, `${file}: use matching warning ink`);
}
const chatHome = await readFile("src/hosted/ui/HostedHome.svelte", "utf8");
assert.match(chatHome, /\.storage-notice\s*\{[^}]*background:\s*var\(--surface-card\)/, "storage guidance uses the shared card material");
assert.match(chatHome, /background:\s*var\(--warning-action\)/, "storage protection keeps its yellow action button");
for (const file of ["src/hosted/ui/HostedHome.svelte", "src/lib/builder/ValidationBlock.svelte", "src/panels/SteeringRack.svelte"]) {
  const source = await readFile(file, "utf8");

  assert.doesNotMatch(source, /border-(?:left|right|inline-start|inline-end)\s*:/, `${file}: no decorative side borders`);
  assert.doesNotMatch(source, /inset\s+[1-9]\d*px\s+0\s+0\s+var\(--(?:warning|accent)/, `${file}: no decorative side shadows`);
}
for (const file of ["src/hosted/ui/HostedHome.svelte", "src/hosted/ui/HostedApp.svelte", "src/drawers/RackDrawer.svelte", "src/drawers/ProbeInspectorDrawer.svelte", "src/panels/loom/LoomWeave.svelte"]) {
  const source = await readFile(file, "utf8");

  for (const opening of source.matchAll(/<details\b[^>]*>/g)) {
    assert.match(opening[0], /use:animatedDetails/, `${file}: native disclosures share reversible motion`);
  }
}
for (const file of [
  "src/lib/ui/Button.svelte",
  "src/panels/Chat.svelte",
  "src/panels/loom/LoomSidebar.svelte",
  "src/drawers/LoadConversationDrawer.svelte",
]) {
  const source = await readFile(file, "utf8");

  assert.match(source, /background:\s*(?:var\(--control-sheen\),\s*)?var\(--danger-bg\)/, `${file}: use the shared destructive surface`);
  assert.match(source, /background:\s*(?:var\(--control-sheen\),\s*)?var\(--danger-hover\)/, `${file}: use matching destructive hover`);
}

const globalCss = await readFile("src/lib/style/global.css", "utf8");
assert.match(globalCss, /:focus-visible\s*\{/);
assert.match(globalCss, /prefers-reduced-motion:\s*reduce/);
assert.match(
  globalCss,
  /max-width:\s*760px\)\s*and\s*\(pointer:\s*coarse\)[\s\S]*font-size:\s*var\(--text-input-touch\)\s*!important/,
);
assert.match(globalCss, /scroll-behavior:\s*auto\s*!important/);

const motionTokens = await readFile("src/lib/style/tokens.css", "utf8");
assert.match(motionTokens, /--ease-enter:/);
assert.match(motionTokens, /--ease-exit:/);
assert.match(motionTokens, /--press-scale:\s*0\.96/);
assert.match(motionTokens, /--control-target:\s*(?:4\d|[5-9]\d|\d{3,})px/);
assert.match(motionTokens, /--surface-radius:\s*12px/);
for (const [radius, value] of Object.entries({
  "radius-sm": "4px",
  "radius": "8px",
  "radius-inset": "var(--radius)",
  "radius-group": "12px",
  "radius-lg": "12px",
  "radius-pill": "999px",
})) {
  assert.ok(motionTokens.includes(`--${radius}: ${value};`), `${radius} follows the concentric geometry hierarchy`);
}
assert.match(motionTokens, /--space-unit:\s*8px/);
assert.match(motionTokens, /--surface-padding:\s*24px/);
for (const padding of ["surface-gutter", "drawer-gutter-inline", "drawer-gutter-block", "panel-padding", "page-gutter", "workspace-gutter"]) {
  assert.ok(motionTokens.includes(`--${padding}: var(--surface-padding);`), `${padding} uses the 24px content inset`);
  assert.equal([...motionTokens.matchAll(new RegExp(`--${padding}:`, "g"))].length, 1, `${padding} has no breakpoint override`);
}
for (const match of motionTokens.matchAll(/--space-[\w-]+:\s*([\d.]+)px/g)) {
  assert.equal(Number(match[1]) % 8, 0, `${match[0]} stays on the 8px grid`);
}
assert.deepEqual(
  [...new Set([...motionTokens.matchAll(/--weight-(?!display:)[\w-]+:\s*(\d+)/g)].map((match) => Number(match[1])))].sort(),
  [400, 500, 600],
  "Text and data retain three font weights",
);
assert.match(motionTokens, /--weight-display:\s*780;/, "Display text retains its original heavy weight");
assert.match(globalCss, /b,\s*strong\s*\{\s*font-weight:\s*var\(--weight-bold\)/);
for (const [file, selector, compact = false] of [
  ["src/hosted/ui/HostedApp.svelte", ".model-grid > button"],
  ["src/hosted/ui/HostedHome.svelte", ".chat-card"],
  ["src/hosted/ui/Credits.svelte", ".team-member"],
  ["src/hosted/ui/HostedRoot.svelte", ".runtime-dialog"],
  ["src/drawers/DownloadChatDrawer.svelte", ".download-chat"],
  ["src/lib/Select.svelte", ".sk-select-popover", true],
  ["src/lib/Combobox.svelte", ".popover", true],
  ["src/lib/Toaster.svelte", ".toast"],
  ["src/panels/Chat.svelte", ".msg"],
  ["src/panels/Chat.svelte", ".input-row"],
  ["src/panels/SteeringRack.svelte", ".rack"],
  ["src/panels/ProbeRack.svelte", ".rack"],
  ["src/panels/rack/RackCard.svelte", ".card"],
]) {
  const css = ((await readFile(file, "utf8")).match(/<style[^>]*>([\s\S]*?)<\/style>/)?.[1] ?? "").replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter((match) => match[1].trim() === selector);
  const paddedRules = rules.filter((match) => /(?:^|;)\s*padding\s*:/.test(match[2]));
  assert.ok(paddedRules.length > 0, `${file}: ${selector} defines its content inset`);
  for (const rule of paddedRules) {
    if (file === "src/panels/Chat.svelte" && selector === ".input-row" && /padding:\s*var\(--space-[12]\)\s*;/.test(rule[2])) continue;
    assert.match(rule[2], compact ? /padding:\s*var\(--space-2\)\s*;/ : /padding:\s*var\(--surface-padding\)(?:\s+var\(--surface-padding\))*\s*;/, `${file}: ${selector} uses ${compact ? "compact menu" : "24px panel"} padding`);
  }
}
assert.match(motionTokens, /--popup-radius:\s*var\(--radius-lg\)/);
assert.match(motionTokens, /--shadow-control:/);
assert.match(motionTokens, /--shadow-control-hover:/);
assert.match(motionTokens, /--page-gutter:/);
assert.match(motionTokens, /--section-space:/);
assert.match(motionTokens, /--shadow-loom-node:/);
assert.match(motionTokens, /--loom-edge-depth:/);
assert.match(motionTokens, /--type-control:\s*1rem/);
assert.match(motionTokens, /--type-body:\s*1rem/);
assert.match(motionTokens, /--type-title:\s*1\.714285714rem/);
assert.doesNotMatch(
  motionTokens,
  /@media\s*\(max-width:\s*760px\)\s*\{\s*:root\s*\{[^}]*--text-/,
);
assert.match(globalCss, /-webkit-font-smoothing:\s*antialiased/);
assert.match(globalCss, /h1,[\s\S]*text-wrap:\s*balance/);
assert.match(globalCss, /p\s*\{[\s\S]*text-wrap:\s*pretty/);

const motionHelpers = await readFile("src/lib/motion.ts", "utf8");
assert.match(motionHelpers, /prefers-reduced-motion:\s*reduce/);
assert.match(motionHelpers, /cubicIn/);
assert.match(motionHelpers, /cubicOut/);
const interactionDurations = [...motionHelpers.matchAll(/motionDuration\((\d+)\)/g)]
  .map((match) => Number(match[1]));
assert.ok(interactionDurations.length > 0);
assert.ok(interactionDurations.every((duration) => duration <= 300));

const loom = await readFile("src/panels/loom/LoomSidebar.svelte", "utf8");
const menuStart = loom.indexOf('class="loom-menu"');
const menuEnd = loom.indexOf("{/if}", menuStart);
const menuOpening = loom.slice(menuStart, menuEnd);
assert.doesNotMatch(menuOpening, /\bin:/);
assert.match(menuOpening, /out:fade/);
assert.match(loom, /class="loom-depth-field"\s+aria-hidden="true"/);
assert.match(loom, /--loom-grid-size/);
assert.doesNotMatch(loom, /loom-depth-light|loom-focus-halo/);
assert.match(loom, /prefers-reduced-motion:\s*reduce[\s\S]*\.loom-depth-field/);
assert.match(loom, /touchPointers/);
assert.match(loom, /scaleFromPinch/);
assert.match(loom, /onpointercancel=\{endViewportGesture\}/);

const probeInspector = await readFile("src/drawers/ProbeInspectorDrawer.svelte", "utf8");
assert.match(probeInspector, /touchPointers/);
assert.match(probeInspector, /scaleFromPinch/);
assert.match(probeInspector, /onpointercancel=\{onPointerUp\}/);
assert.match(probeInspector, /drag · scroll or pinch/);

const toaster = await readFile("src/lib/Toaster.svelte", "utf8");
assert.doesNotMatch(toaster, /border-(?:left|right)(?:-[a-z]+)?\s*:/i);
assert.doesNotMatch(toaster, /border-inline-start(?:-color)?\s*:/);
assert.match(toaster, /background:\s*var\(--surface-sheen\),\s*var\(--popup-bg\)/);
assert.match(toaster, /in:fly/);
assert.match(toaster, /out:fly/);

const tokenDrilldown = await readFile("src/drawers/TokenDrilldownDrawer.svelte", "utf8");
assert.match(tokenDrilldown, /instrumentFamily\("lens"\)\?\.capabilities\.token_readout/);
assert.match(tokenDrilldown, /instrumentFamily\("sae"\)\?\.capabilities\.token_readout/);
assert.match(tokenDrilldown, /instrumentFamily\("geometry"\)\?\.capabilities\.token_readout/);
assert.match(tokenDrilldown, /if \(!lensTokenReplayAvailable\) return/);
assert.match(tokenDrilldown, /if \(!saeTokenReplayAvailable\) return/);
assert.match(tokenDrilldown, /if \(!geometryTokenReplayAvailable\) return/);

const instrumentStore = await readFile("src/lib/stores/instruments.svelte.ts", "utf8");
assert.match(
  instrumentStore,
  /sessionState\.info\?\.jlens_fitted === true &&\s*instrumentFamily\("lens"\)\?\.capabilities\.token_readout === true/,
);
assert.match(
  instrumentStore,
  /saeLoaded\(\) &&\s*instrumentFamily\("sae"\)\?\.capabilities\.token_readout === true/,
);

const geometryTab = await readFile("src/drawers/token/GeometryTab.svelte", "utf8");
assert.match(
  geometryTab,
  /replayAvailable && \(\(readout\.data\?\.steering \?\? null\) !== null \|\| !steered\)/,
);

const nodeCompare = await readFile("src/drawers/NodeCompareDrawer.svelte", "utf8");
assert.match(nodeCompare, /apiTree\.replayCapabilities\(\)/);
assert.match(nodeCompare, /if \(!availability\.available\)/);

for (const [file, redundantLabel] of [
  ["src/hosted/ui/HostedHome.svelte", /class="eyebrow"/],
  ["src/drawers/HelpDrawer.svelte", /class="eyebrow"/],
  ["src/drawers/CastDrawer.svelte", /class="eyebrow"/],
  ["src/drawers/LocalRuntimeDrawer.svelte", /class="eyebrow"/],
  ["src/hosted/ui/HostedRoot.svelte", /runtime-dialog-label/],
  ["src/panels/loom/LoomSidebar.svelte", /projection-kicker/],
  ["src/panels/InspectorPanel.svelte", /<p>(Sampling|Guidance and readings)<\/p>/],
]) {
  assert.doesNotMatch(await readFile(file, "utf8"), redundantLabel, `${file}: redundant heading label`);
}

const savedChatLayout = await readFile("src/hosted/ui/HostedHome.svelte", "utf8");
for (const [page, current] of [["Landing", "home"], ["Credits", "credits"], ["HostedHome", "chats"], ["HostedApp", "models"]]) {
  const source = await readFile(`src/hosted/ui/${page}.svelte`, "utf8");
  assert.ok(source.includes(`<PageHeader current="${current}"`), `${page} uses the shared page header`);
  assert.ok(source.includes("<PageFooter"), `${page} uses the shared page footer`);
  assert.doesNotMatch(source, /<header class="page-header"|\.header-actions\s*\{/, `${page} must not recreate header styling`);
}
const pageHeader = await readFile("src/hosted/ui/PageHeader.svelte", "utf8");
const modelCardGeometry = await readFile("src/hosted/ui/HostedApp.svelte", "utf8");
assert.match(modelCardGeometry, /\.model-grid > button \{[^}]*border-radius: var\(--radius\);/, "Model cards use restrained 8px corners");
assert.match(pageHeader, /grid-template-columns: auto minmax\(0, 1fr\) auto;/, "Site header keeps brand, navigation, and appearance in one row");
assert.match(pageHeader, /flex-wrap: nowrap;/, "Primary navigation never wraps");
assert.doesNotMatch(pageHeader, /grid-row:\s*2|grid-column:\s*1\s*\/\s*-1/, "Mobile navigation must not move to a second row");
const compactThemeToggle = await readFile("src/lib/ui/ThemeToggle.svelte", "utf8");
assert.match(compactThemeToggle, /--radius-inset: var\(--radius-pill\)/, "The sliding theme selection stays circular");
assert.match(compactThemeToggle, /width: var\(--control-target\);\s*height: var\(--control-target\);\s*min-width: 40px;\s*min-height: 40px;/, "Theme options have non-overlapping 40px minimum circular targets");
assert.match(compactThemeToggle, /padding: calc\(var\(--space-xs\) \/ 4\)/, "Theme pill uses a compact 2px inset");
assert.match(compactThemeToggle, /border-radius: var\(--radius-pill\)/, "Theme container follows the circular options");
assert.match(compactThemeToggle, /background: var\(--surface-sheen\)/, "Theme pill retains the shared finish");
for (const path of ["src/App.svelte", "src/hosted/ui/NotFound.svelte", "src/hosted/ui/HostedRoot.svelte"]) {
  const source = await readFile(path, "utf8");
  assert.ok(source.includes("<PageHeader"), `${path} shares the page header`);
  if (path !== "src/App.svelte") assert.ok(source.includes("<PageFooter"), `${path} shares the page footer`);
  assert.doesNotMatch(source, /<a class="page-brand"/, `${path} must not duplicate the wordmark`);
}
const creditsLayout = await readFile("src/hosted/ui/Credits.svelte", "utf8");
assert.match(creditsLayout, /class="credits-art-frame" aria-hidden="true"/, "Credits artwork remains decorative");
assert.match(creditsLayout, /\.credits-art-frame \{[^}]*overflow: hidden; pointer-events: none;/, "Oversized artwork is clipped without blocking controls");
assert.match(creditsLayout, /\.credits-shell \{[^}]*overflow-x: clip;/, "Full-width artwork does not create a horizontal page scrollbar");
assert.match(creditsLayout, /\.credits-art \{[^}]*width: clamp\(44rem, 92vw, 96rem\)/, "Credits artwork uses an oversized responsive canvas");
assert.doesNotMatch(creditsLayout, /opacity: 0\.35|width: 1[25]rem/, "Mobile artwork must not regress to a tiny faded illustration");
assert.match(pageHeader, /column-gap: var\(--space-8\)/, "Navigation and appearance keep a full group gap");
assert.match(pageHeader, /aria-current=/, "Current page stays identifiable");
assert.match(pageHeader, /event\.metaKey \|\| event\.ctrlKey/, "Modified clicks retain browser navigation behavior");
assert.match(savedChatLayout, /class="chat-meta">\s*<time[\s\S]*?<\/time>\s*<SavedChatMenu/, "Timestamp sits immediately before the chat options menu");
assert.doesNotMatch(savedChatLayout, /class="chat-status"/, "Timestamp must not get a separate mobile row");
assert.match(savedChatLayout, /\.chat-model, \.chat-counts \{ font-weight: var\(--weight-structure-bold\); line-height: 24px;/);
assert.match(savedChatLayout, /\.chat-copy \{[^}]*gap: var\(--space-4\)/, "Metadata uses one spacing step");
assert.match(savedChatLayout, /\.chat-footer \{[^}]*display: flex;[^}]*flex-wrap: wrap;/, "Saved-chat actions form a compact wrapping toolbar");
assert.match(savedChatLayout, /\.chat-footer > \.primary \{ margin-inline-end: auto;/, "The primary chat action leads the toolbar");
assert.match(savedChatLayout, /\.chat-card \{[^}]*background: var\(--surface-card\);\s*box-shadow: var\(--shadow-card\);/);
assert.match(savedChatLayout, /button\.primary \{ color: var\(--action-ink\); background: var\(--control-sheen\), var\(--action-bg\);/);

for (const [file, selector] of [
  ["src/lib/ui/Chip.svelte", ".sk-chip"],
  ["src/lib/builder/ModeTabs.svelte", ".sk-mode-tabs"],
  ["src/panels/rack/SteerCard.svelte", ".operation"],
  ["src/panels/rack/AtomSteerCard.svelte", ".operation"],
  ["src/panels/loom/LoomSidebar.svelte", ".loom-view-controls"],
]) {
  const source = await readFile(file, "utf8");

  const css = (source.match(/<style[^>]*>([\s\S]*?)<\/style>/)?.[1] ?? "").replace(/\/\*[\s\S]*?\*\//g, "");
  const rule = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].find(match => match[1].trim() === selector)?.[2];
  assert.ok(rule, `${file}: compact group exists`);
  assert.match(rule, /padding:\s*var\(--space-(?:xs|1)\)/, `${file}: group inset is 8px`);
  assert.match(rule, /border-radius:\s*var\(--radius-group\)/, `${file}: group radius includes the child radius and inset`);
  assert.match(rule, /background:\s*var\(--surface-sheen\)/, `${file}: group keeps the shared finish`);
}
const samplingGeometry = await readFile("src/panels/SamplingStrip.svelte", "utf8");
assert.match(samplingGeometry, /\.control\s*\{[^}]*padding:\s*0;[^}]*background:\s*transparent;/, "Sampling fields are grouped by spacing without individual cards");
assert.match(samplingGeometry, /\.sampling-strip\s*\{[^}]*padding:\s*0 var\(--panel-padding\) var\(--panel-padding\)/, "Sampling controls align with the heading and footer");
assert.match(samplingGeometry, /\.row\.actions > :only-child\s*\{\s*grid-column:\s*1 \/ -1/, "A base-model action fills the row instead of leaving an empty column");
assert.doesNotMatch(globalCss.match(/:focus-visible\s*\{([^}]+)\}/)?.[1] ?? "", /border-radius/, "Keyboard focus must not change control geometry");
assert.match(globalCss, /:is\(\.drawer, \.drawer-shell\) > \.body\s*\{\s*scrollbar-gutter:\s*stable both-edges/, "Drawer scrollbars preserve balanced insets");
const inspectorGeometry = await readFile("src/panels/InspectorPanel.svelte", "utf8");
assert.match(inspectorGeometry, /\.control-grid\s*\{[^}]*grid-template-rows:\s*minmax\(36rem, 1fr\)/, "Short windows scroll the instrument layout rather than collapsing its cards");
assert.match(inspectorGeometry, /@media \(max-width: 920px\)[\s\S]*?\.control-grid\s*\{[^}]*scrollbar-gutter:\s*stable both-edges;\s*padding:\s*0;/, "Single-column controls avoid a redundant inset");
assert.match(inspectorGeometry, /\.rack-grid :global\(\.rack\)\s*\{\s*padding:\s*0;/, "Rack grids own the shared gutter without doubled child padding");
const numberGeometry = await readFile("src/lib/NumberInput.svelte", "utf8");
assert.match(numberGeometry, /\.sk-number\s*\{[^}]*border-radius:\s*var\(--radius\)/);
assert.match(numberGeometry, /\.sk-number-input\s*\{[^}]*border-radius:\s*inherit/, "The input follows its wrapper instead of introducing another corner radius");
const selectionMaterial = await readFile("src/lib/style/sliding-selection.css", "utf8");
assert.match(selectionMaterial, /\.selection-indicator\s*\{[^}]*background:\s*var\(--control-sheen\)/, "The moving selection carries the gradient");
assert.match(selectionMaterial, /background-image:\s*none !important/, "Child buttons do not double the selection gradient");
const recipeGeometry = await readFile("src/panels/RecipeBar.svelte", "utf8");
assert.ok(recipeGeometry.includes("max-height: calc((var(--control-target) + var(--space-xs) * 2) * 2 + var(--space-xs));"), "Recipe scrolling fits two complete chip rows at each target size");

const workspaceGeometry = await readFile("src/App.svelte", "utf8");
for (const selector of ["shell", "workspace-frame", "workspace-page", "workspace-surface"]) {
  assert.match(workspaceGeometry, new RegExp(`\\.${selector}\\s*\\{[^}]*overflow:\\s*clip;`), `${selector} cannot scroll when a nested control receives focus`);
}
assert.match(workspaceGeometry, /class:loom-workspace=\{workspaceView === "branches"\}/);
assert.match(workspaceGeometry, /\.workspace-page\.branches-page\s*\{\s*padding:\s*0;/, "Loom fills the workspace without a page inset");
assert.match(workspaceGeometry, /\.workspace-surface\.loom-zone\s*\{\s*border-radius:\s*0;[^}]*box-shadow:\s*none;/, "The canvas has no outer rounded card");
assert.match(workspaceGeometry, /conversationToolsVisible = \$state\(false\)/, "Chat tools are opt-in");
assert.match(workspaceGeometry, /loomToolsVisible = \$state\(false\)/, "Loom tools keep an independent opt-in preference");
assert.doesNotMatch(workspaceGeometry, /headers-collapsed|header-toggle/, "Main navigation never disappears or grows a second disclosure row");
assert.match(workspaceGeometry, /onToggleTools=\{toggleViewTools\}/, "View tools remain available in the menu");
assert.match(workspaceGeometry, /\.workspace-navigation\s*\{[^}]*justify-content:\s*center;[^}]*width:\s*100%;[^}]*max-width:\s*100%;/, "Compact workspace navigation centers its tabs across the available width");
assert.match(workspaceGeometry, /@media \(max-width: 420px\)[\s\S]*\.workspace-navigation \{ justify-content: center; \}/, "Narrow navigation remains centered");
assert.doesNotMatch(workspaceGeometry, /--(?:radius(?:-[a-z]+)?|popup-radius):/, "Workbench and detached overlays share the same global corner geometry");
assert.match(workspaceGeometry, /\.workspace-surface\s*\{[^}]*border-radius:\s*0;[^}]*background:\s*transparent;/, "The workspace has no decorative outer container");
assert.match(workspaceGeometry, /--left-sidebar-width:\s*13\.25rem/, "Desktop retains the compact sidebar by default");
assert.match(workspaceGeometry, /grid-template-columns:\s*var\(--left-sidebar-width\) minmax\(0, 1fr\)/, "Collapsing the sidebar gives its space to the workspace");
assert.doesNotMatch(workspaceGeometry, /@media \(max-width: (?:900|1120)px\)/, "Tablet widths retain the sidebar");
assert.match(workspaceGeometry, /<PageHeader current="workbench" compact/);
assert.match(workspaceGeometry, /<WorkbenchMenu/);
assert.doesNotMatch(workspaceGeometry, /<PageFooter|home-button|help-action/, "Secondary actions do not form permanent extra rows");
assert.match(workspaceGeometry, /grid-template-rows:\s*max-content minmax\(0, calc\(var\(--control-target\) \+ var\(--space-2\) \* 2 \+ var\(--space-1\)\)\) minmax\(0, 1fr\) minmax\(0, 0fr\)/, "Compact chrome reserves room for the inset tab group and leaves remaining height for content");
const workspaceMenu = await readFile("src/lib/ui/WorkbenchMenu.svelte", "utf8");
assert.match(workspaceMenu, /\{#if toolsLabel\}/, "Only relevant view tools are offered");
assert.match(workspaceMenu, /\{#if hasChat\}/, "An empty chat has no download action");
assert.match(workspaceMenu, /disabled=\{generating\}/, "Chat downloads wait for generation to finish");
assert.match(workspaceMenu, /dropdownMotion\(\)/, "Menu uses the shared interruptible motion");
assert.match(workspaceMenu, /popover="manual"/, "Menu escapes clipped shell containers");
assert.match(pageHeader, /\{#if current === "workbench"\}/, "Workbench does not repeat the site's navigation row");
assert.doesNotMatch(workspaceGeometry, /togglePalette|key\.toLowerCase\(\) === "k"/, "Command-K is not intercepted");
const contextualChat = await readFile("src/panels/Chat.svelte", "utf8");
assert.ok(contextualChat.indexOf('id="chat-tools-header"') < contextualChat.indexOf('<div class="chat"'), "Revealed tools sit outside the conversation scroller");
assert.match(workspaceGeometry, /\.app-header\s*\{[^}]*background:\s*var\(--workspace-panel-bg\);/, "The site header shares the quiet workspace material");
for (const file of ["src/panels/Chat.svelte", "src/panels/loom/LoomSidebar.svelte"]) {
  const source = await readFile(file, "utf8");

  assert.match(source, /headersVisible = true/, `${file}: standalone panels retain their headers`);
  assert.match(source, /\{#if headersVisible\}/, `${file}: tool headers follow the shell disclosure`);
  assert.match(source, /in:slide=\{collapseIn\(\)\} out:slide=\{collapseOut\(\)\}/, `${file}: reveal and dismissal use shared motion`);
  assert.doesNotMatch(source, /Cmd\+K|Ctrl\+K|Meta\+K|Control\+K|commandShortcut|kbd-hint|tool-key/, `${file}: no shortcut behavior or hint remains`);
}
const buttonType = await readFile("src/lib/ui/Button.svelte", "utf8");
for (const size of ["sm", "md"]) assert.match(buttonType, new RegExp(`\\.${size}\\s*\\{[^}]*font-size:\\s*var\\(--text-sm\\)`), "Compact and regular buttons use the body-text size");
assert.match(loom, /\.tree-scroll\s*\{[^}]*overflow:\s*clip;/, "Focus cannot scroll the canvas behind the camera transform");
assert.match(loom, /observer\.observe\(viewport\);\s*return \(\) => observer\.disconnect\(\)/, "Resizing is observed for a viewport mounted after the initial empty loom");
assert.match(loom, /resizeLoomCamera\(camera, lastViewportSize, size\)/, "Loom preserves the viewed point across resizes");
assert.match(loom, /function focusInitialView\(\): void\s*\{\s*if \(!active \|\| loomUiState.view !== "map" \|\| !viewportEl\) return;/, "The camera is not initialized against the hidden workspace's smaller dimensions");
const controlsGeometry = await readFile("src/panels/ControlsPanel.svelte", "utf8");
assert.match(controlsGeometry, /\.controls-nav :global\(\.sk-tabs\)\s*\{[^}]*padding:\s*0;[^}]*background:\s*transparent;/, "Controls tabs do not sit inside another container");
assert.match(controlsGeometry, /\.controls\s*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\)/, "Controls do not grow their grid to the tab row's intrinsic width");
assert.match(controlsGeometry, /\.controls-nav :global\(\.sk-tabs\)\s*\{[^}]*min-width:\s*0;/, "The nested tab group can shrink on narrow screens");
assert.match(controlsGeometry, /@media \(max-width: 680px\)[\s\S]*\.controls-nav :global\(\.tab\)\s*\{[^}]*flex:\s*1 1 auto;[^}]*padding-inline:\s*var\(--space-xs\)/, "Compact tabs retain the complete label and equal insets");
const completionGeometry = await readFile("src/panels/RawBuffer.svelte", "utf8");
assert.match(completionGeometry, /\.raw-buffer\s*\{[^}]*flex:\s*1 0 auto;[^}]*min-height:\s*0;/, "Completion grows naturally rather than overflowing its parent's full height");
assert.doesNotMatch(completionGeometry, /min-height:\s*24rem/, "The completion buffer has no oversized fixed minimum");
assert.match(completionGeometry, /\.surface\s*\{[^}]*flex:\s*1 0 calc\(2lh \+ var\(--surface-padding\) \* 2\);/, "The editor keeps two text lines and its padding while yielding space to actions and model identity");
const chatGeometry = await readFile("src/panels/Chat.svelte", "utf8");
assert.match(chatGeometry, /\.input-actions\.has-clear :global\(button:first-child\)\s*\{\s*grid-column:\s*1 \/ -1;/, "The primary mobile chat action keeps its own row instead of three cramped columns");
assert.match(chatGeometry, /\.chat\s*\{[^}]*overflow-y:\s*auto;[^}]*scrollbar-gutter:\s*auto;/, "Chat surfaces do not add scrollbar gutters to the horizontal inset");
assert.match(chatGeometry, /--chat-radius:\s*calc\(var\(--radius-lg\) \+ var\(--chat-inset\)\)/, "The chat frame radius follows the composer radius and inset");
const statusGeometry = await readFile("src/panels/StatusFooter.svelte", "utf8");
assert.match(statusGeometry, /\.status-footer\s*\{[^}]*flex:\s*0 0 auto;/, "Status controls do not shrink into adjacent actions");
assert.ok(statusGeometry.includes("min-height: var(--workspace-status-height, calc(var(--control-target) + var(--space-2) * 2));"), "The status row supports compact workspace spacing with equal standalone insets");

await import("./workspace-density.test.mjs");
await import("./appearance-lifecycle.test.mjs");
console.log(`interface policy passed across ${files.length} source files`);
