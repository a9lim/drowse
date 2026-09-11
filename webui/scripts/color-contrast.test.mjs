import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { runInNewContext } from "node:vm";
import { highlightHue, SURPRISE_TARGET, PROBABILITY_TARGET, ENTROPY_TARGET } from "../src/lib/tokens.ts";
import { icon } from "./tab-icon-artwork.mjs";
import { CHAT_ACCENTS } from "../src/lib/chatAccent.ts";

const [css, themeRuntime, localBootstrap, hostedBootstrap] = await Promise.all([
  readFile("src/lib/style/tokens.css", "utf8"),
  readFile("src/lib/theme.ts", "utf8"),
  readFile("public/theme-init.js", "utf8"),
  readFile("public-hosted/theme-init.js", "utf8"),
]);

function ruleBody(selector) {
  const start = css.indexOf(selector);
  assert.notEqual(start, -1, `missing ${selector}`);
  const open = css.indexOf("{", start);
  let depth = 0;
  for (let index = open; index < css.length; index += 1) {
    if (css[index] === "{") depth += 1;
    if (css[index] === "}") depth -= 1;
    if (depth === 0) return css.slice(open + 1, index);
  }
  throw new Error(`unclosed ${selector}`);
}

function colors(body) {
  return Object.fromEntries(
    [...body.matchAll(/--([\w-]+):\s*(#[0-9a-f]{6}|rgba?\([^;]+\)|var\(--[\w-]+\)|[\d.]+%?)\s*;/gi)]
      .map((match) => [`--${match[1]}`, match[2].toLowerCase()]),
  );
}

const dark = colors(ruleBody(":root {"));
const light = { ...dark, ...colors(ruleBody(':root[data-theme="light"]')) };

for (const bootstrap of [localBootstrap, hostedBootstrap]) {
  for (const [saved, expected] of [[null, "dark"], ["invalid", "dark"], ["light", "light"], ["dark", "dark"]]) {
    const root = { dataset: {}, style: {} };
    const meta = {};
    runInNewContext(bootstrap, {
      window: {
        addEventListener() {},
        localStorage: { getItem: key => key === "drowse.theme" ? saved : null },
      },
      document: { documentElement: root, querySelector: () => meta },
    });
    assert.equal(root.dataset.theme, expected);
    assert.equal(root.style.colorScheme, expected);
    assert.equal(meta.content, expected === "dark" ? "#0b0e17" : "#f2f4f8");
  }
  const root = { dataset: {}, style: {} };
  runInNewContext(bootstrap, {
    window: {
      addEventListener() {},
      localStorage: { getItem() { throw new Error("Storage unavailable"); } },
    },
    document: { documentElement: root, querySelector: () => null },
  });
  assert.equal(root.dataset.theme, "dark", "blocked storage still opens in dark mode");

  const handlers = new Map();
  let initialTransitionHandled = false;
  runInNewContext(bootstrap, {
    window: {
      addEventListener: (name, handler) => handlers.set(name, handler),
      localStorage: { getItem: () => "dark" },
    },
    document: {
      activeViewTransition: { ready: { catch(callback) {
        callback(new Error("Transition was skipped"));
        initialTransitionHandled = true;
      } } },
      documentElement: { dataset: {}, style: {} }, querySelector: () => null,
    },
  });
  assert.ok(initialTransitionHandled, "An inbound transition can be skipped before pagereveal exposes it");
  for (const name of ["pageswap", "pagereveal"]) {
    const handler = handlers.get(name);
    assert.equal(typeof handler, "function");
    assert.doesNotThrow(() => handler({ viewTransition: null }));
    let handled = false;
    handler({ viewTransition: { ready: { catch: (callback) => {
      callback(new Error("Transition was skipped"));
      handled = true;
    } } } });
    assert.ok(handled, `${name} must handle skipped transitions before the app mounts`);
  }
}

for (const [name, source] of Object.entries({ themeRuntime, localBootstrap, hostedBootstrap })) {
  assert.ok(source.includes(light["--bg"]), `${name} must use the light canvas as its theme color`);
  assert.ok(source.includes(dark["--bg"]), `${name} must use the dark canvas as its theme color`);
}

function parseColor(value) {
  const hex = value.match(/^#([0-9a-f]{6})$/i);
  if (hex) {
    return {
      rgb: [0, 2, 4].map((index) => Number.parseInt(hex[1].slice(index, index + 2), 16) / 255),
      alpha: 1,
    };
  }
  const functional = value.match(/^rgba?\(([^)]+)\)$/i);
  assert.ok(functional, `unsupported color value ${value}`);
  const fields = functional[1].trim().split(/[\s,/]+/);
  const parts = fields.map((part) => Number.parseFloat(part));
  assert.ok(parts.length === 3 || parts.length === 4, `invalid color value ${value}`);
  const alpha = parts[3] === undefined ? 1 : parts[3] / (fields[3].endsWith("%") ? 100 : 1);
  return { rgb: parts.slice(0, 3).map((channel) => channel / 255), alpha };
}

function composite(foreground, background, alpha) {
  return foreground.map((channel, index) => channel * alpha + background[index] * (1 - alpha));
}

function rendered(theme, token, background = null) {
  const value = theme[token];
  assert.ok(value, `missing ${token}`);
  const alias = value.match(/^var\((--[\w-]+)\)$/);
  if (alias) return rendered(theme, alias[1], background);
  const color = parseColor(value);
  if (color.alpha === 1) return color.rgb;
  const backdrop = background ?? rendered(theme, "--bg");
  return composite(color.rgb, backdrop, color.alpha);
}

function wcagLuminance(rgb) {
  const [red, green, blue] = rgb.map((channel) => channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function wcagContrast(foreground, background) {
  const [lighter, darker] = [wcagLuminance(foreground), wcagLuminance(background)]
    .sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
}

function apcaLuminance([red, green, blue]) {
  return 0.2126729 * red ** 2.4 + 0.7151522 * green ** 2.4 + 0.072175 * blue ** 2.4;
}

function apcaContrast(foreground, background) {
  const clamp = (luminance) => luminance < 0.022
    ? luminance + (0.022 - luminance) ** 1.414
    : luminance;
  const text = clamp(apcaLuminance(foreground));
  const surface = clamp(apcaLuminance(background));
  const raw = surface > text
    ? (surface ** 0.56 - text ** 0.57) * 1.14
    : (surface ** 0.65 - text ** 0.62) * 1.14;
  if (Math.abs(raw) < 0.1) return 0;
  return (raw > 0 ? raw - 0.027 : raw + 0.027) * 100;
}

function expectPair(theme, name, foreground, background, minWcag, minApca) {
  const surface = rendered(theme, background);
  const ink = rendered(theme, foreground, surface);
  const wcag = wcagContrast(ink, surface);
  const apca = Math.abs(apcaContrast(ink, surface));
  assert.ok(wcag >= minWcag, `${name}: WCAG ${wcag.toFixed(2)} < ${minWcag}`);
  assert.ok(apca >= minApca, `${name}: APCA ${apca.toFixed(1)} < ${minApca}`);
  return `${name} ${wcag.toFixed(2)}:1 / Lc ${apca.toFixed(1)}`;
}

function expectSurface(theme, name, foreground, background, minWcag, minApca) {
  const backdrop = rendered(theme, background);
  const surface = rendered(theme, foreground, backdrop);
  const wcag = wcagContrast(surface, backdrop);
  const apca = Math.abs(apcaContrast(surface, backdrop));
  assert.ok(wcag >= minWcag, `${name}: WCAG ${wcag.toFixed(2)} < ${minWcag}`);
  assert.ok(apca >= minApca, `${name}: APCA ${apca.toFixed(1)} < ${minApca}`);
  return `${name} ${wcag.toFixed(2)}:1 / Lc ${apca.toFixed(1)}`;
}

const reports = [];
for (const palette of CHAT_ACCENTS) {
  for (const [name, theme] of Object.entries({ dark, light })) {
    const accentTheme = { ...theme, "--accent": palette[name] };
    reports.push(expectPair(accentTheme, `${palette.name} ${name} button`, "--text-on-accent", "--accent", 4.5, 60));
    const actionTheme = { ...theme, "--action-bg": palette.dark };
    reports.push(expectPair(actionTheme, `${palette.name} ${name} pastel action`, "--action-ink", "--action-bg", 4.5, 60));
    const controlLight = parseColor(theme["--control-light"]);
    const surfaceShade = parseColor(theme["--control-shade"]);
    const actionLight = parseColor(theme["--action-light"]);
    const actionShade = parseColor(theme["--action-shade"]);
    for (const surface of [
      composite(controlLight.rgb, parseColor(palette.dark).rgb, controlLight.alpha),
      composite(surfaceShade.rgb, parseColor(palette.dark).rgb, surfaceShade.alpha),
      composite(actionLight.rgb, parseColor(palette.dark).rgb, actionLight.alpha),
      composite(actionShade.rgb, parseColor(palette.dark).rgb, actionShade.alpha),
    ]) {
      assert.ok(wcagContrast(rendered(theme, "--action-ink"), surface) >= 4.5, `${palette.name} ${name} control material contrast`);
      assert.ok(Math.abs(apcaContrast(rendered(theme, "--action-ink"), surface)) >= 60, `${palette.name} ${name} control material APCA`);
    }
    reports.push(expectPair(accentTheme, `${palette.name} ${name} accent`, "--accent", "--bg-alt", 4.5, 60));
    const accent = parseColor(palette[name]).rgb;
    const light = parseColor(theme["--surface-light"]);
    const ink = rendered(theme, "--fg-muted");
    for (const hover of [false, true]) for (const strength of [0, 0.5, 1]) {
      const base = hover ? composite(rendered(theme, "--fg"), rendered(theme, "--bg-alt"), 0.04) : rendered(theme, "--bg-alt");
      const sheen = composite(light.rgb, base, light.alpha * strength);
      const material = sheen;
      assert.ok(wcagContrast(ink, material) >= 4.5, `${palette.name} ${name} card material contrast`);
      assert.ok(Math.abs(apcaContrast(ink, material)) >= 60, `${palette.name} ${name} card material APCA ${Math.abs(apcaContrast(ink, material)).toFixed(1)}`);
    }
    const buttonSheen = composite([1, 1, 1], accent, 0.08);
    assert.ok(wcagContrast(rendered(theme, "--text-on-accent"), buttonSheen) >= 4.5, `${palette.name} ${name} button sheen contrast`);
  }
}
for (const target of [SURPRISE_TARGET, PROBABILITY_TARGET, ENTROPY_TARGET, "jlens/word"]) {
  assert.equal(highlightHue(target), "surprise", `${target} must use the vocabulary data hue`);
}
assert.equal(highlightHue("sae/1"), "sae");
assert.equal(highlightHue("local/concept"), "signed");
for (const [name, theme] of Object.entries({ dark, light })) {
  const accentChannels = rendered(theme, "--accent");
  const [red, green, blue] = accentChannels;
  assert.ok(blue > red && red > green && blue - green > 50 / 255,
    `${name} interaction accent must remain purple`);
  assert.ok(icon("home", 0, name).includes(`--tile:${theme["--accent"]};`),
    `${name} tab icon must match the interaction accent`);
  assert.equal(theme["--focus-ring"], theme["--accent"]);
  for (const surface of ["--surface-hi"]) {
    reports.push(expectPair(theme, `${name} error text on ${surface}`, "--error-text", surface, 4.5, 75));
  }
  for (const surface of [
    "--bg",
    "--bg-alt",
    "--bg-elev",
    "--surface-hi",
    "--glass",
    "--glass-strong",
    "--glass-bright",
    "--input-well",
    "--workspace-field-bg",
  ]) {
    reports.push(expectPair(theme, `${name} primary on ${surface}`, "--fg", surface, 7, 75));
    reports.push(expectPair(theme, `${name} body on ${surface}`, "--fg-strong", surface, 7, 75));
    reports.push(expectPair(theme, `${name} secondary on ${surface}`, "--fg-dim", surface, 4.5, 60));
  }
  for (const surface of ["--bg", "--bg-alt", "--surface-hi"]) {
    reports.push(expectPair(theme, `${name} preferred primary on ${surface}`, "--fg", surface, 7, 90));
  }
  for (const surface of ["--bg", "--bg-alt", "--bg-elev", "--surface-hi", "--glass", "--input-well", "--workspace-field-bg"]) {
    reports.push(expectPair(theme, `${name} muted on ${surface}`, "--fg-muted", surface, 4.5, 60));
  }
  reports.push(expectPair(theme, `${name} disabled`, "--fg-muted", "--glass-strong", 3, 30));
  reports.push(expectPair(theme, `${name} warning copy`, "--warning-ink", "--warning-bg", 7, 75));
  for (const status of ["success", "warning", "danger"]) {
    reports.push(expectPair(theme, `${name} device ${status}`, `--${status}-ink`, `--${status}-bg`, 7, 75));
  }
  for (const surface of ["--warning-action", "--warning-action-hover"]) {
    reports.push(expectPair(theme, `${name} warning action on ${surface}`, "--warning-action-ink", surface, 7, 60));
  }
  for (const surface of ["--danger-bg", "--danger-hover"]) {
    reports.push(expectPair(theme, `${name} destructive action on ${surface}`, "--accent-red", surface, 4.5, 60));
  }
  reports.push(expectPair(theme, `${name} primary action`, "--text-on-accent", "--accent", 4.5, 60));
  reports.push(expectPair(theme, `${name} focus`, "--focus-ring", "--bg", 3, 60));

  for (const token of [
    "--accent",
    "--accent-light",
    "--accent-green",
    "--accent-red",
    "--accent-amber",
    "--accent-yellow",
    "--pillar-subspace",
    "--pillar-manifold",
    "--pillar-sae",
    "--pillar-lens",
  ]) {
    reports.push(expectPair(theme, `${name} ${token}`, token, "--surface-hi", 4.5, 60));
    reports.push(expectPair(theme, `${name} solid ${token}`, "--text-on-accent", token, 4.5, 60));
  }

  assert.ok(
    wcagContrast(rendered(theme, "--data-track"), rendered(theme, "--surface-hi")) >= 3,
    `${name} data track must remain visible against elevated surfaces`,
  );
}

assert.equal(light["--glass"], light["--bg-alt"], "Light cards share the white panel surface");
assert.ok(ruleBody(':root[data-theme="light"]').includes('--shadow-rack: var(--shadow-card)'), "Light cards share the borderless panel shadow");
reports.push(expectSurface(light, "light raised controls on cards", "--glass-strong", "--glass", 1.1, 0));
reports.push(expectSurface(light, "light selected hierarchy", "--glass-bright", "--bg", 1.35, 15));
for (const surface of ["--input-well", "--workspace-field-bg"]) {
  assert.ok(wcagLuminance(rendered(light, surface)) >= 0.9, "Light text fields stay near white");
  reports.push(expectPair(light, `light input focus on ${surface}`, "--focus-ring", surface, 3, 60));
}
reports.push(expectSurface(light, "light separator", "--glass-line", "--bg", 1.7, 30));
reports.push(expectSurface(light, "light grid", "--grid-line", "--bg", 1.45, 20));

const hero = await readFile("src/hosted/ui/HeroShader.svelte", "utf8");
const matrix = hero.match(/<feColorMatrix type="matrix" values="([^"]+)"/)[1].trim().split(/\s+/).map(Number);
assert.equal(matrix.length, 20);
const baseDarkAmplitude = Number(hero.match(/<feFuncR type="gamma" amplitude="([\d.]+)"/)[1]);
for (const [name, theme] of Object.entries({ dark, light })) {
  const panel = parseColor(theme["--landing-panel-bg"]);
  assert.equal(panel.alpha, 0.24, "Landing glass has a stronger tint while retaining transparency");
  for (let corner = 0; corner < 8; corner++) {
    const input = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
    const backdrop = name === "dark" ? input.map(channel => channel * baseDarkAmplitude) : [0, 1, 2].map(row =>
      matrix[row * 5 + 4] + input.reduce((sum, channel, index) => sum + channel * matrix[row * 5 + index], 0));
    const background = composite(panel.rgb, backdrop, panel.alpha);
    for (const token of ["--landing-panel-ink", "--landing-panel-muted"]) {
      assert.ok(wcagContrast(rendered(theme, token), background) >= 4.5,
        `${name} ${token} stays readable over the base shader palette`);
    }
  }
}
assert.match(dark["--landing-panel-ink"], /#ffffff/);
assert.match(css, /--landing-panel-text-shadow: 0 1px 3px rgb\(0 0 0 \/ 70%\)/,
  "dark glass uses a local text shadow for moving boosted highlights, not a panel-wide scrim");
reports.push("landing glass base-palette text contrast passes; boosted highlights use local text shadows");
let minimumHeroContrast = Infinity;
for (let corner = 0; corner < 8; corner++) {
  const input = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
  const output = [0, 1, 2].map(row => matrix[row * 5 + 4] + input.reduce((sum, channel, index) => sum + channel * matrix[row * 5 + index], 0));
  assert.ok(output.every(channel => channel >= 0 && channel <= 1), "Hero palette must stay inside sRGB without clipping");
  minimumHeroContrast = Math.min(minimumHeroContrast, wcagContrast(rendered(light, "--fg"), output));
}
assert.ok(minimumHeroContrast >= 4.5, "Light-mode hero text must stay readable across the entire shader output");
reports.push(`light hero worst-case text ${minimumHeroContrast.toFixed(2)}:1`);
const [, amplitudeText, exponentText] = hero.match(/<feFuncR type="gamma" amplitude="([\d.]+)" exponent="([\d.]+)"/);
const amplitude = Number(amplitudeText), exponent = Number(exponentText);
const darkHeroContrast = wcagContrast(rendered(dark, "--fg"), [amplitude, amplitude, amplitude]);
assert.ok(darkHeroContrast >= 4.5, "Dark hero highlights must preserve readable text");
assert.ok(amplitude * 0.3 ** exponent > 0.38 * 0.3 * 1.8, "Dark hero midtones should be visibly brighter than the old filter");
reports.push(`dark hero worst-case text ${darkHeroContrast.toFixed(2)}:1`);

for (const [name, theme] of Object.entries({ dark, light })) {
  const image = name === "dark" ? [1, 1, 1] : [0, 0, 0];
  const backdrop = composite(image, rendered(theme, "--bg"), 0.14 * (name === "light" ? 0.55 : 1));
  for (const token of ["--fg", "--fg-dim", "--fg-muted"]) {
    assert.ok(wcagContrast(rendered(theme, token, backdrop), backdrop) >= 4.5, `${name} ${token}: background image must preserve text contrast`);
    for (const material of ["--workspace-panel-bg", "--workspace-field-bg", "--workspace-neutral-bg", "--workspace-neutral-hover"]) {
      const surface = rendered(theme, material, backdrop);
      assert.ok(wcagContrast(rendered(theme, token === "--fg-muted" ? "--fg-dim" : token), surface) >= 4.5,
        `${name} ${token}: clear chat surfaces must stay readable over a visible custom background (${material})`);
    }
  }
  assert.equal(parseColor(theme["--workspace-panel-bg"]).alpha, 0.03);
  if (name === "dark") assert.equal(parseColor(theme["--workspace-field-bg"]).alpha, 0.05);
}
console.log(`color contrast passed (${reports.join(", ")})`);
