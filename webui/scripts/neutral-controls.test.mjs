import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const tokens = await readFile("src/lib/style/tokens.css", "utf8");
const globals = await readFile("src/lib/style/global.css", "utf8");
function rule(source, selector) {
  const start = source.indexOf(`${selector} {`);
  assert.ok(start >= 0, `missing ${selector}`);
  return source.slice(source.indexOf("{", start) + 1, source.indexOf("}", start));
}
const light = rule(tokens, ':root[data-theme="light"]');
const value = (body, prop) => body.match(new RegExp(`(?:^|[;\\n])\\s*${prop}:\\s*([^;]+);`))?.[1].trim();
assert.equal(value(light, "--control-neutral-bg"), "#ffffff");
assert.equal(value(light, "--control-neutral-bg"), value(light, "--bg-alt"));
assert.notEqual(value(light, "--bg-elev"), "#ffffff", "content wells keep their own surface tier");
const controls = rule(globals, ':root[data-theme="light"] :is(button, summary, select)');
for (const prop of ["--glass", "--glass-strong", "--glass-bright", "--bg-elev", "--bg-deep", "--input-well"]) {
  assert.equal(value(controls, prop), "var(--control-neutral-bg)", `${prop}: legacy neutral controls must share the white material`);
}
assert.equal(value(controls, "--bg-hover"), "var(--control-neutral-hover)");
for (const prop of ["background", "background-color", "--accent", "--action-bg", "--danger-bg", "--focus-ring", "--data-track"]) {
  assert.equal(value(controls, prop), undefined, `${prop}: semantic and interaction states must not be overwritten`);
}
assert.ok(!controls.includes("!important"));
for (const file of ["src/lib/ui/Button.svelte", "src/lib/ui/ChatAccentPicker.svelte", "src/lib/Select.svelte", "src/hosted/ui/HostedHome.svelte"]) {
  assert.match(await readFile(file, "utf8"), /gap: var\(--control-label-gap\)/, `${file}: consistent label spacing`);
}
const header = await readFile("src/panels/rack/RackSectionHeader.svelte", "utf8");
assert.match(rule(header, ".header"), /flex-wrap: wrap/);
assert.match(rule(header, ".header-text"), /flex-wrap: wrap/);
console.log("Neutral controls: shared white light material, retained semantic colors, scoped dark-mode isolation, and label spacing passed");
