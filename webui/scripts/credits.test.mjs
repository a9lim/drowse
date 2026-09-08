import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: VerificationBadge } = await server.ssrLoadModule("/src/hosted/ui/VerificationBadge.svelte");
  for (const [status, label] of [
    ["blue", "Blue verified account on X"],
    ["gold", "Gold verified organization on X"],
    ["gray", "Gray verified government account on X"],
  ]) {
    const html = render(VerificationBadge, { props: { status } }).body;
    assert.ok(html.includes(`data-status="${status}"`));
    assert.ok(html.includes(`aria-label="${label}"`));
    assert.ok(!html.includes(`title="${label}"`));
  }
  assert.doesNotMatch(render(VerificationBadge, { props: { status: null } }).body, /<svg|verification-badge/,
    "unverified accounts have no badge, not a gray government badge");
  const credits = await readFile("src/hosted/ui/Credits.svelte", "utf8");
  for (const [handle, status] of [["_a9lim", "blue"], ["treetowntree", "blue"], ["transkatgirl", "blue"], ["voooooogel", "blue"], ["motion_so", "gold"]]) {
    assert.match(credits, new RegExp(`handle: "@${handle}"[^\\n]+verification: "${status}"`));
  }
  assert.match(credits, /checked 2026-09-06/, "verification is a dated snapshot, not a live lookup");
  assert.match(credits, /<VerificationBadge status=\{member.verification\}/);
  assert.match(credits, /<VerificationBadge status=\{contributor.verification\}/);
  assert.ok((await readFile("src/assets/credits/motion_so.jpg")).length > 0);
  assert.ok((await readFile("src/assets/credits/voooooogel.jpg")).length > 0);
  const thanks = credits.split("const thanks = [")[1].split("] as const")[0];
  assert.deepEqual([...thanks.matchAll(/handle: "(@[^"]+)"/g)].map(match => match[1]),
    ["@transkatgirl", "@voooooogel", "@motion_so"]);
  console.log("Credits: three verification states, unverified omission, profile mappings, and Motion portrait passed");
} finally {
  await server.close();
}
