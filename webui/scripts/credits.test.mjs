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
  assert.match(credits, /name: "GPT-5\.6 Sol", provider: "openai"/);
  assert.doesNotMatch(credits, /GPT-6 Sol/);
  assert.match(credits, /name: "GPT-6 Astra", provider: "openai"/);
  assert.match(await readFile("src/lib/style/tokens.css", "utf8"), /--brand-openai:\s*#000;/);
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
  for (const heading of ["Models", "Research Drowse builds on", "Interpretability tools and data", "Software, fonts and icons", "Cite Drowse"]) {
    assert.ok(credits.includes(`title: "${heading}"`) || credits.includes(`>${heading}</h2>`), `Credits has a ${heading} section`);
  }
  for (const source of ["arxiv.org/abs/2310.01405", "github.com/vgel/repeng", "arxiv.org/abs/2502.02716", "arxiv.org/abs/2605.05115", "arxiv.org/abs/2601.10387", "transformer-circuits.pub/2026/workspace"]) {
    assert.ok(credits.includes(source), `research lineage links ${source}`);
  }
  assert.match(credits, /Gemma is provided under and subject to the Gemma Terms of Use found at ai\.google\.dev\/gemma\/terms\./, "Gemma attribution notice is shown");
  for (const file of ["/LICENSE-Wix-Madefor.txt", "/LICENSE-Martian-Mono.txt", "/LICENSE-Fluent-System-Icons.txt", "/LICENSE-Torph.txt"]) {
    assert.ok(credits.includes(file), `Credits links ${file}`);
    assert.ok((await readFile(`public-hosted${file}`)).length > 0, `${file} ships with the hosted site`);
  }
  assert.match(credits, /class="sr-only"> by \{providerNames\[model\.provider\]\}/, "screen readers hear who made each credited model");
  assert.match(credits, /role="status">\{citationStatus\}/, "copying the citation is announced");
  assert.doesNotMatch(credits, /<svelte:head>/, "Credits updates the existing description instead of adding a second one");
  assert.doesNotMatch(credits, /__DROWSE_SOURCE_URL__/, "Contribute goes to the on-site guide");
  assert.match(credits, /href="\/contribute"/);
  assert.match(credits, /People and groups who inspired, supported or contributed to Drowse\./);
  console.log("Credits: three verification states, unverified omission, profile mappings, and Motion portrait passed");
} finally {
  await server.close();
}
