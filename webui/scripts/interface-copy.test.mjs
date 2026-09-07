import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "svelte/compiler";
import ts from "typescript";

const root = fileURLToPath(new URL("..", import.meta.url));
const violations = [];
let checked = 0;

function check(value, file, source, start = 0) {
  if (typeof value !== "string") return;
  checked++;
  if (value.includes("\u2014")) {
    const line = source.slice(0, start).split("\n").length;
    violations.push(`${relative(root, file)}:${line}: ${value.trim().slice(0, 100)}`);
  }
}

function walkSvelte(node, file, source) {
  if (!node || typeof node !== "object") return;
  if (node.type === "Comment") return;
  if (node.type === "Text") check(node.data, file, source, node.start);
  if (node.type === "Literal") check(node.value, file, source, node.start);
  if (node.type === "TemplateElement") check(node.value.cooked, file, source, node.start);
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(child => walkSvelte(child, file, source));
    else if (value && typeof value === "object") walkSvelte(value, file, source);
  }
}

const files = [resolve(root, "scripts/site-metadata.mjs")];
for (const folder of ["src", "hosted"]) {
  for (const entry of readdirSync(resolve(root, folder), { recursive: true, withFileTypes: true })) {
    if (entry.isFile() && /\.(svelte|ts|js|html|css)$/.test(entry.name)) {
      files.push(resolve(entry.parentPath, entry.name));
    }
  }
}
for (const file of files) {
  const source = readFileSync(file, "utf8");
  if (/\.(svelte|html)$/.test(file)) {
    walkSvelte(parse(source, { modern: true }), file, source);
  } else if (file.endsWith(".css")) {
    check(source.replace(/\/\*[\s\S]*?\*\//g, ""), file, source);
  } else {
    const tree = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
    const visit = node => {
      if (ts.isStringLiteralLike(node) || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
        check(node.text, file, source, node.getStart(tree));
      }
      ts.forEachChild(node, visit);
    };
    visit(tree);
  }
}
assert.deepEqual(violations, [], `Use sentences, commas, or parentheses in app copy:\n${violations.join("\n")}`);
console.log(`Interface copy: ${checked} authored text segments in ${files.length} files contain no em dashes`);
