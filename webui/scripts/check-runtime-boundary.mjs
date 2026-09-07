import { readdir, readFile } from "node:fs/promises";
import { relative, resolve } from "node:path";
import ts from "typescript";

const root = resolve(import.meta.dirname, "../src");
const allowed = new Set(["lib/runtime/http-client.ts"]);
const violations = [];

async function visit(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      await visit(path);
      continue;
    }
    if (!entry.name.endsWith(".ts") && !entry.name.endsWith(".svelte")) continue;
    const local = relative(root, path);
    if (allowed.has(local)) continue;
    const source = await readFile(path, "utf8");
    const scripts = entry.name.endsWith(".svelte")
      ? [...source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)].map(
          (match) => match[1],
        )
      : [source];
    if (scripts.some((script) => importsApi(script, local))) violations.push(local);
  }
}

function importsApi(source, fileName) {
  const file = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS,
  );
  let found = false;
  const visitNode = (node) => {
    if (found) return;
    if (
      (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
      isApiModule(moduleName(node.moduleSpecifier))
    ) {
      found = true;
      return;
    }
    if (
      ts.isImportEqualsDeclaration(node) &&
      ts.isExternalModuleReference(node.moduleReference) &&
      isApiModule(moduleName(node.moduleReference.expression))
    ) {
      found = true;
      return;
    }
    if (
      ts.isCallExpression(node) && isImportCall(node) &&
      isApiModule(moduleName(node.arguments[0]))
    ) {
      found = true;
      return;
    }
    ts.forEachChild(node, visitNode);
  };
  visitNode(file);
  return found;
}

function isImportCall(node) {
  if (node.expression.kind === ts.SyntaxKind.ImportKeyword) return true;
  if (ts.isIdentifier(node.expression) && node.expression.text === "require") return true;
  return ts.isPropertyAccessExpression(node.expression) &&
    ts.isIdentifier(node.expression.expression) &&
    node.expression.expression.text === "require" &&
    node.expression.name.text === "resolve";
}

function moduleName(node) {
  if (!node) return null;
  if (ts.isStringLiteralLike(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
    return node.text;
  }
  return null;
}

function isApiModule(value) {
  if (value === null) return false;
  const clean = value.split(/[?#]/, 1)[0].replaceAll("\\", "/");
  return /(?:^|\/)api(?:\.[cm]?[jt]sx?)?$/.test(clean);
}

for (const fixture of [
  'import "../api";',
  'export { api } from "../api.ts";',
  'const api = await import(`../api.js`);',
  'const api = require("../api");',
  'import api = require("../api");',
]) {
  if (!importsApi(fixture, "boundary-fixture.ts")) {
    throw new Error(`runtime boundary fixture was not detected: ${fixture}`);
  }
}
if (importsApi('import "../runtime/services";', "boundary-safe-fixture.ts")) {
  throw new Error("runtime boundary rejected a transport-neutral import");
}

await visit(root);
if (violations.length) {
  throw new Error(
    `Direct HTTP client imports are restricted to HttpRuntimeClient:\n${violations.join("\n")}`,
  );
}
console.log("runtime boundary check passed");
