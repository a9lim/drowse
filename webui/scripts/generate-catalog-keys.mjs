#!/usr/bin/env node

import { generateKeyPairSync } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const [outputArgument, currentKeyId, nextKeyId] = process.argv.slice(2);
if (!outputArgument || !currentKeyId || !nextKeyId) {
  throw new Error(
    "usage: node generate-catalog-keys.mjs OUTPUT_DIR CURRENT_KEY_ID NEXT_KEY_ID",
  );
}
if (currentKeyId === nextKeyId || !safeKeyId(currentKeyId) || !safeKeyId(nextKeyId)) {
  throw new Error("catalog key IDs must be distinct safe slugs");
}

const outputDirectory = resolve(outputArgument);
await mkdir(outputDirectory, { mode: 0o700 });
const entries = [];
for (const [role, keyId] of [["current", currentKeyId], ["next", nextKeyId]]) {
  const pair = generateKeyPairSync("ed25519");
  const privateKey = pair.privateKey.export({ type: "pkcs8", format: "pem" });
  const publicDer = pair.publicKey.export({ type: "spki", format: "der" });
  const rawPublicKey = publicDer.subarray(-32);
  if (rawPublicKey.length !== 32) throw new Error("failed to export Ed25519 public key");
  await writeFile(resolve(outputDirectory, `${role}.private.pem`), privateKey, {
    flag: "wx",
    mode: 0o600,
  });
  entries.push({
    keyId,
    ed25519PublicKeyBase64: rawPublicKey.toString("base64"),
  });
}
await writeFile(
  resolve(outputDirectory, "public-keys.json"),
  `${JSON.stringify(entries, null, 2)}\n`,
  { flag: "wx", mode: 0o644 },
);
process.stdout.write(`${JSON.stringify({ outputDirectory, publicKeys: entries }, null, 2)}\n`);

function safeKeyId(value) {
  return /^[a-z0-9][a-z0-9._-]{0,63}$/u.test(value);
}
