import { readFile, writeFile } from "node:fs/promises";
import { createPrivateKey, sign } from "node:crypto";

const [catalogPath, keyPath, keyId, outputPath] = process.argv.slice(2);
if (!catalogPath || !keyPath || !keyId || !outputPath) {
  throw new Error("usage: node sign-catalog.mjs CATALOG PRIVATE_KEY KEY_ID OUTPUT");
}

const catalogBytes = await readFile(catalogPath);
JSON.parse(catalogBytes.toString("utf8"));
const key = createPrivateKey(await readFile(keyPath));
if (key.asymmetricKeyType !== "ed25519") throw new Error("catalog key must be Ed25519");

const envelope = {
  schemaVersion: 1,
  algorithm: "Ed25519",
  keyId,
  signature: sign(null, catalogBytes, key).toString("base64"),
};
await writeFile(outputPath, `${JSON.stringify(envelope, null, 2)}\n`, { flag: "wx" });

