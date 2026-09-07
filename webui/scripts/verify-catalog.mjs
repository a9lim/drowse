import { readFile } from "node:fs/promises";
import { createPublicKey, verify } from "node:crypto";

const [catalogPath, signaturePath, publicKeyPath] = process.argv.slice(2);
if (!catalogPath || !signaturePath || !publicKeyPath) {
  throw new Error("usage: node verify-catalog.mjs CATALOG SIGNATURE PUBLIC_KEY");
}

const catalogBytes = await readFile(catalogPath);
const catalog = JSON.parse(catalogBytes.toString("utf8"));
const envelope = JSON.parse(await readFile(signaturePath, "utf8"));
const key = createPublicKey(await readFile(publicKeyPath));

if (envelope.schemaVersion !== 1 || envelope.algorithm !== "Ed25519") {
  throw new Error("unsupported catalog signature envelope");
}
if (key.asymmetricKeyType !== "ed25519") throw new Error("catalog key must be Ed25519");
if (!verify(null, catalogBytes, key, Buffer.from(envelope.signature, "base64"))) {
  throw new Error("catalog signature verification failed");
}
if (catalog.schemaVersion !== 1 || !Number.isSafeInteger(catalog.sequence)) {
  throw new Error("invalid signed catalog metadata");
}

console.log(`Verified catalog sequence ${catalog.sequence} with key ${envelope.keyId}`);
