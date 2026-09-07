import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";

export async function verifyArtifactFile(path, artifact, label) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`${label} is not a regular file: ${path}`);
  if (!Number.isSafeInteger(artifact?.bytes) || artifact.bytes < 0) {
    throw new Error(`${label} has an invalid declared byte count`);
  }
  if (!/^[0-9a-f]{64}$/.test(artifact?.sha256 ?? "")) {
    throw new Error(`${label} has an invalid declared SHA-256`);
  }
  if (info.size !== artifact.bytes) {
    throw new Error(
      `${label} byte count differs: expected ${artifact.bytes}, found ${info.size}`,
    );
  }
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  const actual = digest.digest("hex");
  if (actual !== artifact.sha256) {
    throw new Error(
      `${label} SHA-256 differs: expected ${artifact.sha256}, found ${actual}`,
    );
  }
}
