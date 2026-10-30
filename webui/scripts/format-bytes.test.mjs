import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const { formatBytes } = await server.ssrLoadModule("/src/lib/format/bytes.ts");
  for (const [bytes, expected] of [
    [0, "0 B"],
    [-1, "0 B"],
    [Number.NaN, "0 B"],
    [512, "512 B"],
    [999, "999 B"],
    [1_000, "1.0 KB"],
    [9_940, "9.9 KB"],
    [9_960, "10 KB"],
    [512_000, "512 KB"],
    [999_400, "999 KB"],
    [999_500, "1.0 MB"],
    [907_300_000, "907 MB"],
    [907_600_000, "908 MB"],
    [999_499_999, "999 MB"],
    [999_500_000, "1.0 GB"],
    [1_000_000_000, "1.0 GB"],
    [1_249_000_000, "1.2 GB"],
    [2_865_904_392, "2.9 GB"],
    [12_400_000_000, "12 GB"],
    [999_600_000_000, "1.0 TB"],
  ]) {
    assert.equal(formatBytes(bytes), expected, `${bytes} bytes`);
  }
  assert.doesNotMatch(formatBytes(999_999_999), /1000/, "sizes switch unit before reading as 1000");
  console.log("Byte formatting: decimal SI units, unit switching at 999.5, and consistent precision passed");
} finally {
  await server.close();
}
