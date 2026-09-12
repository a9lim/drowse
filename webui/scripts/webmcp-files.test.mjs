import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "vite";
import { fileURLToPath } from "node:url";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)), appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { FileTransfers, createFileTools, getTransferredFile } = await server.ssrLoadModule("/src/lib/webmcp/files.ts");
  const transfers = new FileTransfers();
  for (const name of ["../backup.json", "folder/file", "bad\u0000name", "folder\\file"]) assert.throws(() => transfers.begin(name, 1), /filename/i);
  assert.throws(() => transfers.begin("file", -1), /size|files/i);
  assert.throws(() => transfers.begin("file", 1024 ** 3 + 1), /1 GiB/);
  const bytes = Buffer.from("A pirate's backup: naïve 船");
  const file = transfers.begin("chat.json", bytes.length, "application/json", createHash("sha256").update(bytes).digest("hex"));
  assert.throws(() => transfers.get(file.file_id), /Finish/);
  await assert.rejects(() => transfers.finish(file.file_id), /Uploaded/);
  await assert.rejects(() => transfers.write(file.file_id, 1, bytes.toString("base64")), /chunk|size/i);
  await transfers.write(file.file_id, 0, bytes.toString("base64"));
  await transfers.write(file.file_id, 0, bytes.toString("base64"));
  await assert.rejects(() => transfers.write(file.file_id, 0, Buffer.alloc(bytes.length).toString("base64")), /different content/);
  const finished = await transfers.finish(file.file_id);
  assert.equal(finished.state, "complete");
  assert.equal(await transfers.get(file.file_id).text(), bytes.toString());
  assert.deepEqual(await transfers.finish(file.file_id), finished);
  const mismatch = transfers.begin("bad", 1, undefined, "0".repeat(64));
  await transfers.write(mismatch.file_id, 0, "YQ==");
  await assert.rejects(() => transfers.finish(mismatch.file_id), /checksum/);
  transfers.delete(mismatch.file_id);
  assert.throws(() => transfers.get(mismatch.file_id), /expired/);
  const chunk = Buffer.alloc(512 * 1024, 37), size = 65 * 1024 * 1024;
  const large = transfers.begin("maximum-chat-backup.json", size, "application/json");
  const expected = createHash("sha256");
  for (let offset = 0; offset < size; offset += chunk.length) {
    await transfers.write(large.file_id, offset, chunk.toString("base64"));
    expected.update(chunk);
  }
  assert.equal((await transfers.finish(large.file_id)).sha256, expected.digest("hex"));
  assert.equal(transfers.get(large.file_id).size, size, "the full existing 65 MiB backup size is accepted");
  const tools = new Map(createFileTools().map(tool => [tool.name, tool]));
  const call = (name, input) => tools.get(name).execute(input, {});
  const upload = await call("drowse_begin_file", { name: "example.drowse", size: bytes.length });
  await call("drowse_write_file", { file_id: upload.file_id, offset: 0, base64: bytes.toString("base64") });
  await call("drowse_finish_file", { file_id: upload.file_id });
  assert.equal(getTransferredFile(upload.file_id).name, "example.drowse");
  const first = await call("drowse_read_file", { file_id: upload.file_id, offset: 0, limit: 7 });
  assert.deepEqual(Buffer.from(first.base64, "base64"), bytes.subarray(0, 7));
  assert.equal(first.next_offset, 7);
  await assert.rejects(() => call("drowse_read_file", { file_id: upload.file_id, offset: bytes.length + 1 }), /exceeds/);
  await call("drowse_delete_file", { file_id: upload.file_id });
  transfers.clear();
  console.log("WebMCP file transfer: 65 MiB uploads, byte offsets, checksums, identical retries, conflict rejection and typed import handles passed");
} finally { await server.close(); }
