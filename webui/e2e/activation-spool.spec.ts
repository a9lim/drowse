import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const moduleUrl = `/@fs/${resolve("src/hosted/fitting/activationSpool.ts")}`;

test("browser activation spools commit bounded rows and recover storage loss", async ({ page }) => {
  await page.goto(`${devUrl}/LICENSE-Martian-Mono.txt`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const fitting = await import(moduleUrl);
    const descriptor = {
      runtimeIdentitySha256: "1".repeat(64),
      contextBindingSha256: "2".repeat(64),
      captureSha256: "3".repeat(64),
      layers: [
        { layer: 2, rows: 2, width: 2, expectedBytes: 16 },
        { layer: 5, rows: 1, width: 3, expectedBytes: 12 },
      ],
    };
    const spool = new fitting.BrowserActivationSpool();
    await spool.clear();
    const writer = await spool.begin(descriptor);
    await writer.appendRows(2, 0, new Float32Array([1, 2]));
    await writer.checkpoint(2);
    await writer.appendRows(2, 1, new Float32Array([3, 4]));
    await writer.sealLayer(2);
    await writer.appendRows(5, 0, new Float32Array([5, 6, 7]));
    await writer.sealLayer(5);
    const reader = await writer.commit();
    const committedRows: number[][] = [];
    for await (const chunk of reader.readChunks(2, { maxChunkBytes: 8 })) {
      committedRows.push([...chunk.values]);
    }
    spool.close();

    const reopened = new fitting.BrowserActivationSpool();
    const reopenedReader = await reopened.openCommitted(descriptor);
    const reopenedRows: number[][] = [];
    for await (const chunk of reopenedReader.readChunks(5, { maxChunkBytes: 12 })) {
      reopenedRows.push([...chunk.values]);
    }
    reopened.close();

    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const fittingDirectory = await drowse.getDirectoryHandle("fitting");
    const activations = await fittingDirectory.getDirectoryHandle("activation-spool");
    await activations.removeEntry(
      fitting.activationSpoolFileName(descriptor.captureSha256, 2),
    );
    const recovered = new fitting.BrowserActivationSpool();
    await recovered.recover();
    let recoveryCode: string | null = null;
    try {
      await recovered.openCommitted(descriptor);
    } catch (error) {
      recoveryCode = error instanceof fitting.ActivationSpoolError ? error.code : String(error);
    }
    const remainingEntries: string[] = [];
    for await (const [name] of activations.entries()) remainingEntries.push(name);
    await recovered.clear();
    recovered.close();
    return { committedRows, reopenedRows, recoveryCode, remainingEntries };
  }, { moduleUrl });

  expect(result.committedRows).toEqual([[1, 2], [3, 4]]);
  expect(result.reopenedRows).toEqual([[5, 6, 7]]);
  expect(result.recoveryCode).toBe("CAPTURE_NOT_FOUND");
  expect(result.remainingEntries).toEqual([]);
});
