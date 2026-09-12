import { sha256 } from "@noble/hashes/sha2.js";
import { ToolError } from "./types";

type StoragePort = Pick<Storage, "getItem" | "setItem">;
interface Receipt { fingerprint: string; state: "pending" | "complete"; result?: unknown }
const KEY = "drowse.webmcp.requests.v1";

export class MutationReceipts {
  constructor(private readonly storage: StoragePort) {}

  fingerprint(input: string): string {
    return Array.from(sha256(new TextEncoder().encode(input)), byte => byte.toString(16).padStart(2, "0")).join("");
  }

  get(key: string, fingerprint: string): Receipt | null {
    const receipt = this.read().get(key);
    if (receipt && receipt.fingerprint !== fingerprint) throw new ToolError("request_conflict", "This request ID was already used with different inputs.");
    return receipt ?? null;
  }

  start(key: string, fingerprint: string): void {
    const receipts = this.read();
    receipts.set(key, { fingerprint, state: "pending" });
    this.write(receipts);
  }

  finish(key: string, fingerprint: string, result: unknown): void {
    const receipts = this.read();
    const serialized = JSON.stringify(result);
    receipts.set(key, { fingerprint, state: "complete", result: serialized.length <= 16000 && !(result as { result_id?: unknown } | null)?.result_id ? result : {
      ok: true, reconcile_required: true, message: "The action completed, but its large result belonged to the previous page. Read current state or its job; do not repeat the mutation.",
    } });
    this.write(receipts);
  }

  private read(): Map<string, Receipt> {
    try {
      const raw = this.storage.getItem(KEY);
      if (!raw) return new Map();
      const rows: unknown = JSON.parse(raw);
      if (!Array.isArray(rows) || rows.length > 128 || rows.some(row => !Array.isArray(row) || typeof row[0] !== "string" || !row[1] || typeof row[1].fingerprint !== "string" || !["pending", "complete"].includes(row[1].state))) throw Error("Invalid receipt data");
      return new Map(rows as [string, Receipt][]);
    } catch {
      throw new ToolError("REQUEST_STORAGE_UNAVAILABLE", "Request recovery storage could not be read. Inspect current state before retrying with a new request ID.");
    }
  }

  private write(receipts: Map<string, Receipt>): void {
    while (receipts.size > 128) receipts.delete(receipts.keys().next().value!);
    try { this.storage.setItem(KEY, JSON.stringify([...receipts])); }
    catch { throw new ToolError("REQUEST_STORAGE_UNAVAILABLE", "Request recovery storage could not be saved. Inspect current state and jobs before retrying."); }
  }
}
