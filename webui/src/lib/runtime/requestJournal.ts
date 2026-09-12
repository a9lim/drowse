import type { WSClientMessage, WSDoneEvent, WSErrorEvent, WSServerMessage } from "../types";

export interface RuntimeRequestStatus {
  request_id: string;
  state: "unknown" | "running" | "completed" | "cancelled" | "failed" | "interrupted";
  results: WSDoneEvent[];
  error?: WSErrorEvent;
}

interface Entry extends RuntimeRequestStatus { fingerprint: string }

export class RuntimeRequestJournal {
  private readonly records = new Map<string, Entry>();
  private readonly storage: Storage | null;
  private readonly key = "drowse.runtime.requests.v1";

  constructor() {
    try { this.storage = typeof sessionStorage === "undefined" ? null : sessionStorage; }
    catch { this.storage = null; }
    try {
      const rows = JSON.parse(this.storage?.getItem(this.key) ?? "[]") as Entry[];
      for (const entry of rows) {
        if (typeof entry.request_id !== "string" || !Array.isArray(entry.results)) continue;
        if (entry.state === "running") entry.state = "interrupted";
        this.records.set(entry.request_id, entry);
      }
    } catch { /* Missing recovery evidence remains unknown. */ }
  }

  claim(message: WSClientMessage): "new" | "existing" | "conflict" {
    if (!message.request_id) return "new";
    const fingerprint = JSON.stringify(message);
    const previous = this.records.get(message.request_id);
    if (previous) return previous.fingerprint === fingerprint ? "existing" : "conflict";
    this.records.set(message.request_id, { request_id: message.request_id, state: "running", results: [], fingerprint });
    this.persist();
    return "new";
  }

  observe(event: WSServerMessage): void {
    if (!("request_id" in event) || !event.request_id) return;
    const entry = this.records.get(event.request_id);
    if (!entry) return;
    if (event.type === "done") {
      entry.results = [...entry.results.filter(row => row.sibling_index !== event.sibling_index), structuredClone(event)];
    } else if (event.type === "error") {
      if (["REQUEST_ID_CONFLICT", "REQUEST_RECONCILE_REQUIRED"].includes(event.code ?? "")) return;
      entry.error = structuredClone(event);
    }
    else if (event.type === "request_complete") entry.state = event.state;
    else return;
    this.persist();
  }

  get(id: string): RuntimeRequestStatus {
    const entry = this.records.get(id);
    if (!entry) return { request_id: id, state: "unknown", results: [] };
    const { fingerprint: _fingerprint, ...status } = entry;
    return structuredClone(status);
  }

  interruptRunning(): void {
    for (const entry of this.records.values()) if (entry.state === "running") entry.state = "interrupted";
    this.persist();
  }

  private persist(): void {
    const terminal = [...this.records.values()].filter(row => row.state !== "running");
    for (const entry of terminal.slice(0, Math.max(0, terminal.length - 100))) this.records.delete(entry.request_id);
    try { this.storage?.setItem(this.key, JSON.stringify([...this.records.values()])); }
    catch { /* The job registry separately reports persistence failures; missing journal evidence is never success. */ }
  }
}

export function replayRequest(status: RuntimeRequestStatus, emit: (event: WSServerMessage) => void): void {
  for (const result of status.results) emit(result);
  if (status.error) emit(status.error);
  if (["completed", "cancelled", "failed"].includes(status.state)) emit({ type: "request_complete",
    request_id: status.request_id, state: status.state as "completed" | "cancelled" | "failed", completed_siblings: status.results.length });
  else emit({ type: "error", request_id: status.request_id, code: "REQUEST_RECONCILE_REQUIRED",
    message: "This request was already admitted. Query its recorded status instead of submitting it again." });
}
