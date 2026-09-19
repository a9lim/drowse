import { ToolError, objectSchema, type AppTool, type InputSchema, type ToolContext } from "./types";
import { validateInput } from "./validation";
import { MutationReceipts, requestFingerprint } from "./requests";

export interface ModelContextProvider {
  registerTool(tool: {
    name: string;
    title: string;
    description: string;
    inputSchema: InputSchema;
    annotations: AppTool["annotations"];
    execute: (input: unknown, options: { signal: AbortSignal }) => Promise<unknown>;
  }, options: { signal: AbortSignal }): Promise<void> | void;
}

interface AdapterOptions {
  provider: ModelContextProvider | null;
  context: (signal: AbortSignal) => ToolContext;
  revision?: () => string;
  visible?: (tool: AppTool) => boolean;
  requestStorage?: Pick<Storage, "getItem" | "setItem">;
}

function canonicalInput(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalInput);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, canonicalInput(item)]));
  return value;
}

export class WebMcpAdapter {
  private tools = new Map<string, AppTool>();
  private registrations = new Map<string, { lifetime: AbortController; tool: AppTool }>();
  private calls = new Map<string, { input: string; result: Promise<unknown>; pending: boolean }>();
  private pendingCalls = 0;
  private pendingInputCharacters = 0;
  private results = new Map<string, string>();
  private resultBytes = 0;
  private mutationTail: Promise<unknown> = Promise.resolve();
  private refreshTail: Promise<void> = Promise.resolve();
  private disposed = false;
  private lifetime = new AbortController();
  private actionLifetime = new AbortController();
  private dataEpoch = 0;
  readonly errors = new Map<string, string>();
  private receipts: MutationReceipts | null;

  constructor(private options: AdapterOptions) { this.receipts = options.requestStorage ? new MutationReceipts(options.requestStorage) : null; }

  setTools(tools: readonly AppTool[]): Promise<void> {
    const names = new Set<string>();
    for (const tool of tools) {
      if (names.has(tool.name)) throw new Error(`Duplicate WebMCP tool: ${tool.name}`);
      names.add(tool.name);
    }
    this.tools = new Map(tools.map(tool => [tool.name, tool]));
    return this.refresh();
  }

  list(): Array<{ name: string; title: string; group: string; available: boolean; reason: string | null }> {
    const context = this.options.context(this.lifetime.signal);
    return [...this.tools.values()].map(tool => {
      const reason = tool.available?.(context) ?? null;
      return { name: tool.name, title: tool.title, group: tool.group, available: reason === null, reason };
    });
  }

  describe(name: string): unknown {
    const tool = this.tools.get(name);
    if (!tool) throw new ToolError("not_found", "Unknown action. List actions for this route first.");
    const reason = tool.available?.(this.options.context(this.lifetime.signal)) ?? null;
    return {
      name: tool.name, title: tool.title, description: tool.description,
      group: tool.group, scope: tool.scope, input_schema: this.schema(tool), annotations: tool.annotations,
      available: reason === null, reason, registered: this.registrations.has(name),
      interfaces: tool.surfaces ?? [], runtime_methods: tool.services ?? [],
      validation: "Application schema validation followed by the shared action's model, identity and capability checks.",
      result: "JSON {ok,data,revision} or {ok:false,error}; a job receipt requires drowse_get_job until terminal. Large results use drowse_read_result.",
    };
  }

  refresh(): Promise<void> {
    this.refreshTail = this.refreshTail.then(async () => {
      if (this.disposed || !this.options.provider) return;
      const context = this.options.context(this.lifetime.signal);
      for (const [name, registration] of this.registrations) {
        const tool = this.tools.get(name);
        if (tool !== registration.tool || (this.options.visible && !this.options.visible(tool)) || tool.available?.(context)) {
          registration.lifetime.abort();
          this.registrations.delete(name);
        }
      }
      for (const tool of this.tools.values()) {
        if (this.disposed || this.registrations.has(tool.name) || (this.options.visible && !this.options.visible(tool)) || tool.available?.(context)) continue;
        const lifetime = new AbortController();
        const abort = () => lifetime.abort();
        this.lifetime.signal.addEventListener("abort", abort, { once: true });
        try {
          await this.options.provider.registerTool({
            name: tool.name,
            title: tool.title,
            description: tool.description,
            inputSchema: this.schema(tool),
            annotations: tool.annotations,
            execute: (input, options) => lifetime.signal.aborted || this.tools.get(tool.name) !== tool
              ? Promise.resolve({ ok: false, error: { code: "unavailable", message: "This registration expired. Discover tools again." } })
              : this.execute(tool.name, input, options?.signal),
          }, { signal: lifetime.signal });
          this.registrations.set(tool.name, { lifetime, tool });
          this.errors.delete(tool.name);
        } catch (error) {
          lifetime.abort();
          this.errors.set(tool.name, error instanceof Error ? error.message : "Registration failed");
        } finally {
          this.lifetime.signal.removeEventListener("abort", abort);
          if (this.disposed) lifetime.abort();
        }
      }
    }).catch(error => {
      this.errors.set("registration", error instanceof Error ? error.message : "Registration failed");
    });
    return this.refreshTail;
  }

  private schema(tool: AppTool): InputSchema {
    if (tool.annotations.readOnlyHint) return tool.inputSchema;
    const extend = (schema: InputSchema): InputSchema => ({
      ...schema,
      ...(schema.anyOf ? { anyOf: schema.anyOf.map(extend) } : {}),
      properties: {
        ...schema.properties,
        request_id: { type: "string", minLength: 1, maxLength: 128, description: "Reuse this ID when reconciling an uncertain call; changed inputs require a new ID." },
        expected_revision: { type: "string", description: "Revision from drowse_get_state; a mismatch prevents changes to a newer workspace." },
      },
    });
    return extend(tool.inputSchema);
  }

  async execute(name: string, value: unknown, signal = new AbortController().signal): Promise<unknown> {
    const epoch = this.dataEpoch;
    signal = AbortSignal.any([signal, this.lifetime.signal, this.actionLifetime.signal]);
    try {
      if (this.disposed) throw new ToolError("unavailable", "This page's tools have been disposed. Discover tools again.");
      const tool = this.tools.get(name);
      if (!tool) throw new ToolError("unavailable", "This tool is no longer available. Discover tools again.");
      validateInput(this.schema(tool), value);
      const input = structuredClone(value) as Record<string, unknown>;
      const serialized = JSON.stringify(canonicalInput(input));
      if (serialized.length > 2 * 1024 * 1024) throw new ToolError("invalid_input", "Use the file import interface for inputs larger than 2 MiB.");
      const key = !tool.annotations.readOnlyHint && typeof input.request_id === "string" ? `${name}:${input.request_id}` : null;
      const fingerprint = key ? requestFingerprint(serialized) : null;
      const run = async () => {
        signal.throwIfAborted();
        if (this.disposed || this.tools.get(name) !== tool) throw new ToolError("unavailable", "The page or runtime changed. Discover tools again.");
        const context = this.options.context(signal);
        const reason = tool.available?.(context);
        if (reason) throw new ToolError("unavailable", reason);
        if (input.expected_revision !== undefined && input.expected_revision !== this.options.revision?.()) {
          throw new ToolError("stale_state", "The workspace changed. Read its current state before applying this change.");
        }
        if (key && fingerprint) this.receipts?.start(key, fingerprint);
        delete input.expected_revision;
        let output: unknown;
        try {
          const result = await tool.execute(input, context);
          if (this.disposed || epoch !== this.dataEpoch) throw new DOMException("The page context changed during this action.", "AbortError");
          output = this.bound({ ok: true, data: result ?? null, ...(this.options.revision ? { revision: this.options.revision() } : {}) });
        } catch (error) {
          output = this.disposed || epoch !== this.dataEpoch
            ? { ok: false, error: { code: tool.annotations.readOnlyHint ? "cancelled" : "reconcile_required", message: "The page context changed during this action. It may have completed; read current state before retrying." } }
            : this.bound({ ok: false, error: actionError(error) });
        }
        if (key && fingerprint && !this.disposed && epoch === this.dataEpoch) this.receipts?.finish(key, fingerprint, output);
        return output;
      };
      if (tool.annotations.readOnlyHint) return await run();
      const prior = key ? this.calls.get(key) : null;
      if (prior) {
        if (prior.input !== fingerprint) throw new ToolError("request_conflict", "This request ID was already used with different inputs.");
        return await prior.result;
      }
      const recovered = key && fingerprint ? this.receipts?.get(key, fingerprint) : null;
      if (recovered) {
        if (recovered.state === "complete") return recovered.result;
        const job = this.options.context(signal).jobs.list().find(item => item.requestId === input.request_id);
        if (job) return this.bound({ ok: true, data: job, recovered: true, revision: this.options.revision?.() });
        throw new ToolError("reconcile_required", "This request was admitted before the page changed. Inspect current state or jobs to determine its outcome; it will not be automatically repeated.", { request_id: input.request_id, action: name });
      }
      const inputCharacters = serialized.length;
      if (this.pendingCalls >= 128 || this.pendingInputCharacters + inputCharacters > 8 * 1024 * 1024) {
        throw new ToolError("tool_busy", "The agent mutation queue is full. Wait for pending actions to finish before retrying.");
      }
      this.pendingCalls++;
      this.pendingInputCharacters += inputCharacters;
      let call: { input: string; result: Promise<unknown>; pending: boolean } | undefined;
      const result = this.mutationTail.then(run).finally(() => {
        this.pendingCalls--;
        this.pendingInputCharacters -= inputCharacters;
        if (call) call.pending = false;
        this.trimCalls();
      });
      this.mutationTail = result.catch(() => undefined);
      if (key && fingerprint) {
        call = { input: fingerprint, result, pending: true };
        this.calls.set(key, call);
        this.trimCalls();
      }
      return await result;
    } catch (error) {
      return this.bound({ ok: false, error: actionError(error) });
    }
  }

  private trimCalls(): void {
    for (const [key, call] of this.calls) {
      if (this.calls.size <= 128) break;
      if (!call.pending) this.calls.delete(key);
    }
  }

  private bound(value: unknown): unknown {
    const text = JSON.stringify(value);
    if (text.length <= 12_000) return JSON.parse(text);
    if (text.length > 8 * 1024 * 1024) return { ok: true, result_too_large: true, message: "Use a narrower query or the download/export interface to retrieve this result." };
    while (this.results.size >= 32 || this.resultBytes + text.length > 16 * 1024 * 1024) {
      const first = this.results.keys().next().value!;
      this.resultBytes -= this.results.get(first)!.length;
      this.results.delete(first);
    }
    const resultId = crypto.randomUUID();
    this.results.set(resultId, text);
    this.resultBytes += text.length;
    return { ok: true, result_id: resultId, encoding: "json", characters: text.length, preview: text.slice(0, 800), next: "drowse_read_result" };
  }

  resultTool(): AppTool {
    return {
      name: "drowse_read_result", title: "Read a result page", group: "context", scope: "public",
      description: "Read a bounded chunk of a previous large JSON result. Join text chunks in order before parsing. Handles last for this page only.",
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      inputSchema: objectSchema({ result_id: { type: "string" }, offset: { type: "integer", minimum: 0 }, limit: { type: "integer", minimum: 1, maximum: 8000 } }, ["result_id"]),
      execute: input => {
        const result = this.results.get(input.result_id as string);
        if (result === undefined) throw new ToolError("not_found", "Result expired. Run the read-only query again; reconcile mutations through their job or request ID.");
        const offset = (input.offset as number | undefined) ?? 0;
        const end = Math.min(result.length, offset + ((input.limit as number | undefined) ?? 4000));
        return { encoding: "json", text: result.slice(offset, end), next_offset: end < result.length ? end : null, characters: result.length };
      },
    };
  }

  clearRetainedData(): void {
    this.dataEpoch++;
    this.actionLifetime.abort();
    this.actionLifetime = new AbortController();
    this.results.clear();
    this.calls.clear();
    this.resultBytes = 0;
    this.receipts?.clear();
  }

  dispose(): void {
    this.disposed = true;
    this.lifetime.abort();
    for (const registration of this.registrations.values()) registration.lifetime.abort();
    this.registrations.clear();
    this.results.clear();
    this.calls.clear();
    this.resultBytes = 0;
  }
}

function actionError(error: unknown): { code: string; message: string; details?: unknown } {
  const value = error as { code?: unknown; status?: unknown; body?: unknown } | null;
  const body = value?.body && typeof value.body === "object" ? value.body as Record<string, unknown> : null;
  const nested = body?.error && typeof body.error === "object" ? body.error as Record<string, unknown> : null;
  const code = typeof value?.code === "string" ? value.code : typeof body?.code === "string" ? body.code
    : typeof nested?.code === "string" ? nested.code : typeof value?.status === "number" ? `HTTP_${value.status}`
    : error instanceof DOMException && error.name === "AbortError" ? "cancelled" : "action_failed";
  return { code, message: error instanceof Error ? error.message : "The action failed. Read current state before retrying.",
    ...(error instanceof ToolError && error.details !== undefined ? { details: error.details }
      : typeof value?.status === "number" ? { details: { status: value.status, ...(body ? { response: redactCredentials(body) } : {}) } } : {}) };
}

function redactCredentials(value: unknown, depth = 0): unknown {
  if (depth > 12) return "[nested detail omitted]";
  if (Array.isArray(value)) return value.slice(0, 100).map(item => redactCredentials(item, depth + 1));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).slice(0, 100).map(([key, item]) =>
    [key, /api.?key|authorization|password|secret|credential|access.?token|refresh.?token/i.test(key) ? "[redacted]" : redactCredentials(item, depth + 1)]));
  return typeof value === "string" && value.length > 8000 ? `${value.slice(0, 8000)} [truncated]` : value;
}

export function nativeModelContext(): ModelContextProvider | null {
  if (typeof document === "undefined") return null;
  const provider = (document as Document & { modelContext?: ModelContextProvider }).modelContext;
  return typeof provider?.registerTool === "function" ? provider : null;
}
