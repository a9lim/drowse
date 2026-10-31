// WebMCP is an experimental browser API.  Its tool registry (about 90 KB of
// schemas and handlers) loads only on pages that expose ``document.modelContext``.
type WebMcpModule = typeof import("./index");

let module: Promise<WebMcpModule> | null = null;

/** The WebMCP module when the browser supports it, else null. */
export function loadWebMcp(): Promise<WebMcpModule> | null {
  if (typeof document === "undefined") return null;
  const provider = (document as Document & { modelContext?: { registerTool?: unknown } }).modelContext;
  if (typeof provider?.registerTool !== "function") return null;
  return module ??= import("./index");
}
