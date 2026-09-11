import { DurableObject } from "cloudflare:workers";
import { CONTACT_ADDRESS, CONTACT_REASONS, CONTACT_TOPICS, parseContact, validateContact, type ContactFields } from "../../src/lib/contact";

const MAX_BYTES = 40_000;
const RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
type Result = { status: "sent" | "pending" | "failed" | "conflict"; reference: string };
type Receipt = { hash: string; status: string; message_id: string | null };

export function allowedOrigin(origin: string | null): boolean {
  if (!origin) return false;
  try {
    const url = new URL(origin);
    if (url.origin !== origin) return false;
    return (url.protocol === "https:" && ["drowse.ai", "www.drowse.ai"].includes(url.hostname))
      || (["http:", "https:"].includes(url.protocol) && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname));
  } catch { return false; }
}

export function emailContent(fields: ContactFields, reference: string, receivedAt: string) {
  const reason = CONTACT_REASONS.find(option => option.value === fields.reason)!.label;
  const topic = CONTACT_TOPICS.find(option => option.value === fields.topic)!.label;
  const text = [
    fields.title.trim(), "",
    `From form: ${fields.source}`, `Reason: ${reason}`, `Topic: ${topic}`,
    `Reply email: ${fields.email.trim() || "Not provided"}`,
    `Received: ${receivedAt}`, `Reference: ${reference}`, "", fields.body.trim(),
  ].join("\n");
  const escape = (value: string) => value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!);
  return {
    to: CONTACT_ADDRESS,
    from: { email: "feedback@drowse.ai", name: "Drowse contact form" },
    ...(fields.email.trim() ? { replyTo: fields.email.trim() } : {}),
    subject: `[Drowse · ${reason}] ${fields.title.trim()}`,
    text,
    html: `<div style="font-family:system-ui,sans-serif;line-height:1.6;white-space:pre-wrap">${escape(text)}</div>`,
  };
}

export class ContactSubmission extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS receipt (id INTEGER PRIMARY KEY CHECK (id = 1), hash TEXT NOT NULL, status TEXT NOT NULL, message_id TEXT)");
  }

  async submit(fields: ContactFields, reference: string, hash: string): Promise<Result> {
    const existing = this.ctx.storage.sql.exec<Receipt>("SELECT hash, status, message_id FROM receipt WHERE id = 1").toArray()[0];
    if (existing && existing.hash !== hash) return { status: "conflict", reference };
    if (existing?.status === "sent") return { status: "sent", reference };
    if (existing?.status === "pending") return { status: "pending", reference };

    // Claim before external I/O; a retry can observe pending but cannot send twice.
    this.ctx.storage.sql.exec("INSERT INTO receipt (id, hash, status) VALUES (1, ?, 'pending') ON CONFLICT(id) DO UPDATE SET status = 'pending'", hash);
    await this.ctx.storage.setAlarm(Date.now() + RETENTION_MS);
    let messageId: string;
    try {
      const result = await this.env.EMAIL.send(emailContent(fields, reference, new Date().toISOString()));
      messageId = result.messageId;
      if (!messageId) throw new Error("missing_receipt");
    } catch (error) {
      const code = error && typeof error === "object" && "code" in error && typeof error.code === "string" ? error.code : "unknown";
      // Only explicit provider rejections are retryable. Unknown outcomes stay pending.
      const rejected = ["E_VALIDATION_ERROR", "E_FIELD_MISSING", "E_SENDER_NOT_VERIFIED", "E_RECIPIENT_NOT_ALLOWED", "E_RECIPIENT_SUPPRESSED", "E_SENDER_DOMAIN_NOT_AVAILABLE", "E_RATE_LIMIT_EXCEEDED", "E_DAILY_LIMIT_EXCEEDED", "E_DELIVERY_FAILED"].includes(code);
      if (rejected) this.ctx.storage.sql.exec("UPDATE receipt SET status = 'failed' WHERE id = 1");
      console.error(JSON.stringify({ event: "contact_send_failed", reference, code }));
      return { status: rejected ? "failed" : "pending", reference };
    }
    this.ctx.storage.sql.exec("UPDATE receipt SET status = 'sent', message_id = ? WHERE id = 1", messageId);
    console.info(JSON.stringify({ event: "contact_accepted", reference, messageId }));
    return { status: "sent", reference };
  }

  alarm(): void { this.ctx.storage.sql.exec("DELETE FROM receipt"); }
}

async function boundedJson(request: Request): Promise<unknown> {
  if (Number(request.headers.get("Content-Length")) > MAX_BYTES) throw new RangeError();
  if (!request.body) throw new SyntaxError();
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BYTES) { await reader.cancel(); throw new RangeError(); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return JSON.parse(new TextDecoder().decode(bytes));
}

export default {
  async fetch(request, env): Promise<Response> {
    const origin = request.headers.get("Origin");
    const headers = new Headers({ "Cache-Control": "no-store", "Vary": "Origin", "X-Content-Type-Options": "nosniff" });
    const reply = (body: unknown, status: number) => Response.json(body, { status, headers });
    if (new URL(request.url).pathname !== "/api/contact") return reply({ error: "not_found" }, 404);
    if (!allowedOrigin(origin)) return reply({ error: "origin_not_allowed" }, 403);
    headers.set("Access-Control-Allow-Origin", origin!);
    if (request.method === "OPTIONS") {
      headers.set("Access-Control-Allow-Methods", "POST, OPTIONS");
      headers.set("Access-Control-Allow-Headers", "Content-Type");
      headers.set("Access-Control-Max-Age", "600");
      return new Response(null, { status: 204, headers });
    }
    if (request.method !== "POST") { headers.set("Allow", "POST, OPTIONS"); return reply({ error: "method_not_allowed" }, 405); }
    if (env.CONTACT_ENABLED !== "true") return reply({ error: "contact_unavailable" }, 503);
    if (request.headers.get("Content-Type")?.split(";")[0].trim() !== "application/json") return reply({ error: "json_required" }, 415);
    const ip = request.headers.get("CF-Connecting-IP");
    if (!ip) return reply({ error: "client_address_required" }, 403);
    try {
      const limited = await env.ATTEMPTS.limit({ key: ip });
      if (!limited.success) { headers.set("Retry-After", "60"); return reply({ error: "rate_limited" }, 429); }
      let body: unknown;
      try { body = await boundedJson(request); }
      catch (error) { return reply({ error: "invalid_request" }, error instanceof RangeError ? 413 : 400); }
      const fields = parseContact(body);
      if (!fields || !body || typeof body !== "object" || !("requestId" in body) || typeof body.requestId !== "string" || !UUID.test(body.requestId)) return reply({ error: "invalid_request" }, 400);
      if (fields.website) return reply({ error: "invalid_request" }, 400);
      const errors = validateContact(fields);
      if (Object.keys(errors).length) return reply({ error: "invalid_fields", fields: errors }, 400);
      if (!(await env.VOLUME.limit({ key: "contact" })).success) { headers.set("Retry-After", "60"); return reply({ error: "rate_limited" }, 429); }
      const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(fields)));
      const hash = Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, "0")).join("");
      const result = await env.SUBMISSIONS.getByName(body.requestId).submit(fields, body.requestId, hash);
      return reply(result, { sent: 200, pending: 202, failed: 502, conflict: 409 }[result.status]);
    } catch {
      console.error(JSON.stringify({ event: "contact_request_failed" }));
      return reply({ error: "contact_unavailable" }, 503);
    }
  },
} satisfies ExportedHandler<Env>;
