import { DurableObject } from "cloudflare:workers";
import { CONTACT_ADDRESS, CONTACT_REASONS, CONTACT_TOPICS, parseContact, validateContact, type ContactFields } from "../../src/lib/contact";

const MAX_BYTES = 40_000;
const RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const DEFAULT_DAILY_LIMIT = 200;
// One Durable Object instance (of the receipt class, with its own table) holds the global daily send count.
const DAILY_CAP_NAME = "global-daily-cap";
const TURNSTILE_VERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify";
// Bidi overrides/isolates and other format characters, C1 controls (including NEL), and line/paragraph
// separators can spoof or split the subject line. ZWNJ/ZWJ stay allowed for scripts and emoji sequences.
const UNSAFE_TITLE = /[\u0080-\u009f\u2028\u2029]|(?![\u200c\u200d])\p{Cf}/u;
type Result = { status: "sent" | "pending" | "failed" | "conflict" | "limited"; reference: string; retryAfter?: number };
type Receipt = { hash: string; status: string; message_id: string | null };
// Optional settings not yet in generated types: vars in wrangler.jsonc, TURNSTILE_SECRET as a Worker secret.
type Settings = { DAILY_LIMIT?: string; ALLOW_LOCAL_ORIGINS?: string; TURNSTILE_SECRET?: string };

export function allowedOrigin(origin: string | null, allowLocal = false): boolean {
  if (!origin) return false;
  try {
    const url = new URL(origin);
    if (url.origin !== origin) return false;
    return (url.protocol === "https:" && url.hostname === "drowse.ai")
      || (allowLocal && ["http:", "https:"].includes(url.protocol) && ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname));
  } catch { return false; }
}

export function titleError(title: string): string | undefined {
  return UNSAFE_TITLE.test(title) ? "Use plain text in the title, without hidden formatting characters." : undefined;
}

export function dailyLimit(value: string | undefined): number {
  const limit = Number(value);
  return value?.trim() && Number.isSafeInteger(limit) && limit >= 0 ? limit : DEFAULT_DAILY_LIMIT;
}

async function verifyTurnstile(secret: string, token: string, ip: string, reference: string): Promise<boolean | null> {
  try {
    const response = await fetch(TURNSTILE_VERIFY, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // The idempotency key lets a retry of the same reference re-verify its token after a transient failure.
      body: JSON.stringify({ secret, response: token, remoteip: ip, idempotency_key: reference }),
    });
    if (!response.ok) return null;
    const result: unknown = await response.json();
    return !!result && typeof result === "object" && "success" in result && result.success === true;
  } catch { return null; }
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
    // Only real send attempts count toward the global cap. A refused attempt is retryable under the same reference.
    const cap = await this.env.SUBMISSIONS.getByName(DAILY_CAP_NAME).consumeDaily(dailyLimit((this.env as Env & Settings).DAILY_LIMIT));
    if (!cap.allowed) {
      this.ctx.storage.sql.exec("UPDATE receipt SET status = 'failed' WHERE id = 1");
      console.warn(JSON.stringify({ event: "contact_daily_limit", reference }));
      return { status: "limited", reference, retryAfter: cap.retryAfter };
    }
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

  /** Used only on the DAILY_CAP_NAME instance. Synchronous, so concurrent callers cannot overshoot the limit. */
  consumeDaily(limit: number, now = Date.now()): { allowed: boolean; retryAfter: number } {
    const day = new Date(now).toISOString().slice(0, 10);
    const retryAfter = Math.max(1, Math.ceil((Date.parse(`${day}T00:00:00Z`) + 86_400_000 - now) / 1000));
    const sql = this.ctx.storage.sql;
    sql.exec("CREATE TABLE IF NOT EXISTS daily_count (day TEXT PRIMARY KEY, count INTEGER NOT NULL)");
    sql.exec("DELETE FROM daily_count WHERE day <> ?", day);
    const count = sql.exec<{ count: number }>("SELECT count FROM daily_count WHERE day = ?", day).toArray()[0]?.count ?? 0;
    if (count >= limit) return { allowed: false, retryAfter };
    sql.exec("INSERT INTO daily_count (day, count) VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET count = count + 1", day);
    return { allowed: true, retryAfter };
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
    const settings = env as Env & Settings;
    if (!allowedOrigin(origin, settings.ALLOW_LOCAL_ORIGINS === "true")) return reply({ error: "origin_not_allowed" }, 403);
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
      const unsafeTitle = errors.title ? undefined : titleError(fields.title);
      if (unsafeTitle) errors.title = unsafeTitle;
      if (Object.keys(errors).length) return reply({ error: "invalid_fields", fields: errors }, 400);
      // Turnstile is enforced only when its secret is configured. Verify before the shared limits so bots cannot drain them.
      if (settings.TURNSTILE_SECRET) {
        const token = "turnstileToken" in body && typeof body.turnstileToken === "string" ? body.turnstileToken : "";
        if (!token || token.length > 2048) return reply({ error: "verification_required" }, 400);
        const verified = await verifyTurnstile(settings.TURNSTILE_SECRET, token, ip, body.requestId);
        if (verified === null) return reply({ error: "verification_unavailable" }, 503);
        if (!verified) return reply({ error: "verification_failed" }, 403);
      }
      if (!(await env.VOLUME.limit({ key: "contact" })).success) { headers.set("Retry-After", "60"); return reply({ error: "rate_limited" }, 429); }
      const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(fields)));
      const hash = Array.from(new Uint8Array(bytes), byte => byte.toString(16).padStart(2, "0")).join("");
      const result = await env.SUBMISSIONS.getByName(body.requestId).submit(fields, body.requestId, hash);
      if (result.status === "limited") {
        headers.set("Retry-After", String(result.retryAfter));
        return reply({ error: "daily_limit_reached", reference: result.reference }, 429);
      }
      return reply(result, { sent: 200, pending: 202, failed: 502, conflict: 409 }[result.status]);
    } catch {
      console.error(JSON.stringify({ event: "contact_request_failed" }));
      return reply({ error: "contact_unavailable" }, 503);
    }
  },
} satisfies ExportedHandler<Env>;
