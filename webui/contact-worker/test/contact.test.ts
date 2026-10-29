import { env } from "cloudflare:workers";
import { createExecutionContext, evictDurableObject, runDurableObjectAlarm, runInDurableObject } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import worker, { allowedOrigin, dailyLimit, emailContent, titleError } from "../src/index";
import { emptyContact, parseContact, validateContact } from "../../src/lib/contact";

const fields = () => ({ ...emptyContact("feedback"), title: "A useful improvement", body: "Please make this easier to find." });
const request = (body: unknown, overrides: RequestInit = {}) => new Request("https://drowse.ai/api/contact", {
  method: "POST", headers: { Origin: "https://drowse.ai", "Content-Type": "application/json", "CF-Connecting-IP": "192.0.2.7" },
  body: JSON.stringify(body), ...overrides,
});
const enabled = (overrides: Record<string, string> = {}) => ({ ...env, CONTACT_ENABLED: "true", ...overrides });
const allowAll = () => {
  vi.spyOn(env.ATTEMPTS, "limit").mockResolvedValue({ success: true });
  vi.spyOn(env.VOLUME, "limit").mockResolvedValue({ success: true });
};
const today = () => new Date().toISOString().slice(0, 10);
const capStub = () => env.SUBMISSIONS.getByName("global-daily-cap");
const setDailyCount = (count: number | null) => runInDurableObject(capStub(), (_instance, state) => {
  state.storage.sql.exec("CREATE TABLE IF NOT EXISTS daily_count (day TEXT PRIMARY KEY, count INTEGER NOT NULL)");
  state.storage.sql.exec("DELETE FROM daily_count");
  if (count !== null) state.storage.sql.exec("INSERT INTO daily_count (day, count) VALUES (?, ?)", today(), count);
});
afterEach(() => vi.restoreAllMocks());

describe("contact validation and email", () => {
  it("allows an omitted reply address and every field round-trips", () => {
    expect(validateContact(fields())).toEqual({});
    expect(parseContact({ ...fields(), recipient: "ignored@example.com" })).toEqual(fields());
    expect(parseContact({ ...fields(), title: 3 })).toBeNull();
    expect(parseContact(null)).toBeNull();
  });
  it.each([
    ["email", "someone\r\nBcc: x@example.com"], ["email", "broken"], ["email", "user\u0001@example.com"],
    ["title", "\nBcc: person@example.com"], ["title", " "], ["title", "x".repeat(121)],
    ["body", " "], ["body", "x".repeat(8001)], ["topic", "not-real"], ["reason", "support"],
  ])("rejects invalid %s", (key, value) => expect(validateContact({ ...fields(), [key]: value })).toHaveProperty(key));
  it.each([
    ["support", "getting-started", "Getting started"],
    ["bug", "models", "Models and downloads"],
    ["feature", "python", "Python library and API"],
    ["media", "interview", "Interview request"],
    ["research", "study", "Using Drowse in a study"],
    ["research", "funding", "Grants and research funding"],
    ["funding", "sponsorship", "Sponsoring Drowse"],
    ["teaching", "course", "Using Drowse in a course"],
    ["other", "none", "Not applicable"],
  ])("accepts %s topics and keeps their labels in the email", (reason, topic, label) => {
    const submission = { ...fields(), source: "contact" as const, reason, topic };
    expect(validateContact(submission)).toEqual({});
    expect(emailContent(submission, "reference", "now").text).toContain(`Topic: ${label}`);
  });
  it.each([
    ["feedback", "interview"], ["support", "study"], ["bug", "getting-started"],
    ["feature", "none"], ["media", "funding"], ["other", "general"], ["media", "chat"], ["research", "models"], ["other", "press-materials"],
  ])("rejects a topic that does not belong to %s", (reason, topic) => {
    expect(validateContact({ ...fields(), source: "contact", reason, topic })).toHaveProperty("topic");
  });
  it("fixes sender and recipient, escapes HTML, and includes all submitted details", () => {
    const message = emailContent({ ...fields(), source: "contact", reason: "media", topic: "press-materials", email: " reader@example.com ", body: '<script>"test" & text</script>' }, "reference", "2026-09-07T00:00:00Z");
    expect(message.to).toBe("contact@drowse.ai");
    expect(message.from.email).toBe("feedback@drowse.ai");
    expect(message.replyTo).toBe("reader@example.com");
    expect(message.text).toContain("Media and press");
    expect(message.text).toContain("Images and press materials");
    expect(message.text).toContain("reference");
    expect(message.html).not.toContain("<script>");
    expect(message.html).toContain("&lt;script&gt;");
    expect(emailContent(fields(), "reference", "now")).not.toHaveProperty("replyTo");
  });
  it("allows only the public site, plus loopback origins when enabled", () => {
    const loopback = ["http://localhost:3000", "http://127.0.0.1:4176", "http://[::1]:8080"];
    for (const origin of ["https://drowse.ai", ...loopback]) expect(allowedOrigin(origin, true)).toBe(true);
    expect(allowedOrigin("https://drowse.ai")).toBe(true);
    for (const origin of loopback) expect(allowedOrigin(origin)).toBe(false);
    for (const origin of [null, "null", "https://evil.test", "https://drowse.ai.evil.test", "https://drowse.ai/path", "http://drowse.ai", "https://www.drowse.ai"]) expect(allowedOrigin(origin, true)).toBe(false);
  });
  it.each([
    ["right-to-left override", "Invoice \u202Egnp.exe"], ["isolate", "A \u2066hidden\u2069 title"], ["zero-width space", "a\u200Bb"],
    ["byte order mark", "\uFEFFtitle"], ["line separator", "one\u2028two"], ["paragraph separator", "one\u2029two"],
    ["next line", "one\u0085two"], ["C1 control", "one\u009Btwo"],
  ])("rejects a title containing a %s", (_name, title) => {
    expect(titleError(title)).toBeDefined();
  });
  it("keeps ordinary Unicode titles, including emoji joiner sequences", () => {
    for (const title of ["A useful improvement", "Café ☕ résumé", "\u05E9\u05DC\u05D5\u05DD", "Family \u{1F468}\u200D\u{1F469}\u200D\u{1F467}"]) expect(titleError(title)).toBeUndefined();
  });
  it("parses the daily limit with a safe default", () => {
    expect(dailyLimit(undefined)).toBe(200);
    expect(dailyLimit("")).toBe(200);
    expect(dailyLimit("abc")).toBe(200);
    expect(dailyLimit("-1")).toBe(200);
    expect(dailyLimit("1.5")).toBe(200);
    expect(dailyLimit("0")).toBe(0);
    expect(dailyLimit("50")).toBe(50);
  });
});

describe("HTTP boundary", () => {
  it("fails closed until the sender is enabled", async () => {
    const result = await worker.fetch(request(fields()), env, createExecutionContext());
    expect(result.status).toBe(503);
    expect(result.headers.get("Cache-Control")).toBe("no-store");
  });
  it("preflights without credentials and refuses unknown origins", async () => {
    const preflight = await worker.fetch(request(undefined, { method: "OPTIONS", body: undefined }), env, createExecutionContext());
    expect(preflight.status).toBe(204);
    expect(preflight.headers.get("Access-Control-Allow-Origin")).toBe("https://drowse.ai");
    expect(preflight.headers.has("Access-Control-Allow-Credentials")).toBe(false);
    const blocked = await worker.fetch(request(fields(), { headers: { Origin: "https://evil.test" } }), enabled(), createExecutionContext());
    expect(blocked.status).toBe(403);
    expect(blocked.headers.has("Access-Control-Allow-Origin")).toBe(false);
  });
  it("rejects malformed, overlong, non-JSON, and honeypot submissions", async () => {
    const attempts = vi.spyOn(env.ATTEMPTS, "limit").mockResolvedValue({ success: true });
    const bad = [
      request(null), request({ ...fields(), requestId: "no" }),
      request({ ...fields(), requestId: crypto.randomUUID(), website: "bot" }),
      request({ ...fields(), requestId: crypto.randomUUID(), title: "\r\nInjected" }),
      request({ ...fields(), source: "contact", reason: "media", topic: "chat", requestId: crypto.randomUUID() }),
    ];
    for (const req of bad) expect((await worker.fetch(req, enabled(), createExecutionContext())).status).toBe(400);
    const large = request("x".repeat(41_000));
    expect((await worker.fetch(large, enabled(), createExecutionContext())).status).toBe(413);
    const nonJson = request(fields()); nonJson.headers.set("Content-Type", "text/plain");
    expect((await worker.fetch(nonJson, enabled(), createExecutionContext())).status).toBe(415);
    expect(attempts).toHaveBeenCalled();
  });
  it("accepts loopback origins only when ALLOW_LOCAL_ORIGINS is true", async () => {
    const local = { headers: { Origin: "http://localhost:8000", "Content-Type": "application/json", "CF-Connecting-IP": "192.0.2.7" } };
    const blocked = await worker.fetch(request(fields(), local), enabled({ ALLOW_LOCAL_ORIGINS: "false" }), createExecutionContext());
    expect(blocked.status).toBe(403);
    expect(await blocked.json()).toEqual({ error: "origin_not_allowed" });
    const preflight = await worker.fetch(request(undefined, { ...local, method: "OPTIONS", body: undefined }), enabled({ ALLOW_LOCAL_ORIGINS: "true" }), createExecutionContext());
    expect(preflight.status).toBe(204);
    expect(preflight.headers.get("Access-Control-Allow-Origin")).toBe("http://localhost:8000");
  });
  it("rejects hidden formatting characters in the title at the HTTP boundary", async () => {
    allowAll();
    const send = vi.spyOn(env.EMAIL, "send");
    const result = await worker.fetch(request({ ...fields(), title: "Invoice \u202Egnp.exe", requestId: crypto.randomUUID() }), enabled(), createExecutionContext());
    expect(result.status).toBe(400);
    expect(await result.json()).toMatchObject({ error: "invalid_fields", fields: { title: expect.any(String) } });
    expect(send).not.toHaveBeenCalled();
  });
  it("enforces rate limits without invoking email", async () => {
    vi.spyOn(env.ATTEMPTS, "limit").mockResolvedValue({ success: false });
    const result = await worker.fetch(request({ ...fields(), requestId: crypto.randomUUID() }), enabled(), createExecutionContext());
    expect(result.status).toBe(429);
    expect(result.headers.get("Retry-After")).toBe("60");
  });
  it("returns success only with the reference accepted by the sender", async () => {
    vi.spyOn(env.ATTEMPTS, "limit").mockResolvedValue({ success: true });
    vi.spyOn(env.VOLUME, "limit").mockResolvedValue({ success: true });
    vi.spyOn(env.EMAIL, "send").mockResolvedValue({ messageId: "provider-receipt" });
    const requestId = crypto.randomUUID();
    const result = await worker.fetch(request({ ...fields(), requestId }), enabled(), createExecutionContext());
    expect(result.status).toBe(200);
    expect(await result.json()).toEqual({ status: "sent", reference: requestId });
  });
});

describe("Turnstile verification", () => {
  const siteverify = (body: unknown, status = 200) => vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json(body, { status }));
  it("is not required when no secret is configured", async () => {
    allowAll();
    const verify = vi.spyOn(globalThis, "fetch");
    vi.spyOn(env.EMAIL, "send").mockResolvedValue({ messageId: "provider-receipt" });
    const result = await worker.fetch(request({ ...fields(), requestId: crypto.randomUUID() }), enabled({ TURNSTILE_SECRET: "" }), createExecutionContext());
    expect(result.status).toBe(200);
    expect(verify).not.toHaveBeenCalled();
  });
  it("requires a token when a secret is configured", async () => {
    allowAll();
    const verify = siteverify({ success: true });
    const result = await worker.fetch(request({ ...fields(), requestId: crypto.randomUUID() }), enabled({ TURNSTILE_SECRET: "secret" }), createExecutionContext());
    expect(result.status).toBe(400);
    expect(await result.json()).toEqual({ error: "verification_required" });
    expect(verify).not.toHaveBeenCalled();
  });
  it("rejects a failed token before shared limits or email", async () => {
    vi.spyOn(env.ATTEMPTS, "limit").mockResolvedValue({ success: true });
    const volume = vi.spyOn(env.VOLUME, "limit").mockResolvedValue({ success: true });
    const send = vi.spyOn(env.EMAIL, "send");
    siteverify({ success: false, "error-codes": ["invalid-input-response"] });
    const result = await worker.fetch(request({ ...fields(), requestId: crypto.randomUUID(), turnstileToken: "bad" }), enabled({ TURNSTILE_SECRET: "secret" }), createExecutionContext());
    expect(result.status).toBe(403);
    expect(await result.json()).toEqual({ error: "verification_failed" });
    expect(volume).not.toHaveBeenCalled();
    expect(send).not.toHaveBeenCalled();
  });
  it("fails closed when siteverify is unreachable", async () => {
    allowAll();
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network"));
    const result = await worker.fetch(request({ ...fields(), requestId: crypto.randomUUID(), turnstileToken: "token" }), enabled({ TURNSTILE_SECRET: "secret" }), createExecutionContext());
    expect(result.status).toBe(503);
    expect(await result.json()).toEqual({ error: "verification_unavailable" });
  });
  it("sends after a verified token and passes the secret, token, IP, and reference", async () => {
    allowAll();
    const verify = siteverify({ success: true });
    vi.spyOn(env.EMAIL, "send").mockResolvedValue({ messageId: "provider-receipt" });
    const requestId = crypto.randomUUID();
    const result = await worker.fetch(request({ ...fields(), requestId, turnstileToken: "token" }), enabled({ TURNSTILE_SECRET: "secret" }), createExecutionContext());
    expect(result.status).toBe(200);
    expect(await result.json()).toEqual({ status: "sent", reference: requestId });
    const [url, init] = verify.mock.calls[0];
    expect(String(url)).toBe("https://challenges.cloudflare.com/turnstile/v0/siteverify");
    expect(JSON.parse(String(init?.body))).toEqual({ secret: "secret", response: "token", remoteip: "192.0.2.7", idempotency_key: requestId });
  });
});

describe("global daily cap", () => {
  it("counts per UTC day and resets on the next day", async () => {
    await setDailyCount(null);
    const stub = capStub();
    const now = Date.parse("2026-09-07T23:59:00Z");
    expect(await stub.consumeDaily(2, now)).toEqual({ allowed: true, retryAfter: 60 });
    expect((await stub.consumeDaily(2, now)).allowed).toBe(true);
    expect(await stub.consumeDaily(2, now)).toEqual({ allowed: false, retryAfter: 60 });
    expect((await stub.consumeDaily(2, now + 120_000)).allowed).toBe(true);
    expect((await stub.consumeDaily(0, now + 120_000)).allowed).toBe(false);
    await setDailyCount(null);
  });
  it("refuses sends over the cap with a clear code and allows a retry under the same reference", async () => {
    allowAll();
    const send = vi.spyOn(env.EMAIL, "send").mockResolvedValue({ messageId: "provider-receipt" });
    const requestId = crypto.randomUUID();
    try {
      await setDailyCount(dailyLimit((env as typeof env & { DAILY_LIMIT?: string }).DAILY_LIMIT));
      const limited = await worker.fetch(request({ ...fields(), requestId }), enabled(), createExecutionContext());
      expect(limited.status).toBe(429);
      expect(await limited.json()).toEqual({ error: "daily_limit_reached", reference: requestId });
      expect(Number(limited.headers.get("Retry-After"))).toBeGreaterThan(0);
      expect(send).not.toHaveBeenCalled();
      await setDailyCount(null);
      const retried = await worker.fetch(request({ ...fields(), requestId }), enabled(), createExecutionContext());
      expect(retried.status).toBe(200);
      expect(send).toHaveBeenCalledTimes(1);
      const counted = await runInDurableObject(capStub(), (_instance, state) => state.storage.sql.exec("SELECT count FROM daily_count").toArray());
      expect(counted).toEqual([{ count: 1 }]);
    } finally { await setDailyCount(null); }
  });
});

describe("durable delivery receipts", () => {
  it("deduplicates across retries and eviction without storing form text", async () => {
    const send = vi.spyOn(env.EMAIL, "send").mockResolvedValue({ messageId: "provider-id" });
    const reference = crypto.randomUUID();
    const stub = env.SUBMISSIONS.getByName(reference);
    expect(await stub.submit(fields(), reference, "hash")).toEqual({ status: "sent", reference });
    expect(await stub.submit(fields(), reference, "hash")).toEqual({ status: "sent", reference });
    await evictDurableObject(stub);
    expect(await stub.submit(fields(), reference, "hash")).toEqual({ status: "sent", reference });
    expect(send).toHaveBeenCalledTimes(1);
    expect(await stub.submit(fields(), reference, "changed")).toEqual({ status: "conflict", reference });
    const rows = await runInDurableObject(stub, (_instance, state) => state.storage.sql.exec("SELECT * FROM receipt").toArray());
    expect(rows).toEqual([{ id: 1, hash: "hash", status: "sent", message_id: "provider-id" }]);
    expect(await runDurableObjectAlarm(stub)).toBe(true);
    expect(await runInDurableObject(stub, (_instance, state) => state.storage.sql.exec("SELECT * FROM receipt").toArray())).toEqual([]);
  });
  it("does not send a second email while the first send is in flight", async () => {
    const reference = crypto.randomUUID();
    const stub = env.SUBMISSIONS.getByName(reference);
    await runInDurableObject(stub, async instance => {
      let accept!: (value: { messageId: string }) => void;
      const send = vi.spyOn(env.EMAIL, "send").mockImplementation(() => new Promise(resolve => { accept = resolve; }));
      const first = instance.submit(fields(), reference, "hash");
      await vi.waitFor(() => expect(send).toHaveBeenCalledTimes(1));
      expect(await instance.submit(fields(), reference, "hash")).toEqual({ status: "pending", reference });
      accept({ messageId: "provider-id" });
      expect((await first).status).toBe("sent");
      expect(send).toHaveBeenCalledTimes(1);
    });
  });
  it("retries explicit rejection but never blindly retries an ambiguous provider result", async () => {
    const send = vi.spyOn(env.EMAIL, "send").mockRejectedValueOnce(Object.assign(new Error("private provider detail"), { code: "E_RATE_LIMIT_EXCEEDED" })).mockResolvedValue({ messageId: "provider-id" });
    const reference = crypto.randomUUID();
    const stub = env.SUBMISSIONS.getByName(reference);
    expect((await stub.submit(fields(), reference, "hash")).status).toBe("failed");
    expect((await stub.submit(fields(), reference, "hash")).status).toBe("sent");
    expect(send).toHaveBeenCalledTimes(2);
    send.mockRejectedValue(new Error("connection lost"));
    const uncertain = env.SUBMISSIONS.getByName(crypto.randomUUID());
    expect((await uncertain.submit(fields(), reference, "hash")).status).toBe("pending");
    expect((await uncertain.submit(fields(), reference, "hash")).status).toBe("pending");
    expect(send).toHaveBeenCalledTimes(3);
  });
});
