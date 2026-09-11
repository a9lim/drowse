import { env } from "cloudflare:workers";
import { createExecutionContext, evictDurableObject, runDurableObjectAlarm, runInDurableObject } from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";
import worker, { allowedOrigin, emailContent } from "../src/index";
import { emptyContact, parseContact, validateContact } from "../../src/lib/contact";

const fields = () => ({ ...emptyContact("feedback"), title: "A useful improvement", body: "Please make this easier to find." });
const request = (body: unknown, overrides: RequestInit = {}) => new Request("https://drowse.ai/api/contact", {
  method: "POST", headers: { Origin: "https://drowse.ai", "Content-Type": "application/json", "CF-Connecting-IP": "192.0.2.7" },
  body: JSON.stringify(body), ...overrides,
});
const enabled = () => ({ ...env, CONTACT_ENABLED: "true" });
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
    ["research", "funding", "Funding & Sponsorship"],
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
  it("allows only the public site and loopback origins", () => {
    for (const origin of ["https://drowse.ai", "http://localhost:3000", "http://127.0.0.1:4176", "http://[::1]:8080"]) expect(allowedOrigin(origin)).toBe(true);
    for (const origin of [null, "null", "https://evil.test", "https://drowse.ai.evil.test", "https://drowse.ai/path", "http://drowse.ai"]) expect(allowedOrigin(origin)).toBe(false);
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
