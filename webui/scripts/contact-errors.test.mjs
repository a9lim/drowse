import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const originalFetch = globalThis.fetch;
const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
try {
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { onLine: true } });
  const store = await server.ssrLoadModule("/src/lib/stores/contact.svelte.ts");
  const { CONTACT_REASONS, contactTopics } = await server.ssrLoadModule("/src/lib/contact.ts");

  for (const [value, label] of [["funding", "Funding and sponsorship"], ["teaching", "Teaching with Drowse"]]) {
    assert.ok(CONTACT_REASONS.some(reason => reason.value === value && reason.label === label), `${value} is a contact reason`);
    assert.ok(contactTopics(value).length >= 2, `${value} has its own topics plus Something else`);
  }

  const respond = (status, body, type = "application/json") => {
    globalThis.fetch = async () => new Response(typeof body === "string" ? body : JSON.stringify(body), { status, headers: { "Content-Type": type } });
  };
  const fill = () => {
    store.resetContact("contact");
    Object.assign(store.contactDrafts.contact.fields, { reason: "support", topic: "general", title: "Help", body: "A question" });
    return store.contactDrafts.contact;
  };

  let current = fill();
  respond(400, { error: "invalid_fields", fields: { title: "Add a short title.", body: 7, unknown: "x" } });
  await store.submitContact("contact");
  assert.deepEqual({ ...current.errors }, { title: "Add a short title." }, "server field errors reach the matching fields only");
  assert.match(current.message, /Fix the marked fields/);

  current = fill();
  respond(503, { error: "contact_unavailable" });
  await store.submitContact("contact");
  assert.match(current.message, /turned off for now/);
  assert.match(current.message, /contact@drowse\.ai/);

  current = fill();
  respond(502, "<html>Bad gateway</html>", "text/html");
  await store.submitContact("contact");
  assert.match(current.message, /contact service isn't responding/, "an HTML error page is a server problem, not a connection problem");
  assert.equal(current.uncertain, true);

  current = fill();
  globalThis.fetch = async () => { throw new TypeError("Failed to fetch"); };
  await store.submitContact("contact");
  assert.match(current.message, /couldn't reach Drowse\. Check your connection/);

  current = fill();
  respond(413, { error: "invalid_request" });
  await store.submitContact("contact");
  assert.ok(current.errors.body, "an oversized request marks the message field");

  current = fill();
  globalThis.fetch = async (_url, init) => new Response(JSON.stringify({ status: "sent", reference: JSON.parse(init.body).requestId }), { status: 200, headers: { "Content-Type": "application/json" } });
  await store.submitContact("contact");
  assert.equal(current.status, "sent");

  const form = await readFile("src/lib/ui/ContactForm.svelte", "utf8");
  assert.match(form, /ariaLabelledby=\{`\$\{uid\}-reason-label`\}/, "Reason announces its label and current value");
  assert.match(form, /All fields are required unless marked optional\./);
  const select = await readFile("src/lib/Select.svelte", "utf8");
  assert.match(select, /aria-labelledby=\{ariaLabelledby \? `\$\{ariaLabelledby\} \$\{uid\}-value` : undefined\}/);
  const storeSource = await readFile("src/lib/stores/contact.svelte.ts", "utf8");
  assert.match(storeSource, /sessionStorage/, "drafts survive a reload in the same tab");
  assert.match(storeSource, /removeItem\(key\)/, "a sent draft is cleared from storage");
  const contact = await readFile("src/hosted/ui/Contact.svelte", "utf8");
  assert.match(contact, /\{#if mounted\}<PwaUpdatePrompt \/>\{\/if\}/);
  assert.doesNotMatch(contact, /<svelte:head>/, "Contact updates the existing description instead of adding a second one");
  console.log("Contact: new reasons, server field errors, unavailable form, server vs network errors, and labelled selects passed");
} finally {
  globalThis.fetch = originalFetch;
  if (originalNavigator) Object.defineProperty(globalThis, "navigator", originalNavigator);
  await server.close();
}
