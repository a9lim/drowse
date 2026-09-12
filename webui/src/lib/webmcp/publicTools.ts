import { CONTACT_ADDRESS, CONTACT_LIMITS, CONTACT_REASONS, contactTopics, validateContact, type ContactSource } from "../contact";
import { contactDrafts, resetContact, submitContact } from "../stores/contact.svelte";
import { setTheme } from "../theme";
import { pageActions, updateActions } from "./lifecycle";
import { objectSchema, ToolError, type AppTool } from "./types";

const sourceSchema = { type: "string", enum: ["contact", "feedback"] };
const contactSource = (input: Record<string, unknown>): ContactSource => (input.source as ContactSource | undefined) ?? "contact";

export function createPublicTools(hostedSite: boolean): AppTool[] {
  const tool = (definition: Omit<AppTool, "scope" | "group"> & { group?: string }): AppTool => ({ scope: "public", group: "site", ...definition });
  return [
    tool({
      name: "drowse_about", title: "About Drowse", annotations: { readOnlyHint: true }, inputSchema: objectSchema(),
      description: "Read what Drowse does, its runtime modes, and where to start. Behavior settings and steering create reusable configurations without changing model weights.",
      execute: () => ({
        description: "Drowse runs language models, shapes responses with activation steering, and inspects their internal representations through geometry, lens, and SAE instruments.",
        runtime: hostedSite ? "Hosted app: inference and saved data live in this browser." : "Python dashboard: the connected server owns the model.",
        examples: ["Speak like a pirate: use style instructions for a chat model.", "Compare activation interventions: discover compatible controls and compare matched branches.", "Base models: raw continuations use a text scaffold or compatible steering."],
        links: { app: hostedSite ? "/app" : "/", documentation: "https://github.com/a9lim/drowse", contact: "https://drowse.ai/contact", credits: "https://drowse.ai/credits" },
      }),
    }),
    tool({
      name: "drowse_open_page", title: "Open a Drowse page", annotations: { readOnlyHint: false },
      description: "Navigate to Drowse's app, saved chats, model picker, contact or credits. Active hosted work is saved before leaving the workspace.",
      inputSchema: objectSchema({ page: { type: "string", enum: ["home", "app", "models", "chats", "contact", "credits"] } }, ["page"]),
      execute: async (input, context) => {
        const destination = input.page as string;
        const page = pageActions();
        if (page && (destination === "chats" || destination === "models")) {
          await (destination === "models" ? page.models() : page.home());
          return page.state();
        }
        const paths: Record<string, string> = { home: "/", app: hostedSite ? "/app" : "/", models: "/app?choose=1", chats: "/app", contact: "/contact", credits: "/credits" };
        const local = hostedSite || destination === "app" || destination === "home";
        const url = new URL(paths[destination], local ? window.location.origin : "https://drowse.ai").href;
        if (page) await page.leave(url);
        else {
          if (context.runtime) await (await import("../stores/savedConversations.svelte")).flushConversationAutosave();
          window.setTimeout(() => window.location.assign(url), 0);
        }
        return { state: "navigating", url };
      },
    }),
    tool({
      name: "drowse_set_theme", title: "Set theme", annotations: { readOnlyHint: false },
      description: "Set the visible light or dark theme. This changes presentation only.",
      inputSchema: objectSchema({ theme: { type: "string", enum: ["light", "dark"] } }, ["theme"]),
      execute: input => { setTheme(input.theme as "light" | "dark"); return { theme: input.theme }; },
    }),
    tool({
      name: "drowse_contact_draft", title: "Read or edit contact draft", group: "contact", annotations: { readOnlyHint: false, untrustedContentHint: true },
      description: "Prepare a contact or feedback message without sending it. Topics depend on reason. Explicit sending uses drowse_send_contact and transmits the draft to Drowse support.",
      inputSchema: objectSchema({ source: sourceSchema, reason: { type: "string", enum: CONTACT_REASONS.map(row => row.value) }, topic: { type: "string" }, email: { type: "string", maxLength: CONTACT_LIMITS.email }, title: { type: "string", maxLength: CONTACT_LIMITS.title }, body: { type: "string", maxLength: CONTACT_LIMITS.body } }),
      execute: input => {
        const source = contactSource(input);
        let draft = contactDrafts[source];
        if (draft.status === "sending") throw new ToolError("busy", "Wait for the current send to settle before editing this draft.");
        if (draft.status === "sent") { resetContact(source); draft = contactDrafts[source]; }
        const fields = { ...draft.fields };
        for (const key of ["reason", "topic", "email", "title", "body"] as const) if (typeof input[key] === "string") fields[key] = input[key];
        if (source === "feedback") fields.reason = "feedback";
        const topics = contactTopics(fields.reason);
        if (!topics.some(topic => topic.value === fields.topic) && input.topic === undefined) fields.topic = topics[0]?.value ?? "";
        draft.fields = fields;
        draft.errors = validateContact(fields);
        return { source, fields: { reason: fields.reason, topic: fields.topic, email: fields.email, title: fields.title, body: fields.body }, topics, errors: draft.errors, destination: CONTACT_ADDRESS, state: "draft" };
      },
    }),
    tool({
      name: "drowse_read_contact", title: "Read contact draft and delivery", group: "contact", annotations: { readOnlyHint: true, untrustedContentHint: true },
      description: "Read the contact draft, validation and last delivery state. Uncertain delivery is not confirmed success; retrying an unchanged draft preserves its request identity.",
      inputSchema: objectSchema({ source: sourceSchema }),
      execute: input => {
        const draft = contactDrafts[contactSource(input)];
        return { fields: draft.fields, state: draft.status, errors: draft.errors, message: draft.message, uncertain: draft.uncertain, reference: draft.requestId || null, destination: CONTACT_ADDRESS };
      },
    }),
    tool({
      name: "drowse_send_contact", title: "Send contact message", group: "contact", annotations: { readOnlyHint: false, consequentialHint: true, untrustedContentHint: true },
      description: "Send the reviewed contact/feedback draft to Drowse support. Use only when the user requests sending this message; includes its email, title and body. An uncertain result requires delivery reconciliation.",
      inputSchema: objectSchema({ source: sourceSchema }),
      execute: async input => {
        const source = contactSource(input);
        const draft = contactDrafts[source];
        const errors = validateContact(draft.fields);
        if (Object.keys(errors).length) throw new ToolError("invalid_input", "Correct the contact draft before sending.", errors);
        await submitContact(source);
        return { state: draft.status, reference: draft.requestId, uncertain: draft.uncertain, message: draft.message, destination: CONTACT_ADDRESS };
      },
    }),
    tool({
      name: "drowse_app_update", title: "Inspect or apply app update", annotations: { readOnlyHint: false },
      description: "Inspect a pending app update, remind later, or apply it through the app's save-before-reload flow. Applying reloads this page and invalidates tool handles.",
      inputSchema: objectSchema({ action: { type: "string", enum: ["inspect", "apply", "remind_later"] } }, ["action"]),
      available: () => hostedSite ? null : "App updates are managed by the Python installation for this dashboard.",
      execute: async input => {
        const update = updateActions();
        if (!update) return { available: false, reason: "Open the app to manage offline updates." };
        if (input.action !== "inspect") {
          if (!update.state().available) throw new ToolError("unavailable", "There is no pending app update.");
          if (input.action === "apply") await update.apply(); else update.remindLater();
        }
        return update.state();
      },
    }),
  ];
}
