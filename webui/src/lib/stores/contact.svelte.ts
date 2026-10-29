import { CONTACT_ADDRESS, emptyContact, parseContact, validateContact, type ContactErrors, type ContactSource } from "../contact";

const DRAFT_STORAGE_PREFIX = "drowse.contact-draft.";
const FIELD_ERROR_KEYS = ["email", "title", "body", "topic", "reason"] as const;

function createDraft(source: ContactSource) {
  return {
    fields: emptyContact(source),
    status: "idle" as "idle" | "sending" | "sent" | "error",
    errors: {} as ContactErrors,
    message: "",
    requestId: "",
    submittedContent: "",
    uncertain: false,
  };
}

function draftStorage(): Storage | null {
  try { return typeof sessionStorage === "undefined" ? null : sessionStorage; }
  catch { return null; }
}

// Drafts live in this tab's session storage so a reload or an accidental
// navigation doesn't lose them. They're removed once a message is sent.
function restoreDraft(source: ContactSource) {
  const draft = createDraft(source);
  try {
    const saved: unknown = JSON.parse(draftStorage()?.getItem(DRAFT_STORAGE_PREFIX + source) ?? "null");
    if (!saved || typeof saved !== "object" || !("fields" in saved)) return draft;
    const fields = parseContact(saved.fields);
    if (!fields || fields.source !== source) return draft;
    draft.fields = { ...fields, website: "" };
    if ("requestId" in saved && typeof saved.requestId === "string" && "submittedContent" in saved && typeof saved.submittedContent === "string") {
      draft.requestId = saved.requestId;
      draft.submittedContent = saved.submittedContent;
      draft.uncertain = "uncertain" in saved && saved.uncertain === true;
    }
  } catch {
    // A missing, blocked, or malformed saved draft starts a fresh one.
  }
  return draft;
}

function saveDraft(source: ContactSource, draft: ReturnType<typeof createDraft>) {
  const storage = draftStorage();
  if (!storage) return;
  const key = DRAFT_STORAGE_PREFIX + source;
  const { title, body, email } = draft.fields;
  try {
    if (draft.status === "sent" || !(title || body || email)) storage.removeItem(key);
    else storage.setItem(key, JSON.stringify({ fields: { ...draft.fields, website: "" }, requestId: draft.requestId, submittedContent: draft.submittedContent, uncertain: draft.uncertain }));
  } catch {
    // Storage can be full or blocked. The draft still stays in memory.
  }
}

export const contactDrafts = $state({ feedback: restoreDraft("feedback"), contact: restoreDraft("contact") });

if (typeof window !== "undefined") {
  $effect.root(() => {
    for (const source of ["feedback", "contact"] as const) {
      $effect(() => {
        const draft = contactDrafts[source];
        // Reading every field subscribes this effect to edits.
        JSON.stringify(draft.fields);
        void draft.status; void draft.requestId; void draft.uncertain;
        saveDraft(source, draft);
      });
    }
  });
}

export function resetContact(source: ContactSource) {
  if (contactDrafts[source].status !== "sending") contactDrafts[source] = createDraft(source);
}

function serverFieldErrors(result: unknown): ContactErrors {
  const errors: ContactErrors = {};
  if (!result || typeof result !== "object" || !("fields" in result) || !result.fields || typeof result.fields !== "object") return errors;
  const fields = result.fields as Record<string, unknown>;
  for (const key of FIELD_ERROR_KEYS) if (typeof fields[key] === "string" && fields[key]) errors[key] = fields[key];
  return errors;
}

function errorCode(result: unknown): string {
  return result && typeof result === "object" && "error" in result && typeof result.error === "string" ? result.error : "";
}

export async function submitContact(source: ContactSource): Promise<void> {
  const draft = contactDrafts[source];
  if (draft.status === "sending" || draft.status === "sent") return;
  draft.errors = validateContact(draft.fields);
  draft.message = "";
  if (Object.keys(draft.errors).length) return;
  if (!navigator.onLine) {
    draft.status = "error";
    draft.message = "You're offline. Your draft is still here. Reconnect, then try again.";
    return;
  }
  const content = JSON.stringify(draft.fields);
  if (!draft.requestId || draft.submittedContent !== content) {
    draft.requestId = crypto.randomUUID();
    draft.submittedContent = content;
    draft.uncertain = false;
  }
  draft.status = "sending";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25_000);
  let response: Response;
  try {
    const endpoint = import.meta.env.VITE_DROWSE_CONTACT_ENDPOINT || "https://drowse.ai/api/contact";
    response = await fetch(endpoint, {
      method: "POST",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...draft.fields, requestId: draft.requestId }),
      signal: controller.signal,
    });
  } catch {
    clearTimeout(timer);
    draft.status = "error";
    draft.uncertain = true;
    draft.message = controller.signal.aborted
      ? "The send took too long, so we can't tell if it arrived. Your draft is still here. Check delivery to try again safely."
      : navigator.onLine
        ? "We couldn't reach Drowse. Check your connection and try again. Your draft is still here."
        : "Your connection was lost. Reconnect and try again. Your draft is still here.";
    return;
  }
  try {
    let result: unknown = null;
    if (response.headers.get("Content-Type")?.includes("application/json")) {
      try { result = await response.json(); } catch { result = null; }
    }
    if (response.ok && result && typeof result === "object" && "status" in result && result.status === "sent" && "reference" in result && result.reference === draft.requestId) {
      draft.status = "sent";
      draft.uncertain = false;
      return;
    }
    const code = errorCode(result);
    draft.uncertain = response.status === 202 || response.status >= 500;
    if (response.status === 429) draft.message = "Too many attempts. Wait a minute, then try again. Your draft is still here.";
    else if (response.status === 202) draft.message = "We haven't confirmed the send yet. Wait a moment, then check delivery again.";
    else if (response.status === 400 && code === "invalid_fields") {
      const errors = serverFieldErrors(result);
      draft.errors = errors;
      draft.message = Object.keys(errors).length
        ? "We couldn't send this. Fix the marked fields, then try again."
        : "We couldn't send this. Check each field, then try again.";
    }
    else if (response.status === 413) {
      draft.errors = { body: "This message is too long to send. Shorten it, then try again." };
      draft.message = "We couldn't send this. Fix the marked fields, then try again.";
    }
    else if (response.status === 409) {
      // The reference was already used for different content. A fresh one is made on the next send.
      draft.submittedContent = "";
      draft.message = "This draft changed after an earlier send. Send it again to deliver the new version.";
    }
    else if (response.status === 503 && code === "contact_unavailable") draft.message = `The contact form is turned off for now. Your draft is still here. Email us at ${CONTACT_ADDRESS} instead.`;
    else if (response.status === 502 && result && typeof result === "object" && "status" in result && result.status === "failed") {
      draft.uncertain = false;
      draft.message = `Our mail service didn't accept this message. Your draft is still here. Try again later, or email ${CONTACT_ADDRESS}.`;
    }
    else if (response.status >= 500) draft.message = `Drowse's contact service isn't responding. Your draft is still here. Try again in a few minutes, or email ${CONTACT_ADDRESS}.`;
    else draft.message = `We couldn't send this. Your draft is still here. Try again, or email ${CONTACT_ADDRESS} directly.`;
    draft.status = "error";
  } finally {
    clearTimeout(timer);
  }
}
