import { CONTACT_ADDRESS, emptyContact, validateContact, type ContactErrors, type ContactSource } from "../contact";

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

export const contactDrafts = $state({ feedback: createDraft("feedback"), contact: createDraft("contact") });

export function resetContact(source: ContactSource) {
  if (contactDrafts[source].status !== "sending") contactDrafts[source] = createDraft(source);
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
  try {
    const endpoint = import.meta.env.VITE_DROWSE_CONTACT_ENDPOINT || "https://drowse.ai/api/contact";
    const response = await fetch(endpoint, {
      method: "POST",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...draft.fields, requestId: draft.requestId }),
      signal: controller.signal,
    });
    if (!response.headers.get("Content-Type")?.includes("application/json")) throw new Error("unavailable");
    const result: unknown = await response.json();
    if (response.ok && result && typeof result === "object" && "status" in result && result.status === "sent" && "reference" in result && result.reference === draft.requestId) {
      draft.status = "sent";
      draft.uncertain = false;
      return;
    }
    draft.uncertain = response.status === 202 || response.status >= 500;
    if (response.status === 429) draft.message = "Too many attempts. Wait a minute, then try again. Your draft is still here.";
    else if (response.status === 202) draft.message = "We haven't confirmed the send yet. Wait a moment, then check delivery again.";
    else if (response.status === 400) draft.message = "We couldn't send this. Check the fields above, then try again.";
    else draft.message = `We couldn't confirm the send. Your draft is still here. Try again, or email ${CONTACT_ADDRESS} directly.`;
    draft.status = "error";
  } catch {
    draft.status = "error";
    draft.uncertain = true;
    draft.message = navigator.onLine
      ? "We couldn't confirm the send. Check your connection and try again. Your draft is still here."
      : "Your connection was lost. Reconnect and try again. Your draft is still here.";
  } finally {
    clearTimeout(timer);
  }
}
