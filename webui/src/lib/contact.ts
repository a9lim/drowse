export type ContactSource = "feedback" | "contact";
export const CONTACT_ADDRESS = "contact@drowse.ai";
export const CONTACT_LIMITS = { email: 254, title: 120, body: 8000 } as const;
export const CONTACT_REASONS = [
  { value: "feedback", label: "General feedback" },
  { value: "support", label: "Help and support" },
  { value: "bug", label: "Bug report" },
  { value: "feature", label: "Feature suggestion" },
  { value: "media", label: "Media and press" },
  { value: "research", label: "Research and collaboration" },
  { value: "other", label: "Other inquiry" },
];

const productReasons = ["feedback", "support", "bug", "feature"];
export const CONTACT_TOPICS = [
  { value: "general", label: "Drowse in general", reasons: ["feedback", "support"] },
  { value: "getting-started", label: "Getting started", reasons: ["support"] },
  { value: "chat", label: "Chat and text completion", reasons: productReasons },
  { value: "loom", label: "Loom and branching", reasons: productReasons },
  { value: "controls", label: "Steering and readings", reasons: productReasons },
  { value: "models", label: "Models and downloads", reasons: productReasons },
  { value: "interface", label: "Interface and accessibility", reasons: productReasons },
  { value: "python", label: "Python library and API", reasons: productReasons },
  { value: "docs", label: "Documentation", reasons: productReasons },
  { value: "interview", label: "Interview request", reasons: ["media"] },
  { value: "coverage", label: "Coverage and fact checking", reasons: ["media"] },
  { value: "press-materials", label: "Images and press materials", reasons: ["media"] },
  { value: "collaboration", label: "Research collaboration", reasons: ["research"] },
  { value: "study", label: "Using Drowse in a study", reasons: ["research"] },
  { value: "methods", label: "Methods and validation", reasons: ["research"] },
  { value: "results", label: "Sharing results", reasons: ["research"] },
  { value: "funding", label: "Funding & Sponsorship", reasons: ["research"] },
  { value: "none", label: "Not applicable", reasons: ["other"] },
  { value: "other", label: "Something else", reasons: CONTACT_REASONS.filter(({ value }) => value !== "other").map(({ value }) => value) },
];

export function contactTopics(reason: string) {
  return CONTACT_TOPICS.filter(topic => topic.reasons.includes(reason));
}

export interface ContactFields {
  source: ContactSource;
  reason: string;
  topic: string;
  email: string;
  title: string;
  body: string;
  website: string;
}
export type ContactErrors = Partial<Record<"email" | "title" | "body" | "topic" | "reason", string>>;

export function emptyContact(source: ContactSource): ContactFields {
  return { source, reason: "feedback", topic: "general", email: "", title: "", body: "", website: "" };
}

export function validateContact(fields: ContactFields): ContactErrors {
  const errors: ContactErrors = {};
  if (!CONTACT_REASONS.some(({ value }) => value === fields.reason) || (fields.source === "feedback" && fields.reason !== "feedback")) errors.reason = "Choose a reason for getting in touch.";
  if (!contactTopics(fields.reason).some(({ value }) => value === fields.topic)) errors.topic = "Choose a topic for this reason.";
  const email = fields.email.trim();
  if (email && (email.length > CONTACT_LIMITS.email || /[\u0000-\u001f\u007f]/.test(email) || !/^[^\s@<>(),;:\\"\[\]]+@[^\s@<>(),;:\\"\[\]]+\.[^\s@<>(),;:\\"\[\]]+$/.test(email))) errors.email = "Enter a valid email address, or leave this blank.";
  if (!fields.title.trim()) errors.title = "Add a short title.";
  else if (fields.title.length > CONTACT_LIMITS.title || /[\r\n\u0000-\u001f\u007f]/.test(fields.title)) errors.title = `Use one line, up to ${CONTACT_LIMITS.title} characters.`;
  if (!fields.body.trim()) errors.body = "Add a message before sending.";
  else if (fields.body.length > CONTACT_LIMITS.body || /\u0000/.test(fields.body)) errors.body = `Keep your message under ${CONTACT_LIMITS.body.toLocaleString("en-US")} characters.`;
  return errors;
}

export function parseContact(value: unknown): ContactFields | null {
  if (!value || typeof value !== "object") return null;
  if (!("source" in value) || (value.source !== "feedback" && value.source !== "contact")
    || !("reason" in value) || typeof value.reason !== "string"
    || !("topic" in value) || typeof value.topic !== "string"
    || !("email" in value) || typeof value.email !== "string"
    || !("title" in value) || typeof value.title !== "string"
    || !("body" in value) || typeof value.body !== "string"
    || !("website" in value) || typeof value.website !== "string") return null;
  return { source: value.source, reason: value.reason, topic: value.topic, email: value.email, title: value.title, body: value.body, website: value.website };
}
