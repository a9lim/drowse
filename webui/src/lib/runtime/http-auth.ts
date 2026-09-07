let apiKey: string | null = readApiKeyFromMeta();

function readApiKeyFromMeta(): string | null {
  if (typeof document === "undefined") return null;
  const tag = document.querySelector<HTMLMetaElement>('meta[name="api-key"]');
  const value = tag?.content?.trim();
  return value || null;
}

export function setApiKey(key: string | null): void {
  apiKey = key?.trim() || null;
}

export function getApiKey(): string | null {
  return apiKey;
}

export function httpAuthHeaders(extra: HeadersInit = {}): HeadersInit {
  const headers: Record<string, string> = {};
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  if (extra instanceof Headers) {
    extra.forEach((value, key) => (headers[key] = value));
  } else if (Array.isArray(extra)) {
    for (const [key, value] of extra) headers[key] = value;
  } else {
    Object.assign(headers, extra);
  }
  return headers;
}
