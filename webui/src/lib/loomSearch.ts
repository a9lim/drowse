export function searchableLoomText(text: string): string {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

export function loomTextMatches(text: string, query: string): boolean {
  const needle = searchableLoomText(query);
  return needle.length > 0 && searchableLoomText(text).includes(needle);
}

export function loomSearchExcerpt(text: string, query: string): { before: string; match: string; after: string } {
  const readable = text.replace(/\s+/g, " ").trim();
  const needle = searchableLoomText(query);
  const index = needle ? readable.toLowerCase().indexOf(needle) : -1;
  if (index < 0) return { before: readable.slice(0, 150), match: "", after: readable.length > 150 ? "…" : "" };
  const start = Math.max(0, index - 45);
  const end = Math.min(readable.length, index + needle.length + 90);
  return {
    before: (start > 0 ? "…" : "") + readable.slice(start, index),
    match: readable.slice(index, index + needle.length),
    after: readable.slice(index + needle.length, end) + (end < readable.length ? "…" : ""),
  };
}
