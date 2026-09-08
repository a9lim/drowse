export function createTokenViewCache<T extends { text: string }>() {
  type View = { readonly tok: T; originalIdx: number };
  const cache = new WeakMap<T[], { entries: View[]; visible: View[]; start: number; length: number }>();
  return (tokens: T[]): View[] => {
    let saved = cache.get(tokens);
    if (!saved) {
      saved = { entries: [], visible: [], start: -1, length: -1 };
      cache.set(tokens, saved);
    }
    saved.entries.length = Math.min(saved.entries.length, tokens.length);
    while (saved.entries.length < tokens.length) {
      const originalIdx = saved.entries.length;
      saved.entries.push({ get tok() { return tokens[originalIdx]; }, originalIdx });
    }
    let start = 0;
    while (start < tokens.length && !tokens[start].text.trim()) start += 1;
    if (saved.length !== tokens.length || saved.start !== start) {
      saved.visible = saved.entries.slice(start);
      saved.length = tokens.length;
      saved.start = start;
    }
    return saved.visible;
  };
}
