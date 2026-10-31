// Pure helpers behind the loom dialogs: the fan-out strength list and the
// go-to-turn id lookup.

/** Parse a fan-out alpha string (comma list / ``linspace(a, b, n)`` /
 *  ``start:stop:step``) into ``number[]``. */
export function parseStrengthList(raw: string): number[] {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const linspaceMatch = trimmed.match(
    /^linspace\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,)]+?)\s*\)\s*$/i,
  );
  if (linspaceMatch) {
    const start = Number(linspaceMatch[1]);
    const stop = Number(linspaceMatch[2]);
    const count = Number(linspaceMatch[3]);
    if (!Number.isFinite(start) || !Number.isFinite(stop)) return [];
    if (!Number.isInteger(count) || count < 1) return [];
    if (count === 1) return [start];
    const step = (stop - start) / (count - 1);
    const out: number[] = [];
    for (let i = 0; i < count; i++) out.push(start + step * i);
    return out;
  }

  if (trimmed.includes(":")) {
    const parts = trimmed.split(":").map((p) => p.trim());
    if (parts.length !== 3) return [];
    const start = Number(parts[0]);
    const stop = Number(parts[1]);
    const step = Number(parts[2]);
    if (![start, stop, step].every(Number.isFinite)) return [];
    if (step === 0) return [];
    if ((stop - start) * step < 0) return [];
    const out: number[] = [];
    const eps = Math.abs(step) * 1e-9;
    const ascending = step > 0;
    let v = start;
    let guard = 0;
    while (
      (ascending ? v <= stop + eps : v >= stop - eps) &&
      guard++ < 10000
    ) {
      out.push(Number.parseFloat(v.toPrecision(12)));
      v += step;
    }
    if (guard >= 10000) return [];
    return out;
  }

  const out: number[] = [];
  for (const part of trimmed.split(",")) {
    const t = part.trim();
    if (!t) continue;
    const v = Number(t);
    if (!Number.isFinite(v)) return [];
    out.push(v);
  }
  return out;
}

export interface PrefixResolution {
  /** Single unambiguous match — caller can navigate immediately. */
  id: string | null;
  /** Every node id whose ulid starts with the prefix.  ``length > 1``
   *  signals ambiguity; ``length == 0`` signals no match. */
  matches: string[];
}

/** Resolve a node by full id or id prefix (``root`` names the root). */
export function resolveNodeIdPrefix(prefix: string, ids: Iterable<string>, rootId: string | null): PrefixResolution {
  const p = prefix.trim();
  if (!p) return { id: null, matches: [] };
  if (p === "root") {
    return rootId ? { id: rootId, matches: [rootId] } : { id: null, matches: [] };
  }
  const matches: string[] = [];
  for (const id of ids) {
    if (id === p) return { id, matches: [id] };  // exact match wins
    if (id.startsWith(p)) matches.push(id);
  }
  if (matches.length === 1) return { id: matches[0], matches };
  return { id: null, matches };
}
