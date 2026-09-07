import type { TokenScore } from "./types";

export function tokenProbabilityRows(token: TokenScore) {
  const rows = (token.topAlts ?? []).map(alt => ({
    id: alt.id, text: alt.text, logprob: alt.logprob,
    chosen: token.tokenId != null && alt.id === token.tokenId,
  }));
  if (!rows.some(row => row.chosen) && token.logprob != null) {
    rows.push({ id: token.tokenId ?? -1, text: token.text, logprob: token.logprob, chosen: true });
  }
  return rows.sort((a, b) => b.logprob - a.logprob);
}

export function visibleTokenText(text: string): string {
  return text.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
}
