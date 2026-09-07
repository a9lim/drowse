import type { ChatTurn, LoomNodeJSON, TokenScore } from "../../lib/types";

function tokenRow(token: TokenScore): NonNullable<LoomNodeJSON["tokens"]>[number] {
  return {
    text: token.text,
    ...(token.tokenId == null ? {} : { token_id: token.tokenId }),
    ...(token.rawIndex == null ? {} : { raw_index: token.rawIndex }),
    logprob: token.logprob ?? null,
    perplexity: token.perplexity ?? null,
    ...(token.samplerEntropy == null ? {} : { sampler_entropy: token.samplerEntropy }),
    ...(token.topAlts ? { top_alts: token.topAlts } : {}),
    ...(token.measurements ? { measurements: token.measurements } : {}),
  };
}

/** Display the received stream without changing the saved tree or its revision. */
export function loomLiveNode(node: LoomNodeJSON, turn: ChatTurn): LoomNodeJSON {
  if (turn.nodeId !== node.id || node.finish_reason !== null) return node;
  const tokens = turn.tokens?.map(tokenRow) ?? node.tokens;
  const thinkingTokens = turn.thinkingTokens?.map(tokenRow) ?? node.thinking_tokens;
  const rawIds = [...(node.raw_token_ids ?? [])];
  const rows = [...(tokens ?? []), ...(thinkingTokens ?? [])]
    .filter(row => row.raw_index != null && row.token_id != null)
    .sort((a, b) => a.raw_index! - b.raw_index!);
  let complete = true;
  for (const row of rows) {
    if (row.raw_index! > rawIds.length) { complete = false; break; }
    rawIds[row.raw_index!] = row.token_id!;
  }
  return {
    ...node,
    text: turn.text ?? node.text,
    tokens,
    thinking_tokens: thinkingTokens,
    thinking_text: thinkingTokens?.map(token => token.text).join("") ?? node.thinking_text,
    raw_token_ids: complete ? rawIds : null,
  };
}
