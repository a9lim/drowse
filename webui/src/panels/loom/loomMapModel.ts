// View-model for the loom map: the live overlay of the streaming reply and
// the per-segment sizes the layout consumes.  Everything here is memoized on
// identity so a streamed frame costs work proportional to the new tokens,
// not to the whole tree: saved nodes keep their object identity until the
// tree replaces them, and the live overlay reuses each converted token row.
import type { ChatTurn, LoomNodeJSON, TokenScore } from "../../lib/types";
import { loomMessageWidth } from "./loomLayout";
import type { LoomSegment } from "./loomSegments";
import { splitTokenSentences } from "./loomSentences";

type TokenRow = NonNullable<LoomNodeJSON["tokens"]>[number];

function tokenRow(token: TokenScore): TokenRow {
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

interface LiveStream {
  node: LoomNodeJSON;
  turn: ChatTurn;
  tokens: TokenRow[];
  thinking: TokenRow[];
  thinkingText: string;
  rawIds: number[];
  /** Rows whose raw index is past the filled prefix (a hidden control-token gap). */
  gaps: Map<number, number>;
}

/** The streaming reply as a display-only node.  Returns a function that
 *  overlays a live chat turn on its saved (unfinished) loom node without
 *  touching the saved tree or its revision.  Repeated calls for the same
 *  node and turn convert only the tokens that arrived since the last call;
 *  each call returns a fresh node object (fresh arrays too) so identity-
 *  keyed consumers see the change, while the token rows themselves are
 *  reused across frames. */
export function createLoomLiveNode(): (node: LoomNodeJSON, turn: ChatTurn) => LoomNodeJSON {
  let stream: LiveStream | null = null;

  function absorb(row: TokenRow): void {
    const live = stream!;
    const index = row.raw_index;
    if (index == null || row.token_id == null) return;
    if (index > live.rawIds.length) {
      live.gaps.set(index, row.token_id);
      return;
    }
    live.rawIds[index] = row.token_id;
    while (live.gaps.has(live.rawIds.length)) {
      const next = live.rawIds.length;
      live.rawIds.push(live.gaps.get(next)!);
      live.gaps.delete(next);
    }
  }

  return (node, turn) => {
    if (turn.nodeId !== node.id || node.finish_reason !== null) return node;
    const tokens = turn.tokens;
    const thinking = turn.thinkingTokens;
    if (!stream || stream.node !== node || stream.turn !== turn ||
      (tokens?.length ?? 0) < stream.tokens.length || (thinking?.length ?? 0) < stream.thinking.length) {
      stream = { node, turn, tokens: [], thinking: [], thinkingText: "", rawIds: [...(node.raw_token_ids ?? [])], gaps: new Map() };
    }
    const live = stream;
    for (let index = live.tokens.length; index < (tokens?.length ?? 0); index += 1) {
      const row = tokenRow(tokens![index]);
      live.tokens.push(row);
      absorb(row);
    }
    for (let index = live.thinking.length; index < (thinking?.length ?? 0); index += 1) {
      const row = tokenRow(thinking![index]);
      live.thinking.push(row);
      live.thinkingText += row.text;
      absorb(row);
    }
    return {
      ...node,
      text: turn.text ?? node.text,
      tokens: tokens ? live.tokens.slice() : node.tokens,
      thinking_tokens: thinking ? live.thinking.slice() : node.thinking_tokens,
      thinking_text: thinking ? live.thinkingText : node.thinking_text,
      raw_token_ids: live.gaps.size === 0 ? live.rawIds.slice() : null,
    };
  };
}

// ------------------------------------------------------- segment sizes --

type SizeCache = Map<string, number>;
const widths = new WeakMap<LoomNodeJSON, SizeCache>();
const heights = new WeakMap<LoomNodeJSON, SizeCache>();

function cached(store: WeakMap<LoomNodeJSON, SizeCache>, node: LoomNodeJSON, key: string, compute: () => number): number {
  let sizes = store.get(node);
  if (!sizes) store.set(node, sizes = new Map());
  let value = sizes.get(key);
  if (value === undefined) sizes.set(key, value = compute());
  return value;
}

/** Card width for a segment's token span (or its node text when untokenized). */
export function segmentWidth(segment: LoomSegment): number {
  return cached(widths, segment.node, `${segment.start}:${segment.end}`, () => {
    const tokens = segment.node.tokens?.slice(segment.start, segment.end);
    return loomMessageWidth(tokens?.length ? tokens.map(token => token.text).join("") : segment.node.text ?? "", tokens?.length ?? 0);
  });
}

/** Card height estimate used until the rendered card reports its measured
 *  height: sentence rows of token chips wrapped at the card width. */
export function segmentHeightEstimate(segment: LoomSegment, width: number, coarse: boolean): number {
  return cached(heights, segment.node, `${segment.start}:${segment.end}:${width}:${coarse ? 1 : 0}`, () => {
    const tokens = segment.node.tokens?.slice(segment.start, segment.end) ?? [];
    if (tokens.length > 0) {
      const availableWidth = width - 60;
      const sentences = splitTokenSentences(tokens);
      let rows = 0;
      for (const sentence of sentences) {
        rows += 1;
        let usedWidth = 0;
        for (let index = sentence.start; index <= sentence.end; index += 1) {
          const label = (tokens[index].text || "∅").replace(/\s/g, " ");
          const tokenWidth = Math.min(availableWidth, Math.max(coarse ? 24 : 18, 16 + label.length * 7));
          if (usedWidth > 0 && usedWidth + 3 + tokenWidth > availableWidth) {
            rows += 1;
            usedWidth = tokenWidth;
          } else {
            usedWidth += (usedWidth > 0 ? 3 : 0) + tokenWidth;
          }
        }
      }
      return coarse
        ? 120 + rows * 47 + sentences.length * 56
        : 105 + rows * 27 + sentences.length * 38;
    }
    const textLength = (segment.node.text ?? "").trim().length;
    const lines = Math.min(6, Math.max(1, Math.ceil(textLength / Math.max(1, (width - 40) / 7))));
    return 98 + lines * 18;
  });
}
