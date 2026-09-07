import type { ChatTurn, TokenScore } from "./types";

export interface RawTokenView {
  text: string;
  turnIdx: number | null;
  nodeId: string | null;
  tokenIdx: number;
  isThinking: boolean;
  tok: TokenScore | null;
  source: "user" | "model" | "draft";
  contextChanged: boolean;
}

export function completionTokenViews(turns: ChatTurn[]): RawTokenView[] {
  return turns.flatMap<RawTokenView>((turn, turnIdx) => {
    const source = turn.generated ? "model" as const : "user" as const;
    const base = { nodeId: turn.nodeId ?? null, source, contextChanged: false, isThinking: false };
    if (turn.tokens?.length && turn.tokens.map(token => token.text).join("") === turn.text) {
      return turn.tokens.map((tok, tokenIdx) => ({ ...base, text: tok.text, turnIdx, tokenIdx, tok }));
    }
    return [{ ...base, text: turn.text, turnIdx: null, tokenIdx: 0, tok: null }];
  });
}

export function projectEditedTokens(views: RawTokenView[], text: string): RawTokenView[] {
  const previous = views.map(view => view.text).join("");
  if (text === previous) return views;
  let prefix = 0;
  while (prefix < previous.length && prefix < text.length && previous[prefix] === text[prefix]) prefix++;
  if (prefix > 0 && /[\uD800-\uDBFF]/.test(text[prefix - 1])) prefix--;
  let suffix = 0;
  while (suffix < previous.length - prefix && suffix < text.length - prefix &&
    previous[previous.length - 1 - suffix] === text[text.length - 1 - suffix]) suffix++;
  if (suffix > 0 && /[\uDC00-\uDFFF]/.test(text[text.length - suffix])) suffix--;

  const draft = (value: string): RawTokenView => ({
    text: value, turnIdx: null, nodeId: null, tokenIdx: 0, isThinking: false,
    tok: null, source: "draft", contextChanged: false,
  });
  const slice = (start: number, end: number, contextChanged: boolean): RawTokenView[] => {
    let offset = 0;
    return views.flatMap(view => {
      const viewStart = offset;
      offset += view.text.length;
      const from = Math.max(start, viewStart);
      const to = Math.min(end, offset);
      if (to <= from) return [];
      if (from === viewStart && to === offset) {
        return [{ ...view, contextChanged: view.contextChanged || contextChanged }];
      }
      const text = view.text.slice(from - viewStart, to - viewStart);
      return [view.tok ? draft(text) : { ...view, text }];
    });
  };
  const inserted = text.slice(prefix, text.length - suffix);
  return [
    ...slice(0, prefix, false),
    ...(inserted ? [draft(inserted)] : []),
    ...slice(previous.length - suffix, previous.length, true),
  ];
}
