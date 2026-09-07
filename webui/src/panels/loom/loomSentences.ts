export interface SentenceToken {
  text: string;
}

export interface TokenSentence {
  start: number;
  end: number;
  text: string;
}

const SENTENCE_END = /[.!?…](?:["'”’)\]}]+)?$/u;

/** Group model tokens into human-readable sentence branch points without
 * changing their indexes or text. The end index is always an exact token
 * boundary that can be replayed by the runtime. */
export function splitTokenSentences(
  tokens: readonly SentenceToken[],
): TokenSentence[] {
  const sentences: TokenSentence[] = [];
  let start = 0;

  for (let index = 0; index < tokens.length; index += 1) {
    const text = tokens[index]?.text ?? "";
    const visible = text.trimEnd();
    const endsSentence = SENTENCE_END.test(visible);
    if (!endsSentence) continue;

    sentences.push({
      start,
      end: index,
      text: tokens.slice(start, index + 1).map((token) => token.text).join(""),
    });
    start = index + 1;
  }

  if (start < tokens.length) {
    sentences.push({
      start,
      end: tokens.length - 1,
      text: tokens.slice(start).map((token) => token.text).join(""),
    });
  }

  return sentences;
}
