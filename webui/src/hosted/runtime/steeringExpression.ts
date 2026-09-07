export const DEFAULT_STEERING_COEFFICIENT = 0.5;
export const DEFAULT_ABLATION_COEFFICIENT = 1;

export type SteeringProjectionOperator = "~" | "|";
export type SteeringComparisonOperator = ">" | ">=" | "<" | "<=";

export type SteeringPhase =
  | { kind: "both" }
  | { kind: "after_thinking" }
  | { kind: "prompt_only" }
  | { kind: "thinking_only" }
  | { kind: "generated_only" }
  | { kind: "first"; tokens: number }
  | { kind: "after"; tokens: number };

export interface SteeringProbeGate {
  probe: string;
  operator: SteeringComparisonOperator;
  threshold: number;
}

export interface SteeringTrigger {
  phase: SteeringPhase;
  gate: SteeringProbeGate | null;
}

export interface SteeringAtom {
  namespace: string | null;
  concept: string;
  variant: string;
  column: number;
}

export interface SteeringSelector {
  base: SteeringAtom;
  projection: SteeringProjectionOperator | null;
  onto: SteeringAtom | null;
  manifoldPosition: readonly number[] | string | null;
}

export interface SteeringExpressionTerm {
  coefficients: readonly number[];
  selector: SteeringSelector;
  trigger: SteeringTrigger | null;
  ablation: boolean;
}

export interface SteeringExpression {
  terms: readonly SteeringExpressionTerm[];
}

export class SteeringExpressionError extends Error {
  constructor(message: string, readonly column: number | null = null) {
    super(column === null ? message : `${message} (column ${column})`);
    this.name = "SteeringExpressionError";
  }
}

type TokenKind =
  | "AMP"
  | "AT"
  | "BANG"
  | "COLON"
  | "COMMA"
  | "DOT"
  | "EOF"
  | "GT"
  | "GTE"
  | "IDENT"
  | "LBRACK"
  | "LT"
  | "LTE"
  | "MINUS"
  | "NUM"
  | "ORTHO"
  | "PERCENT"
  | "PLUS"
  | "RBRACK"
  | "SLASH"
  | "STAR"
  | "TILDE";

interface Token {
  kind: TokenKind;
  value: string | number;
  column: number;
}

const SYMBOLS: Record<string, TokenKind> = {
  ".": "DOT",
  "/": "SLASH",
  ":": "COLON",
  "*": "STAR",
  "+": "PLUS",
  "-": "MINUS",
  "@": "AT",
  "~": "TILDE",
  "|": "ORTHO",
  "!": "BANG",
  "%": "PERCENT",
  ",": "COMMA",
  "[": "LBRACK",
  "]": "RBRACK",
  "&": "AMP",
};

const PHASES: Record<string, SteeringPhase> = {
  both: { kind: "both" },
  after: { kind: "after_thinking" },
  before: { kind: "prompt_only" },
  thinking: { kind: "thinking_only" },
  response: { kind: "generated_only" },
  prompt: { kind: "prompt_only" },
  generated: { kind: "generated_only" },
};

export function parseSteeringExpression(text: string): SteeringExpression {
  if (!text.trim()) throw new SteeringExpressionError("empty steering expression");
  return { terms: new Parser(lex(text)).parse() };
}

export function referencedSteeringSelectors(
  text: string,
): Array<Pick<SteeringAtom, "namespace" | "concept" | "variant">> {
  if (!text.trim()) return [];
  const result: Array<Pick<SteeringAtom, "namespace" | "concept" | "variant">> = [];
  for (const term of parseSteeringExpression(text).terms) {
    if (term.selector.manifoldPosition !== null) continue;
    result.push(stripAtomLocation(term.selector.base));
    if (term.selector.onto !== null) result.push(stripAtomLocation(term.selector.onto));
  }
  return result;
}

function stripAtomLocation(
  atom: SteeringAtom,
): Pick<SteeringAtom, "namespace" | "concept" | "variant"> {
  return { namespace: atom.namespace, concept: atom.concept, variant: atom.variant };
}

class Parser {
  private position = 0;

  constructor(private readonly tokens: readonly Token[]) {}

  parse(): SteeringExpressionTerm[] {
    let sign = 1;
    if (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      sign = this.consume().kind === "MINUS" ? -1 : 1;
    }
    const terms = [this.term(sign)];
    while (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      terms.push(this.term(this.consume().kind === "MINUS" ? -1 : 1));
    }
    const trailing = this.peek();
    if (trailing.kind !== "EOF") {
      if (trailing.kind === "IDENT") {
        throw this.error(
          "unexpected identifier after a complete term; use underscores for multi-word concepts",
          trailing,
        );
      }
      throw this.error(`unexpected token ${trailing.kind}`, trailing);
    }
    return terms;
  }

  private term(sign: number): SteeringExpressionTerm {
    let explicit = false;
    let coefficients = [sign * DEFAULT_STEERING_COEFFICIENT];
    if (this.peek().kind === "NUM") {
      explicit = true;
      coefficients = [sign * this.number(this.consume())];
      while (this.peek().kind === "COMMA") {
        const comma = this.consume();
        coefficients.push(sign * this.signedNumber());
        if (coefficients.length > 2) {
          throw this.error(
            "a manifold coefficient slot takes at most 2 comma-separated values",
            comma,
          );
        }
      }
      if (this.peek().kind === "STAR") this.consume();
    }
    let ablation = false;
    if (this.peek().kind === "BANG") {
      this.consume();
      ablation = true;
      if (!explicit) coefficients = [sign * DEFAULT_ABLATION_COEFFICIENT];
    }
    const selector = this.selector();
    const trigger = this.peek().kind === "AT"
      ? (this.consume(), this.trigger())
      : null;
    if (selector.manifoldPosition !== null && ablation) {
      throw this.error("a manifold term does not compose with ablation", selector.base.column);
    }
    if (selector.manifoldPosition === null && coefficients.length > 1) {
      throw this.error(
        "comma-separated coefficients are only valid for manifold position terms",
        selector.base.column,
      );
    }
    if (ablation && selector.projection !== null) {
      throw this.error("ablation does not compose with projection operators", selector.base.column);
    }
    return { coefficients, selector, trigger, ablation };
  }

  private trigger(): SteeringTrigger {
    const head = this.expect("IDENT");
    if (head.value === "when") return this.gateClause(head.column);
    const phase = this.phase(head);
    if (this.peek().kind !== "AMP") return { phase, gate: null };
    this.consume();
    const gate = this.expect("IDENT");
    if (gate.value !== "when") {
      throw this.error("a compound trigger must end with a when probe gate", gate);
    }
    return { phase, gate: this.gateClause(gate.column).gate };
  }

  private phase(token: Token): SteeringPhase {
    const keyword = String(token.value);
    if ((keyword === "first" || keyword === "after") && this.peek().kind === "COLON") {
      this.consume();
      const count = this.expect("NUM");
      const tokens = this.number(count);
      if (!Number.isSafeInteger(tokens) || tokens < 0) {
        throw this.error("a counted trigger requires a non-negative integer", count);
      }
      return { kind: keyword, tokens };
    }
    if (keyword === "first") {
      throw this.error("trigger @first requires a token count", token);
    }
    const phase = PHASES[keyword];
    if (phase === undefined) throw this.error(`unknown trigger @${keyword}`, token);
    return phase;
  }

  private gateClause(column: number): SteeringTrigger {
    if (this.peek().kind !== "COLON") {
      throw this.error("trigger @when requires a probe gate", this.peek());
    }
    this.consume();
    return {
      phase: { kind: "generated_only" },
      gate: this.probeGate(column),
    };
  }

  private probeGate(_column: number): SteeringProbeGate {
    const head = this.expect("IDENT");
    let probe = String(head.value);
    if (this.peek().kind === "SLASH") {
      this.consume();
      if (probe === "sae" && this.peek().kind === "NUM") {
        const feature = this.expect("NUM");
        const value = this.number(feature);
        if (!Number.isSafeInteger(value) || value < 0) {
          throw this.error("SAE feature id must be a non-negative integer", feature);
        }
        probe = `sae/${value}`;
      } else {
        probe = `${probe}/${String(this.expect("IDENT").value)}`;
      }
    }
    const discriminator = this.peek().kind;
    if (discriminator === "COLON") {
      this.consume();
      const channel = this.expect("IDENT");
      if (channel.value !== "fraction" && channel.value !== "membership") {
        throw this.error("unknown manifold gate channel", channel);
      }
      probe = `${probe}:${channel.value}`;
    } else if (discriminator === "AT" || discriminator === "TILDE") {
      this.consume();
      const label = this.expect("IDENT");
      probe = `${probe}${discriminator === "AT" ? "@" : "~"}${label.value}`;
    } else {
      if (discriminator === "DOT") {
        this.consume();
        probe = `${probe}.${String(this.expect("IDENT").value)}`;
      }
      if (this.peek().kind === "LBRACK") {
        this.consume();
        const index = this.expect("NUM");
        const value = this.number(index);
        if (!Number.isSafeInteger(value) || value < 0) {
          throw this.error("coordinate index must be a non-negative integer", index);
        }
        this.expect("RBRACK");
        probe = `${probe}[${value}]`;
      }
    }
    const operator = this.consume();
    if (!["GT", "GTE", "LT", "LTE"].includes(operator.kind)) {
      throw this.error("expected comparison operator in @when clause", operator);
    }
    return {
      probe,
      operator: ({ GT: ">", GTE: ">=", LT: "<", LTE: "<=" } as const)[
        operator.kind as "GT" | "GTE" | "LT" | "LTE"
      ],
      threshold: this.signedNumber(),
    };
  }

  private selector(): SteeringSelector {
    const base = this.atom();
    if (this.peek().kind === "PERCENT") {
      this.consume();
      let manifoldPosition: readonly number[] | string;
      if (this.peek().kind === "IDENT") {
        manifoldPosition = String(this.consume().value);
      } else {
        const coordinates = [this.signedNumber()];
        while (this.peek().kind === "COMMA") {
          this.consume();
          coordinates.push(this.signedNumber());
        }
        manifoldPosition = coordinates;
      }
      if (["TILDE", "ORTHO", "PERCENT"].includes(this.peek().kind)) {
        throw this.error("a manifold term does not compose with projection", this.peek());
      }
      return { base, projection: null, onto: null, manifoldPosition };
    }
    if (this.peek().kind === "TILDE" || this.peek().kind === "ORTHO") {
      const projection = this.consume().kind === "TILDE" ? "~" : "|";
      const onto = this.atom();
      if (this.peek().kind === "TILDE" || this.peek().kind === "ORTHO") {
        throw this.error("chained projection is not allowed", this.peek());
      }
      return { base, projection, onto, manifoldPosition: null };
    }
    return { base, projection: null, onto: null, manifoldPosition: null };
  }

  private atom(): SteeringAtom {
    const first = this.expect("IDENT");
    let namespace: string | null = null;
    let concept = String(first.value);
    if (this.peek().kind === "SLASH") {
      this.consume();
      namespace = concept;
      if (namespace === "sae" && this.peek().kind === "NUM") {
        const feature = this.consume();
        const value = this.number(feature);
        if (!Number.isSafeInteger(value) || value < 0) {
          throw this.error("SAE feature id must be a non-negative integer", feature);
        }
        concept = String(value);
      } else {
        concept = String(this.expect("IDENT").value);
      }
    }
    if (this.peek().kind === "DOT") {
      this.consume();
      concept = `${concept}.${String(this.expect("IDENT").value)}`;
    }
    let variant = "raw";
    if (this.peek().kind === "COLON") {
      this.consume();
      variant = String(this.expect("IDENT").value);
    }
    return { namespace, concept, variant, column: first.column };
  }

  private signedNumber(): number {
    let sign = 1;
    if (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      sign = this.consume().kind === "MINUS" ? -1 : 1;
    }
    return sign * this.number(this.expect("NUM"));
  }

  private number(token: Token): number {
    const value = Number(token.value);
    if (!Number.isFinite(value)) throw this.error("number must be finite", token);
    return value;
  }

  private peek(offset = 0): Token {
    return this.tokens[this.position + offset];
  }

  private consume(): Token {
    return this.tokens[this.position++];
  }

  private expect(kind: TokenKind): Token {
    const token = this.peek();
    if (token.kind !== kind) throw this.error(`expected ${kind}, got ${token.kind}`, token);
    return this.consume();
  }

  private error(message: string, location: Token | number): SteeringExpressionError {
    return new SteeringExpressionError(
      message,
      typeof location === "number" ? location : location.column,
    );
  }
}

function lex(text: string): Token[] {
  const tokens: Token[] = [];
  let index = 0;
  while (index < text.length) {
    const character = text[index];
    if (/\s/u.test(character)) {
      index += 1;
      continue;
    }
    if (/\d/u.test(character) || (character === "." && /\d/u.test(text[index + 1] ?? ""))) {
      const match = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/u.exec(text.slice(index));
      if (match === null) throw new SteeringExpressionError("malformed number", index);
      const value = Number(match[0]);
      if (!Number.isFinite(value)) throw new SteeringExpressionError("number must be finite", index);
      tokens.push({ kind: "NUM", value, column: index });
      index += match[0].length;
      continue;
    }
    if (character === ">" || character === "<") {
      const equals = text[index + 1] === "=";
      tokens.push({
        kind: character === ">" ? (equals ? "GTE" : "GT") : (equals ? "LTE" : "LT"),
        value: equals ? `${character}=` : character,
        column: index,
      });
      index += equals ? 2 : 1;
      continue;
    }
    const symbol = SYMBOLS[character];
    if (symbol !== undefined) {
      tokens.push({ kind: symbol, value: character, column: index });
      index += 1;
      continue;
    }
    if (/[A-Za-z]/u.test(character)) {
      const start = index++;
      while (index < text.length) {
        const next = text[index];
        if (/[A-Za-z0-9_]/u.test(next)) {
          index += 1;
          continue;
        }
        if (next === "-" && /[A-Za-z0-9_]/u.test(text[index + 1] ?? "")) {
          index += 1;
          continue;
        }
        break;
      }
      tokens.push({ kind: "IDENT", value: text.slice(start, index), column: start });
      continue;
    }
    if (character === '"' || character === "'") {
      throw new SteeringExpressionError(
        "quoted identifiers are not supported; use underscores for multi-word concepts",
        index,
      );
    }
    throw new SteeringExpressionError(`unexpected character ${JSON.stringify(character)}`, index);
  }
  tokens.push({ kind: "EOF", value: "", column: text.length });
  return tokens;
}
