import { DrowseArchiveError } from "./types";

const decoder = new TextDecoder("utf-8", { fatal: true });

export function parseExactJson(bytes: Uint8Array, label: string): unknown {
  let text: string;
  try {
    text = decoder.decode(bytes);
  } catch {
    throw new DrowseArchiveError("JSON_NOT_UTF8", `${label} is not UTF-8`);
  }
  return new ExactJsonParser(text, label).parse();
}

class ExactJsonParser {
  private offset = 0;

  constructor(
    private readonly text: string,
    private readonly label: string,
  ) {}

  parse(): unknown {
    this.whitespace();
    const value = this.value(0);
    this.whitespace();
    if (this.offset !== this.text.length) this.fail("has trailing data");
    return value;
  }

  private value(depth: number): unknown {
    if (depth > 128) this.fail("is nested too deeply");
    const char = this.text[this.offset];
    if (char === "{") return this.object(depth + 1);
    if (char === "[") return this.array(depth + 1);
    if (char === '"') return this.string();
    if (char === "t") return this.literal("true", true);
    if (char === "f") return this.literal("false", false);
    if (char === "n") return this.literal("null", null);
    if (char === "-" || (char >= "0" && char <= "9")) return this.number();
    this.fail("contains an invalid value");
  }

  private object(depth: number): Record<string, unknown> {
    this.offset += 1;
    this.whitespace();
    const result = Object.create(null) as Record<string, unknown>;
    const keys = new Set<string>();
    if (this.take("}")) return result;
    while (true) {
      if (this.text[this.offset] !== '"') this.fail("has a non-string object key");
      const key = this.string();
      if (keys.has(key)) {
        throw new DrowseArchiveError(
          "JSON_DUPLICATE_KEY",
          `${this.label} has duplicate key ${JSON.stringify(key)}`,
        );
      }
      keys.add(key);
      this.whitespace();
      if (!this.take(":")) this.fail("is missing ':' after an object key");
      this.whitespace();
      result[key] = this.value(depth);
      this.whitespace();
      if (this.take("}")) return result;
      if (!this.take(",")) this.fail("is missing ',' between object entries");
      this.whitespace();
    }
  }

  private array(depth: number): unknown[] {
    this.offset += 1;
    this.whitespace();
    const result: unknown[] = [];
    if (this.take("]")) return result;
    while (true) {
      result.push(this.value(depth));
      this.whitespace();
      if (this.take("]")) return result;
      if (!this.take(",")) this.fail("is missing ',' between array entries");
      this.whitespace();
    }
  }

  private string(): string {
    const start = this.offset;
    this.offset += 1;
    while (this.offset < this.text.length) {
      const code = this.text.charCodeAt(this.offset);
      if (code === 0x22) {
        this.offset += 1;
        try {
          return JSON.parse(this.text.slice(start, this.offset)) as string;
        } catch {
          this.fail("contains an invalid string");
        }
      }
      if (code < 0x20) this.fail("contains an unescaped control character");
      if (code === 0x5c) {
        this.offset += 1;
        if (this.offset >= this.text.length) this.fail("has a truncated escape");
        if (this.text[this.offset] === "u") {
          if (!/^[0-9a-fA-F]{4}$/.test(this.text.slice(this.offset + 1, this.offset + 5))) {
            this.fail("contains an invalid Unicode escape");
          }
          this.offset += 4;
        } else if (!'"\\/bfnrt'.includes(this.text[this.offset])) {
          this.fail("contains an invalid escape");
        }
      }
      this.offset += 1;
    }
    this.fail("contains an unterminated string");
  }

  private number(): number {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(
      this.text.slice(this.offset),
    );
    if (match === null) this.fail("contains an invalid number");
    this.offset += match[0].length;
    const value = Number(match[0]);
    if (!Number.isFinite(value)) this.fail("contains a non-finite number");
    return value;
  }

  private literal<T>(text: string, value: T): T {
    if (!this.text.startsWith(text, this.offset)) this.fail("contains an invalid literal");
    this.offset += text.length;
    return value;
  }

  private whitespace(): void {
    while (/\s/.test(this.text[this.offset] ?? "") && this.offset < this.text.length) {
      const code = this.text.charCodeAt(this.offset);
      if (code !== 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d) {
        this.fail("contains non-JSON whitespace");
      }
      this.offset += 1;
    }
  }

  private take(char: string): boolean {
    if (this.text[this.offset] !== char) return false;
    this.offset += 1;
    return true;
  }

  private fail(message: string): never {
    throw new DrowseArchiveError(
      "JSON_INVALID",
      `${this.label} ${message} near byte ${this.offset}`,
    );
  }
}
