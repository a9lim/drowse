import type { LoomNodeJSON, LoomTokenRowJSON } from "../../lib/types";

type AggregateOperator = "agg" | "any" | "last";
type ComparisonOperator = ">" | ">=" | "<" | "<=";

interface FilterClause {
  aggregate: AggregateOperator;
  probe: string;
  comparison: ComparisonOperator;
  threshold: number;
}

const AGGREGATE_OPERATORS = new Set<AggregateOperator>(["agg", "any", "last"]);
const PROBE_NAME = /^(?:[A-Za-z][A-Za-z0-9_-]*\/)?[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)?(?:\[[0-9]+\])?$/u;
const COMPARISON = /(>=|<=|>|<)/u;
const PYTHON_FLOAT = /^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|inf(?:inity)?|nan)$/iu;

export class BrowserTreeFilterError extends Error {
  readonly code = "INVALID_TREE_FILTER";
  readonly status = 400;
  readonly recoverable = true;

  constructor(message: string) {
    super(message);
    this.name = "BrowserTreeFilterError";
  }
}

export function filterBrowserTree(
  nodes: readonly LoomNodeJSON[],
  expression: string,
): string[] {
  const normalized = expression.trim();
  if (!normalized) return [];
  const clauses = parseBrowserTreeFilter(normalized);
  return nodes
    .filter((node) => clauses.every((clause) => matchesClause(node, clause)))
    .map((node) => node.id)
    .sort();
}

export function parseBrowserTreeFilter(expression: string): readonly FilterClause[] {
  if (!expression.trim()) throw new BrowserTreeFilterError("empty filter expression");
  const parts = expression.split(",").map((part) => part.trim()).filter(Boolean);
  if (parts.length === 0) {
    throw new BrowserTreeFilterError(
      `filter expression ${JSON.stringify(expression)} yielded no clauses`,
    );
  }
  return parts.map(parseClause);
}

function parseClause(raw: string): FilterClause {
  let aggregate: AggregateOperator = "agg";
  let remainder = raw;
  const colon = raw.indexOf(":");
  if (colon >= 0) {
    const prefix = raw.slice(0, colon).trim();
    if (!AGGREGATE_OPERATORS.has(prefix as AggregateOperator)) {
      throw new BrowserTreeFilterError(
        `unknown agg op ${JSON.stringify(prefix)}; expected one of agg, any, last (or drop the prefix to use 'agg:' by default)`,
      );
    }
    aggregate = prefix as AggregateOperator;
    remainder = raw.slice(colon + 1);
  }

  const comparisonMatch = COMPARISON.exec(remainder);
  if (!comparisonMatch || comparisonMatch.index === undefined) {
    throw new BrowserTreeFilterError(
      `clause ${JSON.stringify(raw)} missing comparison op (>, >=, <, <=)`,
    );
  }
  const comparison = comparisonMatch[1] as ComparisonOperator;
  const probe = remainder.slice(0, comparisonMatch.index).trim();
  const thresholdText = remainder.slice(comparisonMatch.index + comparison.length).trim();
  if (!probe) {
    throw new BrowserTreeFilterError(
      `clause ${JSON.stringify(raw)} missing probe name before ${JSON.stringify(comparison)}`,
    );
  }
  if (!PROBE_NAME.test(probe)) {
    throw new BrowserTreeFilterError(
      `clause ${JSON.stringify(raw)}: probe ${JSON.stringify(probe)} is not a valid identifier (letter, then [A-Za-z0-9_-], optional .pole)`,
    );
  }
  if (!thresholdText) {
    throw new BrowserTreeFilterError(
      `clause ${JSON.stringify(raw)}: missing threshold after ${JSON.stringify(comparison)}`,
    );
  }
  if (!PYTHON_FLOAT.test(thresholdText)) {
    throw new BrowserTreeFilterError(
      `clause ${JSON.stringify(raw)}: threshold ${JSON.stringify(thresholdText)} is not a number`,
    );
  }
  const threshold = pythonFloat(thresholdText);
  return { aggregate, probe, comparison, threshold };
}

function matchesClause(node: LoomNodeJSON, clause: FilterClause): boolean {
  if (clause.aggregate === "agg") {
    if (!Object.hasOwn(node.aggregate_readings, clause.probe)) return false;
    return compare(node.aggregate_readings[clause.probe], clause.comparison, clause.threshold);
  }
  const values = tokenSeries(node, clause.probe);
  if (values.length === 0) return false;
  if (clause.aggregate === "last") {
    return compare(values[values.length - 1], clause.comparison, clause.threshold);
  }
  let extreme = values[0];
  const wantsMaximum = clause.comparison === ">" || clause.comparison === ">=";
  for (let index = 1; index < values.length; index += 1) {
    if (wantsMaximum ? values[index] > extreme : values[index] < extreme) {
      extreme = values[index];
    }
  }
  return compare(extreme, clause.comparison, clause.threshold);
}

function tokenSeries(node: LoomNodeJSON, probe: string): number[] {
  const values: number[] = [];
  for (const rows of [node.thinking_tokens, node.tokens]) {
    for (const row of rows ?? []) {
      const scores = rowScores(row);
      const value = scores?.[probe];
      if (typeof value === "number" && Number.isFinite(value)) values.push(value);
    }
  }
  return values;
}

function rowScores(row: LoomTokenRowJSON): Record<string, number> | undefined {
  return row.measurements?.scores ?? row.probes;
}

function compare(left: number, operator: ComparisonOperator, right: number): boolean {
  if (operator === ">") return left > right;
  if (operator === ">=") return left >= right;
  if (operator === "<") return left < right;
  return left <= right;
}

function pythonFloat(value: string): number {
  const normalized = value.toLowerCase();
  const sign = normalized.startsWith("-") ? -1 : 1;
  const unsigned = normalized.replace(/^[+-]/u, "");
  if (unsigned === "inf" || unsigned === "infinity") return sign * Infinity;
  if (unsigned === "nan") return NaN;
  return Number(value);
}
