import { ToolError, type InputSchema } from "./types";

export function validateInput(schema: InputSchema, value: unknown, path = "input", depth = 0): void {
  const fail = (message: string): never => { throw new ToolError("invalid_input", `${path}: ${message}`); };
  if (depth > 32) fail("nesting is too deep");
  if (schema.anyOf) {
    if (!schema.anyOf.some(candidate => {
      try { validateInput(candidate, value, path, depth + 1); return true; }
      catch (error) { if (error instanceof ToolError) return false; throw error; }
    })) fail("does not match an accepted input shape");
  }
  const matches = (type: string): boolean => {
    if (type === "null") return value === null;
    if (type === "array") return Array.isArray(value);
    if (type === "integer") return typeof value === "number" && Number.isSafeInteger(value);
    if (type === "number") return typeof value === "number" && Number.isFinite(value);
    if (type === "object") return value !== null && typeof value === "object" && !Array.isArray(value);
    return typeof value === type;
  };
  const types = schema.type ? (Array.isArray(schema.type) ? schema.type : [schema.type]) : [];
  if (types.length && !types.some(matches)) fail(`expected ${types.join(" or ")}`);
  if (schema.enum && !schema.enum.some(option => Object.is(option, value))) fail(`expected one of ${schema.enum.join(", ")}`);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) fail("must be finite");
    if (schema.minimum !== undefined && value < schema.minimum) fail(`must be at least ${schema.minimum}`);
    if (schema.maximum !== undefined && value > schema.maximum) fail(`must be at most ${schema.maximum}`);
  }
  if (typeof value === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) fail(`must contain at least ${schema.minLength} characters`);
    if (schema.maxLength !== undefined && value.length > schema.maxLength) fail(`must contain at most ${schema.maxLength} characters`);
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) fail(`requires at least ${schema.minItems} items`);
    if (value.length > (schema.maxItems ?? 100_000)) fail("contains too many items");
    if (schema.items) value.forEach((item, index) => validateInput(schema.items!, item, `${path}[${index}]`, depth + 1));
  } else if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of schema.required ?? []) if (!Object.hasOwn(record, key)) fail(`missing ${key}`);
    for (const [key, item] of Object.entries(record)) {
      if (["__proto__", "prototype", "constructor"].includes(key)) fail(`unsupported property ${key}`);
      if (schema.properties && Object.hasOwn(schema.properties, key)) {
        validateInput(schema.properties[key], item, `${path}.${key}`, depth + 1);
      } else if (schema.additionalProperties === false) fail(`unknown property ${key}`);
      else if (typeof schema.additionalProperties === "object") validateInput(schema.additionalProperties, item, `${path}.${key}`, depth + 1);
    }
  }
}
