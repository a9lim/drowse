import { objectSchema, type InputSchema } from "./webmcp/types";

const text: InputSchema = { type: "string", maxLength: 200000 };
const flag: InputSchema = { type: "boolean" };
const real: InputSchema = { type: "number" };
const int = (minimum: number, maximum = Number.MAX_SAFE_INTEGER): InputSchema => ({ type: "integer", minimum, maximum });
const choices = (...values: string[]): InputSchema => ({ type: "string", enum: values });
const nullable = (schema: InputSchema): InputSchema => ({ anyOf: [schema, { type: "null" }] });
const list = (items: InputSchema, maxItems = 1000): InputSchema => ({ type: "array", items, maxItems });
const tuning = objectSchema({ fitMode: choices("auto", "pca", "spectral"), maxDim: int(1), varThreshold: { type: "number", minimum: 0, maximum: 1 }, kNN: nullable(int(1)), bandwidth: nullable({ type: "number", minimum: 0 }) });
export const manifoldIdentitySchema = objectSchema({ mode: choices("discover", "authored", "templated"), namespace: text, name: text, description: text });
export const discoverDraftSchema = objectSchema({ concepts_text: text, kind: choices("abstract", "concrete", "custom"), custom_system: text, samples_per_prompt: int(1), role_per_node: flag, force: flag, sae_release: text, also_fit: flag, advanced_open: flag, tuning });
export const authoredDraftSchema = objectSchema({
  auto_domain: flag, domain_kind: choices("box", "sphere", "klein", "projective"), box_dim: int(1, 3), sphere_dim: int(1, 3), advanced_open: flag, tuning,
  axes: { ...list(objectSchema({ name: text, lo: real, hi: real, periodic: flag }, ["name", "lo", "hi", "periodic"]), 3), minItems: 3 },
  nodes: list(objectSchema({ label: text, coords: list(real, 3), statements: text, role: text, expanded: flag }, ["label", "coords", "statements", "role", "expanded"])),
});
export const templatedDraftSchema = objectSchema({ selected_key: text, max_dim: nullable(int(1)), also_fit: flag, advanced_open: flag, tuning });
export const templateDraftSchema = objectSchema({
  tab: choices("score", "build"), selected_key: text, steering: text, score_by: choices("sum", "mean"), name: text, slot: text, values_text: text,
  contexts: list(objectSchema({ turns: list(objectSchema({ role: choices("user", "assistant", "system"), content: text }, ["role", "content"])), assistant: text }, ["turns", "assistant"])),
});
export const mergeDraftSchema = objectSchema({ sources: list(text), target_name: text, fit_mode: choices("", "pca", "spectral") });
export const nodeCompareViewSchema = objectSchema({ layout: choices("side-by-side", "unified"), sort_by: choices("magnitude", "name") });
export const profileCompareViewSchema = objectSchema({ profile_a: text, profile_b: text });
export const geometryViewSchema = objectSchema({ layer: int(0), zoom: { type: "number", minimum: 0.3, maximum: 6 }, rotate: objectSchema({ dx: real, dy: real }, ["dx", "dy"]), reset: flag });
export const loomViewSchema = objectSchema({ zoom: { type: "number", minimum: 0.01, maximum: 1.4 }, camera: choices("fit", "current"), search_match: choices("next", "previous", "first"), collapsed_ids: list(text) });
export const chatViewSchema = objectSchema({ comparison_visible: flag });
export const tokenInspectorSchema = objectSchema({
  cursor: objectSchema({ turnIdx: int(0), seg: choices("thinking", "response"), tokenIdx: int(0) }, ["turnIdx", "seg", "tokenIdx"]),
  branch: choices("primary", "shadow"), tab: choices("logits", "geometry", "sae", "lens"),
  apply_recipe: objectSchema({ geometry: flag, sae: flag, lens: flag }),
  move: choices("next", "previous", "next_turn", "previous_turn", "segment_start", "segment_end", "anchor", "other_segment"), presentation: choices("dock", "drawer", "hide"),
});
