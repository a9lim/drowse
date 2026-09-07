import assert from "node:assert/strict";
import { join } from "node:path";
import {
  parseArguments,
  semanticValidationArguments,
  targetRepository,
} from "./publish-hosted-pack.mjs";

assert.deepEqual(parseArguments(["qwen3-0.6b", "core", "./core"]), {
  modelId: "qwen3-0.6b",
  kind: "core",
  directory: join(process.cwd(), "core"),
  upload: false,
});
assert.equal(
  parseArguments(["qwen3-0.6b", "sae", "./sae", "--upload"]).upload,
  true,
);
assert.throws(() => parseArguments(["qwen3-0.6b", "unknown", "./pack"]), /usage/);
assert.throws(() => parseArguments(["qwen3-0.6b", "core", "./pack", "--force"]), /usage/);

const model = { convertedRepository: "logitsml/drowse-web-qwen3-0.6b" };
assert.equal(targetRepository(model, "core"), model.convertedRepository);
assert.equal(
  targetRepository(model, "sae"),
  "logitsml/drowse-web-qwen3-0.6b-instruments",
);
assert.equal(
  targetRepository(model, "jlens"),
  "logitsml/drowse-web-qwen3-0.6b-instruments",
);
assert.throws(() => targetRepository({}, "core"), /converted repository/);
assert.deepEqual(
  semanticValidationArguments("fixture", "core", "/tmp/core").slice(1),
  ["fixture", "/tmp/core"],
);
assert.deepEqual(
  semanticValidationArguments("fixture", "jlens", "/tmp/jlens").slice(1),
  ["fixture", "jlens", "/tmp/jlens"],
);
assert.deepEqual(
  semanticValidationArguments("fixture", "jlens", "/tmp/jlens", true).slice(1),
  ["fixture", "jlens", "/tmp/jlens"],
);

console.log("hosted pack publisher checks passed");
