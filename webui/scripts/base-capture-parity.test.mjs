import assert from "node:assert/strict";
import { compareCaptures } from "./base-capture-parity.mjs";

const capture = values => ({ layerCount: 2, positionCount: 1, hiddenSize: 2, values });
const reference = capture([1, 2, 10, 20]);
assert.equal(compareCaptures(reference, reference).passed, true);
const rounding = capture([1, 2, 10.00003, 20]);
assert.equal(compareCaptures(reference, rounding).passed, false);
assert.equal(compareCaptures(reference, rounding, "fp32").passed, true);
assert.equal(compareCaptures(reference, capture([1, 2, 10.001, 20]), "fp32").passed, false);
assert.equal(compareCaptures(capture([0.001, 0.002, 10, 20]),
  capture([0.00101, 0.002, 10, 20]), "fp32").passed, false,
"a quiet layer cannot hide behind the larger norm of other layers");
assert.equal(compareCaptures(capture([0, 0, 0, 0]), capture([0, 0, 0, 0]), "fp32").passed, true);
assert.throws(() => compareCaptures(reference, capture([1, NaN, 10, 20])), /non-finite/);
assert.throws(() => compareCaptures(reference, capture([1, 2, Infinity, 20])), /non-finite/);
assert.throws(() => compareCaptures(reference, capture([1, 2])), /shape/);
assert.throws(() => compareCaptures(reference, { ...reference, layerCount: 1 }), /shapes/);
assert.throws(() => compareCaptures(reference, reference, "loose"), /policy/);
console.log("Capture replay policies: strict default, explicit dual-bound FP32, layer isolation, and invalid values passed");
