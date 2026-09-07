import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    parseSteeringExpression,
    referencedSteeringSelectors,
    SteeringExpressionError,
  } = await server.ssrLoadModule("/src/hosted/runtime/steeringExpression.ts");
  const sharedFixture = JSON.parse(await readFile(
    new URL("../../tests/fixtures/browser_steering_expression.json", import.meta.url),
    "utf8",
  ));
  assert.equal(sharedFixture.schemaVersion, 1);
  for (const testCase of sharedFixture.accepted) {
    assert.doesNotThrow(
      () => parseSteeringExpression(testCase.expression),
      testCase.expression,
    );
    assert.deepEqual(
      referencedSteeringSelectors(testCase.expression).map((item) => [
        item.namespace,
        item.concept,
        item.variant,
      ]),
      testCase.selectors,
      testCase.expression,
    );
  }
  for (const expression of sharedFixture.rejected) {
    assert.throws(
      () => parseSteeringExpression(expression),
      (error) => error instanceof SteeringExpressionError,
      expression,
    );
  }

  const parsed = parseSteeringExpression(
    "-0.6,0.3 default/emotions:role%pirate@after&when:personas[3]>=-0.4",
  );
  assert.equal(parsed.terms.length, 1);
  assert.deepEqual(parsed.terms[0].coefficients, [-0.6, -0.3]);
  assert.deepEqual(parsed.terms[0].selector, {
    base: {
      namespace: "default",
      concept: "emotions",
      variant: "role",
      column: 9,
    },
    projection: null,
    onto: null,
    manifoldPosition: "pirate",
  });
  assert.deepEqual(parsed.terms[0].trigger, {
    phase: { kind: "after_thinking" },
    gate: { probe: "personas[3]", operator: ">=", threshold: -0.4 },
  });

  assert.deepEqual(
    referencedSteeringSelectors(
      "0.4 alice/happy.sad:sae-release~jlens/fake + !sae/12@first:5 + mood%calm",
    ),
    [
      { namespace: "alice", concept: "happy.sad", variant: "sae-release" },
      { namespace: "jlens", concept: "fake", variant: "raw" },
      { namespace: "sae", concept: "12", variant: "raw" },
    ],
  );

  const gates = [
    "x@when:default/emotions:fraction>0.2",
    "x@when:emotions:membership<=.5",
    "x@when:emotions@happy>-2",
    "x@when:emotions~happy>=+0.25",
    "x@when:sae/42<1",
    "x@thinking&when:angry.calm[0]>0",
  ].map((expression) => parseSteeringExpression(expression).terms[0].trigger);
  assert.deepEqual(gates.map((trigger) => trigger.gate.probe), [
    "default/emotions:fraction",
    "emotions:membership",
    "emotions@happy",
    "emotions~happy",
    "sae/42",
    "angry.calm[0]",
  ]);

  console.log("Steering expression checks passed");
} finally {
  await server.close();
}
