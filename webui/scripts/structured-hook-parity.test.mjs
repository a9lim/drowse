import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";


const root = fileURLToPath(new URL("..", import.meta.url));
const fixture = JSON.parse(await readFile(
  new URL("../../browser-runtime/fixtures/structured-program-parity-v1.json", import.meta.url),
  "utf8",
));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const [hooks, profiles, expression, instruments] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
    server.ssrLoadModule("/src/hosted/runtime/steeringExpression.ts"),
    server.ssrLoadModule("/src/hosted/runtime/browserInstrumentPacks.ts"),
  ]);
  const profile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE;
  const geometryProfile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V2;

  const multi = hooks.compileStructuredHookProgram(
    fixture.multiCurve.hiddenSize,
    [{ curves: fixture.multiCurve.curves }],
    profile,
  );
  const multiResult = hooks.runStructuredHookLayer(
    multi,
    hooks.initialStructuredHookState(multi),
    0,
    fixture.multiCurve.residual,
    { decode: false },
  );
  close(multiResult.residual, fixture.multiCurve.expected.residual, 4e-5);
  fixture.multiCurve.expected.preSlideFeet.forEach((expected, curve) => {
    const offset = curve * profile.maxIntrinsicDim;
    close(
      multiResult.curveFeet.slice(offset, offset + expected.length),
      expected,
      4e-5,
    );
  });

  const sphere = hooks.compileStructuredHookProgram(
    fixture.sphereCurve.hiddenSize,
    [{ curves: [fixture.sphereCurve.curve] }],
    geometryProfile,
  );
  const sphereResult = hooks.runStructuredHookLayer(
    sphere,
    hooks.initialStructuredHookState(sphere),
    0,
    fixture.sphereCurve.residual,
    { decode: false },
  );
  close(sphereResult.residual, fixture.sphereCurve.expected.residual, 4e-5);
  close(
    sphereResult.curveFeet.slice(0, fixture.sphereCurve.expected.preSlideFoot.length),
    fixture.sphereCurve.expected.preSlideFoot,
    4e-5,
  );

  const saeProbe = hooks.saeFeatureProbe(
    fixture.sae.encoderDirection,
    fixture.sae.encoderBias,
    fixture.sae.decoderBias,
    fixture.sae.maxAct,
  );
  const sae = hooks.compileStructuredHookProgram(
    fixture.sae.residual.length,
    [{ probes: [saeProbe] }],
    profile,
  );
  const saeResult = hooks.runStructuredHookLayer(
    sae,
    hooks.initialStructuredHookState(sae),
    0,
    fixture.sae.residual,
  );
  close(
    saeResult.probes.slice(0, 1),
    [fixture.sae.expectedNormalizedActivation],
    2e-6,
  );
  close(
    instruments.saeDecoderDirection(
      fixture.sae.decoderDirection,
      fixture.sae.decoderDirection.length,
      0,
    ),
    fixture.sae.expectedDecoderDirection,
    2e-6,
  );
  close(
    hooks.jLensTokenDirection(
      fixture.jlens.unembeddingRow,
      fixture.jlens.jacobian.flat(),
    ),
    fixture.jlens.expectedDirection,
    2e-6,
  );

  for (const gateFixture of fixture.gates) {
    const trigger = expression.parseSteeringExpression(
      gateFixture.expression,
    ).terms[0].trigger;
    const gateProgram = hooks.compileStructuredHookProgram(1, [{
      affineGroups: [hooks.additiveDirectionGroup([1], 1)],
    }], profile);
    gateProgram.controls = {
      affine: Array.from(
        { length: gateProgram.affineActive.length },
        (_, index) => index === 0
          ? {
              enabled: true,
              phase: trigger.phase,
              gate: trigger.gate === null
                ? null
                : {
                    slots: [],
                    scoreKey: trigger.gate.probe,
                    operator: trigger.gate.operator,
                    threshold: trigger.gate.threshold,
                  },
            }
          : null,
      ),
      curve: Array.from({ length: gateProgram.curveActive.length }, () => null),
    };
    for (const context of gateFixture.contexts) {
      const controls = hooks.structuredHookControlsFor(gateProgram, {
        prefill: context.prefill,
        thinking: context.thinking,
        generatedTokens: context.generatedTokens,
        priorScores: context.scores,
      });
      assert.equal(controls.affineActive[0] === 1, context.active);
    }
  }

  console.log("Shared curved, periodic, sphere, gated, SAE, and J-lens parity checks passed");
} finally {
  await server.close();
}


function close(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `${index}: ${actual[index]} != ${expected[index]} within ${tolerance}`,
    );
  }
}
