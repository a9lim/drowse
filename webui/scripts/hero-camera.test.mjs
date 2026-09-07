import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { HERO_FOCAL_LENGTH, HERO_VOLUME_RADIUS, heroCameraPosition } from "../src/hosted/ui/heroCamera.ts";

for (const [aspect, mobile] of [[16 / 9, false], [390 / 844, true]]) {
  const start = heroCameraPosition(0, aspect, mobile);
  const end = heroCameraPosition(1, aspect, mobile);
  assert.ok(Math.hypot(...start) > HERO_VOLUME_RADIUS);
  assert.ok(Math.hypot(...end) < HERO_VOLUME_RADIUS, "Camera must cross into the volume");
  assert.ok(end[2] < 0, "Camera must pass the center plane");
  let previous = start;
  for (let step = 1; step <= 100; step++) {
    const camera = heroCameraPosition(step / 100, aspect, mobile);
    assert.ok(camera[2] < previous[2]);
    assert.ok(Math.hypot(...camera.map((value, i) => value - previous[i])) < 0.15);
    previous = camera;
  }
  assert.deepEqual(heroCameraPosition(-1, aspect, mobile), start);
  assert.deepEqual(heroCameraPosition(2, aspect, mobile), end);
  const middle = heroCameraPosition(0.5, aspect, mobile);
  let previousZ = heroCameraPosition(0, aspect, mobile, 0)[2];
  let previousStep = Infinity;
  for (let step = 1; step <= 100; step++) {
    const z = heroCameraPosition(0, aspect, mobile, step / 100)[2];
    const distance = previousZ - z;
    assert.ok(distance >= 0 && distance <= previousStep, "Entrance must ease toward the orb without overshoot");
    previousZ = z;
    previousStep = distance;
  }
  assert.equal(previousZ, start[2]);
  assert.deepEqual(heroCameraPosition(1, aspect, mobile, 0), end, "Entrance must not pull a restored scroll position outside the orb");
  const projectX = (point, camera) => (point[0] - camera[0]) * HERO_FOCAL_LENGTH / (camera[2] - point[2]);
  const near = [0.2, 0, 1], far = [0.2, 0, -1];
  const nearMagnification = projectX(near, middle) / projectX(near, start);
  const farMagnification = projectX(far, middle) / projectX(far, start);
  assert.ok(Math.abs(nearMagnification - farMagnification) > 0.1, "Depth layers must move differently, not scale together");
}
const shader = await readFile(new URL("../src/hosted/ui/etherealShaders.ts", import.meta.url), "utf8");
assert.ok(shader.includes("uCamera + ray * distance"));
assert.ok(shader.includes("vec3 relative = p - uCamera"));
assert.ok(shader.includes("depth <= 0.06"));
assert.ok(!shader.includes("uFit") && !shader.includes("sourceUV"));
const component = await readFile(new URL("../src/hosted/ui/HeroShader.svelte", import.meta.url), "utf8");
assert.ok(!component.includes("pointermove") && !shader.includes("uPointer"));
console.log("hero camera: volume crossing, depth parallax, near clipping, and fixed projection passed");
