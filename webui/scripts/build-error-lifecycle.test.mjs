import assert from "node:assert/strict";
import { createServer } from "vite";
import { rolldown } from "rolldown";

const server = await createServer({
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null },
  plugins: [{
    name: "error-lifecycle-regression",
    resolveId(id) {
      return id.startsWith("/rejected-module-") ? `\0${id}` : null;
    },
    async load(id) {
      if (!id.startsWith("\0/rejected-module-")) return null;
      await Promise.resolve();
      throw Object.assign(new Error(`expected rejection: ${id}`), { code: "EXPECTED_BUILD_REJECTION" });
    },
  }],
});

try {
  for (let batch = 0; batch < 8; batch++) {
    await Promise.all(Array.from({ length: 8 }, (_, index) =>
      assert.rejects(server.transformRequest(`/rejected-module-${batch}-${index}.js`), (error) => {
        assert.match(error.message, /expected rejection:/);
        assert.equal(error.code, "EXPECTED_BUILD_REJECTION");
        assert.match(error.stack, /build-error-lifecycle/);
        return true;
      }),
    ));
  }
} finally {
  await server.close();
}

for (let batch = 0; batch < 8; batch++) {
  await Promise.all(Array.from({ length: 8 }, async (_, index) => {
    const marker = `native-${batch}-${index}`;
    const bundle = await rolldown({
      input: marker,
      logLevel: "silent",
      plugins: [{
        name: "error-lifecycle-regression",
        resolveId: (id) => id,
        async load() {
          await Promise.resolve();
          throw Object.assign(new Error(marker), { marker });
        },
      }],
    });
    try {
      await assert.rejects(bundle.generate({ format: "esm" }), (error) => {
        assert.equal(error.errors.length, 1);
        assert.equal(error.errors[0].message, marker);
        assert.equal(error.errors[0].marker, marker);
        assert.match(error.errors[0].stack, /build-error-lifecycle/);
        return true;
      });
    } finally {
      await bundle.close();
    }
  }));
}
console.log("build error lifecycle: 128 concurrent SSR/native plugin rejections preserve error details and close cleanly");
