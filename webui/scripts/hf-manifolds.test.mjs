import assert from "node:assert/strict";
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

const tests = [];
const test = (name, run) => tests.push({ name, run });
const sha = "a".repeat(40);

try {
  const { BrowserHfManifoldClient } = await server.ssrLoadModule(
    "/src/hosted/artifacts/hfManifolds.ts",
  );
  const { hostedHfManifoldDownloadOrigins } = await server.ssrLoadModule(
    "/src/hosted/artifacts/service.ts",
  );

  test("hosted imports use the same explicit origin union as the CSP", () => {
    assert.deepEqual([...hostedHfManifoldDownloadOrigins({
      catalogUrl: "https://huggingface.co/catalog.json",
      signatureUrl: "https://signatures.example.test/catalog.sig.json",
      allowedCatalogRedirectOrigins: new Set([
        "https://catalog-cdn.example.test",
      ]),
      allowedArtifactRedirectOrigins: new Set([
        "https://artifact-cdn.example.test",
      ]),
    })].sort(), [
      "https://artifact-cdn.example.test",
      "https://catalog-cdn.example.test",
      "https://huggingface.co",
      "https://signatures.example.test",
    ]);
  });

  test("search returns only immutable browser-compatible Drowse repositories", async () => {
    const requests = [];
    const fetchImpl = async (input, init) => {
      const url = new URL(input);
      requests.push({ url, init });
      if (url.pathname === "/api/models") {
        return response([{
          id: "fixture/portable",
          sha,
          tags: ["drowse-manifold"],
          siblings: [
            { rfilename: "manifold.json", size: 400 },
            { rfilename: "portable.drowse", size: 1234 },
            { rfilename: "_zdGVzdC9tb2RlbA.safetensors", size: 50 },
          ],
        }, {
          id: "fixture/legacy",
          sha,
          tags: ["drowse-manifold"],
          siblings: [{ rfilename: "manifold.json", size: 400 }],
        }, {
          id: "fixture/untagged",
          sha,
          tags: [],
          siblings: [{ rfilename: "untagged.drowse", size: 200 }],
        }], url.href);
      }
      if (url.pathname.endsWith("/manifold.json")) {
        return response({
          format_version: 10,
          name: "portable",
          description: "portable fixture",
          fit_mode: "pca",
          hyperparams: { max_dim: 2 },
          nodes: [{ label: "cold" }, { label: "hot" }],
          files: {},
          source: `hf://fixture/portable@${sha}`,
          tags: ["fixture"],
          template_ref: null,
        }, url.href);
      }
      throw new Error(`Unexpected request ${url}`);
    };
    const client = new BrowserHfManifoldClient({ fetchImpl });
    const result = await client.search(" portable ", 3);
    assert.equal(result.query, "portable");
    assert.deepEqual(result.results, [{
      name: "portable",
      namespace: "fixture",
      description: "portable fixture",
      tags: ["fixture"],
      node_count: 2,
      domain_label: "discover-pca",
      fit_mode: "pca",
      tensor_models: ["_zdGVzdC9tb2RlbA"],
      repository: "fixture/portable",
      revision: sha,
      archive_filename: "portable.drowse",
      archive_bytes: 1234,
      browser_compatible: true,
    }]);
    assert.equal(requests[0].url.searchParams.get("filter"), "drowse-manifold");
    assert.equal(requests[0].url.searchParams.get("search"), "portable");
    assert.equal(requests[0].init.credentials, "omit");
    assert.equal(requests[0].init.referrerPolicy, "no-referrer");
  });

  test("search fails closed on response bounds and redirect origins", async () => {
    const oversized = new BrowserHfManifoldClient({
      fetchImpl: async () => response([], "https://huggingface.co/api/models", {
        "Content-Length": String(2 * 1024 * 1024 + 1),
      }),
    });
    await assert.rejects(oversized.search("x"), /response is too large/);

    const redirected = new BrowserHfManifoldClient({
      fetchImpl: async () => response([], "https://example.com/api/models"),
    });
    await assert.rejects(redirected.search("x"), /untrusted origin/);
    await assert.rejects(redirected.search("x".repeat(201)), /query is too long/);
    await assert.rejects(redirected.search("x", 21), /between 1 and 20/);
  });

  test("search accepts only an explicitly approved download redirect", async () => {
    const fetchImpl = async (input) => {
      const url = new URL(input);
      if (url.pathname === "/api/models") {
        return response([{
          id: "fixture/portable",
          sha,
          tags: ["drowse-manifold"],
          siblings: [
            { rfilename: "manifold.json", size: 400 },
            { rfilename: "portable.drowse", size: 1234 },
          ],
        }], url.href);
      }
      if (url.pathname.endsWith("/manifold.json")) {
        return response({
          format_version: 10,
          name: "portable",
          description: "redirected fixture",
          fit_mode: "pca",
          nodes: [{ label: "cold" }, { label: "hot" }],
          tags: [],
        }, "https://artifact-cdn.example.test/manifold.json");
      }
      throw new Error(`Unexpected request ${url}`);
    };
    const approved = new BrowserHfManifoldClient({
      fetchImpl,
      allowedDownloadOrigins: new Set([
        "https://huggingface.co",
        "https://artifact-cdn.example.test",
      ]),
    });
    assert.equal((await approved.search("portable")).results.length, 1);

    const rejected = new BrowserHfManifoldClient({ fetchImpl });
    assert.deepEqual((await rejected.search("portable")).results, []);
  });

  test("archive resolution requires one tagged immutable pack before OPFS access", async () => {
    const untagged = new BrowserHfManifoldClient({
      fetchImpl: async (input) => response({
        id: "fixture/portable",
        sha,
        tags: [],
        siblings: [{ rfilename: "portable.drowse", size: 10 }],
      }, String(input)),
    });
    await assert.rejects(
      untagged.withArchive("fixture/portable", {}, async () => null),
      /not tagged/,
    );

    const ambiguous = new BrowserHfManifoldClient({
      fetchImpl: async (input) => response({
        id: "fixture/portable",
        sha,
        tags: ["drowse-manifold"],
        siblings: [
          { rfilename: "one.drowse", size: 10 },
          { rfilename: "two.drowse", size: 10 },
        ],
      }, String(input)),
    });
    await assert.rejects(
      ambiguous.withArchive(`fixture/portable@${sha}`, {}, async () => null),
      /exactly one root/,
    );
    await assert.rejects(
      ambiguous.withArchive("../escape", {}, async () => null),
      /owner\/repository/,
    );
  });

  let passed = 0;
  for (const { name, run } of tests) {
    await run();
    passed += 1;
    process.stdout.write(`ok ${passed} - ${name}\n`);
  }
  process.stdout.write(`1..${passed}\n`);
} finally {
  await server.close();
}

function response(value, url, headers = {}) {
  const body = JSON.stringify(value);
  const result = new Response(body, {
    status: 200,
    headers: { "Content-Type": "application/json", ...headers },
  });
  Object.defineProperty(result, "url", { value: url });
  return result;
}
