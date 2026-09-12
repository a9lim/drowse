import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const browserRuntime = fileURLToPath(new URL("../../browser-runtime/", import.meta.url));
const PREFLIGHT_ORIGIN = "https://drowse.invalid";

export async function main(environment = process.env) {
  const [distribution, runtimeLock] = await Promise.all([
    readJson(new URL("distribution-lock.json", `file://${browserRuntime}/`)),
    readJson(new URL("runtime-lock.json", `file://${browserRuntime}/`)),
  ]);
  if (distribution.status !== "verified") {
    throw new Error("Distribution preflight requires a verified distribution lock");
  }
  if (runtimeLock.status !== "verified") {
    throw new Error("Distribution preflight requires a verified runtime lock");
  }

  const artifactTimeoutMs = positiveTimeout(
    environment.DROWSE_PREFLIGHT_ARTIFACT_TIMEOUT_MS,
    2 * 60 * 60 * 1_000,
  );
  const catalogOrigins = new Set(distribution.allowedCatalogRedirectOrigins);
  const artifactOrigins = new Set(distribution.allowedArtifactRedirectOrigins);
  const server = await createServer({
    root,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
  server: { middlewareMode: true, watch: null },
  });

  try {
    const [{ verifyCatalog, WebCryptoEd25519Verifier }, { HOSTED_DISTRIBUTION_CONFIG }] =
      await Promise.all([
        server.ssrLoadModule("/src/lib/runtime/catalog.ts"),
        server.ssrLoadModule("/src/hosted/runtime/distributionConfig.ts"),
      ]);
    const [catalogResponse, signatureResponse] = await Promise.all([
      fetchChecked(distribution.catalogUrl, catalogOrigins, 60_000),
      fetchChecked(distribution.signatureUrl, catalogOrigins, 60_000),
    ]);
    const [catalogBytes, signatureBytes] = await Promise.all([
      readBounded(catalogResponse, 4 * 1024 * 1024, "catalog"),
      readBounded(signatureResponse, 16 * 1024, "catalog signature"),
    ]);
    const signature = parseJson(signatureBytes, "catalog signature");
    const catalog = await verifyCatalog(
      catalogBytes,
      signature,
      new WebCryptoEd25519Verifier(HOSTED_DISTRIBUTION_CONFIG.publicKeys),
      {
        now: Date.now(),
        online: true,
        minimumSequence: HOSTED_DISTRIBUTION_CONFIG.minimumAcceptedSequence,
        lastAcceptedSequence: null,
        expectedRuntimeAbi: HOSTED_DISTRIBUTION_CONFIG.expectedRuntimeAbi,
        runtimeLockModels: HOSTED_DISTRIBUTION_CONFIG.runtimeLockModels,
      },
    );
    assertCatalogSequenceFloor(
      catalog.document.sequence,
      HOSTED_DISTRIBUTION_CONFIG.minimumAcceptedSequence,
    );
    const files = uniqueCatalogFiles(catalog.document);
    let verifiedBytes = 0;
    for (const file of files) {
      await preflightArtifact(file, artifactOrigins, artifactTimeoutMs);
      verifiedBytes += file.bytes;
      console.log(`Verified ${file.path} (${file.bytes} bytes)`);
    }
    console.log(
      `Hosted distribution preflight passed: catalog sequence ${catalog.document.sequence}, ` +
        `${files.length} immutable files, ${verifiedBytes} bytes`,
    );
  } finally {
    await server.close();
  }
}

export function assertCatalogSequenceFloor(sequence, minimumAcceptedSequence) {
  if (sequence < minimumAcceptedSequence) {
    throw new Error(
      `Catalog sequence ${sequence} is below the embedded rollback floor ` +
        `${minimumAcceptedSequence}`,
    );
  }
}

class TransientDownloadError extends Error {
  constructor(message, retryAfterMs) {
    super(message);
    this.retryAfterMs = retryAfterMs;
  }
}

export function retryAfterMilliseconds(value, now = Date.now()) {
  if (value === null || value.trim() === "") return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return seconds >= 0 ? seconds * 1_000 : null;
  const deadline = Date.parse(value);
  return Number.isFinite(deadline) ? Math.max(0, deadline - now) : null;
}

export async function preflightArtifact(
  file, allowedOrigins, timeoutMs, fetcher = fetch,
  wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      return await verifyArtifact(file, allowedOrigins, timeoutMs, fetcher);
    } catch (error) {
      const transient = error instanceof TransientDownloadError || [
        "UND_ERR_SOCKET", "UND_ERR_CONNECT_TIMEOUT", "UND_ERR_HEADERS_TIMEOUT",
        "UND_ERR_BODY_TIMEOUT", "ECONNRESET", "ETIMEDOUT", "EAI_AGAIN",
      ].includes(error?.cause?.code ?? error?.code);
      if (!transient || attempt === 2) throw error;
      const delay = error.retryAfterMs ?? 5_000 * 2 ** attempt;
      if (delay > Math.min(timeoutMs, 300_000)) throw error;
      console.warn(`Retrying ${file.path} after a transient download failure in ${delay} ms (${attempt + 1}/2)`);
      await wait(delay);
    }
  }
}

async function verifyArtifact(file, allowedOrigins, timeoutMs, fetcher) {
  const end = Math.min(file.bytes, 8 * 1024 * 1024) - 1;
  const rangeResponse = await fetchChecked(file.url, allowedOrigins, timeoutMs, {
    Range: `bytes=0-${end}`,
  }, fetcher);
  if (rangeResponse.status !== 206) {
    throw new Error(`${file.path} range request returned HTTP ${rangeResponse.status}`);
  }
  const expectedRange = `bytes 0-${end}/${file.bytes}`;
  if (rangeResponse.headers.get("Content-Range") !== expectedRange) {
    throw new Error(`${file.path} returned an incompatible Content-Range`);
  }
  requireIntegerHeader(rangeResponse, "Content-Length", end + 1, file.path);
  const exposed = exposedHeaders(rangeResponse);
  if (!exposed.has("content-range") && !exposed.has("*")) {
    throw new Error(`${file.path} does not expose Content-Range to browsers`);
  }
  if (rangeResponse.headers.has("ETag") && !exposed.has("etag") && !exposed.has("*")) {
    throw new Error(`${file.path} returns ETag without exposing it to browsers`);
  }
  await consumeExact(rangeResponse, end + 1, `${file.path} range`);

  const fullResponse = await fetchChecked(file.url, allowedOrigins, timeoutMs, {}, fetcher);
  if (fullResponse.status !== 200) {
    throw new Error(`${file.path} full request returned HTTP ${fullResponse.status}`);
  }
  requireFullResponseLength(fullResponse, file.bytes, file.path);
  const digest = createHash("sha256");
  const received = await consume(fullResponse, (chunk) => {
    digest.update(chunk);
  });
  if (received !== file.bytes) {
    throw new Error(`${file.path} returned ${received} bytes instead of ${file.bytes}`);
  }
  if (digest.digest("hex") !== file.sha256) {
    throw new Error(`${file.path} failed its SHA-256 preflight`);
  }
  return undefined;
}

export async function fetchChecked(
  url,
  allowedOrigins,
  timeoutMs,
  extraHeaders = {},
  fetcher = fetch,
) {
  assertHttpsUrl(url, "distribution URL");
  const approvedOrigins = new Set([new URL(url).origin, ...allowedOrigins]);
  let currentUrl = url;
  for (let redirects = 0; redirects <= 5; redirects += 1) {
    const current = validatedApprovedUrl(currentUrl, approvedOrigins, url);
    const response = await fetcher(current, {
      cache: "no-store",
      credentials: "omit",
      redirect: "manual",
      referrerPolicy: "no-referrer",
      headers: { Origin: PREFLIGHT_ORIGIN, ...extraHeaders },
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (response.status === 429 || response.status === 503) {
      const retryAfterMs = retryAfterMilliseconds(response.headers.get("Retry-After"))
        ?? (response.status === 429 ? 300_000 : null);
      await response.body?.cancel().catch(() => undefined);
      const location = new URL(current);
      throw new TransientDownloadError(
        `${location.origin}${location.pathname} returned HTTP ${response.status}`,
        retryAfterMs,
      );
    }
    const allowedOrigin = response.headers.get("Access-Control-Allow-Origin");
    if (allowedOrigin !== "*" && allowedOrigin !== PREFLIGHT_ORIGIN) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`${current} did not allow the requested browser origin`);
    }
    if ([301, 302, 303, 307, 308].includes(response.status)) {
      const location = response.headers.get("Location");
      await response.body?.cancel().catch(() => undefined);
      if (!location) throw new Error(`${current} returned a redirect without Location`);
      if (redirects === 5) throw new Error(`${url} exceeded five approved redirects`);
      currentUrl = new URL(location, current).href;
      continue;
    }
    if (!response.ok) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`${current} returned HTTP ${response.status}`);
    }
    return response;
  }
  throw new Error(`${url} exceeded five approved redirects`);
}

export function validatedApprovedUrl(value, approvedOrigins, signedUrl) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${signedUrl} resolved to an invalid URL`);
  }
  if (
    url.protocol !== "https:" || url.username || url.password ||
    !approvedOrigins.has(url.origin)
  ) {
    throw new Error(`${signedUrl} redirected to unapproved URL ${url.href}`);
  }
  return url.href;
}

function uniqueCatalogFiles(document) {
  const files = document.models.flatMap((model) => model.variants.flatMap((variant) => [
    ...variant.files,
    ...variant.packs.flatMap((pack) => pack.files),
  ]));
  const urls = new Map();
  for (const file of files) {
    const previous = urls.get(file.url);
    if (previous && (
      previous.bytes !== file.bytes || previous.sha256 !== file.sha256 ||
      previous.revision !== file.revision
    )) throw new Error(`Catalog URL ${file.url} has conflicting immutable metadata`);
    urls.set(file.url, file);
  }
  return [...urls.values()];
}

async function readBounded(response, maximumBytes, label) {
  const chunks = [];
  let total = 0;
  await consume(response, (chunk) => {
    total += chunk.byteLength;
    if (total > maximumBytes) throw new Error(`${label} exceeds ${maximumBytes} bytes`);
    chunks.push(chunk);
  });
  if (total === 0) throw new Error(`${label} is empty`);
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function consumeExact(response, expectedBytes, label) {
  const received = await consume(response, () => undefined);
  if (received !== expectedBytes) {
    throw new Error(`${label} returned ${received} bytes instead of ${expectedBytes}`);
  }
}

async function consume(response, onChunk) {
  if (!response.body) throw new Error(`${response.url} returned no response body`);
  const reader = response.body.getReader();
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return total;
      if (!value?.byteLength) continue;
      total += value.byteLength;
      onChunk(value);
    }
  } finally {
    reader.releaseLock();
  }
}

function exposedHeaders(response) {
  return new Set(
    (response.headers.get("Access-Control-Expose-Headers") ?? "")
      .split(",")
      .map((value) => value.trim().toLowerCase())
      .filter(Boolean),
  );
}

function requireIntegerHeader(response, name, expected, label) {
  const value = response.headers.get(name);
  if (value === null || !/^\d+$/.test(value) || Number(value) !== expected) {
    throw new Error(`${label} has an invalid ${name}`);
  }
}

function requireFullResponseLength(response, expected, label) {
  const value = response.headers.get("Content-Length");
  if (value === null && response.headers.has("Content-Encoding")) return;
  if (value === null || !/^\d+$/.test(value) || Number(value) !== expected) {
    throw new Error(`${label} has an invalid Content-Length`);
  }
}

function parseJson(bytes, label) {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new Error(`${label} is not valid UTF-8 JSON`);
  }
}

async function readJson(url) {
  return JSON.parse(await readFile(url, "utf8"));
}

function assertHttpsUrl(value, label) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${label} is invalid`);
  }
  if (url.protocol !== "https:") throw new Error(`${label} must use HTTPS`);
}

function positiveTimeout(value, fallback) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error("DROWSE_PREFLIGHT_ARTIFACT_TIMEOUT_MS must be a positive integer");
  }
  return parsed;
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  await main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
