#!/usr/bin/env node

import { execFile as execFileCallback } from "node:child_process";
import { createHash, createPublicKey, generateKeyPairSync } from "node:crypto";
import { createReadStream } from "node:fs";
import { createServer as createHttpsServer } from "node:https";
import {
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, resolve, sep } from "node:path";
import { pipeline } from "node:stream/promises";
import { promisify } from "node:util";
import { build as viteBuild, createServer as createViteServer } from "vite";
import {
  BROWSER_PRODUCTION_APP_TOOL_VERSION,
  classifyInstalledAppState,
  createProductionAppCatalog,
  createProductionAppFailureReport,
  isExpectedOfflineCatalogFailure,
  isOfflineServiceWorkerMaintenanceRequest,
  isPublishedSaeDescriptionUrl,
  isVerifiedLocalArtifactAbort,
  legacyWebLlmPersistentState,
  parseBrowserProductionAppArguments,
  parseSingleByteRange,
  stripHostedDevelopmentFixture,
  validatorRuntimeLockArguments,
} from "./browser-production-app-runtime-contract.mjs";
import {
  isBrowserFullRuntimeWebGpuError,
  validateBrowserFullRuntimeModelClosure,
} from "./browser-full-runtime-contract.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
} from "./release-tool-storage-quota.mjs";

const execFile = promisify(execFileCallback);
const CATALOG_REPOSITORY = "logitsml/drowse-web-catalog-e2e";
const PROMPT_SENTINEL = "DROWSE_PRODUCTION_APP_E2E_PROMPT";
const CORE_GEOMETRY_SELECTOR = "default/welcoming.detached";
const KEY_ID = "production-app-current";
const SECONDARY_KEY_ID = "production-app-next";
const GENERATION_TIMEOUT_MS = 5 * 60_000;
const SERVICE_WORKER_TIMEOUT_MS = 30_000;
const options = parseBrowserProductionAppArguments(process.argv.slice(2));
const nativeWebMcp = process.env.DROWSE_WEBMCP_NATIVE_EVIDENCE === "1"
  ? await (await import("./webmcp-live-scenario.mjs")).nativeEvidenceBrowserOptions() : null;
const isDescriptionRequest = (url) => isPublishedSaeDescriptionUrl(url, options.modelId);
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const runtimeLockPath = options.runtimeLock === null
  ? resolve(repositoryRoot, "browser-runtime/runtime-lock.json")
  : resolve(options.runtimeLock);
const temporaryRoot = await mkdtemp(resolve(tmpdir(), "drowse-production-app-e2e-"));
const outputDirectory = resolve(temporaryRoot, "dist-hosted");
const cacheDirectory = resolve(temporaryRoot, "vite-cache");
const catalogPath = resolve(temporaryRoot, "catalog.json");
const signaturePath = resolve(temporaryRoot, "catalog.sig.json");
const privateKeyPath = resolve(temporaryRoot, "catalog-private.pem");
const serverRequests = [];
const offlineServerRequests = [];
const offlineWorkbenchRequests = [];
const runMeasurements = {};
let offlineArmed = false;
let offlineWorkbenchArmed = false;
let browser = null;
let releaseBrowser = null;
let server = null;
let currentStage = "initialization";
const lifecycleEvents = [];
const diagnosticArtifactRequests = [];

try {
  const runtimeLock = JSON.parse(
    await readFile(runtimeLockPath, "utf8"),
  );
  const sourceDistributionLock = JSON.parse(
    await readFile(resolve(repositoryRoot, "browser-runtime/distribution-lock.json"), "utf8"),
  );
  const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
  if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);

  const sets = {
    model: await artifactSet(resolve(options.modelDirectory), "hosted-artifacts.json", {
      modelLibrary: resolve(options.modelLibrary),
    }),
    core: await artifactSet(resolve(options.coreDirectory), "hosted-pack-artifacts.json", {
      role: "core_pack",
    }),
    jlens: await artifactSet(resolve(options.jlensDirectory), "hosted-pack-artifacts.json", {
      role: "instrument",
    }),
    sae: await artifactSet(resolve(options.saeDirectory), "hosted-pack-artifacts.json", {
      role: "instrument",
    }),
  };
  Object.assign(runMeasurements, {
    modelFiles: sets.model.files.length,
    modelBytes: sumBytes(sets.model.files),
    coreFiles: sets.core.files.length,
    coreBytes: sumBytes(sets.core.files),
    jlensFiles: sets.jlens.files.length,
    jlensBytes: sumBytes(sets.jlens.files),
    saeFiles: sets.sae.files.length,
    saeBytes: sumBytes(sets.sae.files),
  });
  validateBrowserFullRuntimeModelClosure({
    lock,
    runtimeLock,
    manifest: sets.model.manifest,
    files: sets.model.files,
    contextTokens: options.contextTokens,
  });
  lockedModelExecutionProfiles(lock, sets.model.manifest);
  const validatorRuntimeLock = validatorRuntimeLockArguments(
    options.runtimeLock === null ? null : runtimeLockPath,
  );
  await runNode("validate-hosted-core-pack.mjs", [
    options.modelId,
    sets.core.directory,
    ...validatorRuntimeLock,
  ]);
  await runNode("validate-hosted-instrument-pack.mjs", [
    options.modelId,
    "jlens",
    sets.jlens.directory,
    ...validatorRuntimeLock,
  ]);
  await runNode("validate-hosted-instrument-pack.mjs", [
    options.modelId,
    "sae",
    sets.sae.directory,
    ...validatorRuntimeLock,
  ]);

  const revision = deterministicRevision(sets);
  const now = Date.now();
  const catalog = createProductionAppCatalog({
    runtimeLock,
    lock,
    revision,
    modelFiles: sets.model.files,
    coreFiles: sets.core.files,
    jlensFiles: sets.jlens.files,
    saeFiles: sets.sae.files,
    contextTokens: options.contextTokens,
    issuedAt: new Date(now - 60_000).toISOString(),
    expiresAt: new Date(now + 24 * 60 * 60 * 1000).toISOString(),
  });
  await validateCatalogDocument(catalog.document);
  const catalogBytes = `${JSON.stringify(catalog.document, null, 2)}\n`;
  await writeFile(catalogPath, catalogBytes, { flag: "wx" });

  const currentKey = generateKeyPairSync("ed25519");
  const nextKey = generateKeyPairSync("ed25519");
  await writeFile(
    privateKeyPath,
    currentKey.privateKey.export({ format: "pem", type: "pkcs8" }),
    { flag: "wx", mode: 0o600 },
  );
  await runNode("sign-catalog.mjs", [catalogPath, privateKeyPath, KEY_ID, signaturePath]);
  const signatureBytes = await readFile(signaturePath);
  const publicKeys = [
    { keyId: KEY_ID, ed25519PublicKeyBase64: rawEd25519(currentKey.publicKey) },
    { keyId: SECONDARY_KEY_ID, ed25519PublicKeyBase64: rawEd25519(nextKey.publicKey) },
  ];
  const catalogUrl = artifactUrl(CATALOG_REPOSITORY, revision, "catalog.json");
  const signatureUrl = artifactUrl(CATALOG_REPOSITORY, revision, "catalog.sig.json");
  const localArtifacts = new Map();
  const tlsKeyPath = resolve(temporaryRoot, "loopback-key.pem");
  const tlsCertificatePath = resolve(temporaryRoot, "loopback-certificate.pem");
  await execFile("openssl", [
    "req", "-x509", "-newkey", "rsa:2048", "-nodes",
    "-keyout", tlsKeyPath,
    "-out", tlsCertificatePath,
    "-subj", "/CN=127.0.0.1",
    "-addext", "subjectAltName=IP:127.0.0.1",
    "-days", "1",
  ]);
  const tlsCertificate = await readFile(tlsCertificatePath);
  const certificateSpkiSha256 = createHash("sha256")
    .update(createPublicKey(tlsCertificate).export({ type: "spki", format: "der" }))
    .digest("base64");
  server = createStaticServer(
    outputDirectory,
    serverRequests,
    offlineServerRequests,
    offlineWorkbenchRequests,
    () => offlineArmed,
    () => offlineWorkbenchArmed,
    localArtifacts,
    {
      key: await readFile(tlsKeyPath),
      cert: tlsCertificate,
    },
  );
  await listen(server);
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("failed to bind production-app server");
  }
  const origin = `https://127.0.0.1:${address.port}`;
  const runtimeOverride = {
    ...runtimeLock,
    status: "verified",
    models: [{ ...lock, convertedRevision: revision }],
  };
  const distributionOverride = {
    ...sourceDistributionLock,
    status: "verified",
    catalogUrl,
    signatureUrl,
    minimumAcceptedSequence: catalog.document.sequence,
    publicKeys,
    allowedCatalogRedirectOrigins: ["https://huggingface.co"],
    allowedArtifactRedirectOrigins: ["https://huggingface.co", origin],
  };
  const evidenceOverrides = {
    "runtime-lock.json": runtimeOverride,
    "distribution-lock.json": distributionOverride,
  };
  const loadedEvidence = new Set();

  await viteBuild({
    configFile: resolve(webuiRoot, "vite.hosted.config.ts"),
    cacheDir: cacheDirectory,
    logLevel: "warn",
    plugins: [productionEvidencePlugin(evidenceOverrides, loadedEvidence)],
    worker: {
      plugins: () => [productionEvidencePlugin(evidenceOverrides, loadedEvidence)],
    },
    build: { outDir: outputDirectory, emptyOutDir: true },
  });
  const missingEvidence = Object.keys(evidenceOverrides).filter((name) => !loadedEvidence.has(name));
  if (missingEvidence.length > 0) {
    throw new Error(`production-app build did not load evidence modules: ${missingEvidence.join(", ")}`);
  }
  await runNode("check-build-isolation.mjs", ["hosted", outputDirectory]);

  const artifactEntries = createArtifactEntries(catalog.document, sets);
  for (const entry of artifactEntries) {
    const existing = localArtifacts.get(entry.sha256);
    if (existing && (existing.bytes !== entry.bytes || existing.absolute !== entry.absolute)) {
      const [left, right] = await Promise.all([
        readFile(existing.absolute),
        readFile(entry.absolute),
      ]);
      if (!left.equals(right)) {
        throw new Error(`duplicate artifact digest has different bytes: ${entry.sha256}`);
      }
    }
    if (!existing) localArtifacts.set(entry.sha256, entry);
  }
  const routedResources = new Map([
    [catalogUrl, { kind: "catalog", bytes: Buffer.from(catalogBytes) }],
    [signatureUrl, { kind: "signature", bytes: signatureBytes }],
    ...artifactEntries.map((entry) => [entry.url, { kind: "artifact", ...entry }]),
  ]);
  const expectedHashes = new Set(artifactEntries.map((entry) => entry.sha256));
  Object.assign(runMeasurements, {
    signedFiles: artifactEntries.length,
    uniqueSignedHashes: expectedHashes.size,
    artifactRequests: 0,
    rangeResponses: 0,
  });
  const artifactRequests = [];
  const artifactRequestsOutsideDownload = [];
  const unexpectedExternalRequests = [];
  const descriptionRequests = [];
  const expectedResourceErrors = [];
  const offlineCatalogRefreshes = [];
  let routePhase = "catalog";
  let artifactRequestsAtInstall = null;
  const markArtifactClosureServed = () => {
    const requestedHashes = new Set(artifactRequests.map((request) => request.sha256));
    if ([...expectedHashes].every((sha256) => requestedHashes.has(sha256))) {
      artifactRequestsAtInstall ??= artifactRequests.length;
      routePhase = "load";
    }
  };

  const launchArgs = options.browserFlags === "compatibility"
    ? [
        "--enable-unsafe-webgpu",
        "--disable-gpu-sandbox",
        ...(process.platform === "darwin" ? ["--use-angle=metal"] : []),
      ]
    : [];
  launchArgs.push(`--ignore-certificate-errors-spki-list=${certificateSpkiSha256}`);
  if (nativeWebMcp) launchArgs.push(...nativeWebMcp.args);
  releaseBrowser = await launchReleaseToolContext({
    channel: options.browserChannel,
    ...(nativeWebMcp ? { executablePath: nativeWebMcp.executablePath } : {}),
    headless: !options.headed,
    args: launchArgs,
    serviceWorkers: "allow",
    ignoreHTTPSErrors: true,
  });
  const context = releaseBrowser.context;
  const page = releaseBrowser.page;
  browser = context.browser();
  if (browser === null) throw new Error("persistent release browser is unavailable");
  page.setDefaultTimeout(30_000);
  page.setDefaultNavigationTimeout(60_000);
  stage("browser launched");
  const failedRequests = [];
  const pageErrors = [];
  const consoleErrors = [];
  const webGpuErrors = [];
  const requestBodies = [];
  const workerCreations = [];
  const workerDestructions = [];
  const recordLifecycle = (event) => {
    lifecycleEvents.push({ event, stage: currentStage, at: Date.now() });
    process.stderr.write(`[production-app] browser lifecycle: ${event} during ${currentStage}\n`);
  };
  browser.on("disconnected", () => recordLifecycle("browser-disconnected"));
  context.on("close", () => recordLifecycle("context-closed"));
  page.on("close", () => recordLifecycle("page-closed"));
  page.on("crash", () => recordLifecycle("page-crashed"));
  page.on("worker", (worker) => {
    workerCreations.push(worker.url());
    worker.on("close", () => workerDestructions.push(worker.url()));
  });
  page.on("request", (request) => {
    const postData = request.postData();
    if (postData) requestBodies.push({ url: request.url(), postData });
  });
  page.on("requestfailed", (request) => {
    failedRequests.push({
      url: request.url(),
      errorText: request.failure()?.errorText ?? "unknown",
      stage: currentStage,
      phase: routePhase,
    });
  });
  page.on("pageerror", (error) => pageErrors.push(error.stack ?? error.message));
  page.on("console", (message) => {
    const text = message.text();
    if (message.type() === "error") {
      const url = message.location().url;
      const offlineTransportError = routePhase === "offline" && text === "Failed to load resource: net::ERR_INTERNET_DISCONNECTED";
      if (isDescriptionRequest(url) || offlineTransportError || (routePhase === "offline" && (url === catalogUrl || url === signatureUrl))) {
        expectedResourceErrors.push({ url, text, phase: routePhase });
      } else consoleErrors.push(text);
    }
    if (isBrowserFullRuntimeWebGpuError(text)) webGpuErrors.push(text);
    if (message.type() === "error" || message.type() === "warning") {
      process.stderr.write(`browser ${message.type()}: ${text}\n`);
    }
  });
  await context.route("**/*", async (route) => {
    const request = route.request();
    const url = request.url();
    if (url.startsWith(`${origin}/`) || url.startsWith(`blob:${origin}/`) || url.startsWith("data:")) {
      await route.continue();
      return;
    }
    if (request.method() === "GET" && isDescriptionRequest(url)) {
      descriptionRequests.push({ url, phase: routePhase });
      await route.continue();
      return;
    }
    const resource = routedResources.get(url);
    if (!resource || request.method() !== "GET") {
      unexpectedExternalRequests.push(`${request.method()} ${url} during ${routePhase}`);
      process.stderr.write(`[production-app] blocked unexpected request: ${request.method()} ${url}\n`);
      await route.abort("blockedbyclient");
      return;
    }
    if (resource.kind !== "artifact") {
      if (routePhase === "offline") {
        offlineCatalogRefreshes.push(`${resource.kind} ${url}`);
        await route.abort("internetdisconnected");
        return;
      }
      await route.fulfill({
        status: 200,
        body: resource.bytes,
        headers: resourceHeaders(resource.bytes.byteLength, null, null),
      });
      return;
    }
    if (routePhase !== "download") {
      artifactRequestsOutsideDownload.push(`${request.method()} ${url} during ${routePhase}`);
      await route.abort(routePhase === "offline" ? "internetdisconnected" : "blockedbyclient");
      return;
    }
    const headers = request.headers();
    const requestedRange = headers.range ?? null;
    const ifRange = headers["if-range"] ?? null;
    const etag = `"sha256-${resource.sha256}"`;
    const range = ifRange !== null && ifRange !== etag
      ? parseSingleByteRange(null, resource.bytes)
      : parseSingleByteRange(requestedRange, resource.bytes);
    const responseBytes = range.end - range.start + 1;
    const requestRecord = {
      url,
      path: resource.catalogPath,
      sha256: resource.sha256,
      range: requestedRange,
      status: range.status,
      bytes: responseBytes,
    };
    artifactRequests.push(requestRecord);
    diagnosticArtifactRequests.push(requestRecord);
    runMeasurements.artifactRequests = artifactRequests.length;
    runMeasurements.rangeResponses = artifactRequests.filter(
      (artifactRequest) => artifactRequest.status === 206,
    ).length;
    await route.fulfill({
      status: 307,
      headers: {
        "access-control-allow-origin": "*",
        "cache-control": "no-store",
        "cross-origin-resource-policy": "cross-origin",
        location: `${origin}/__artifacts/${resource.sha256}`,
      },
    });
    markArtifactClosureServed();
    return;
  });

  await page.goto(`${origin}/app`, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(
    page,
    origin,
    Object.values(sets).flatMap((set) => set.files),
  );
  stage("hosted /app loaded");
  await waitForReadyPage(page);
  stage("device check ready");
  const environment = await page.evaluate(() => ({
    secureContext: isSecureContext,
    crossOriginIsolated,
    webGpu: typeof navigator.gpu !== "undefined",
  }));
  requireCondition(environment.secureContext, "the hosted /app is not a secure context");
  requireCondition(environment.crossOriginIsolated, "the hosted /app is not cross-origin isolated");
  requireCondition(environment.webGpu, "the hosted /app has no WebGPU API");

  const modelButtons = page.locator(".model-grid > button").filter({ hasNotText: "Download unavailable" });
  const modelButton = modelButtons.first();
  await modelButton.waitFor({ timeout: 30_000 });
  requireEqual(await modelButtons.count(), 1, "production catalog model choice count");
  await modelButton.click();
  await verifyFirstRunPacks(page);
  stage(`${options.modelId} ${options.contextTokens}-token profile selected`);
  routePhase = "download";
  await clickDownload(page);
  stage("signed setup download started");
  await waitForAuthoritativeInstalledState(page, catalog.variantId, "signed setup completion", options.timeoutMs);
  stage("signed setup installed and verified");
  const downloadedHashes = new Set(artifactRequests.map((request) => request.sha256));
  const missingHashes = [...expectedHashes].filter((hash) => !downloadedHashes.has(hash));
  requireEqual(missingHashes.length, 0, `signed hashes not fetched: ${missingHashes.join(", ")}`);
  artifactRequestsAtInstall ??= artifactRequests.length;

  routePhase = "load";
  await clickOpen(page);
  stage("real backend load started");
  await page.locator(".shell").waitFor({ timeout: options.timeoutMs });
  await page.getByRole("textbox", { name: /^Compose as /u }).waitFor({ timeout: options.timeoutMs });
  requireEqual(artifactRequestsOutsideDownload.length, 0, "artifact requests during model load");
  requireCondition((await page.title()).includes("Drowse"), "workbench title");
  stage("real Drowse workbench opened");

  const initialDocumentMarker = await page.evaluate(() => crypto.randomUUID());
  const initialPersistentState = await persistentStorageInventory(page);
  requireNoLegacyWebLlmPersistentState(initialPersistentState, "initial model load");
  routePhase = "explicit-unload-reload";
  await openWorkspace(page, "Controls");
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Model", exact: true }).click();
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await modelControls.getByRole("button", { name: "Close model", exact: true }).click();
  await waitForReadyPage(page);
  await page.getByText("Installed and verified", { exact: true }).waitFor();
  const reopenedDocumentMarker = await page.evaluate(() => crypto.randomUUID());
  requireCondition(
    reopenedDocumentMarker !== initialDocumentMarker,
    "explicit model unload did not recreate the production app document",
  );
  await clickOpen(page);
  await page.getByRole("textbox", { name: /^Compose as /u }).waitFor({ timeout: options.timeoutMs });
  requireEqual(
    artifactRequests.length,
    artifactRequestsAtInstall,
    "artifact requests after explicit unload/reload",
  );
  const reopenedPersistentState = await persistentStorageInventory(page);
  requireNoLegacyWebLlmPersistentState(reopenedPersistentState, "explicit unload/reload");
  stage("verified OPFS model reopened after explicit unload and page recreation");

  const firstReply = await generate(page, { prompt: "hiho", maxTokens: options.maxTokens });
  const followupReply = await generate(page, { prompt: "In one sentence, say hello again.", maxTokens: options.maxTokens });
  runMeasurements.plainConversation = { firstReply, followupReply };
  if (options.output) await page.screenshot({ path: `${resolve(options.output)}.png`, fullPage: true });
  stage("first and follow-up replies completed with default token alternatives");
  const instrumentResult = await attachInstrumentProbes(page, options);
  stage("J-lens and SAE probes attached");
  const onlineGeneration = await generate(page, {
    prompt: `${PROMPT_SENTINEL}: Reply with the single word blue.`,
    maxTokens: options.maxTokens,
  });
  stage("online real-backend generation completed");
  const probeResult = await assertProbeMeasurements(page, options);
  stage("live instrument measurements verified");
  const tokenReadoutResult = await assertTokenInstrumentReadouts(page);
  stage("token SAE and geometry readouts verified");

  routePhase = "catalog-cache";
  await waitForServiceWorkerReady(page);
  if (!(await page.evaluate(() => navigator.serviceWorker.controller !== null))) {
    await page.reload({ waitUntil: "load" });
    await waitForServiceWorkerController(page);
  }
  await waitForAuthoritativeInstalledState(
    page,
    catalog.variantId,
    "installed model state after service-worker control",
  );
  stage("service worker controls the installed shell");

  routePhase = "offline";
  offlineArmed = true;
  await context.setOffline(true);
  // Playwright updates navigator.onLine for an emulated offline context but
  // does not consistently deliver the browser's offline event before the next
  // service-worker navigation. Deliver that standard event when needed so the
  // app can persist the same cross-document hint a physical browser emits.
  await page.evaluate(() => {
    if (navigator.onLine === false) window.dispatchEvent(new Event("offline"));
  });
  runMeasurements.offlineNavigationHint = await page.evaluate(() =>
    sessionStorage.getItem("drowse.hosted.offline")
  );
  requireEqual(
    runMeasurements.offlineNavigationHint,
    "1",
    "offline navigation hint before service-worker navigation",
  );
  const beforeOfflineDocumentMarker = await page.evaluate(() => crypto.randomUUID());
  runMeasurements.navigatorOnlineBeforeOfflineNavigation = await page.evaluate(
    () => navigator.onLine,
  );
  await page.goto(`${origin}/app`, { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "Open chat", exact: true }).click();
  await waitForAuthoritativeInstalledState(
    page,
    catalog.variantId,
    "installed model state after offline navigation",
  );
  const offlineDocumentMarker = await page.evaluate(() => crypto.randomUUID());
  requireCondition(
    offlineDocumentMarker !== beforeOfflineDocumentMarker,
    "offline navigation did not recreate the production app document",
  );
  runMeasurements.navigatorOnlineAfterOfflineNavigation = await page.evaluate(
    () => navigator.onLine,
  );
  stage("offline /app restored from cache");
  await waitForRequestQuiescence(serverRequests);
  offlineWorkbenchArmed = true;
  await clickOpen(page);
  await page.getByRole("textbox", { name: /^Compose as /u }).waitFor({ timeout: options.timeoutMs });
  const offlineGeneration = await generate(page, {
    prompt: `${PROMPT_SENTINEL}_OFFLINE: Reply with the single word green.`,
    maxTokens: options.maxTokens,
  });
  offlineWorkbenchArmed = false;
  stage("offline real-backend generation completed");
  const offlinePersistentState = await persistentStorageInventory(page);
  requireNoLegacyWebLlmPersistentState(offlinePersistentState, "offline worker recreation");
  await page.waitForTimeout(250);
  await context.setOffline(false);

  const browserRuntimeWorkerCreations = workerCreations.filter(isBrowserRuntimeWorkerUrl);
  const browserRuntimeWorkerDestructions = workerDestructions.filter(isBrowserRuntimeWorkerUrl);
  requireCondition(
    browserRuntimeWorkerCreations.length >= 3,
    `expected at least three production browser-worker creations, got ${browserRuntimeWorkerCreations.length}`,
  );
  requireCondition(
    browserRuntimeWorkerDestructions.length >= 2,
    `expected at least two production browser-worker destructions, got ${browserRuntimeWorkerDestructions.length}`,
  );
  requireEqual(
    artifactRequests.length,
    artifactRequestsAtInstall,
    "artifact requests after installed unload/reload and offline reopen",
  );

  const promptUploads = requestBodies.filter(({ postData }) => postData.includes(PROMPT_SENTINEL));
  const verifiedLocalArtifactAborts = failedRequests.filter((request) =>
    isVerifiedLocalArtifactAbort({
      ...request,
      artifactOrigin: origin,
      expectedHashes,
      downloadedHashes,
    })
  );
  const verifiedLocalArtifactAbortSet = new Set(verifiedLocalArtifactAborts);
  const expectedOfflineCatalogFailureSet = new Set(
    failedRequests.filter((request) =>
      isExpectedOfflineCatalogFailure(request, { catalogUrl, signatureUrl })
    ),
  );
  runMeasurements.verifiedLocalArtifactAborts = verifiedLocalArtifactAborts.length;
  runMeasurements.offlineCatalogRefreshes = offlineCatalogRefreshes.length;
  const unexpectedOfflineServerRequests = offlineServerRequests.filter(
    (request) => !isOfflineServiceWorkerMaintenanceRequest(request),
  );
  const unexpectedOfflineWorkbenchRequests = offlineWorkbenchRequests.filter(
    (request) => !isOfflineServiceWorkerMaintenanceRequest(request),
  );
  const audit = {
    artifactRequestsOutsideDownload,
    consoleErrors,
    failedRequests: failedRequests.filter(
      (request) =>
        !verifiedLocalArtifactAbortSet.has(request) &&
        !expectedOfflineCatalogFailureSet.has(request) &&
        !(isDescriptionRequest(request.url) && (request.errorText === "net::ERR_ABORTED" ||
          (request.phase === "offline" && request.errorText === "net::ERR_INTERNET_DISCONNECTED"))),
    ),
    pageErrors,
    promptUploads,
    unexpectedExternalRequests,
    unexpectedOfflineServerRequests,
    unexpectedOfflineWorkbenchRequests,
    webGpuErrors,
  };
  Object.assign(runMeasurements, { onlineGeneration, offlineGeneration, probes: probeResult, tokenReadouts: tokenReadoutResult, audit, descriptionRequests, expectedResourceErrors });
  for (const [name, entries] of Object.entries(audit)) {
    if (entries.length > 0) throw new Error(`${name} is non-empty: ${JSON.stringify(entries)}`);
  }
  const webmcp = nativeWebMcp ? await (await import("./webmcp-live-scenario.mjs"))
    .runConnectedNativeEvidence(nativeWebMcp.port, page.url(), { runtime: "browser", control: CORE_GEOMETRY_SELECTOR }) : undefined;

  const report = {
    schemaVersion: 1,
    toolVersion: BROWSER_PRODUCTION_APP_TOOL_VERSION,
    releaseEvidence: false,
    passed: true,
    route: "/app",
    modelId: options.modelId,
    contextTokens: options.contextTokens,
    variantId: catalog.variantId,
    artifactRevision: revision,
    browser: {
      channel: options.browserChannel,
      version: browser.version(),
      flags: options.browserFlags,
      headed: options.headed,
      runtimeLockSource: options.runtimeLock === null ? "repository" : "override",
      platform: process.platform,
      architecture: process.arch,
    },
    checks: {
      localArtifactHashes: true,
      semanticPackValidation: true,
      productionBundleIsolation: true,
      signedCatalogAdmission: true,
      verifiedDownload: true,
      browserContentStoreReopen: true,
      explicitUnloadReload: true,
      workerPageRecreation: true,
      noLegacyWebLlmPersistence: true,
      zeroArtifactRequestsAfterInstall: true,
      realBackendGeneration: true,
      repeatedPlainConversation: true,
      instrumentMeasurements: true,
      serviceWorkerOfflineReopen: true,
      promptNetworkIsolation: true,
      browserErrorAudit: true,
    },
    measurements: {
      environment,
      signedFiles: artifactEntries.length,
      uniqueSignedHashes: expectedHashes.size,
      artifactRequests: artifactRequests.length,
      artifactRequestsAfterInstall: artifactRequests.length - artifactRequestsAtInstall,
      rangeResponses: artifactRequests.filter((request) => request.status === 206).length,
      modelBytes: sumBytes(sets.model.files),
      coreBytes: sumBytes(sets.core.files),
      jlensBytes: sumBytes(sets.jlens.files),
      saeBytes: sumBytes(sets.sae.files),
      plainConversation: runMeasurements.plainConversation,
      ...(webmcp ? { webmcp } : {}),
      descriptionRequests,
      expectedResourceErrors,
      onlineGeneration,
      offlineGeneration,
      instruments: instrumentResult,
      probes: probeResult,
      tokenReadouts: tokenReadoutResult,
      browserRuntimeWorkerCreations,
      browserRuntimeWorkerDestructions,
      persistentState: {
        initial: initialPersistentState,
        reopened: reopenedPersistentState,
        offline: offlinePersistentState,
      },
      lifecycleEvents,
      serverRequests: serverRequests.length,
      offlineServerRequests: offlineServerRequests.length,
      offlineWorkbenchRequests: offlineWorkbenchRequests.length,
      offlineCatalogRefreshes: offlineCatalogRefreshes.length,
    },
    audit,
  };
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (options.output) await writeFile(resolve(options.output), output, { flag: "wx" });
  process.stdout.write(output);
} catch (error) {
  if (releaseBrowser?.page && !releaseBrowser.page.isClosed()) {
    runMeasurements.pageText = await releaseBrowser.page.locator("body").innerText().catch(() => "");
    if (options.output) {
      await releaseBrowser.page.screenshot({ path: `${resolve(options.output)}.failure.png`, fullPage: true }).catch(() => undefined);
    }
  }
  const failureReport = createProductionAppFailureReport({
    stage: currentStage,
    error,
    measurements: {
      ...runMeasurements,
      artifactRequestTail: diagnosticArtifactRequests.slice(-10),
      lifecycleEvents,
      serverRequests: serverRequests.length,
      offlineServerRequests: offlineServerRequests.length,
      offlineWorkbenchRequests: offlineWorkbenchRequests.length,
    },
  });
  const output = `${JSON.stringify(failureReport, null, 2)}\n`;
  if (options.output) await writeFile(resolve(options.output), output, { flag: "wx" });
  process.stdout.write(output);
  throw error;
} finally {
  if (releaseBrowser) await releaseBrowser.close().catch(() => undefined);
  else if (browser) await browser.close().catch(() => undefined);
  if (server?.listening) {
    await new Promise((resolvePromise) => server.close(resolvePromise));
  }
  await rm(temporaryRoot, { recursive: true, force: true });
}

async function attachInstrumentProbes(page, options) {
  await openWorkspace(page, "Controls");
  const tabs = page.getByRole("group", { name: "Response guidance type" });
  await tabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  const lensPackId = `${options.modelId}-jlens`;
  await locatorContains(lens.getByRole("button", { name: "Compatible data pack" }), lensPackId);
  await lens.getByRole("button", { name: "In use", exact: true }).waitFor();
  const lensLiveOn = lens.getByRole("button", { name: "Turn on live predicted-word readings", exact: true });
  if (await lensLiveOn.isVisible()) {
    await lensLiveOn.click();
  }
  await lens.getByRole("button", { name: "Turn off live predicted-word readings", exact: true }).waitFor();
  await lens.getByRole("textbox", { name: "Watch a prediction word" }).fill(options.jlensWord);
  await lens.getByRole("button", { name: "Watch word", exact: true }).click();
  await lens.getByRole("list", { name: "J-lens probe tokens" }).waitFor();

  await tabs.getByRole("button", { name: "SAE", exact: true }).click();
  const sae = page.getByLabel("Model feature controls");
  const saePackId = `${options.modelId}-sae`;
  await locatorContains(sae.getByRole("button", { name: "Compatible data pack" }), saePackId);
  await sae.getByRole("button", { name: "In use", exact: true }).waitFor();
  const saeLiveOn = sae.getByRole("button", { name: "Turn on live model-feature readings", exact: true });
  if (await saeLiveOn.isVisible()) {
    await saeLiveOn.click();
  }
  await sae.getByRole("button", { name: "Turn off live model-feature readings", exact: true }).waitFor();
  await sae.getByRole("textbox", { name: "Watch a model feature" }).fill(String(options.saeFeature));
  await sae.getByRole("button", { name: "Watch feature", exact: true }).click();
  await sae.getByRole("list", { name: "SAE feature probes" }).waitFor();

  await tabs.getByRole("button", { name: "Subspace", exact: true }).click();
  await page.getByRole("button", { name: "Add subspace probe", exact: true }).click();
  const geometryDrawer = page.getByRole("dialog", { name: "Add subspace" });
  await geometryDrawer.getByText("attach selector", { exact: true }).click();
  await geometryDrawer.getByRole("textbox", { name: "Selector" })
    .fill(CORE_GEOMETRY_SELECTOR);
  await geometryDrawer.getByRole("button", { name: "+ attach", exact: true }).click();
  await page.getByText(`probe ${CORE_GEOMETRY_SELECTOR}`, { exact: true }).waitFor();
  await geometryDrawer.getByRole("button", { name: "Close drawer", exact: true }).click();

  return {
    lensPackId,
    saePackId,
    jlensProbe: `jlens/${options.jlensWord}`,
    saeProbe: `sae/${options.saeFeature}`,
    geometryProbe: CORE_GEOMETRY_SELECTOR,
  };
}

async function assertProbeMeasurements(page, options) {
  await openWorkspace(page, "Controls");
  const tabs = page.getByRole("group", { name: "Response guidance type" });
  await tabs.getByRole("button", { name: "J-lens", exact: true }).click();
  const lens = page.getByLabel("Layer prediction controls");
  const lensList = lens.getByRole("list", { name: "J-lens probe tokens" });
  await lensList.waitFor();
  const layerStrip = lens.getByRole("slider", {
    name: `Per-layer strength for ${options.jlensWord}`,
  }).first();
  await layerStrip.waitFor();
  const lensLayerValue = await layerStrip.getAttribute("aria-valuetext");
  requireCondition(
    typeof lensLayerValue === "string" && !/no (?:layer )?data/iu.test(lensLayerValue),
    `J-lens layer strip has no data: ${lensLayerValue}`,
  );
  const lensStrength = await finiteAriaMeasurement(lensList, /Strength (-?[0-9.]+(?:e[+-]?[0-9]+)?)/u);

  await tabs.getByRole("button", { name: "SAE", exact: true }).click();
  const sae = page.getByLabel("Model feature controls");
  const saeList = sae.getByRole("list", { name: "SAE feature probes" });
  await saeList.locator('[aria-label^="Strength "], [aria-label^="Activation "]')
    .first()
    .waitFor();
  const saeMeasurement = await finiteAriaMeasurement(saeList, /(?:Strength|Activation) (-?[0-9.]+)/u);

  await tabs.getByRole("button", { name: "Subspace", exact: true }).click();
  const geometryStrip = page.getByLabel("Probe rack").getByRole("slider", {
    name: `Per-layer readings for ${CORE_GEOMETRY_SELECTOR}`,
  });
  await geometryStrip.waitFor();
  const geometryLayerValue = await geometryStrip.getAttribute("aria-valuetext");
  requireCondition(
    typeof geometryLayerValue === "string" && !/no (?:layer )?data/iu.test(geometryLayerValue),
    `geometry layer strip has no data: ${geometryLayerValue}`,
  );
  return { lensLayerValue, lensStrength, saeMeasurement, geometryLayerValue };
}

async function assertTokenInstrumentReadouts(page) {
  await openWorkspace(page, "Conversation");
  await page.getByRole("button", { name: /Inspect tokens in .* message/u }).last().click();
  const drawer = page.getByRole("dialog", { name: "Generated word details" })
    .or(page.getByRole("complementary", { name: "Generated word details" }));
  await drawer.waitFor();
  const tabs = drawer.getByRole("group", { name: "Token detail view" });

  await tabs.getByRole("button", { name: "sae", exact: true }).click();
  const saeFeatures = drawer.getByRole("list", { name: "Top SAE features" });
  await saeFeatures.waitFor();
  const saeFeatureCount = await saeFeatures.getByRole("listitem").count();
  requireCondition(saeFeatureCount > 0, "token SAE readout returned no active features");
  requireEqual(
    await drawer.getByText(/^readout:/u).count(),
    0,
    "token SAE readout errors",
  );

  await tabs.getByRole("button", { name: "geometry", exact: true }).click();
  await drawer.getByText("PROBE READINGS", { exact: true }).waitFor();
  await drawer.getByText(CORE_GEOMETRY_SELECTOR, { exact: true }).waitFor();
  requireEqual(
    await drawer.getByText(/^readout:/u).count(),
    0,
    "token geometry readout errors",
  );

  await drawer.getByRole("button", { name: "Close drawer", exact: true }).click();
  return { saeFeatureCount, geometryProbe: CORE_GEOMETRY_SELECTOR };
}

async function finiteAriaMeasurement(root, pattern) {
  const labels = await root.locator("[aria-label]").evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute("aria-label")).filter(Boolean)
  );
  const matched = labels.map((label) => pattern.exec(label)).find(Boolean);
  if (!matched) throw new Error(`no finite instrument measurement matched ${pattern}`);
  const value = Number(matched[1]);
  requireCondition(Number.isFinite(value), `instrument measurement is non-finite: ${matched[0]}`);
  return { label: matched[0], value };
}

async function generate(page, { prompt, maxTokens }) {
  const dismissNotice = page.getByRole("button", { name: "Dismiss", exact: true });
  if (await dismissNotice.isVisible()) await dismissNotice.click();
  await openWorkspace(page, "Controls");
  const length = page.getByRole("spinbutton", { name: "Max tokens", exact: true });
  await length.fill(String(maxTokens));
  await length.press("Enter");
  await openWorkspace(page, "Conversation");
  const composer = page.getByRole("textbox", { name: /^Compose as /u });
  await composer.fill(prompt);
  requireEqual(await composer.inputValue(), prompt, "composer prompt before generation");
  const statusLocator = page.getByLabel("Generation status", { exact: true });
  const responseLocator = page.locator(".msg .response-body").last();
  const generatedTurns = page.getByRole("button", { name: /Inspect tokens in .* message/u });
  const previousGeneratedTurns = await generatedTurns.count();
  const previousStatus = await statusLocator.innerText({ timeout: 5_000 }).catch(() => "");
  const previousResponse = await responseLocator.innerText({ timeout: 5_000 }).catch(() => "");
  const deadline = Date.now() + GENERATION_TIMEOUT_MS;
  const clickState = { settled: false, error: null };
  const send = page.getByRole("button", { name: /^(Send|Generate reply|Add message)$/u });
  requireCondition(await send.isEnabled(), "generation control is disabled after composing a prompt");
  void send
    .click({ timeout: GENERATION_TIMEOUT_MS })
    .then(
      () => { clickState.settled = true; },
      (error) => {
        clickState.settled = true;
        clickState.error = error;
      },
    );
  stage("generation click dispatched");
  const stop = page.getByRole("button", { name: /^Stop$/iu });
  let nextProgressLog = Date.now();
  let sawActive = false;
  let status = "";
  let response = "";
  while (Date.now() < deadline) {
    if (clickState.error) throw clickState.error;
    const alerts = await page.getByRole("alert").allTextContents();
    if (alerts.some((text) => /Generation:|Drowse stopped:/u.test(text))) {
      throw new Error(`generation failed: ${alerts.join(" ")}`);
    }
    const statusResult = await boundedLocatorRead(
      () => statusLocator.innerText({ timeout: 5_000 }),
      status,
    );
    const responseResult = await boundedLocatorRead(
      () => responseLocator.innerText({ timeout: 5_000 }),
      response,
    );
    const stopDisabledResult = await boundedLocatorRead(
      () => stop.isDisabled({ timeout: 5_000 }),
      null,
    );
    const doneResult = await boundedLocatorRead(
      () => page.locator('[aria-label="Generation status"] .done-label')
        .isVisible({ timeout: 5_000 }),
      false,
    );
    status = statusResult.value.trim();
    response = responseResult.value.trim();
    const active = stopDisabledResult.value === null ? null : !stopDisabledResult.value;
    sawActive ||= active === true;
    if (Date.now() >= nextProgressLog) {
      const unavailable = [statusResult, responseResult, stopDisabledResult, doneResult]
        .some((result) => result.timedOut);
      stage(
        `generation progress: ${status.replaceAll(/\s+/gu, " ") || "status unavailable"}; ` +
        `response chars ${response.length}; UI ${unavailable ? "partly unresponsive" : "responsive"}; ` +
        `click ${clickState.settled ? "settled" : "pending"}`,
      );
      nextProgressLog = Date.now() + 30_000;
    }
    const changed = status !== previousStatus.trim() || response !== previousResponse.trim();
    const newGeneratedTurn = await generatedTurns.count() > previousGeneratedTurns;
    if (clickState.settled && newGeneratedTurn && (sawActive || changed) && active === false && doneResult.value === true) break;
    await delay(5_000);
  }
  if (Date.now() >= deadline) {
    const error = new Error(
      `generation exceeded ${GENERATION_TIMEOUT_MS} ms; status: ${status}; response: ${response.slice(0, 500)}`,
    );
    error.name = "GenerationTimeoutError";
    error.code = "PRODUCTION_APP_GENERATION_TIMEOUT";
    throw error;
  }
  requireCondition(
    await stop.isDisabled({ timeout: 10_000 }),
    "generation stop control stayed enabled after completion",
  );
  response = (await responseLocator.innerText({ timeout: 10_000 })).trim();
  requireCondition(response.length > 0, "assistant response is empty");
  status = (await statusLocator.innerText({ timeout: 10_000 })).replaceAll(/\s+/gu, " ").trim();
  const tokensMatch = /([0-9]+) tokens/u.exec(status);
  const speedMatch = /([0-9]+(?:\.[0-9]+)?) tokens\/s/u.exec(status);
  if (!tokensMatch || !speedMatch) throw new Error(`generation status is incomplete: ${status}`);
  const tokens = Number(tokensMatch[1]);
  const tokensPerSecond = Number(speedMatch[1]);
  if (tokens === 0) {
    const alerts = await page.getByRole("alert").allInnerTexts();
    throw new Error(
      `generation produced no tokens; status: ${status}; alerts: ${JSON.stringify(alerts)}`,
    );
  }
  requireCondition(tokens >= 1 && tokens <= maxTokens, `generated ${tokens}/${maxTokens} tokens`);
  requireCondition(
    Number.isFinite(tokensPerSecond) && tokensPerSecond > 0,
    `generation speed is invalid: ${tokensPerSecond}`,
  );
  return { response, tokens, tokensPerSecond, status };
}

async function boundedLocatorRead(operation, fallback) {
  try {
    return { value: await operation(), timedOut: false };
  } catch {
    return { value: fallback, timedOut: true };
  }
}

async function delay(milliseconds) {
  await new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

async function openWorkspace(page, name) {
  const accessibleName = name === "Conversation"
    ? /^Conversation$/u
    : name === "Controls"
      ? /^Controls$/u
      : /^Branches$/u;
  await page.getByRole("button", { name: accessibleName }).click();
}

async function clickDownload(page) {
  const choice = await waitForVisibleChoice(page, [
    ["direct", page.getByRole("button", {
      name: /^(Download and open|Add tools and open|Resume download|Retry download)$/u,
    })],
    ["warning", page.getByRole("button", { name: "Review warning", exact: true })],
  ], "download control");
  await choice.locator.click();
  if (choice.name === "warning") {
    const override = await waitForVisibleChoice(page, [
      ["override", page.getByRole("button", {
        name: /^(Download anyway|Retry download)$/u,
      })],
    ], "unsafe download confirmation");
    await override.locator.click();
  }
}

async function verifyFirstRunPacks(page) {
  const picker = page.locator(".tool-picker");
  await picker.waitFor({ state: "visible", timeout: 30_000 });
  const checkboxes = picker.getByRole("checkbox");
  requireEqual(
    await checkboxes.count(),
    0,
    "published optional first-run pack choices",
  );
  requireEqual(await picker.locator(".required-tool").count(), 3, "included core, J-lens and SAE setup packs");
}

async function waitForReadyPage(page) {
  try {
    await page.locator(".app-shell.hardware-compatible").waitFor({ timeout: 60_000 });
  } catch (error) {
    const body = (await page.locator("body").innerText().catch(() => "<body unavailable>"))
      .replaceAll(/\s+/gu, " ")
      .slice(0, 4_000);
    throw new Error(`device check did not become ready; page text: ${body}`, { cause: error });
  }
}

async function waitForServiceWorkerReady(page) {
  try {
    await withTimeout(page.evaluate(async (timeoutMs) => {
      if (!("serviceWorker" in navigator)) throw new Error("service workers are unavailable");
      let timeout;
      try {
        await Promise.race([
          navigator.serviceWorker.ready,
          new Promise((_, reject) => {
            timeout = setTimeout(
              () => reject(new Error(`service worker ready timed out after ${timeoutMs} ms`)),
              timeoutMs,
            );
          }),
        ]);
      } finally {
        clearTimeout(timeout);
      }
    }, SERVICE_WORKER_TIMEOUT_MS), SERVICE_WORKER_TIMEOUT_MS + 1_000, "service worker ready");
  } catch (error) {
    const diagnostics = await serviceWorkerDiagnostics(page);
    throw new Error(
      `service worker did not become ready; diagnostics: ${JSON.stringify(diagnostics)}`,
      { cause: error },
    );
  }
}

async function waitForServiceWorkerController(page) {
  try {
    await withTimeout(
      page.waitForFunction(
        () => navigator.serviceWorker.controller !== null,
        undefined,
        { timeout: SERVICE_WORKER_TIMEOUT_MS },
      ),
      SERVICE_WORKER_TIMEOUT_MS + 1_000,
      "service worker controller",
    );
  } catch (error) {
    const diagnostics = await serviceWorkerDiagnostics(page);
    throw new Error(
      `service worker did not control the page after reload; diagnostics: ${JSON.stringify(diagnostics)}`,
      { cause: error },
    );
  }
}

async function waitForAuthoritativeInstalledState(page, expectedModelVariantId, label, timeoutMs = 30_000) {
  try {
    const choice = await waitForVisibleChoice(page, [
      ["verified-onboarding", page.getByText("Installed and verified", { exact: true })],
      ["loaded-workbench", page.locator(".shell")],
    ], label, timeoutMs);
    let loadedModelVariantId = null;
    if (choice.name === "loaded-workbench") {
      loadedModelVariantId = await currentWorkbenchModelVariantId(page);
    }
    const state = classifyInstalledAppState({
      verifiedOnboarding: choice.name === "verified-onboarding",
      loadedModelVariantId,
      expectedModelVariantId,
    });
    if (state === null) {
      throw new Error(
        `loaded workbench model ${JSON.stringify(loadedModelVariantId)} does not match ` +
        `${JSON.stringify(expectedModelVariantId)}`,
      );
    }
    return state;
  } catch (error) {
    const [diagnostics, body] = await Promise.all([
      serviceWorkerDiagnostics(page),
      page.locator("body").innerText().catch(() => "<body unavailable>"),
    ]);
    throw new Error(
      `${label} was not authoritative; service-worker diagnostics: ` +
      `${JSON.stringify(diagnostics)}; page text: ${body.replaceAll(/\s+/gu, " ").slice(0, 2_000)}`,
      { cause: error },
    );
  }
}

async function currentWorkbenchModelVariantId(page) {
  await openWorkspace(page, "Controls");
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Model", exact: true }).click();
  const modelControls = page.getByRole("region", { name: "Model controls", exact: true });
  await modelControls.waitFor({ state: "visible", timeout: 30_000 });
  const modelVariantId = await modelControls.getAttribute("data-model-variant-id");
  await page.getByRole("group", { name: "Controls section" })
    .getByRole("button", { name: "Response", exact: true }).click();
  await openWorkspace(page, "Conversation");
  return modelVariantId?.trim() || null;
}

async function serviceWorkerDiagnostics(page) {
  try {
    return await withTimeout(page.evaluate(async () => {
      if (!("serviceWorker" in navigator)) {
        return { supported: false, controller: null, registrations: [] };
      }
      const describe = (worker) => worker
        ? { scriptURL: worker.scriptURL, state: worker.state }
        : null;
      const registrations = await navigator.serviceWorker.getRegistrations();
      return {
        supported: true,
        controller: describe(navigator.serviceWorker.controller),
        registrations: registrations.map((registration) => ({
          scope: registration.scope,
          updateViaCache: registration.updateViaCache,
          active: describe(registration.active),
          waiting: describe(registration.waiting),
          installing: describe(registration.installing),
        })),
      };
    }), 5_000, "service worker diagnostics");
  } catch (error) {
    return { unavailable: error instanceof Error ? error.message : String(error) };
  }
}

async function withTimeout(operation, timeoutMs, label) {
  let timeout;
  try {
    return await Promise.race([
      operation,
      new Promise((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error(`${label} timed out after ${timeoutMs} ms`)),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

async function clickOpen(page) {
  const choice = await waitForVisibleChoice(page, [
    ["ready", page.locator(".shell")],
    ["direct", page.getByRole("button", { name: "Open Drowse", exact: true })],
    ["warning", page.getByRole("button", { name: "Review warning", exact: true })],
  ], "model open control");
  if (choice.name === "ready") return;
  await choice.locator.click();
  if (choice.name === "warning") {
    const override = await waitForVisibleChoice(page, [
      ["override", page.getByRole("button", {
        name: /^(Load anyway|Retry this model)$/u,
      })],
    ], "unsafe model-load confirmation");
    await override.locator.click();
  }
}

async function waitForVisibleChoice(page, choices, label, timeoutMs = 30_000) {
  try {
    return await Promise.any(choices.map(async ([name, locator]) => {
      await locator.waitFor({ state: "visible", timeout: timeoutMs });
      return { name, locator };
    }));
  } catch (error) {
    const body = (await page.locator("body").innerText().catch(() => ""))
      .replaceAll(/\s+/gu, " ")
      .slice(0, 2_000);
    throw new Error(`${label} did not appear; page text: ${body}`, { cause: error });
  }
}

async function persistentStorageInventory(page) {
  return page.evaluate(async () => ({
    cacheNames: (await caches.keys()).sort(),
    databaseNames: (await indexedDB.databases())
      .map(({ name }) => name)
      .filter((name) => typeof name === "string")
      .sort(),
  }));
}

function requireNoLegacyWebLlmPersistentState(inventory, stageName) {
  const legacy = legacyWebLlmPersistentState(inventory);
  requireEqual(
    legacy.cacheNames.length,
    0,
    `${stageName} legacy WebLLM CacheStorage names: ${legacy.cacheNames.join(", ")}`,
  );
  requireEqual(
    legacy.databaseNames.length,
    0,
    `${stageName} legacy WebLLM IndexedDB names: ${legacy.databaseNames.join(", ")}`,
  );
}

function isBrowserRuntimeWorkerUrl(url) {
  return /\/browser\.worker-[^/]+\.js(?:$|\?)/u.test(url);
}

async function locatorContains(locator, text) {
  await locator.filter({ hasText: text }).waitFor();
}

function productionEvidencePlugin(overrides, loaded) {
  const virtualPrefix = "virtual:drowse-production-app-e2e/";
  const resolvedPrefix = "\0drowse-production-app-e2e:";
  const resolvedSuffix = ":module";
  const hostedMain = resolve(webuiRoot, "hosted/main.ts");
  const importers = new Map([
    [resolve(webuiRoot, "src/hosted/runtime/runtimeLock.ts"), [
      "runtime-lock.json",
    ]],
    [resolve(webuiRoot, "src/hosted/runtime/distributionConfig.ts"), ["distribution-lock.json"]],
  ]);
  return {
    name: "drowse-production-app-e2e-evidence",
    enforce: "pre",
    resolveId(source) {
      return source.startsWith(virtualPrefix)
        ? `${resolvedPrefix}${source.slice(virtualPrefix.length)}${resolvedSuffix}`
        : null;
    },
    load(id) {
      if (!id.startsWith(resolvedPrefix)) return null;
      const name = id.slice(resolvedPrefix.length, -resolvedSuffix.length);
      if (!(name in overrides)) throw new Error(`unknown production-app evidence module ${name}`);
      loaded.add(name);
      return `export default ${JSON.stringify(overrides[name])};`;
    },
    transform(source, id) {
      const path = id.split("?", 1)[0];
      if (path === hostedMain) return stripHostedDevelopmentFixture(source);
      const names = importers.get(path);
      if (!names) return null;
      let transformed = source;
      for (const name of names) {
        const specifier = `../../../../browser-runtime/${name}`;
        if (!transformed.includes(specifier)) {
          throw new Error(`${path} no longer imports ${specifier}`);
        }
        transformed = transformed.replaceAll(specifier, `${virtualPrefix}${name}`);
      }
      return transformed;
    },
  };
}

async function validateCatalogDocument(document) {
  const vite = await createViteServer({
    root: webuiRoot,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null, hmr: false },
  });
  try {
    const module = await vite.ssrLoadModule("/src/lib/runtime/catalog.ts");
    module.validateCatalogDocument(document);
  } finally {
    await vite.close();
  }
}

async function artifactSet(directory, manifestName, options = {}) {
  const directoryInfo = await stat(directory);
  if (!directoryInfo.isDirectory()) throw new Error(`${manifestName} path is not a directory: ${directory}`);
  const manifest = JSON.parse(await readFile(resolve(directory, manifestName), "utf8"));
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error(`${manifestName} contains no files`);
  }
  const paths = new Set();
  const files = [];
  const absoluteByPath = new Map();
  for (const raw of manifest.files) {
    if (
      !raw || typeof raw !== "object" || Array.isArray(raw) ||
      typeof raw.path !== "string" || unsafePath(raw.path) || paths.has(raw.path) ||
      !Number.isSafeInteger(raw.bytes) || raw.bytes < 1 ||
      typeof raw.sha256 !== "string" || !/^[a-f0-9]{64}$/u.test(raw.sha256)
    ) throw new Error(`${manifestName} contains an invalid file entry`);
    paths.add(raw.path);
    const entry = { ...raw, ...(options.role ? { role: options.role } : {}) };
    const absolute = options.modelLibrary && raw.role === "model_library"
      ? options.modelLibrary
      : resolve(directory, ...raw.path.split("/"));
    await verifyFile(absolute, entry, `${manifestName}:${entry.path}`);
    files.push(entry);
    absoluteByPath.set(entry.path, absolute);
  }
  return { directory, manifest, files, absoluteByPath };
}

async function verifyFile(path, entry, label) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`${label} is not a regular file: ${path}`);
  if (info.size !== entry.bytes) throw new Error(`${label} has ${info.size} bytes, expected ${entry.bytes}`);
  const actual = await sha256File(path);
  if (actual !== entry.sha256) {
    throw new Error(`${label} has SHA-256 ${actual}, expected ${entry.sha256}`);
  }
}

async function sha256File(path) {
  const hash = createHash("sha256");
  await new Promise((resolvePromise, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return hash.digest("hex");
}

function deterministicRevision(sets) {
  const hash = createHash("sha256");
  for (const [name, set] of Object.entries(sets)) {
    for (const file of set.files) hash.update(`${name}\0${file.path}\0${file.sha256}\n`);
  }
  return hash.digest("hex").slice(0, 40);
}

function rawEd25519(publicKey) {
  const spki = publicKey.export({ format: "der", type: "spki" });
  if (spki.byteLength !== 44) throw new Error(`unexpected Ed25519 SPKI length ${spki.byteLength}`);
  return spki.subarray(spki.byteLength - 32).toString("base64");
}

function createArtifactEntries(document, sets) {
  const variant = document.models[0].variants[0];
  const groups = [
    [variant.files, sets.model],
    [[...variant.packs[0].files], sets.core],
    [[...variant.packs[1].files], sets.jlens],
    [[...variant.packs[2].files], sets.sae],
  ];
  const entries = [];
  for (const [files, set] of groups) {
    for (const file of files) {
      const absolute = set.absoluteByPath.get(file.path);
      if (!absolute) throw new Error(`catalog path has no local artifact: ${file.path}`);
      entries.push({
        url: file.url,
        catalogPath: file.path,
        sha256: file.sha256,
        bytes: file.bytes,
        absolute,
      });
    }
  }
  return entries;
}

function createStaticServer(
  root,
  requests,
  offlineRequests,
  offlineWorkbenchRequests,
  isOfflineArmed,
  isOfflineWorkbenchArmed,
  localArtifacts,
  tls,
) {
  return createHttpsServer(tls, async (request, response) => {
    const method = request.method ?? "GET";
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    requests.push(`${method} ${url.pathname}`);
    if (isOfflineArmed()) offlineRequests.push(`${method} ${url.pathname}`);
    if (isOfflineWorkbenchArmed()) {
      offlineWorkbenchRequests.push(`${method} ${url.pathname}`);
    }
    for (const [name, value] of Object.entries({
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "no-store",
    })) response.setHeader(name, value);
    try {
      if (method !== "GET" && method !== "HEAD") return send(response, "method not allowed", "text/plain", 405);
      const pathname = decodeURIComponent(url.pathname);
      if (pathname.startsWith("/__artifacts/")) {
        const sha256 = pathname.slice("/__artifacts/".length);
        const artifact = localArtifacts.get(sha256);
        if (!artifact) return send(response, "not found", "text/plain", 404);
        const etag = `"sha256-${artifact.sha256}"`;
        const requestedRange = request.headers.range ?? null;
        const ifRange = request.headers["if-range"] ?? null;
        const range = ifRange !== null && ifRange !== etag
          ? parseSingleByteRange(null, artifact.bytes)
          : parseSingleByteRange(requestedRange, artifact.bytes);
        const responseBytes = range.end - range.start + 1;
        response.writeHead(
          range.status,
          resourceHeaders(
            responseBytes,
            etag,
            range.status === 206
              ? `bytes ${range.start}-${range.end}/${artifact.bytes}`
              : null,
          ),
        );
        if (method === "HEAD") response.end();
        else await pipeline(
          createReadStream(artifact.absolute, { start: range.start, end: range.end }),
          response,
        );
        return;
      }
      let absolute = pathname === "/" || pathname === "/app" || pathname.startsWith("/app/")
        ? resolve(root, "index.html")
        : resolve(root, `.${pathname}`);
      if (absolute !== root && !absolute.startsWith(`${root}${sep}`)) {
        return send(response, "not found", "text/plain", 404);
      }
      let info;
      try {
        info = await stat(absolute);
      } catch {
        if (!extname(pathname)) {
          absolute = resolve(root, "index.html");
          info = await stat(absolute);
        } else return send(response, "not found", "text/plain", 404);
      }
      if (!info.isFile()) return send(response, "not found", "text/plain", 404);
      response.writeHead(200, {
        "Content-Type": contentType(absolute),
        "Content-Length": info.size,
      });
      if (method === "HEAD") response.end();
      else await pipeline(createReadStream(absolute), response);
    } catch (error) {
      console.error(error);
      if (response.headersSent) response.destroy(error instanceof Error ? error : undefined);
      else send(response, "Internal server error", "text/plain", 500);
    }
  });
}

async function waitForRequestQuiescence(requests, quietMs = 500, timeoutMs = 5_000) {
  const startedAt = Date.now();
  let lastChangeAt = startedAt;
  let previousCount = requests.length;
  while (Date.now() - startedAt < timeoutMs) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 50));
    if (requests.length !== previousCount) {
      previousCount = requests.length;
      lastChangeAt = Date.now();
    }
    if (Date.now() - lastChangeAt >= quietMs) return;
  }
  throw new Error("offline shell requests did not become quiescent");
}

function resourceHeaders(bytes, etag, contentRange) {
  return {
    "access-control-allow-origin": "*",
    "accept-ranges": "bytes",
    "cache-control": "no-store",
    "content-length": String(bytes),
    "content-type": "application/octet-stream",
    "cross-origin-resource-policy": "cross-origin",
    "access-control-expose-headers": "Accept-Ranges, Content-Length, Content-Range, ETag",
    ...(etag ? { etag } : {}),
    ...(contentRange ? { "content-range": contentRange } : {}),
  };
}

function contentType(path) {
  const extension = extname(path);
  if (extension === ".html") return "text/html; charset=utf-8";
  if (extension === ".js") return "text/javascript; charset=utf-8";
  if (extension === ".css") return "text/css; charset=utf-8";
  if (extension === ".json" || extension === ".webmanifest") return "application/json; charset=utf-8";
  if (extension === ".wasm") return "application/wasm";
  if (extension === ".svg") return "image/svg+xml";
  if (extension === ".png") return "image/png";
  if (extension === ".woff2") return "font/woff2";
  return "application/octet-stream";
}

function send(response, body, type, statusCode = 200) {
  response.writeHead(statusCode, {
    "Content-Type": type,
    "Content-Length": Buffer.byteLength(body),
  });
  response.end(body);
}

async function listen(server) {
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
}

async function runNode(script, args) {
  const result = await execFile(process.execPath, [resolve(import.meta.dirname, script), ...args], {
    cwd: webuiRoot,
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.stdout) process.stderr.write(result.stdout);
  if (result.stderr) process.stderr.write(result.stderr);
}

function artifactUrl(repository, revision, path) {
  return `https://huggingface.co/${repository}/resolve/${revision}/${
    path.split("/").map(encodeURIComponent).join("/")
  }`;
}

function unsafePath(path) {
  return path.split("/").some((part) => !part || part === "." || part === ".." || part.includes("\\"));
}

function sumBytes(files) {
  return files.reduce((sum, file) => sum + file.bytes, 0);
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function requireEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: received ${actual}, expected ${expected}`);
}

function stage(message) {
  currentStage = message;
  process.stderr.write(`[production-app] ${message}\n`);
}
