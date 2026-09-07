import distributionLock from "../../../../browser-runtime/distribution-lock.json";
import { BROWSER_RUNTIME_LOCK } from "./runtimeLock";
import type { CatalogRuntimeLockModel } from "../../lib/runtime/catalog";

const publicKeys = distributionLock.publicKeys as Array<{
  keyId: string;
  ed25519PublicKeyBase64: string;
}>;

export interface HostedDistributionConfig {
  status: "feasibility-required" | "verified";
  catalogUrl: string;
  signatureUrl: string;
  minimumAcceptedSequence: number;
  expectedRuntimeAbi: string;
  runtimeLockModels: readonly CatalogRuntimeLockModel[];
  publicKeys: ReadonlyMap<string, Uint8Array>;
  allowedCatalogRedirectOrigins: ReadonlySet<string>;
  allowedArtifactRedirectOrigins: ReadonlySet<string>;
}

export const HOSTED_DISTRIBUTION_CONFIG: HostedDistributionConfig = Object.freeze({
  status: distributionLock.status as HostedDistributionConfig["status"],
  catalogUrl: distributionLock.catalogUrl,
  signatureUrl: distributionLock.signatureUrl,
  minimumAcceptedSequence: distributionLock.minimumAcceptedSequence,
  expectedRuntimeAbi: BROWSER_RUNTIME_LOCK.runtimeAbi,
  runtimeLockModels: BROWSER_RUNTIME_LOCK.models.flatMap((model) => {
    if (
      model.convertedRevision === null || model.manifestSha256 === null ||
      model.librarySha256 === null || model.tokenizerSha256 === null ||
      model.chatTemplateSha256 === null || model.hiddenSize === null ||
      model.layerMap === null
    ) return [];
    return [{
      modelType: model.modelType,
      runtimeIdentity: {
        sourceModel: model.sourceRepository,
        sourceRevision: model.sourceRevision,
        convertedManifestSha256: model.manifestSha256,
        quantization: model.quantization,
        tokenizerSha256: model.tokenizerSha256,
        chatTemplateSha256: model.chatTemplateSha256,
        modelLibrarySha256: model.librarySha256,
        runtimeAbi: BROWSER_RUNTIME_LOCK.runtimeAbi,
        hookAbi: BROWSER_RUNTIME_LOCK.hookAbi,
        hiddenSize: model.hiddenSize,
        layerMap: [...model.layerMap],
      },
      structuredHookProfile: model.structuredHookProfile,
      thinkingProfile: model.thinkingProfile,
      convertedRepository: model.convertedRepository,
      convertedRevision: model.convertedRevision,
      contextProfiles: [...model.contextProfiles],
    } satisfies CatalogRuntimeLockModel];
  }),
  publicKeys: new Map(
    publicKeys.map((key) => [
      key.keyId,
      decodeBase64(key.ed25519PublicKeyBase64),
    ]),
  ),
  allowedCatalogRedirectOrigins: new Set(
    distributionLock.allowedCatalogRedirectOrigins,
  ),
  allowedArtifactRedirectOrigins: new Set(
    distributionLock.allowedArtifactRedirectOrigins,
  ),
});

function decodeBase64(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}
