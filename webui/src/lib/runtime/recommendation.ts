import type {
  CapabilityIssue,
  CatalogModel,
  DeviceLoadRecord,
  ModelAssessment,
  ModelPreference,
  ModelRecommendation,
  ModelTier,
  ModelVariant,
  RuntimeCapabilities,
} from "./contracts";
import { MEBIBYTE } from "./contracts";
import { userFacingError } from "./userFacingError";

export function requiredStorageBytes(
  downloadBytes: number,
  requiredCorePackBytes: number,
): number {
  const headroom = Math.max(downloadBytes * 0.2, 256 * MEBIBYTE);
  return Math.ceil(downloadBytes + requiredCorePackBytes + headroom);
}

export interface AssessmentOptions {
  capabilities: RuntimeCapabilities;
  contextTokens: number;
  loadRecords?: DeviceLoadRecord[];
  installedModelVariantIds?: readonly string[];
  explicitOomRetry?: boolean;
}

function issue(
  code: string,
  message: string,
  severity: "hard" | "advisory",
): CapabilityIssue {
  return { code, message, severity };
}

export function assessModelVariant(
  model: CatalogModel,
  variant: ModelVariant,
  options: AssessmentOptions,
): ModelAssessment {
  const { capabilities, contextTokens } = options;
  const hardFailures = capabilities.issues.filter((item) => item.severity === "hard");
  const advisories = capabilities.issues.filter(
    (item) => item.severity === "advisory",
  );
  const context =
    variant.contextProfiles.find((profile) => profile.contextTokens === contextTokens) ??
    null;
  const installed = options.installedModelVariantIds?.includes(variant.id) ?? false;
  const requiredBytes = installed
    ? 0
    : requiredStorageBytes(
        variant.downloadBytes,
        variant.requiredCorePackBytes,
      );

  const requiredOperations = installed
    ? (["generation"] as const)
    : (["download", "generation"] as const);
  for (const operation of requiredOperations) {
    const availability = capabilities.operations[operation];
    if (availability.available) continue;
    if (!availability.reasons.length) {
      hardFailures.push(
        issue(
          "RUNTIME_OPERATION_UNAVAILABLE",
          `The browser runtime cannot perform ${operation}`,
          "hard",
        ),
      );
      continue;
    }
    for (const reason of availability.reasons) {
      if (
        !hardFailures.some(
          (existing) =>
            existing.code === reason.code && existing.message === reason.message,
        )
      ) hardFailures.push({ ...reason, severity: "hard" });
    }
  }

  if (!context) {
    hardFailures.push(
      issue(
        "CONTEXT_UNSUPPORTED",
        `${variant.id} has no measured ${contextTokens}-token profile`,
        "hard",
      ),
    );
  }

  for (const feature of variant.requirements.features) {
    if (!capabilities.webGpu.features.includes(feature)) {
      hardFailures.push(
        issue(
          "WEBGPU_FEATURE_MISSING",
          `${variant.id} requires the WebGPU feature ${feature}`,
          "hard",
        ),
      );
    }
  }

  for (const [name, minimum] of Object.entries(variant.requirements.limits)) {
    const actual = capabilities.webGpu.limits[name];
    if (actual === undefined || actual < minimum) {
      hardFailures.push(
        issue(
          "WEBGPU_LIMIT_TOO_LOW",
          `${variant.id} requires ${name} >= ${minimum}; this adapter reports ${actual ?? "unknown"}`,
          "hard",
        ),
      );
    }
  }

  if (capabilities.signals.appleMobile === true && isFourBModel(model, variant)) {
    hardFailures.push(
      issue(
        "APPLE_MOBILE_MODEL_TOO_LARGE",
        `${model.displayName} is not offered on iPhone or iPad because its browser memory requirements are too high`,
        "hard",
      ),
    );
  }

  if (
    capabilities.storage.availableBytes !== null &&
    capabilities.storage.availableBytes < requiredBytes
  ) {
    hardFailures.push(
      issue(
        "STORAGE_INSUFFICIENT",
        `${variant.id} requires ${requiredBytes} bytes of available storage`,
        "hard",
      ),
    );
  } else if (!installed && capabilities.storage.availableBytes === null) {
    hardFailures.push(
      issue(
        "STORAGE_UNKNOWN",
        "The browser did not report an available-storage estimate",
        "hard",
      ),
    );
  }

  const matchingRecords = (options.loadRecords ?? []).filter(
    (record) =>
      record.modelVariantId === variant.id &&
      record.runtimeIdentitySha256 === variant.runtimeIdentitySha256 &&
      record.deviceSignature === capabilities.deviceSignature &&
      record.contextTokens === contextTokens,
  );
  const orderedRecords = matchingRecords
    .map((record, index) => ({ record, index }))
    .sort(
      (a, b) =>
        a.record.recordedAt - b.record.recordedAt || a.index - b.index,
    );
  const lastSuccess = orderedRecords.findLastIndex(
    ({ record }) => record.result === "success",
  );
  const lastOom = orderedRecords.findLastIndex(
    ({ record }) => record.result === "oom",
  );
  const deviceLossesAfterSuccess = orderedRecords
    .slice(lastSuccess + 1)
    .filter(({ record }) => record.result === "device_lost").length;
  const confirmedOom = lastOom >= 0 && lastOom > lastSuccess;
  const repeatedDeviceLoss = deviceLossesAfterSuccess >= 2;
  const proven =
    lastSuccess >= 0 && !confirmedOom && !repeatedDeviceLoss;

  if (confirmedOom && !options.explicitOomRetry) {
    hardFailures.push(
      issue(
        "CONFIRMED_OOM",
        "This exact model, runtime, context, and device profile previously ran out of memory",
        "hard",
      ),
    );
  }
  if (repeatedDeviceLoss) {
    advisories.push(
      issue(
        "REPEATED_DEVICE_LOSS",
        "This configuration repeatedly lost the WebGPU device",
        "advisory",
      ),
    );
  }

  if (context && context.minimumCalibrationScore !== null) {
    if (capabilities.signals.calibrationScore === null) {
      advisories.push(
        issue(
          "CALIBRATION_UNKNOWN",
          "The WebGPU compute calibration did not produce a usable score",
          "advisory",
        ),
      );
    } else if (
      capabilities.signals.calibrationScore < context.minimumCalibrationScore
    ) {
      advisories.push(
        issue(
          "CALIBRATION_BELOW_PROFILE",
          "The WebGPU compute calibration is below this model's measured profile",
          "advisory",
        ),
      );
    }
  }

  if (
    context &&
    (context.measuredDevices === 0 ||
      (requiresExactBrowserProof(capabilities.signals.runtimeClass) && !proven))
  ) {
    advisories.push(
      issue(
        "PROFILE_UNMEASURED",
        unmeasuredProfileMessage(capabilities),
        "advisory",
      ),
    );
  }

  if (context && context.minimumDeviceMemoryGiB !== null) {
    if (capabilities.signals.deviceMemoryGiB === null) {
      advisories.push(
        issue(
          "DEVICE_MEMORY_UNKNOWN",
          "The browser did not expose its coarse device-memory signal",
          "advisory",
        ),
      );
    } else if (
      capabilities.signals.deviceMemoryGiB < context.minimumDeviceMemoryGiB
    ) {
      advisories.push(
        issue(
          "DEVICE_MEMORY_BELOW_PROFILE",
          "The coarse device-memory signal is below this model's measured profile",
          "advisory",
        ),
      );
    }
  }

  return {
    model,
    variant,
    context,
    installed,
    eligible: hardFailures.length === 0,
    proven,
    hardFailures,
    advisories,
    requiredStorageBytes: requiredBytes,
  };
}

export interface RecommendationOptions extends AssessmentOptions {
  preference: ModelPreference;
  language: string;
}

const tierRank: Record<ModelTier, number> = {
  fastest: 0,
  balanced: 1,
  quality: 2,
};

function preferenceRank(tier: ModelTier, preference: ModelPreference): number {
  if (preference === "speed") return -tierRank[tier];
  if (preference === "quality") return tierRank[tier];
  return tier === "balanced" ? 2 : tier === "fastest" ? 1 : 0;
}

function supportsLanguage(model: CatalogModel, language: string): boolean {
  const requested = language.toLowerCase().split("-")[0];
  return model.languages.some((candidate) => {
    const normalized = candidate.toLowerCase();
    return normalized === "multilingual" || normalized.split("-")[0] === requested;
  });
}

function priorDecodeSpeed(
  assessment: ModelAssessment,
  records: DeviceLoadRecord[],
  capabilities: RuntimeCapabilities,
  contextTokens: number,
): number {
  const latest = records
    .filter(
      (record) =>
        record.result === "success" &&
        record.modelVariantId === assessment.variant.id &&
        record.runtimeIdentitySha256 === assessment.variant.runtimeIdentitySha256 &&
        record.deviceSignature === capabilities.deviceSignature &&
        record.contextTokens === contextTokens &&
        record.decodeTokensPerSecond !== null,
    )
    .sort((a, b) => b.recordedAt - a.recordedAt)[0];
  return latest?.decodeTokensPerSecond ?? 0;
}

export function rankCatalogModels(
  models: CatalogModel[],
  options: RecommendationOptions,
): ModelRecommendation[] {
  const assessments = models.flatMap((model) =>
    model.variants.map((variant) => {
      const baseContext = model.modelType === "base"
        ? variant.contextProfiles.filter((profile) => profile.contextTokens <= options.contextTokens)
          .sort((a, b) => b.contextTokens - a.contextTokens)[0]?.contextTokens
        : undefined;
      return assessModelVariant(model, variant, {
        ...options, contextTokens: baseContext ?? options.contextTokens,
      });
    }),
  );
  const records = options.loadRecords ?? [];
  const unknownDevice =
    !assessments.some((assessment) => assessment.proven) &&
    (options.capabilities.webGpu.fallback === "unknown" ||
      options.capabilities.signals.calibrationScore === null ||
      options.capabilities.signals.deviceMemoryGiB === null);

  assessments.sort((a, b) => {
    if (a.eligible !== b.eligible) return a.eligible ? -1 : 1;
    if (a.proven !== b.proven) return a.proven ? -1 : 1;
    if (unknownDevice) {
      const tier = tierRank[a.variant.tier] - tierRank[b.variant.tier];
      if (tier !== 0) return tier;
    }
    const advisory = a.advisories.length - b.advisories.length;
    if (advisory !== 0) return advisory;
    const preference =
      preferenceRank(b.variant.tier, options.preference) -
      preferenceRank(a.variant.tier, options.preference);
    if (preference !== 0) return preference;
    const speed =
      priorDecodeSpeed(b, records, options.capabilities, b.context?.contextTokens ?? options.contextTokens) -
      priorDecodeSpeed(a, records, options.capabilities, a.context?.contextTokens ?? options.contextTokens);
    if (speed !== 0) return speed;
    const language =
      Number(supportsLanguage(b.model, options.language)) -
      Number(supportsLanguage(a.model, options.language));
    if (language !== 0) return language;
    return a.variant.id.localeCompare(b.variant.id);
  });

  const recommendedIndex = assessments.findIndex(isAutomaticallyRecommendable);
  return assessments.map((assessment, index) => ({
    ...assessment,
    recommended: index === recommendedIndex,
    reason: recommendationReason(assessment, unknownDevice && index === recommendedIndex),
  }));
}

function isAutomaticallyRecommendable(assessment: ModelAssessment): boolean {
  if (assessment.model.modelType === "base") return false;
  if (!assessment.eligible) return false;
  if (assessment.proven) return true;
  if (!assessment.context || assessment.context.measuredDevices < 1) return false;
  return !assessment.advisories.some((item) => MANUAL_ONLY_ADVISORIES.has(item.code));
}

const MANUAL_ONLY_ADVISORIES = new Set([
  "CALIBRATION_BELOW_PROFILE",
  "DEVICE_MEMORY_BELOW_PROFILE",
  "PROFILE_UNMEASURED",
  "REPEATED_DEVICE_LOSS",
]);

function isFourBModel(model: CatalogModel, variant: ModelVariant): boolean {
  return [model.id, model.displayName, variant.id].some((value) =>
    /(^|[^a-z0-9])4b([^a-z0-9]|$)/i.test(value)
  );
}

function requiresExactBrowserProof(
  runtimeClass: RuntimeCapabilities["signals"]["runtimeClass"],
): boolean {
  return runtimeClass === "apple-mobile-webkit" ||
    runtimeClass === "desktop-webkit" ||
    runtimeClass === "desktop-gecko";
}

function unmeasuredProfileMessage(capabilities: RuntimeCapabilities): string {
  if (capabilities.signals.appleMobile === true) {
    return "This model has not yet completed a successful load on this iPhone or iPad";
  }
  if (capabilities.signals.runtimeClass === "desktop-webkit") {
    return "This model has not yet completed a successful load in Safari on this Mac";
  }
  if (capabilities.signals.runtimeClass === "desktop-gecko") {
    return "This model has not yet completed a successful load in Firefox on this computer";
  }
  return "This context profile has not passed a physical-device benchmark";
}

function recommendationReason(
  assessment: ModelAssessment,
  conservativeUnknown: boolean,
): string {
  if (!assessment.eligible) {
    return assessment.hardFailures[0]
      ? userFacingError(
          assessment.hardFailures[0],
          "This model is not compatible with this browser or device. Choose another model.",
        )
      : "This model is not available on this browser or device.";
  }
  if (assessment.proven) return "This exact configuration loaded successfully before.";
  if (!isAutomaticallyRecommendable(assessment)) {
    return assessment.advisories.find((item) => MANUAL_ONLY_ADVISORIES.has(item.code))
      ?.message ?? assessment.advisories[0]?.message ?? (assessment.model.modelType === "base"
        ? "This base model is available for text completion."
        : "This model requires an explicit retry.");
  }
  if (conservativeUnknown) {
    return "This is the conservative choice because browser memory information is incomplete.";
  }
  if (assessment.advisories.length) return assessment.advisories[0].message;
  return "This model fits the measured WebGPU and storage requirements.";
}
