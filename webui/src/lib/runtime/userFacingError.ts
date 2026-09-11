import { WINDOWS_GPU_GUIDANCE, WINDOWS_INTEL_GEN9_BLOCK } from "./gpuRecovery";

type ErrorMessage = string | ((raw: string) => string);

const MESSAGES: Array<[RegExp, ErrorMessage]> = [
  [/MODEL_LOAD_TIMEOUT/u,
    "The model took too long to load. Reload this page and try a smaller model. Your saved chats and downloads have not been removed."],
  [/APP_MODULE_UNAVAILABLE/u,
    "Some app files could not load. Reconnect, reload Drowse, then reopen the model. Your downloaded models and saved chats are kept."],
  [/WEBGPU_WINDOWS_INTEL_GEN9_BLOCKED/u, WINDOWS_INTEL_GEN9_BLOCK],
  [/REPEATED_DEVICE_LOSS/u,
    () => gpuRecoveryMessage("Model loading is blocked because the graphics device repeatedly stopped responding. Update your browser and operating system, or use another device.")],
  [/SAMPLING_RUNTIME_OUTDATED/u,
    "Drowse is using an older model runtime that cannot accept these sampling settings. Reload Drowse, then reopen the model. Your downloaded models and saved chats do not need to be removed."],
  [/WEBGPU_LIMIT_TOO_LOW/u, webGpuLimitMessage],
  [/WEBGPU_FEATURE_(?:MISSING|UNAVAILABLE)/u,
    "This browser’s graphics support is missing a feature this model needs. Update your browser and graphics driver, then run the device check again. If it still fails, use another supported browser."],
  [/WEBGPU_(?:UNAVAILABLE|ADAPTER_|CALIBRATION_|LIMIT_UNAVAILABLE)|SOFTWARE_ADAPTER|ADAPTER_KIND_UNKNOWN/u,
    "Drowse could not confirm compatible hardware-accelerated graphics. Update your browser and graphics driver, then run the device check again."],
  [/WEBGPU_(?:DEVICE|OUT_OF_MEMORY)|GPU_DEVICE|DEVICE_LOST/u,
    () => gpuRecoveryMessage("The graphics device stopped responding. Close other tabs, then reopen Drowse with a smaller model. If it happens again, update your browser and operating system.")],
  [/INSECURE_CONTEXT|CROSS_ORIGIN_ISOLATION_REQUIRED/u,
    "Drowse needs a secure browser connection. Open it over HTTPS or on localhost, then run the device check again."],
  [/WORKER_UNAVAILABLE|WASM_UNAVAILABLE|BROADCAST_CHANNEL_UNAVAILABLE|WEB_LOCKS_UNAVAILABLE/u,
    "This browser is missing a feature Drowse needs. Update the browser and run the device check again. If it still fails, use another supported browser."],
  [/OPFS_|INDEXEDDB_|CONTENT_STORE|CORRUPT_RECORD|INVALID_PERSISTED_SETTINGS/u,
    "Drowse could not use its private storage on this device. Check that site storage is allowed, reload the page, and try again."],
  [/QUOTA|STORAGE_(?:FULL|INSUFFICIENT|LIMIT)|NOT_ENOUGH_SPACE|DOWNLOAD_QUOTA_INSUFFICIENT/u,
    "There is not enough free space on this device. Free some storage or choose a smaller model, then try again."],
  [/STORAGE_(?:UNKNOWN|ESTIMATE_TIMEOUT|PERSIST)|DOWNLOAD_QUOTA_CHECK_FAILED/u,
    "Drowse could not confirm enough free storage. Check that site storage is allowed, then run the device check again."],
  [/CATALOG_(?:FETCH_(?:UNAVAILABLE|FAILED|TIMEOUT)|OFFLINE_UNAVAILABLE|RESPONSE_SIZE_INVALID)/u,
    "Drowse could not load the model list. Check your connection, refresh the page, and run the device check again."],
  [/CATALOG_|SIGNED_CATALOG|SIGNED_DISTRIBUTION|ED25519|HOSTED_DISTRIBUTION|VERIFIED_CATALOG/u,
    "Drowse could not verify the model list, so it did not download anything. Refresh the page and run the device check again. If the error continues, try again later."],
  [/MLC_|HOOK_(?:ABI|FORMAT)|STRUCTURED_HOOK|RUNTIME_IDENTITY_MISMATCH|MODEL_INSTALL_BINDING_MISMATCH/u,
    "This model package is not compatible with this version of Drowse. Update Drowse and try again. If it still fails, remove the model and download it again."],
  [/REQUIRED_CORE_PACK|CORE_PACK_|WHITENER_|MANIFOLD_ARTIFACT_BRIDGE/u,
    "This model’s response-control files are incomplete or incompatible. Remove the model, download it again, and retry."],
  [/JLENS_DEVICE_BUFFER_LIMIT|SAE_DEVICE_BUFFER_LIMIT|JLENS_GPU_BUFFER_TOO_LARGE/u,
    "This model tool needs a larger graphics buffer than this browser can provide. Use the model without this tool, or choose a smaller compatible model or tool."],
  [/OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED/u,
    "These model tools need more graphics memory than Drowse can safely reserve at once. Remove one installed tool, or choose a smaller model or tool."],
  [/OPTIONAL_PACK_|INSTRUMENT_|JLENS_|SAE_|EXACT_(?:JLENS|SAE|READOUT)|MEASUREMENT_|BUNDLED_MEASUREMENT/u,
    "A model tool is incomplete or incompatible. Remove that tool in Model settings, download it again, and retry."],
  [/DOWNLOAD_(?:HTTP_REJECTED|REDIRECT_REJECTED)|ARTIFACT_ROUTE_UNAVAILABLE/u,
    "The model host did not provide an approved download. Refresh the page and try again later."],
  [/ARTIFACT_|VERIFIED_ARTIFACT_|UNVERIFIED_ARTIFACT|CHECKSUM|HASH|INTEGRITY|SIZE_MISMATCH|SAFETENSORS/u,
    "A downloaded file failed its safety check. Resume the download to replace it. If it fails again, remove the model and download it again."],
  [/ARCHIVE_|INVALID_TRANSCRIPT|TRANSCRIPT_MODEL_MISMATCH|JSON_(?:INVALID|NOT_UTF8|DUPLICATE_KEY)/u,
    "Drowse could not read that file. Choose an unmodified Drowse file and try again."],
  [/LOCAL_SESSION_(?:INCOMPATIBLE|SCHEMA_UNSUPPORTED)|UNSUPPORTED_SESSION_SCHEMA/u,
    "This saved conversation is not compatible with the current local model session. Start a new local session to continue."],
  [/LOCK|TAKEOVER|RUNTIME_OWNED|RUNTIME_IN_USE|MODEL_IN_USE|OWNERSHIP_NOT_CLAIMED|STALE_SESSION_OWNER/u,
    "Another Drowse tab is using the local model. Close it or ask that tab to release the model, then try again."],
  [/OUT_OF_MEMORY|\bOOM\b|MODEL_MEMORY|MODEL_OOM|CONFIRMED_OOM/u,
    "This model needs more graphics memory than this device could provide. Try a smaller model or a shorter context."],
  [/MODEL_(?:NOT_INSTALLED|VARIANT_NOT_FOUND|VARIANT_AMBIGUOUS)/u,
    "Drowse could not find the selected model on this device. Choose or download the model again."],
  [/MODEL_(?:NOT_LOADED|RUNTIME_NOT_READY|LOAD_FAILED|RECOVERY)|RUNTIME_(?:CHANNEL_CLOSED|WORKER_FAILED|CLEANUP_FAILED|STOP_FAILED)/u,
    "The local model is not ready. Reopen it and try again."],
  [/DOWNLOAD_(?:CANCELLED|REQUEST_CANCELLED)/u,
    "The download was paused. Resume it when you are ready; your verified progress is saved."],
  [/NETWORK|DOWNLOAD|FETCH|OFFLINE|RANGE_/u,
    "The download was interrupted. Check your connection and resume it; your verified progress is saved."],
  [/MUTATION_DURING_GENERATION|GENERATION_(?:BUSY|IN_PROGRESS)/u,
    "Finish or stop the current reply before changing that part of the conversation."],
  [/GENERATION_|INVALID_GENERATION|MODEL_REQUEST_CANCELLED/u,
    "The model could not finish that reply. Try again. If it keeps happening, reopen the model."],
  [/LOOM_|STALE_TREE|TREE_|EVENT_SEQUENCE_GAP/u,
    "The conversation changed before that action finished. Reload the conversation and try again."],
  [/TOKEN_REPLAY|JOINT_LOGPROB|BROWSER_SERVICE_UNAVAILABLE|REPLAY_SCORING|FORCED_REPLAY|CONTINUATION_REPLAY/u,
    "That analysis is not available in this browser session. Reopen the model and try again."],
  [/AUTHORING_OPERATION_UNAVAILABLE|FITTING_UNAVAILABLE|BROWSER_(?:AUTHORING|FITTING|FIT_)|HOSTED_(?:JLENS_FITTING|SAE_TRAINING)/u,
    "That building method is not available for this model in the browser."],
  [/BROWSER_AUTOMATIC_TOPOLOGY_UNAVAILABLE/u,
    "Choose linear PCA as the fit method. Automatic shape detection is not available in the browser."],
  [/FITTING_|WASM_FITTING/u,
    "Drowse could not finish building that model tool. Check the selected inputs and try again."],
  [/CAPTURE_|MEASUREMENT_|GEOMETRY_|PROBE_|RANK_ONE_CAPTURE|SCALAR_MEASUREMENTS|DROWSE_CAPTURE/u,
    "Drowse could not load that model analysis. Reopen the model and try again."],
  [/PACK_|MANIFEST_|CLOSURE_INVALID|ARTIFACT_REPOSITORY_INVALID/u,
    "That model package is incomplete or incompatible. Remove it, download it again, and retry."],
  [/INVALID_|BROWSER_.*INVALID|NONFINITE_VALUES|LAYER_MISMATCH|IDENTITY_MISMATCH/u,
    "Some information needed for that action was invalid. Review your changes and try again."],
];

export function userFacingError(
  error: unknown,
  fallback = "Drowse could not complete that action. Try again or reopen the model.",
): string {
  const raw = errorMessage(error).replace(/^Error:\s*/u, "").trim();
  const suppliedCode = errorCode(error);
  const genericCode = /^(?:WORKER_OPERATION_FAILED|WORKER_REQUEST_FAILED)$/u.test(suppliedCode);
  const code = (!suppliedCode || genericCode ? inferredCode(raw) : "") || suppliedCode;
  for (const [pattern, message] of MESSAGES) {
    if (pattern.test(code)) return typeof message === "function" ? message(raw) : message;
  }
  if (code || !raw || raw.includes("\n") || looksInternal(raw)) return fallback;
  return raw;
}

function gpuRecoveryMessage(message: string): string {
  return typeof navigator !== "undefined" && /windows/iu.test(navigator.userAgent)
    ? `${message} On Windows: ${WINDOWS_GPU_GUIDANCE}`
    : message;
}

function inferredCode(raw: string): string {
  if (/importing a module script failed|failed to (?:fetch dynamically imported module|load module script)|error loading dynamically imported module|unable to preload css|load failed for module/iu.test(raw)) {
    return "APP_MODULE_UNAVAILABLE";
  }
  if (/\b(?:WEBGPU_DEVICE_LOST|GPU_DEVICE_LOST)\b|(?:WebGPU|GPU|graphics) device (?:was |is |has been )?lost|DeviceLostError/iu.test(raw)) {
    return "WEBGPU_DEVICE_LOST";
  }
  if (/Make sure 0 < top_logprobs <= 5\. Got (?:[6-9]|[1-9]\d+)(?:\.|$)/u.test(raw)) {
    return "SAMPLING_RUNTIME_OUTDATED";
  }
  if (/^[A-Z][A-Z0-9_]{2,}$/u.test(raw)) return raw;
  if (/maxStorageBuffersPerShaderStage|graphics buffers?/iu.test(raw)) {
    return "WEBGPU_LIMIT_TOO_LOW";
  }
  if (/mlc-chat-config\.json|capture special-token IDs?/iu.test(raw)) {
    return "MLC_CAPTURE_TOKEN_IDS_INVALID";
  }
  if (/file\[\d+\]\.url|immutable Drowse Hugging Face URL/iu.test(raw)) {
    return "CATALOG_SCHEMA_INVALID";
  }
  if (/out of (?:graphics )?memory|allocation.+memory/iu.test(raw)) {
    return "OUT_OF_MEMORY";
  }
  if (/quota.+(?:exceeded|insufficient)|not enough (?:free )?space/iu.test(raw)) {
    return "STORAGE_INSUFFICIENT";
  }
  return "";
}

function webGpuLimitMessage(raw: string): string {
  const match = raw.match(
    /requires\s+[A-Za-z][A-Za-z0-9]*\s*>=\s*(\d+);\s*this adapter reports\s*(\d+|unknown)/iu,
  );
  const detail = match && match[2] !== "unknown"
    ? `This browser provides ${match[2]} of the ${match[1]} graphics buffers this model needs. `
    : "This browser does not provide enough graphics buffers for this model. ";
  return `${detail}Update your browser and graphics driver, then run the device check again. If it still fails, use another supported browser.`;
}

function errorCode(error: unknown): string {
  if (!error || typeof error !== "object") return "";
  const record = error as {
    code?: unknown;
    body?: { code?: unknown; error?: { code?: unknown } };
  };
  const value = typeof record.code === "string"
    ? record.code
    : typeof record.body?.code === "string"
      ? record.body.code
      : record.body?.error?.code;
  return typeof value === "string" ? value.toUpperCase() : "";
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  if (
    error && typeof error === "object" &&
    typeof (error as { message?: unknown }).message === "string"
  ) return (error as { message: string }).message;
  return "";
}

function looksInternal(message: string): boolean {
  return /(?:TypeError|ReferenceError|WebAssembly|OPFS|IndexedDB|runtime ABI|hook ABI|SHA-?256|\.json\b|\.wasm\b|(?:max|min)[A-Z][A-Za-z0-9]+|\bfile\[\d+\]|(?:>=|<=)|\b\d+-byte\b|https?:\/\/|\bat \S+[:(]\d|[A-Z][A-Z0-9]+_[A-Z0-9_]{2,})/u.test(message);
}
