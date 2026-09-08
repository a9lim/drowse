export const WINDOWS_GPU_GUIDANCE = "If this laptop has a dedicated GPU, in Chrome copy chrome://flags/#force-high-performance-gpu into the address bar, enable the flag, and restart Chrome. Alternatively, open Windows Settings > System > Display > Graphics, add your browser (chrome.exe for Chrome), and set it to High performance. Restart the browser and run the device check again. Confirm that Selected GPU names the dedicated GPU before opening a model.";

export const WINDOWS_INTEL_GEN9_BLOCK = `Drowse model initialization has been reported to hang Chrome on Windows with Intel Gen9 graphics, so loading is blocked on this configuration. ${WINDOWS_GPU_GUIDANCE} If no dedicated GPU is available, use another device or the Python edition.`;

export function windowsIntelGen9Blocked(
  vendor: string | undefined,
  architecture: string | undefined,
  userAgent: string,
): boolean {
  return windowsIntelGpuHint(vendor, userAgent) !== null &&
    /^gen-9$/iu.test(architecture?.trim() ?? "") && /(?:Chrome|Chromium)\//u.test(userAgent);
}

export function windowsIntelGpuHint(vendor: string | undefined, userAgent: string): string | null {
  return /windows/iu.test(userAgent) && /^intel$/iu.test(vendor?.trim() ?? "")
    ? `The browser selected an Intel GPU. On Windows, Chrome ignores WebGPU's high-performance preference; Drowse cannot select a GPU that Chrome does not expose. ${WINDOWS_GPU_GUIDANCE}`
    : null;
}
