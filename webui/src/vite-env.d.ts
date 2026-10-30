/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

declare const __DROWSE_HOSTED_RELEASE__: boolean;
declare const __DROWSE_SOURCE_REVISION__: string;
declare const __DROWSE_SOURCE_URL__: string;
declare const __DROWSE_VERSION__: string;
declare const __DROWSE_ARTIFACT_SOURCE_REVISION__: string;
declare const __DROWSE_RUNTIME_ABI__: string;
declare const __DROWSE_HOOK_ABI__: string;
declare const __DROWSE_HOSTED_CHANNEL__: "preview" | "release";
/** Digest-versioned URLs of the hosted fitting kernel; defined only by the hosted build. */
declare const __DROWSE_FITTING_WASM__: { module: string; wasm: string };
