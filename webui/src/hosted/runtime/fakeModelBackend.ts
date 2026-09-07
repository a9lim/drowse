import type {
  RuntimeProgressEvent,
  RuntimeServiceRequest,
} from "../../lib/runtime/contracts";
import type {
  SessionInfo,
  WSGenerateRequest,
  WSServerMessage,
  WSSubmitRequest,
} from "../../lib/types";
import { DeterministicFakeRuntime } from "./fakeRuntime";
import type {
  BrowserModelBackend,
  BrowserModelLoadRequest,
  BrowserModelLoadResult,
} from "./modelBackend";

export class DeterministicBrowserModelBackend implements BrowserModelBackend {
  private runtime: DeterministicFakeRuntime | null = null;

  constructor(
    private readonly tokenDelayMs = 0,
    private readonly sessionOverrides: Partial<SessionInfo> = {},
  ) {}

  async load(request: BrowserModelLoadRequest): Promise<BrowserModelLoadResult> {
    if (request.signal.aborted) throw new DOMException("Model load was cancelled", "AbortError");
    await this.unload();
    if (request.signal.aborted) throw new DOMException("Model load was cancelled", "AbortError");
    this.runtime = new DeterministicFakeRuntime({
      modelId: request.variant.id,
      sessionOverrides: this.sessionOverrides,
      tokenDelayMs: this.tokenDelayMs,
      instrumentPacks: request.optionalPacks.map(({ pack }) => pack),
      contextBindingSha256: request.variant.contextProfiles.find(
        (profile) => profile.contextTokens === request.contextTokens,
      )?.bindingSha256,
    });
    await this.runtime.events.open();
    return {
      prefillTokensPerSecond: 120,
      decodeTokensPerSecond: 48,
    };
  }

  async unload(): Promise<void> {
    const runtime = this.runtime;
    this.runtime = null;
    if (runtime) await runtime.dispose();
  }

  async refreshManifolds(): Promise<void> {}

  async request(
    request: RuntimeServiceRequest,
    _onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<unknown> {
    const runtime = this.requireRuntime();
    const service = runtime[request.service] as unknown as Record<
      string,
      (...args: unknown[]) => Promise<unknown>
    >;
    const method = service[request.method];
    if (typeof method !== "function") {
      throw new Error(`Fixture service ${request.service}.${request.method} is unavailable`);
    }
    return method(...request.args);
  }

  generate(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
    onAdmitted: () => void = () => undefined,
  ): Promise<void> {
    const generation = this.requireRuntime().generate(request, emit);
    onAdmitted();
    return generation;
  }

  async stop(): Promise<void> {
    await this.runtime?.stopGeneration();
  }

  private requireRuntime(): DeterministicFakeRuntime {
    if (!this.runtime) throw new Error("The deterministic fixture model is not loaded");
    return this.runtime;
  }
}
