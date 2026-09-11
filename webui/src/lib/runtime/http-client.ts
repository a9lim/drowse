import {
  apiExtractStream,
  apiInstruments,
  apiManifoldFitStream,
  apiManifoldGenerateStream,
  apiManifoldInstallStream,
  apiManifolds,
  apiProbes,
  apiProfiles,
  apiSessions,
  apiTemplates,
  apiTree,
  connectWs,
} from "../api";
import type { WSClientMessage, WSServerMessage } from "../types";
import type {
  InstalledInstrumentPackActivationResult,
  RuntimeClient,
  RuntimeEventChannel,
  RuntimeEventChannelState,
  RuntimeInstrumentsService,
  RuntimeManifoldsService,
  RuntimeProfilesService,
} from "./contracts";
import { ApiError } from "./errors";

class HttpRuntimeEventChannel implements RuntimeEventChannel {
  private socket: WebSocket | null = null;
  private opening: Promise<void> | null = null;
  private readonly listeners = new Set<(message: WSServerMessage) => void>();
  private readonly stateListeners = new Set<(state: RuntimeEventChannelState) => void>();
  private readonly expectedClosures = new WeakSet<WebSocket>();
  private connectionRevision = 0;

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  async open(): Promise<void> {
    if (this.isOpen) return;
    if (this.opening) return this.opening;

    const socket = connectWs();
    const revision = ++this.connectionRevision;
    this.socket = socket;
    socket.addEventListener("message", (event: MessageEvent) => {
      if (this.socket !== socket) return;
      let message: WSServerMessage;
      try {
        message = JSON.parse(String(event.data)) as WSServerMessage;
      } catch {
        return;
      }
      for (const listener of this.listeners) listener(message);
    });
    socket.addEventListener("close", () => {
      if (this.connectionRevision !== revision) return;
      if (this.socket === socket) this.socket = null;
      const expected = this.expectedClosures.has(socket);
      this.expectedClosures.delete(socket);
      this.emitState({
        state: "closed",
        expected,
        reason: expected ? null : "The Drowse runtime connection closed unexpectedly",
      });
    });

    const opening = new Promise<void>((resolve, reject) => {
      const fail = () => reject(new Error("Drowse runtime connection failed"));
      socket.addEventListener("open", () => {
        if (this.socket !== socket) { fail(); return; }
        this.emitState({ state: "open" });
        resolve();
      }, { once: true });
      socket.addEventListener("error", fail, { once: true });
      socket.addEventListener("close", fail, { once: true });
    });
    this.opening = opening;
    try {
      await opening;
    } finally {
      if (this.opening === opening) this.opening = null;
    }
  }

  subscribe(listener: (message: WSServerMessage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  subscribeState(listener: (state: RuntimeEventChannelState) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  send(message: WSClientMessage): void {
    if (!this.isOpen || !this.socket) {
      throw new Error("Drowse runtime event channel is not open");
    }
    this.socket.send(JSON.stringify(message));
  }

  async stop(): Promise<void> {
    if (this.isOpen) this.send({ type: "stop" });
  }

  acknowledgeSnapshot(): void {}

  close(): void {
    this.opening = null;
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      this.expectedClosures.add(socket);
      socket.close();
    }
  }

  private emitState(state: RuntimeEventChannelState): void {
    for (const listener of this.stateListeners) listener(state);
  }
}

const profiles: RuntimeProfilesService = {
  ...apiProfiles,
  extract(request, onEvent, id) {
    return apiExtractStream(request, onEvent, id);
  },
};

const instruments: RuntimeInstrumentsService = {
  ...apiInstruments,
  tokenReadout(family, nodeId, rawIndex, options, id, _onProgress) {
    return apiInstruments.tokenReadout(family, nodeId, rawIndex, options, id);
  },
  async activateInstalledPack(family, body, id) {
    if (family === "lens") {
      const activated = await apiInstruments.setLensSource(body.source, id);
      return {
        state: "active",
        family,
        source: activated.source,
        live: { enabled: true, layers: activated.live_layers },
        contextBindingSha256: body.contextBindingSha256 ?? null,
        reloadRequired: false,
      } satisfies InstalledInstrumentPackActivationResult;
    }
    const status = await apiInstruments.startPreparation("sae", {
      operation: "fetch",
      release: body.source,
      layer: body.layer ?? null,
    }, id);
    if (status.error) {
      throw new ApiError(
        409,
        "runtime:instruments.activation_failed",
        status.error,
        { detail: status.error },
      );
    }
    if (status.state === "running" || status.finished_at === null) {
      throw new ApiError(
        409,
        "runtime:instruments.activation_async",
        "The server started preparing this SAE; follow the preparation status before using it",
        {
          detail: "The server started preparing this SAE; follow the preparation status before using it",
        },
      );
    }
    return {
      state: "active",
      family,
      source: status.source ?? status.release ?? body.source,
      live: {
        enabled: status.error === null,
        layer: status.info?.layer ?? body.layer ?? null,
        source: status.source ?? status.release ?? body.source,
      },
      contextBindingSha256: body.contextBindingSha256 ?? null,
      reloadRequired: false,
    } satisfies InstalledInstrumentPackActivationResult;
  },
};

const manifolds: RuntimeManifoldsService = {
  ...apiManifolds,
  install(request, onEvent) {
    return onEvent
      ? apiManifoldInstallStream(request, onEvent)
      : apiManifolds.install(request);
  },
  fit(namespace, name, request, onEvent) {
    return apiManifoldFitStream(namespace, name, request, onEvent);
  },
  generate(request, onEvent) {
    return apiManifoldGenerateStream(request, onEvent);
  },
  async drowseArchiveList() {
    throw browserArtifactUnavailable();
  },
  async drowseArchiveInstall() {
    throw browserArtifactUnavailable();
  },
  async drowseArchiveExport() {
    throw browserArtifactUnavailable();
  },
  async drowseArchiveDelete() {
    throw browserArtifactUnavailable();
  },
};

function browserArtifactUnavailable(): ApiError {
  return new ApiError(
    501,
    "runtime:manifolds.drowse",
    "Browser .drowse storage is unavailable in server mode",
    { detail: "Use the Drowse pack CLI to import or export .drowse files in server mode." },
  );
}

export class HttpRuntimeClient implements RuntimeClient {
  readonly mode = "http" as const;
  readonly sessions = apiSessions;
  readonly profiles = profiles;
  readonly probes = apiProbes;
  readonly manifolds = manifolds;
  readonly templates = apiTemplates;
  readonly tree = apiTree;
  readonly instruments = instruments;
  readonly events: RuntimeEventChannel = new HttpRuntimeEventChannel();

  async dispose(): Promise<void> {
    this.events.close();
  }
}
