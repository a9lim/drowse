import { createFrameQueue } from "../runtime/frameQueue";
// The singleton runtime event channel and the message dispatcher over it.
//
// One connection owned at module level — the chat panel is not
// responsible for lifecycle.  Subscribers register via ``onWsMessage(cb)``
// and receive every ``WSServerMessage``; ``handleWsMessage`` below is the
// built-in subscriber that owns the gen-status lifecycle, the live token
// stream, the tree deltas, and the per-token instrument fan-out.
//
// Reconnection crosses a rehydration barrier: a socket can reattach to a
// freshly restarted server whose tree has a lower revision and entirely
// different node ids, so wire events buffer until the authoritative
// snapshot has replaced the local cache.
//
// The four send primitives (submit / generate / fork / stop) are the only
// writers.  ``sendSubmit`` is the composer path and respects the pending
// queue; the rest fire immediately because they sit behind an explicit
// user gesture on a specific node.

import { SvelteSet } from "svelte/reactivity";
import type { RuntimeEventChannel } from "../runtime/contracts";
import { runtimeClient } from "../runtime/client";
import { apiTree } from "../runtime/services";
import { userFacingError } from "../runtime/userFacingError";
import type { WSClientMessage, WSServerMessage } from "../types";
import type {
  ChatRole,
  ChatTurn,
  PendingAction,
  ProbeReadingJSON,
  TokenScore,
} from "../types";
import { pushToast } from "./toasts.svelte";
import {
  abState,
  autoRegenState,
  scheduleAutoComparison,
} from "./ab.svelte";
import { chatLog, genStatus, geometricMeanPpl, liveTokenStream } from "./chat.svelte";
import {
  backfillSaeMeta,
  lensReadoutSnapshot,
  lensState,
  mergedReadings,
  recordSaeReadoutFrame,
  saeState,
} from "./instruments.svelte";
import {
  applyTreeDelta,
  applyTreeSnapshot,
  castState,
  loomTree,
  recomputeActivePath,
  refreshLoomTree,
  syncChatLogFromTree,
} from "./loom.svelte";
import {
  drainNextPendingAction,
  enqueuePending,
  isPendingBusy,
  nextPendingId,
} from "./pending.svelte";
import {
  MAX_SPARKLINE,
  highlightState,
  refreshProbeList,
  resetProbeStreams,
  setProbeAggregates,
  snapshotProbeBaseline,
  updateProbesFromReadings,
} from "./probes.svelte";
import { buildSamplingPayload, samplingState } from "./sampling.svelte";
import { refreshSession } from "./session.svelte";
import {
  currentSteeringExpression,
  refreshCorrelation,
  refreshManifoldList,
  refreshVectorList,
} from "./steering.svelte";

type WsListener = (msg: WSServerMessage) => void;

interface RuntimeConnection {
  channel: RuntimeEventChannel | null;
  unsubscribe: (() => void) | null;
  unsubscribeState: (() => void) | null;
  listeners: Set<WsListener>;
  opening: boolean;
  recoveringGap: boolean;
  treeRecovery: Promise<void> | null;
  ready: Promise<void> | null;
  flushTokens: (() => void) | null;
}

const wsConn: RuntimeConnection = {
  channel: null,
  unsubscribe: null,
  unsubscribeState: null,
  listeners: new SvelteSet(),
  opening: false,
  recoveringGap: false,
  treeRecovery: null,
  ready: null,
  flushTokens: null,
};

let firstDecodeTokenAt: number | null = null;

export function onWsMessage(cb: WsListener): () => void {
  wsConn.listeners.add(cb);
  return () => wsConn.listeners.delete(cb);
}

export function ensureRuntimeChannel(): Promise<RuntimeEventChannel> {
  if (wsConn.opening && wsConn.channel && wsConn.ready) {
    return wsConn.ready.then(() => wsConn.channel!);
  }
  if (wsConn.channel?.isOpen) {
    if (wsConn.treeRecovery) {
      return wsConn.treeRecovery.then(() => wsConn.channel!);
    }
    return Promise.resolve(wsConn.channel);
  }

  wsConn.flushTokens?.();
  wsConn.unsubscribe?.();
  wsConn.unsubscribeState?.();
  const channel = runtimeClient.events;
  wsConn.channel = channel;
  wsConn.opening = true;
  // A socket can reconnect to a freshly restarted server whose tree has a
  // lower revision and entirely different node ids.  Buffer wire events until
  // we have replaced the local cache with the new authoritative snapshot;
  // otherwise the first post-restart generation splices new nodes into the
  // stale pre-restart sidebar.
  let rehydrating = true;
  let connectionFailure: string | null = null;
  const bufferedMessages: WSServerMessage[] = [];
  const treeRecoveryBuffer: WSServerMessage[] = [];
  let treeRecovering = false;
  let requiredTreeRevision = 0;
  let treeRecoverySerial = 0;
  const notifyListeners = (msg: WSServerMessage): void => {
    for (const cb of wsConn.listeners) {
      try {
        cb(msg);
      } catch {
        /* ignore subscriber failures */
      }
    }
  };
  const tokenQueue = createFrameQueue<{ message: WSServerMessage; receivedAt: number }>(
    ({ message, receivedAt }) => {
      handleWsMessage(message, receivedAt);
      notifyListeners(message);
    },
    (callback) => requestAnimationFrame(callback),
    (frame) => cancelAnimationFrame(frame),
  );
  wsConn.flushTokens = tokenQueue.flush;
  const dispatch = (msg: WSServerMessage): void => {
    if (treeRecovering) {
      treeRecoveryBuffer.push(msg);
      if (msg.type === "tree_mutated") {
        requiredTreeRevision = Math.max(requiredTreeRevision, msg.rev);
      }
      return;
    }
    if (msg.type === "token" && typeof requestAnimationFrame === "function" &&
      (typeof document === "undefined" || document.visibilityState === "visible")) {
      tokenQueue.push({ message: msg, receivedAt: performance.now() });
      return;
    }
    tokenQueue.flush();
    const disposition = handleWsMessage(msg);
    if (disposition === "tree_resync" && msg.type === "tree_mutated") {
      beginTreeRecovery(msg.rev);
      return;
    }
    notifyListeners(msg);
  };
  const clearConnection = (): void => {
    if (wsConn.channel !== channel) return;
    tokenQueue.flush();
    wsConn.flushTokens = null;
    treeRecoverySerial += 1;
    treeRecovering = false;
    requiredTreeRevision = 0;
    treeRecoveryBuffer.length = 0;
    wsConn.unsubscribe?.();
    wsConn.unsubscribeState?.();
    wsConn.unsubscribe = null;
    wsConn.unsubscribeState = null;
    wsConn.channel = null;
    wsConn.opening = false;
    wsConn.recoveringGap = false;
    wsConn.treeRecovery = null;
    wsConn.ready = null;
  };
  const beginTreeRecovery = (minimumRevision: number): void => {
    requiredTreeRevision = Math.max(requiredTreeRevision, minimumRevision);
    if (treeRecovering || wsConn.channel !== channel) return;
    treeRecovering = true;
    const serial = ++treeRecoverySerial;
    const recovery = (async () => {
      try {
        let snap: Awaited<ReturnType<typeof apiTree.get>> | null = null;
        for (let attempt = 0; attempt < 3; attempt += 1) {
          snap = await apiTree.get();
          if (wsConn.channel !== channel || serial !== treeRecoverySerial) return;
          if (snap.rev >= requiredTreeRevision) break;
        }
        if (!snap || snap.rev < requiredTreeRevision) {
          throw new Error(
            `Authoritative tree revision ${requiredTreeRevision} was not available`,
          );
        }
        if (!applyTreeSnapshot(snap)) {
          throw new Error("The authoritative tree snapshot was older than local state");
        }
        treeRecovering = false;
        requiredTreeRevision = 0;
        const pending = treeRecoveryBuffer.splice(0);
        for (const buffered of pending) {
          if (
            buffered.type === "tree_mutated" &&
            buffered.rev <= loomTree.rev
          ) continue;
          dispatch(buffered);
        }
      } catch (error) {
        if (wsConn.channel !== channel || serial !== treeRecoverySerial) return;
        treeRecovering = false;
        const detail = userFacingError(error, "The conversation could not be restored.");
        channel.close();
        clearConnection();
        const failure: Extract<WSServerMessage, { type: "error" }> = {
          type: "error",
          code: "TREE_RESYNC_FAILED",
          message: `The conversation could not be resynchronized: ${detail}`,
        };
        handleWsMessage(failure);
        notifyListeners(failure);
      }
    })();
    wsConn.treeRecovery = recovery;
    void recovery.finally(() => {
      if (wsConn.channel === channel && wsConn.treeRecovery === recovery) {
        wsConn.treeRecovery = null;
      }
    });
  };
  const recoverSequenceGap = (message: Extract<WSServerMessage, { type: "error" }>): void => {
    tokenQueue.flush();
    if (wsConn.channel !== channel || wsConn.recoveringGap) return;
    wsConn.recoveringGap = true;
    void (async () => {
      try {
        await channel.stop();
        const snap = await apiTree.get();
        if (wsConn.channel !== channel) return;
        applyTreeSnapshot(snap, { preserveLiveTokens: false });
        channel.acknowledgeSnapshot();
        dispatch({
          type: "error",
          code: message.code,
          message: "The event stream lost synchronization. Generation was stopped and the authoritative conversation was restored.",
        });
      } catch (error) {
        if (wsConn.channel !== channel) return;
        const detail = userFacingError(error, "The conversation could not be restored.");
        channel.close();
        clearConnection();
        dispatch({
          type: "error",
          code: "TREE_RESYNC_FAILED",
          message: `The event stream could not be recovered: ${detail}`,
        });
      } finally {
        if (wsConn.channel === channel) wsConn.recoveringGap = false;
      }
    })();
  };
  wsConn.unsubscribe = channel.subscribe((msg) => {
    if (rehydrating) {
      bufferedMessages.push(msg);
    } else if (msg.type === "error" && msg.code === "EVENT_SEQUENCE_GAP") {
      recoverSequenceGap(msg);
    } else {
      dispatch(msg);
    }
  });
  wsConn.unsubscribeState = channel.subscribeState((state) => {
    if (
      state.state !== "closed" || state.expected || wsConn.channel !== channel
    ) return;
    const reason = state.reason ?? "The Drowse runtime connection closed unexpectedly";
    const friendly = userFacingError(
      { code: "RUNTIME_CHANNEL_CLOSED", message: reason },
      "The local model connection closed. Reopen the model and try again.",
    );
    if (rehydrating) {
      connectionFailure = reason;
      return;
    }
    clearConnection();
    loomTree.error = friendly;
    if (
      genStatus.active || chatLog.pendingIndex !== null ||
      loomTree.pendingNodeId !== null || abState.processingAb || isPendingBusy()
    ) {
      dispatch({ type: "error", code: "RUNTIME_CHANNEL_CLOSED", message: reason });
    } else {
      pushToast(friendly, { kind: "error", ttlMs: null });
    }
  });

  const ready = (async () => {
    try {
      await channel.open();
      const snap = await apiTree.get();
      if (connectionFailure || !channel.isOpen) {
        throw new Error(connectionFailure ?? "The Drowse runtime connection closed during setup");
      }
      applyTreeSnapshot(snap, {
        preserveLiveTokens: false,
        allowRevisionRegression: true,
      });
      rehydrating = false;
      for (const msg of bufferedMessages) {
        if (msg.type === "tree_mutated" && msg.rev <= snap.rev) continue;
        if (msg.type === "error" && msg.code === "EVENT_SEQUENCE_GAP") {
          recoverSequenceGap(msg);
        } else {
          dispatch(msg);
        }
      }
      bufferedMessages.length = 0;
      if (wsConn.treeRecovery) await wsConn.treeRecovery;
      if (wsConn.channel !== channel || !channel.isOpen) {
        throw new Error("The Drowse runtime connection closed during tree recovery");
      }
      await Promise.allSettled([
        refreshSession(),
        refreshVectorList(),
        refreshProbeList(),
        refreshCorrelation(),
        refreshManifoldList(),
      ]);
    } catch (error) {
      rehydrating = false;
      loomTree.error = userFacingError(error, "The conversation could not reconnect.");
      pushToast(`reconnect: ${loomTree.error}`, { kind: "error" });
      channel.close();
      if (wsConn.channel === channel) {
        clearConnection();
      }
      throw error;
    } finally {
      if (wsConn.channel === channel) wsConn.opening = false;
    }
  })();
  wsConn.ready = ready;
  return ready.then(() => channel);
}

export function disconnectRuntimeChannel(): void {
  wsConn.flushTokens?.();
  wsConn.flushTokens = null;
  wsConn.unsubscribe?.();
  wsConn.unsubscribeState?.();
  wsConn.channel?.close();
  wsConn.channel = null;
  wsConn.unsubscribe = null;
  wsConn.unsubscribeState = null;
  wsConn.opening = false;
  wsConn.recoveringGap = false;
  wsConn.treeRecovery = null;
  wsConn.ready = null;
}

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", disconnectRuntimeChannel);
}

/** Resolve the assistant turn that's currently receiving streamed tokens.
 *
 * Two modes:
 *   - **Normal**: ``chatLog.pendingIndex`` points at the assistant turn the
 *     ``started`` event allocated; tokens append directly to it.
 *   - **A/B shadow**: ``abState.processingAb`` is true and
 *     ``abState.pendingTurnIdx`` points at the *steered* turn; tokens
 *     append to that turn's ``abPair`` (an inner ``ChatTurn`` initialized
 *     on the shadow's ``started`` event).
 *
 * Returning ``null`` means we don't have a write target — drop the token
 * silently rather than throwing, since a stray event during teardown is
 * harmless. */
function _currentWriteTurn(): ChatTurn | null {
  if (abState.processingAb && abState.pendingTurnIdx !== null) {
    const steered = chatLog.turns[abState.pendingTurnIdx];
    return steered?.abPair ?? null;
  }
  if (chatLog.pendingIndex !== null) {
    return chatLog.turns[chatLog.pendingIndex] ?? null;
  }
  return null;
}

/** Bind an in-flight token stream to a concrete loom assistant node.
 *
 * The tree mutation that creates the node is ordered before its first token;
 * this binds the stream to that already-authoritative node. */
function adoptStreamingNode(nodeId: string | null | undefined): void {
  if (!nodeId || abState.processingAb || !loomTree.loaded) return;
  if (loomTree.pendingNodeId === nodeId && loomTree.active_node_id === nodeId &&
    chatLog.pendingIndex !== null && chatLog.turns[chatLog.pendingIndex]?.nodeId === nodeId) return;
  loomTree.pendingNodeId = nodeId;
  if (!loomTree.nodes.has(nodeId)) {
    loomTree.error = "The conversation lost sync while the reply was arriving. Reload it and try again.";
    pushToast(loomTree.error, { kind: "error" });
    return;
  }
  loomTree.active_node_id = nodeId;
  recomputeActivePath();
  syncChatLogFromTree();
  const idx = chatLog.pendingIndex;
  if (idx !== null) {
    const turn = chatLog.turns[idx];
    if (turn) {
      turn.nodeId = nodeId;
      turn.tokens = turn.tokens ?? [];
      turn.thinkingTokens = turn.thinkingTokens ?? [];
    }
  }
}

// ------------------------------------------------- message dispatch --

/** Default WS message handler — owns the gen-status lifecycle and the
 * live token stream.  External subscribers (panels) layer additional
 * behavior via ``onWsMessage``. */
function handleWsMessage(msg: WSServerMessage, receivedAt = performance.now()): void | "tree_resync" {
  switch (msg.type) {
    case "tree_mutated": {
      // The roster is derived from every observed turn label, so any tree
      // mutation may change it.  The server inlines the effective snapshot.
      if (msg.cast) {
        castState.roster = msg.cast;
      }
      // Whole-tree restore can change parentage and sibling order even when
      // node ids overlap the old tree.  Ordinary add/update/remove deltas do
      // not carry enough information to detach an intersecting node from its
      // former parent, so every connected client crosses an authoritative
      // full-snapshot barrier for this operation.
      if (msg.op === "restore") {
        return "tree_resync";
      }
      // Apply the delta; on rev gap, full re-fetch.
      const ok = applyTreeDelta(msg);
      if (!ok) return "tree_resync";
      return;
    }
    case "started": {
      genStatus.active = true;
      genStatus.replay = null;
      genStatus.tokensSoFar = 0;
      genStatus.startedAt = performance.now();
      firstDecodeTokenAt = null;
      genStatus.finishedAt = null;
      genStatus.tokPerSec = 0;
      genStatus.ppl = { logSum: 0, count: 0, mean: null };
      genStatus.finishReason = null;
      liveTokenStream.responseTokens = [];
      liveTokenStream.thinkingTokens = [];
      // Manifold probes: drop the previous gen's trajectory + aggregate so
      // the inspector mini-map starts blank.  Sparkline carries across.
      if (!abState.processingAb) resetProbeStreams();
      // Loom: record the target node so tree-driven sync attaches the
      // streaming turn to the right active-path entry, and so the chat
      // panel's "streaming" highlight fires on the right turn.
      if (msg.node_id) {
        loomTree.pendingNodeId = msg.node_id;
        syncChatLogFromTree();
      }
      if (abState.processingAb && abState.pendingTurnIdx !== null) {
        // A/B shadow run: attach a fresh same-seat abPair to the generated
        // turn that just finished.  Don't append a new top-level turn —
        // the chat panel renders the abPair in its own column.
        const steered = chatLog.turns[abState.pendingTurnIdx];
        if (steered) {
          steered.abPair = {
            role: abState.pendingRole ?? steered.role,
            roleLabel: abState.pendingRoleLabel ?? steered.roleLabel,
            text: "",
            generated: true,
            tokens: [],
            thinkingTokens: [],
          };
        }
        // pendingIndex points at the steered turn so the streaming
        // pulse on Chat.svelte still highlights "this turn is live".
        chatLog.pendingIndex = abState.pendingTurnIdx;
      } else if (loomTree.loaded && msg.node_id) {
        // Loom path: sync an already-created node when the tree mutation
        // arrived first. When ``started`` arrives first, the pending node id
        // is retained and the following authoritative mutation creates it.
        syncChatLogFromTree();
        const pidx = chatLog.pendingIndex;
        if (pidx !== null) {
          const turn = chatLog.turns[pidx];
          if (turn) {
            turn.tokens = turn.tokens ?? [];
            turn.thinkingTokens = turn.thinkingTokens ?? [];
          }
        }
      } else if (loomTree.loaded) {
        // Loom path with a lazily-created assistant node: wait for
        // the authoritative tree mutation before allocating
        // the assistant turn. Appending a local placeholder here creates
        // a duplicate local assistant and is the source of many branch /
        // highlight misroutes.
        chatLog.pendingIndex = null;
        syncChatLogFromTree();
      } else {
        loomTree.error = "The conversation was not ready when generation started. Reload it and try again.";
        pushToast(loomTree.error, { kind: "error" });
      }
      return;
    }
    case "generation_progress": {
      adoptStreamingNode(msg.node_id);
      genStatus.replay = { completed: msg.completed, total: msg.total };
      return;
    }
    case "token": {
      adoptStreamingNode(msg.node_id);
      const writeTurn = _currentWriteTurn();
      const existingTokens = msg.thinking
        ? writeTurn?.thinkingTokens
        : writeTurn?.tokens;
      if (
        msg.raw_index != null &&
        msg.raw_index <= (existingTokens?.findLast((token) => token.rawIndex != null)?.rawIndex ?? -1)
      ) return;
      const isNewToken = msg.raw_index == null || msg.raw_index >= (genStatus.replay?.total ?? 0);
      if (isNewToken) genStatus.tokensSoFar += 1;
      if (
        typeof msg.perplexity === "number"
        && Number.isFinite(msg.perplexity)
        && msg.perplexity > 0
      ) {
        genStatus.ppl.logSum += Math.log(msg.perplexity);
        genStatus.ppl.count += 1;
        genStatus.ppl.mean = Math.exp(
          genStatus.ppl.logSum / genStatus.ppl.count,
        );
      }
      if (isNewToken) {
        const tokenReceivedAt = receivedAt;
        if (firstDecodeTokenAt === null) {
          firstDecodeTokenAt = tokenReceivedAt;
        } else {
          const elapsed = (tokenReceivedAt - firstDecodeTokenAt) / 1000;
          if (elapsed > 0) {
            genStatus.tokPerSec = (genStatus.tokensSoFar - 1) / elapsed;
          }
        }
      }
      // The 5.x measurement envelope is the single read-side record.  Probe
      // readings merge the three families (the rack keys by name); ``scores``
      // is the flat cross-family axis-0 view (highlight tint);
      // ``per_layer_scores`` the heatmap; the native lens/sae ``readout``
      // blocks feed the workspace/discovery cards.
      const m = msg.measurements;
      const stepReadings = mergedReadings(m);
      const scores = m?.scores;
      const lensSnapshot = lensReadoutSnapshot(m?.instruments.lens?.readout);
      const liveLensReadout = lensSnapshot?.readout;
      const liveLensAggregate = lensSnapshot?.aggregate;
      const liveSaeReadout = m?.instruments.sae?.readout?.features;
      const tokenScore: TokenScore = {
        text: msg.text,
        thinking: msg.thinking,
        tokenId: msg.token_id,
        perLayerScores: m?.per_layer_scores,
        // ``scores`` is the magnitude-weighted aggregate probe row. Using
        // it instead of a deepest-layer slice makes live highlighting match the
        // post-generation projected pass. Absent when no probes are loaded.
        probes: scores,
        // Logit-pass: pipe chosen-token logprob + top-K alternatives onto
        // the per-token row.  Both ride the WS ``token`` event directly
        // from Phase 1's engine capture; absent when ``return_top_k == 0``
        // and no other on-token consumer requested capture.
        logprob: msg.logprob ?? null,
        samplerEntropy: msg.sampler_entropy ?? null,
        perplexity: msg.perplexity ?? null,
        topAlts: msg.top_alts ?? null,
        // Raw decode-step index — the join key the logit fork slices
        // ``raw_token_ids`` on.  Rides the WS ``token`` event directly.
        rawIndex: msg.raw_index ?? null,
        measurements: m,
      };
      // Seed the single-probe ``score`` for the selected highlight so the
      // inline tint paints immediately as the token streams in.  The
      // canonical projected scores overwrite this on ``done``.
      if (scores && highlightState.target) {
        const s = scores[highlightState.target];
        if (typeof s === "number") tokenScore.score = s;
      }
      // Per-PC token highlighting: stash the full per-axis domain coords off
      // the merged family readings so axis targets (``personas[3]``) can tint
      // live.  Only multi-axis probes need it — axis 0 already rides
      // ``scores`` — and the row keeps it through ``done`` (the per-token
      // settle pass is axis-0 only and never clobbers this field).
      if (stepReadings) {
        const byProbe: Record<string, number[]> = {};
        for (const [pname, r] of Object.entries(stepReadings)) {
          const coords = (r as ProbeReadingJSON).coords;
          if (Array.isArray(coords) && coords.length > 1) byProbe[pname] = coords;
        }
        if (Object.keys(byProbe).length > 0) tokenScore.coordsByProbe = byProbe;
      }
      const turn = writeTurn;
      if (turn) {
        if (msg.thinking) {
          turn.thinking = true;
          (turn.thinkingTokens ??= []).push(tokenScore);
          // Live-stream buffer is steered-only — the shadow run doesn't
          // feed the main chat highlight pipeline.
          if (!abState.processingAb) {
            liveTokenStream.thinkingTokens.push(tokenScore);
          }
        } else {
          turn.text = (turn.text ?? "") + msg.text;
          (turn.tokens ??= []).push(tokenScore);
          if (!abState.processingAb) {
            liveTokenStream.responseTokens.push(tokenScore);
          }
        }
      }
      // Probe rack — unified per-token readings.  Every probe shape rides the
      // three families' ``readings`` (a 2-node concept axis is the rank-1
      // case), merged by name; the field is omitted when no probe is attached,
      // so the helper no-ops on undefined.  Skip shadow runs so the rack stays
      // anchored to the steered branch.  ``scores`` / ``per_layer_scores``
      // above still feed highlight tinting + the token-drilldown heatmap.
      if (!abState.processingAb) {
        updateProbesFromReadings(stepReadings);
        // J-LENS tab — the live all-layer readout. Present only while
        // the session's live lens is enabled; shadow runs skipped like
        // the probe rack so the matrix tracks the steered branch.
        if (liveLensReadout) lensState.readout = liveLensReadout;
        if (liveLensAggregate) {
          lensState.aggregate = liveLensAggregate;
          // Rolling strength history for the workspace-card sparklines —
          // one compact [token, strength] frame per step, probe-sparkline
          // cap, carries across generations like probe sparklines.
          const frame: [string, number][] = liveLensAggregate.map(
            ([tok, strength]) => [tok, strength],
          );
          lensState.aggHistory.push(frame);
          if (lensState.aggHistory.length > MAX_SPARKLINE) {
            lensState.aggHistory.shift();
          }
        }
        if (liveSaeReadout) {
          recordSaeReadoutFrame(liveSaeReadout);
        }
      }
      return;
    }
    case "done": {
      adoptStreamingNode(msg.node_id);
      genStatus.active = false;
      genStatus.finishedAt = performance.now();
      genStatus.finishReason = msg.result?.finish_reason ?? "stop";
      // Probe rack — end-of-gen aggregate (the settled ``ProbeReading`` per
      // probe: coords / fraction / nearest / residual + per-layer traces),
      // read out of the ``scope: "aggregate"`` measurement envelope and
      // merged across the three families exactly as the ``token`` path
      // merges them.  Same omitted-when-absent rule.
      if (!abState.processingAb) {
        setProbeAggregates(mergedReadings(msg.result?.measurements));
      }
      const turn = _currentWriteTurn();
      if (turn) {
        turn.finishReason = msg.result?.finish_reason ?? "stop";
        turn.tokensSoFar = msg.result?.tokens ?? genStatus.tokensSoFar;
        // Logit-pass: per-turn mean chosen-token logprob (response span
        // only).  Null when capture wasn't live; the inline surprise
        // mode + loom edge-weighting null-guard on this directly.
        turn.meanLogprob = msg.result?.mean_logprob ?? null;
        const turnPpl = geometricMeanPpl(genStatus);
        if (turnPpl !== null) turn.perplexity = turnPpl;
      }
      // Reconcile the live token counter against the server's
      // authoritative ``token_count``.  The streaming ``token`` events
      // may diverge from the engine's final count when (a) the WS dedupes
      // / batches partial UTF-8 tokens, or (b) the server's actual
      // ``max_new_tokens`` differs from the client's local view (e.g.
      // before the first PATCH lands).  Trust the server on close.
      if (typeof msg.result?.tokens === "number" && Number.isFinite(msg.result.tokens)) {
        genStatus.tokensSoFar = Math.max(0, msg.result.tokens - (genStatus.replay?.total ?? 0));
      }

      const wasShadow = abState.processingAb;
      const steeredIdx = chatLog.pendingIndex;
      chatLog.pendingIndex = null;
      // Loom: drop the pending node-id pointer; the server-emitted
      // ``tree_mutated`` (finalize) event has already merged the
      // finalised text + finish_reason into the node.
      if (loomTree.pendingNodeId) {
        loomTree.pendingNodeId = null;
        // Re-sync so the "streaming" decoration on the just-finished
        // turn switches off.
        if (loomTree.loaded) syncChatLogFromTree();
      }

      if (wasShadow) {
        // Shadow gen done — clear the A/B routing flags.  Do NOT touch
        // the probe baseline or correlation refresh; the steered turn
        // already did that when it finished.
        abState.processingAb = false;
        abState.pendingTurnIdx = null;
        abState.pendingRole = null;
        abState.pendingRoleLabel = null;
        // Drain pending actions queued during the shadow gen — same
        // gen-active gate the steered branch uses.
        void drainNextPendingAction();
        return;
      }

      // Snapshot probe baselines on the steered done event only.
      snapshotProbeBaseline();
      void refreshCorrelation();
      // SAE discovery backfill — fetch Neuronpedia metadata (label +
      // maxActApprox) for features the live top-k surfaced this
      // generation.  Between generations only, never per token.
      void backfillSaeMeta();

      // Automatic comparisons are stateless for every mode. Start only after
      // the transport has fully released this request; sending from inside
      // ``done`` races worker busy guards. Keep pending user actions behind
      // the comparison, then drain them when its own ``done`` arrives.
      const comparisonScheduled =
        autoRegenState.enabled &&
        steeredIdx !== null &&
        chatLog.turns[steeredIdx]?.generated === true &&
        scheduleAutoComparison(steeredIdx);
      if (!comparisonScheduled) void drainNextPendingAction();
      return;
    }
    case "error": {
      genStatus.active = false;
      genStatus.finishedAt = performance.now();
      adoptStreamingNode(msg.node_id);
      const wasShadow = abState.processingAb;
      const friendly = userFacingError(
        msg,
        "Generation stopped before the answer was complete. Try again or reopen the model.",
      );
      // Surface the error inline.  When the steered run errored we don't
      // want to spawn a shadow — clear A/B routing flags so a subsequent
      // successful gen behaves normally.  When the shadow itself errored
      // we still want the steered turn to remain visible as-is; just
      // mark its abPair as a placeholder error stub.
      if (wasShadow && abState.pendingTurnIdx !== null) {
        const steered = chatLog.turns[abState.pendingTurnIdx];
        if (steered) {
          steered.abPair = {
            role: "system",
            text: `Alternative generation stopped: ${friendly}`,
          };
        }
      } else {
        chatLog.turns = [
          ...chatLog.turns,
          { role: "system", text: `Drowse stopped: ${friendly}` },
        ];
      }
      chatLog.pendingIndex = null;
      if (loomTree.pendingNodeId) {
        loomTree.pendingNodeId = null;
        if (loomTree.loaded) syncChatLogFromTree();
      }
      // The system turn appended above is rebuilt away whenever
      // ``syncChatLogFromTree`` runs (the tree knows nothing of it), so a
      // server-owned log rendered generation errors as a silent empty
      // node.  A sticky toast survives every tree sync — errors must
      // never be silent.
      pushToast(`Generation: ${friendly}`, {
        kind: "error",
        ttlMs: null,
      });
      abState.processingAb = false;
      abState.pendingTurnIdx = null;
      abState.pendingRole = null;
      abState.pendingRoleLabel = null;
      // Drain the next pending action even on error so the UI doesn't
      // get stuck in "changes pending" forever.  The failed send
      // already surfaced as the system message above.
      void drainNextPendingAction();
      return;
    }
  }
}

// ------------------------------------------------- send primitives ---

export interface SendGenerateOpts {
  append_same_role?: boolean;
  stateless?: boolean;
  raw?: boolean;
  /** Cast model: which seat the generated turn occupies.  Absent /
   *  "assistant" = the classic flow; "user" needs scene mode server-side.
   *  Callers pass it explicitly (the composer reads its seat toggle) —
   *  the send primitive never defaults off ambient UI state. */
  generate_seat?: "user" | "assistant";
  /** Override the rack-derived steering with an explicit string.  Pass
   * ``""`` for unsteered (A/B mode); ``null``/``undefined`` to use the
   * rack. */
  steering?: string | null;
  /** Loom: attach the result as a child of this node.  ``null`` /
   *  absent = active node. */
  parent_node_id?: string | null;
  /** Loom: n-way regen.  Default 1. */
  n?: number;
  /** Loom phase 5: recipe-override modifier — mode string or partial
   *  recipe expression. */
  recipe_override?: string | null;
}

export interface SendSubmitOpts {
  /** Explicit anchor.  The queue-only sentinel resolves against the live
   * active node when this action reaches the head. */
  parent_node_id?: string | null | "active@drain";
  replaceSlot?: number | null;
  raw?: boolean;
  authored_thinking?: string | null;
  steering?: string | null;
  n?: number;
  recipe_override?: string | null;
}

function submitLabel(
  text: string | null,
  generatedRole: ChatRole | null,
): string {
  if (generatedRole === null) return "append";
  if (text === null) return "generate";
  return "send";
}

function buildSubmitPending(
  text: string | null,
  authoredRole: ChatRole | null,
  generatedRole: ChatRole | null,
  opts: Omit<SendSubmitOpts, "replaceSlot">,
): PendingAction {
  return {
    id: nextPendingId(),
    label: submitLabel(text, generatedRole),
    text,
    apply: () => sendSubmitNow(text, authoredRole, generatedRole, opts),
    awaitsGen: true,
    rebuild: text === null
      ? null
      : (newText: string) =>
          buildSubmitPending(newText, authoredRole, generatedRole, opts),
    createdAt: Date.now(),
    endsOnUserNode:
      (generatedRole ?? authoredRole) === "user"
        ? true
        : (generatedRole ?? authoredRole) === "assistant"
          ? false
          : null,
  };
}

/** Send one native role-neutral submission.
 *
 * Text is appended in ``authoredRole``. ``generatedRole`` optionally
 * follows it with a decode; omit it for append-only.  With no text, the
 * generated role continues directly from the selected leaf. No branch
 * depends on the selected node's role.
 */
export async function sendSubmit(
  text: string | null,
  authoredRole: ChatRole | null,
  generatedRole: ChatRole | null,
  opts: SendSubmitOpts = {},
): Promise<void> {
  if (text !== null && text === "") return;
  if (text !== null && authoredRole === null) {
    throw new Error("A text submission requires an authored role");
  }
  if (text === null && generatedRole === null) return;
  if (isPendingBusy()) {
    const { replaceSlot, ...queuedOpts } = opts;
    const item = buildSubmitPending(
      text,
      authoredRole,
      generatedRole,
      queuedOpts,
    );
    enqueuePending(
      {
        label: item.label,
        text: item.text,
        apply: item.apply,
        awaitsGen: item.awaitsGen,
        rebuild: item.rebuild,
        endsOnUserNode: item.endsOnUserNode,
      },
      { replaceSlot: replaceSlot ?? null },
    );
    return;
  }
  return sendSubmitNow(text, authoredRole, generatedRole, opts);
}

async function sendSubmitNow(
  text: string | null,
  authoredRole: ChatRole | null,
  generatedRole: ChatRole | null,
  opts: Omit<SendSubmitOpts, "replaceSlot"> = {},
): Promise<void> {
  if (!loomTree.loaded) {
    await refreshLoomTree();
    if (!loomTree.loaded) {
      throw new Error("Conversation tree is not ready; retry after it loads");
    }
  }
  const channel = await ensureRuntimeChannel();
  const steering =
    opts.steering === undefined ? currentSteeringExpression() : opts.steering;
  const sampling = buildSamplingPayload();
  genStatus.maxTokens = sampling?.max_tokens ?? samplingState.max_tokens;
  const requestedParent = opts.parent_node_id;
  const parent = requestedParent === "active@drain"
    ? loomTree.active_node_id
    : requestedParent;
  const payload: WSClientMessage = {
    type: "submit",
    text,
    authored_role: authoredRole,
    generated_role: generatedRole,
    steering: steering || null,
    sampling,
    thinking: samplingState.thinking ?? false,
    raw: opts.raw ?? false,
    ...(opts.authored_thinking
      ? { authored_thinking: opts.authored_thinking }
      : {}),
    ...(parent !== undefined ? { parent_node_id: parent } : {}),
    ...(opts.n !== undefined ? { n: opts.n } : {}),
    ...(opts.recipe_override !== undefined
      ? { recipe_override: opts.recipe_override }
      : {}),
  };
  channel.send(payload);
}

/** Send a bare-continuation generate request over the WS — the
 * regenerate / continue-from-committed path.  No authored text travels:
 * the model speaks next from ``opts.parent_node_id`` (or the active
 * leaf) in ``opts.generate_seat``.  Builds the steering expression from
 * the rack live, layers the SamplingConfig overrides when one-shot mode
 * is on, and routes everything through the singleton connection.
 *
 * Fires immediately even mid-generation: these are internal
 * store-to-store calls behind an explicit user gesture on a specific
 * node, not composer sends, so they carry no pending-queue slot. */
export async function sendGenerate(
  opts: SendGenerateOpts = {},
): Promise<void> {
  // The first server snapshot may legitimately be revision 0.  Require the
  // explicit readiness bit instead of guessing from the revision, and retain
  // this defensive fetch even though App gates user interaction during boot:
  // store-level callers and future surfaces should be safe on their own.
  if (!loomTree.loaded) {
    await refreshLoomTree();
    if (!loomTree.loaded) {
      throw new Error("Conversation tree is not ready; retry after it loads");
    }
  }
  const channel = await ensureRuntimeChannel();
  const steering =
    opts.steering === undefined ? currentSteeringExpression() : opts.steering;
  const steeringPayload =
    opts.steering === undefined ? (steering || null) : steering;
  // Build the sampling payload — seed + the advanced extras (penalties,
  // stop, logit-bias, return_top_k).  temperature / top-p / top-k /
  // max-tokens are PATCHed to the session as the user edits them, so the
  // server reads its own (now-updated) defaults for those.
  const sampling = buildSamplingPayload();
  // Update genStatus.maxTokens locally so the progress bar widths know
  // their target before the first token lands.
  genStatus.maxTokens = sampling?.max_tokens ?? samplingState.max_tokens;
  const payload: WSClientMessage = {
    type: "generate",
    ...(opts.append_same_role !== undefined ? { append_same_role: opts.append_same_role } : {}),
    // A continue: no committed turn, the model speaks next from the
    // anchor node.
    input: null,
    steering: steeringPayload,
    sampling,
    // Coerce the current family-level automatic setting to explicit ``false`` so the
    // unchecked checkbox really means "no thinking" — the server's
    // chat-template templates treat ``null`` and ``False`` differently
    // on some families and we promised the user a binary toggle.
    thinking: samplingState.thinking ?? false,
    stateless: opts.stateless ?? false,
    raw: opts.raw ?? false,
    // Loom fields ride only when caller explicitly set them (server
    // ignores unknown fields, but the spec keeps them optional).
    ...(opts.parent_node_id !== undefined
      ? { parent_node_id: opts.parent_node_id }
      : {}),
    ...(opts.n !== undefined ? { n: opts.n } : {}),
    ...(opts.recipe_override !== undefined
      ? { recipe_override: opts.recipe_override }
      : {}),
    ...(opts.generate_seat !== undefined && opts.generate_seat !== "assistant"
      ? { generate_seat: opts.generate_seat }
      : {}),
  };
  channel.send(payload);
}

/** Logit fork — regenerate an existing assistant node as a sibling with
 *  one token swapped.  The server reuses the source node's stamped
 *  recipe (steering / sampling / seed / thinking) and replays its raw
 *  decode sequence up to ``rawIndex``, forcing ``altTokenId`` there
 *  before sampling the continuation. ``resample`` gives Continue a fresh
 *  seed; replacements keep the original seed. Streams in like any regen: the
 *  new sibling lands via the WS ``tree_mutated`` / ``token`` / ``done``
 *  events and becomes the active branch. */
export async function sendFork(
  nodeId: string,
  rawIndex: number,
  altTokenId: number,
  resample = false,
): Promise<void> {
  const channel = await ensureRuntimeChannel();
  const payload: WSClientMessage = {
    type: "generate",
    fork_node_id: nodeId,
    fork_raw_index: rawIndex,
    fork_alt_token_id: altTokenId,
    ...(resample ? { fork_seed: crypto.getRandomValues(new Uint32Array(1))[0]! & 0x7fffffff } : {}),
  };
  channel.send(payload);
}

/** Replace the selected raw token with arbitrary authored text, then sample
 *  the rest of the sibling branch with the source node's saved recipe. */
export async function sendTextFork(
  nodeId: string,
  rawIndex: number,
  replacementText: string,
): Promise<void> {
  const channel = await ensureRuntimeChannel();
  const payload: WSClientMessage = {
    type: "generate",
    fork_node_id: nodeId,
    fork_raw_index: rawIndex,
    fork_replacement_text: replacementText,
  };
  channel.send(payload);
}

export function sendStop(): void {
  wsConn.flushTokens?.();
  const channel = wsConn.channel;
  if (!channel) return;
  void channel.stop().catch((error) => {
    wsConn.flushTokens?.();
    handleWsMessage({
      type: "error",
      code: "RUNTIME_STOP_FAILED",
      message: userFacingError(error, "The model could not be stopped cleanly."),
    });
  });
}
