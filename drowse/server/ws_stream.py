"""Native WebSocket token+probe co-stream (``WS /drowse/v1/sessions/{id}/stream``).

Bidirectional token + per-token probe co-stream — the killer feature of
the native tree API.  The connection runs a single perpetual reader, a
loom-mutation forwarder, and a per-generate-turn worker; see the
``_ws_handle_generate`` docstring for the concurrency design.
"""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar, cast

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from anyio import CancelScope

from drowse.core.errors import DrowseError
from drowse.core.loom import LoomMutated
from drowse.core.results import GenerationResult, TokenAlt
from drowse.core.sampling import SamplingConfig
from drowse.core.session import DrowseSession
from drowse.core.token_callback import TokenConsumer, TokenConsumerOptions
from drowse.core.steering import Steering
from drowse.server.app import acquire_session_lock, ws_auth_ok
from drowse.server.native_common import SINGLE_SESSION_ID
from drowse.server.request_helpers import merge_steering, parse_request_steering
from drowse.server.streaming import finish_worker
from drowse.server.tree_models import cast_json, node_json
from drowse.server.ws_events import build_token_event
from drowse.server.ws_models import (
    WSGenerateMessage,
    WSSubmitMessage,
    build_input,
    build_sampling_config,
    result_to_json,
)

_logger = logging.getLogger(__name__)

MAX_WS_MESSAGE_BYTES = 1024 * 1024
MAX_WS_PENDING_MESSAGES = 16
MAX_WS_PENDING_BYTES = 4 * MAX_WS_MESSAGE_BYTES
MAX_WS_OUTBOUND_EVENTS = 256
WS_SEND_TIMEOUT_SECONDS = 30

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject = dict[str, JSONValue]


@dataclass(frozen=True)
class _Stop:
    pass


@dataclass(frozen=True)
class _Disconnect:
    code: int = 1000
    reason: str = ""


@dataclass(frozen=True)
class _InvalidInbound:
    message: str


_Inbound = (
    WSGenerateMessage | WSSubmitMessage | _Stop | _Disconnect
    | _InvalidInbound
)


class _InboundBuffer:
    """One bounded FIFO, with coalesced controls that never consume its budget."""

    def __init__(self) -> None:
        self.pending: deque[tuple[_Inbound, int]] = deque()
        self.pending_bytes = 0
        self.stop = False
        self.terminal: _Disconnect | None = None
        self.changed = asyncio.Event()
        self.closed = asyncio.Event()

    def put(self, message: _Inbound, size: int = 0) -> bool:
        if self.terminal is not None:
            return False
        if isinstance(message, _Stop):
            self.stop = True
        else:
            if len(self.pending) >= MAX_WS_PENDING_MESSAGES or self.pending_bytes + size > MAX_WS_PENDING_BYTES:
                return False
            self.pending.append((message, size))
            self.pending_bytes += size
        self.changed.set()
        return True

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self.terminal is None:
            self.terminal = _Disconnect(code, reason)
        self.pending.clear()
        self.pending_bytes = 0
        self.closed.set()
        self.changed.set()

    async def get(self, *, control_only: bool = False) -> _Inbound:
        while True:
            if self.terminal is not None:
                return self.terminal
            if self.stop and (control_only or not self.pending):
                self.stop = False
                return _Stop()
            if self.pending and not control_only:
                message, size = self.pending.popleft()
                self.pending_bytes -= size
                return message
            self.changed.clear()
            await self.changed.wait()


_T = TypeVar("_T")


class _OutboundQueue(Generic[_T]):
    """Bound both queued events and callbacks awaiting the event loop."""

    def __init__(self, overflow: Callable[[], None]) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[_T] = asyncio.Queue()
        self._slots = threading.BoundedSemaphore(MAX_WS_OUTBOUND_EVENTS)
        self._overflow = overflow
        self._closed = threading.Event()

    def put(self, item: _T) -> None:
        if self._closed.is_set():
            return
        if not self._slots.acquire(blocking=False):
            self._closed.set()
            self._loop.call_soon_threadsafe(self._overflow)
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self) -> _T:
        item = await self._queue.get()
        self._slots.release()
        return item

    def close(self) -> None:
        self._closed.set()


@dataclass(frozen=True)
class _TokenFrame:
    payload: JSONObject


@dataclass(frozen=True)
class _TokenDone:
    pass


_TokenQueueItem = _TokenFrame | _TokenDone


def _error_frame(
    message: str,
    *,
    code: str = "ValueError",
    status: int = 400,
    node_id: str | None = None,
    sibling_index: int = 0,
) -> JSONObject:
    """Build the one ``{type: "error", …}`` frame shape this stream emits.

    Every error frame carries the same five keys, so a client never has to
    branch on which of them happens to be present.  ``node_id`` is filled only
    once a generation has produced its assistant node.
    """
    return {
        "type": "error",
        "message": message,
        "code": code,
        "status": status,
        "node_id": node_id,
        "sibling_index": sibling_index,
    }


def _validation_message(exc: ValidationError) -> str:
    """Render a pydantic rejection as one line, the native tree's convention.

    Each error reads ``"<field path>: <message>"``, the shape
    ``app._on_validation_error`` gives native REST bodies; a model-level rule
    has no path, so its message stands alone (``"a generate message cannot be
    both a fork and a prefill"``) — ``PydanticCustomError`` in ``ws_models``
    is what keeps that verbatim rather than ``"Value error, …"``.

    Every error is reported, not just the first: ``input`` is a union, so its
    real failure (a bad key inside a message object) is the *second* branch
    error and a first-only render would name the wrong problem.
    """
    parts: list[str] = []
    for error in exc.errors():
        message = str(error.get("msg", "invalid request"))
        loc = ".".join(str(part) for part in error.get("loc", ()))
        parts.append(f"{loc}: {message}" if loc else message)
    return "; ".join(parts) or str(exc)


class _WSRequestError(Exception):
    """A request-shape problem to report as an error frame, not a close.

    Raised by the pre-dispatch validators so the field-consistency rules read
    as straight-line checks instead of a ``send_json``/``return`` pair each.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "ValueError",
        status: int = 400,
        node_id: str | None = None,
        sibling_index: int = 0,
    ) -> None:
        super().__init__(message)
        self.frame = _error_frame(
            message, code=code, status=status,
            node_id=node_id, sibling_index=sibling_index,
        )


@dataclass(frozen=True)
class _AuthoredTurn:
    """The turn a ``submit`` commits before its generation runs."""

    text: str
    role: Literal["user", "assistant"]
    thinking: str | None


def register_ws_stream(app: FastAPI) -> None:
    """Mount the bidirectional WebSocket token+probe co-stream."""
    session = app.state.session

    @app.websocket("/drowse/v1/sessions/{session_id}/stream")
    async def session_stream(websocket: WebSocket, session_id: str):
        if not ws_auth_ok(websocket):
            await websocket.close(code=1008, reason="unauthorized")
            return
        protocol = "drowse.v1" if "drowse.v1" in websocket.scope.get("subprotocols", []) else None
        if session_id != SINGLE_SESSION_ID:
            await websocket.accept(subprotocol=protocol)
            await websocket.close(code=1008, reason="session not found")
            return
        await websocket.accept(subprotocol=protocol)

        # Single perpetual reader.  ``websocket.receive_json()`` is bound
        # to a per-connection ``recv_in_progress`` flag in the underlying
        # ``websockets`` library; cancelling a pending receive doesn't
        # clear the flag immediately, so any handler that called
        # ``receive_json()`` while another concurrent (even just-cancelled)
        # caller was pending tripped a "cannot call recv while another
        # coroutine is already waiting" RuntimeError.  Routing every
        # incoming frame through one queue lets both the outer dispatch
        # loop and the in-flight generation share the read side without
        # ever overlapping calls into the WS.
        incoming = _InboundBuffer()

        async def _reader():
            try:
                while True:
                    frame = await websocket.receive_text()
                    size = len(frame.encode("utf-8"))
                    if size > MAX_WS_MESSAGE_BYTES:
                        incoming.close(1009, "message too large")
                        return
                    message: _Inbound
                    try:
                        raw = json.loads(frame)
                        if not isinstance(raw, dict):
                            message = _InvalidInbound("message must be an object")
                        elif raw.get("type") in ("generate", "submit"):
                            message = (
                                WSSubmitMessage(**raw)
                                if raw.get("type") == "submit"
                                else WSGenerateMessage(**raw)
                            )
                        elif raw.get("type") == "stop":
                            message = _Stop()
                        else:
                            message = _InvalidInbound("unknown message type")
                    except ValidationError as exc:
                        message = _InvalidInbound(_validation_message(exc))
                    except (ValueError, RecursionError):
                        message = _InvalidInbound("invalid JSON message")
                    if not incoming.put(message, size):
                        incoming.close(1013, "too many pending messages")
                        return
            except WebSocketDisconnect:
                incoming.close()
            except Exception:
                _logger.exception("native WebSocket reader failed")
                incoming.close(1011, "request failed")

        reader_task = asyncio.create_task(_reader())

        # Loom: subscribe to ``LoomMutated`` for the connection's
        # lifetime and forward exact ``tree_mutated`` frames. Held
        # in a queue + forwarder task so the EventBus callback (which
        # runs on the gen thread) never touches the WS directly.
        tree_event_queue = _OutboundQueue[JSONObject](
            lambda: incoming.close(1013, "client is not consuming events"),
        )
        # ``websocket.send_json`` is not safe for concurrent callers —
        # starlette serializes per-call but two tasks can interleave
        # bytes on the wire and corrupt the frame sequence.  This lock
        # is the single send-side serializer the connection uses; both
        # the generate-handler and the tree-forwarder acquire it before
        # every send.
        ws_send_lock = asyncio.Lock()

        async def _send_json(payload: JSONObject) -> None:
            async with asyncio.timeout(WS_SEND_TIMEOUT_SECONDS):
                async with ws_send_lock:
                    await websocket.send_json(payload)

        def _queue_tree_event(payload: JSONObject) -> None:
            tree_event_queue.put(payload)

        def _on_loom_event(event: object) -> None:
            if not isinstance(event, LoomMutated):
                return
            added_nodes = [node_json(session, nid) for nid in event.added]
            mutated_payload = cast(JSONObject, {
                "type": "tree_mutated",
                "op": event.op,
                "rev": event.rev,
                "added": added_nodes,
                "removed": list(event.removed),
                "updated": [
                    node_json(session, nid)
                    for nid in event.updated
                    if session.tree.has(nid)
                ],
                "active_node_id": event.active_node_id,
            })
            # The roster is derived from the full tree's observed labels as
            # well as explicit configuration.  Inline the small effective
            # roster on every mutation so adds, deletes, and restores reconcile
            # identity without a refetch or provenance inference client-side.
            mutated_payload["cast"] = cast(JSONValue, cast_json(session))
            _queue_tree_event(mutated_payload)
        loom_unsub = session.events.subscribe(_on_loom_event)

        async def _tree_forwarder():
            """Forward tree-mutated events as WS frames.

            Runs as a dedicated task for the connection's lifetime so
            tree mutations from any source (this WS, a REST route on a
            different connection, the gen loop) reach the client without
            interleaving with the per-turn token loop.
            """
            try:
                while True:
                    payload = await tree_event_queue.get()
                    try:
                        await _send_json(payload)
                    except Exception:
                        incoming.close(1013, "client is not consuming events")
                        return
                    if payload.get("op") == "finalize_assistant":
                        updated = payload.get("updated")
                        if isinstance(updated, list):
                            for node in updated:
                                if not isinstance(node, dict):
                                    continue
                                node_id = node.get("id")
                                if isinstance(node_id, str):
                                    tree_forwarded_finalized.append(node_id)
                    tree_forwarded_event.set()
            except asyncio.CancelledError:
                return

        tree_forwarded_finalized: deque[str] = deque(maxlen=MAX_WS_PENDING_MESSAGES)
        tree_forwarded_event = asyncio.Event()
        forwarder_task = asyncio.create_task(_tree_forwarder())

        async def _wait_for_tree_finalization(node_id: str) -> None:
            """Do not let ``done`` overtake its authoritative tree delta.

            The mutation forwarder and generation dispatcher are separate
            tasks sharing a send lock.  Serialization alone does not impose
            ordering: the dispatcher could acquire the lock first even though
            the worker had already queued its ``finalize`` event, briefly
            leaving clients with a completed but empty assistant node.
            """
            while node_id not in tree_forwarded_finalized:
                tree_forwarded_event.clear()
                if node_id in tree_forwarded_finalized:
                    return
                event_wait = asyncio.create_task(tree_forwarded_event.wait())
                finished, pending = await asyncio.wait(
                    {event_wait, forwarder_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    if task is not forwarder_task:
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                if (
                    forwarder_task in finished
                    and node_id not in tree_forwarded_finalized
                ):
                    raise RuntimeError(
                        "tree mutation stream ended before generation finalization"
                    )

        closed_task = asyncio.create_task(incoming.closed.wait())
        request_task: asyncio.Task[None] | None = None
        try:
            while True:
                msg = await incoming.get()
                if isinstance(msg, _Disconnect):
                    with suppress(Exception):
                        await websocket.close(code=msg.code, reason=msg.reason)
                    return
                if isinstance(msg, (WSGenerateMessage, WSSubmitMessage)):
                    request_task = asyncio.create_task(_ws_handle_generate(
                        session, msg, app.state.default_steering, incoming,
                        _send_json, _wait_for_tree_finalization,
                    ))
                    finished, _ = await asyncio.wait(
                        {request_task, closed_task}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    if closed_task in finished:
                        request_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await finish_worker(request_task)
                    else:
                        await request_task
                    request_task = None
                elif isinstance(msg, _Stop):
                    continue
                else:
                    await _send_json(_error_frame(
                        msg.message, code="ValidationError", status=400,
                    ))
        except WebSocketDisconnect:
            return
        except Exception as e:
            _logger.exception("native WebSocket request failed")
            status, message = e.user_message() if isinstance(e, DrowseError) else (
                500, "Request failed. Check the server log for details.",
            )
            try:
                await _send_json(_error_frame(message, code=type(e).__name__, status=status))
            finally:
                with suppress(Exception):
                    await websocket.close(code=1011)
        finally:
            loom_unsub()
            tree_event_queue.close()
            with CancelScope(shield=True):
                if request_task is not None:
                    request_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await finish_worker(request_task)
                for task in (forwarder_task, reader_task, closed_task):
                    task.cancel()
                await asyncio.gather(forwarder_task, reader_task, closed_task, return_exceptions=True)


def _normalize_submit(
    submit: WSSubmitMessage,
) -> tuple[WSGenerateMessage, _AuthoredTurn | None, bool]:
    """Lower a ``submit`` frame onto the specialist ``generate`` schema.

    ``submit`` is the ONE authored-turn contract on this wire: an explicit
    authored role plus an optional generated role.  Append-only submissions
    take the no-decode commit path; append+generate keeps the authored turn
    for one atomic commit inside the generation worker and generates from
    ``input=None`` once that turn has landed.

    Returns ``(generate_message, authored_turn, commit_only)``.  When
    ``commit_only`` the authored turn is the whole request and the returned
    generate message carries only the commit's context (parent, sampling
    labels, raw); otherwise the authored turn — if any — is committed inside
    the generation worker.  Raises :class:`_WSRequestError` on any
    field-consistency violation.
    """
    if submit.text is None:
        if submit.authored_role is not None:
            raise _WSRequestError("authored_role requires non-empty text")
        if submit.authored_thinking is not None:
            raise _WSRequestError("authored_thinking requires non-empty text")
        authored: _AuthoredTurn | None = None
    else:
        if submit.text == "":
            raise _WSRequestError("submit text must be non-empty when present")
        if submit.authored_role is None:
            raise _WSRequestError("text requires authored_role")
        authored = _AuthoredTurn(
            text=submit.text,
            role=submit.authored_role,
            thinking=submit.authored_thinking,
        )
    if submit.generated_role is None and authored is None:
        raise _WSRequestError("submit requires text or generated_role")

    if submit.generated_role is None:
        assert authored is not None  # guarded by the check above
        return (
            WSGenerateMessage(
                type="generate",
                parent_node_id=submit.parent_node_id,
                sampling=submit.sampling,
                raw=submit.raw,
            ),
            authored,
            True,
        )
    return (
        WSGenerateMessage(
            type="generate",
            input=None,
            steering=submit.steering,
            sampling=submit.sampling,
            thinking=submit.thinking,
            stateless=False,
            raw=submit.raw,
            parent_node_id=submit.parent_node_id,
            n=submit.n,
            recipe_override=submit.recipe_override,
            generate_seat=submit.generated_role,
        ),
        authored,
        False,
    )


async def _ws_handle_commit(
    session: DrowseSession,
    msg: WSGenerateMessage,
    authored: _AuthoredTurn,
    send_json: Callable[[JSONObject], Awaitable[None]],
) -> None:
    """Land one authored turn with no decode.

    Short-circuits the n-way fan-out and the streaming worker entirely: one
    tree mutation, one ``started`` (``node_id`` null — the node id is only
    known after the append), one ``done`` carrying the new node.  No token
    frames.  Seating is free: ``session.append_turn`` is the role-neutral
    primitive, so an assistant-authored opening turn needs no parent.
    """
    parent_node_id = msg.parent_node_id
    generation_id = uuid.uuid4().hex[:12]
    await send_json({
        "type": "started",
        "generation_id": generation_id,
        "node_id": None,
        "sibling_index": 0,
        "sibling_count": 1,
    })
    # Per-message role labels ride the commit's sampling block too
    # (roleplay scaffold).  Raw / flat commits carry no chat-template
    # role, so labels are suppressed there.
    commit_label = None
    if msg.sampling is not None:
        commit_label = (
            msg.sampling.user_role
            if authored.role == "user"
            else msg.sampling.assistant_role
        ) or None
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            await send_json(_error_frame(
                "session locked — try again when the current generation finishes",
                code="SessionLocked", status=503,
            ))
            return
        try:
            worker_task = asyncio.create_task(asyncio.to_thread(
                session.append_turn,
                parent_node_id,
                authored.text,
                role=authored.role,
                raw=msg.raw,
                role_label=None if msg.raw else commit_label,
                thinking=None if msg.raw else (authored.thinking or None),
            ))
            try:
                new_id = await asyncio.shield(worker_task)
            finally:
                await finish_worker(worker_task)
        except DrowseError as e:
            status, message = e.user_message()
            await send_json(_error_frame(
                message, code=type(e).__name__, status=status,
            ))
            return
    await send_json({
        "type": "done",
        "result": {
            "role": authored.role,
            "text": authored.text,
            "node_id": new_id,
            "finish_reason": "stop",
            "mean_logprob": None,
            "mean_surprise": None,
        },
        "node_id": new_id,
        "sibling_index": 0,
        "sibling_count": 1,
    })


async def _ws_handle_generate(
    session: DrowseSession,
    msg: WSGenerateMessage | WSSubmitMessage,
    default_steering: "Steering | None",
    incoming: _InboundBuffer,
    send_json: Callable[[JSONObject], Awaitable[None]],
    wait_for_tree_finalization: Callable[[str], Awaitable[None]],
) -> None:
    """Validate one inbound turn and dispatch it to its mode handler.

    ``generate``'s own field-consistency rules already ran in the schema
    (``WSGenerateMessage`` model validators, rejected by the reader), so the
    only checks left here are ``submit``'s cross-field rules, which have no
    schema home: they decide *which* generate frame to lower onto
    (:func:`_normalize_submit`).  Then the steering expression is composed
    over the server default and the turn goes to exactly one of two handlers:
    :func:`_ws_handle_commit` (no decode) or :func:`_ws_stream_generation`
    (the token fan-out).  Every rejection here is an error frame on a
    connection that stays open — a 400-grade user mistake must not close the
    socket.
    """
    authored: _AuthoredTurn | None = None
    commit_only = False
    if isinstance(msg, WSSubmitMessage):
        try:
            msg, authored, commit_only = _normalize_submit(msg)
        except _WSRequestError as e:
            await send_json(e.frame)
            return

    if commit_only:
        assert authored is not None  # set together by ``_normalize_submit``
        await _ws_handle_commit(session, msg, authored, send_json)
        return

    sampling = build_sampling_config(msg.sampling)
    try:
        req_steering, explicit_clear = parse_request_steering(msg.steering)
        thinking_override: bool | None = None
        if req_steering is not None and req_steering.thinking is not None:
            thinking_override = req_steering.thinking
        steering = merge_steering(
            req_steering, default_steering, explicit_clear, thinking_override,
        )
    except DrowseError as e:
        # ``parse_request_steering`` -> ``parse_expr`` -> ``resolve_bare_atom`` can
        # raise ``SteeringExprError`` / ``AmbiguousSelectorError`` /
        # ``AmbiguousVariantError`` on malformed or colliding input.
        # FastAPI's ``@app.exception_handler(DrowseError)`` doesn't apply
        # to WebSocket routes, so without this guard the exception falls
        # through to the outer reader loop's ``except Exception`` which
        # closes the socket with code 1011. A 400-grade user mistake
        # shouldn't kill the connection — send the error frame and let
        # the client try again on the same WS.
        status, message = e.user_message()
        await send_json(_error_frame(
            message, code=type(e).__name__, status=status,
        ))
        return

    await _ws_stream_generation(
        session, msg, steering, sampling, authored,
        incoming, send_json, wait_for_tree_finalization,
    )


async def _ws_stream_generation(
    session: DrowseSession,
    msg: WSGenerateMessage,
    steering: "Steering | None",
    sampling: SamplingConfig | None,
    authored: _AuthoredTurn | None,
    incoming: _InboundBuffer,
    send_json: Callable[[JSONObject], Awaitable[None]],
    wait_for_tree_finalization: Callable[[str], Awaitable[None]],
) -> None:
    """Run one generate turn and stream token/done/error events.

    Concurrency design: synchronous generation runs in a worker thread
    via ``asyncio.to_thread`` under its own cancellation scope. Its
    ``on_token`` callback bridges into asyncio through a bounded queue.
    The main
    coroutine races two tasks: one pulls ``TokenEvent``s from a local
    queue and forwards them as ``{type: "token", ...}`` frames; the
    other pulls controls from the shared ``incoming`` buffer
    (populated by the connection's single reader task) so an in-flight
    ``{type: "stop"}`` can signal this worker without blocking on
    the token loop.

    ``asyncio.wait(..., FIRST_COMPLETED)`` is used in a loop: whenever
    the incoming task returns a stop frame we signal the session and
    keep draining tokens until the worker joins; whenever the queue
    delivers a sentinel we finish.  The WS stays open across generate
    turns — a client can submit ``{type: "generate", ...}`` again after
    ``done``, and the perpetual reader keeps feeding the shared queue
    between turns so we never have two ``receive_json()`` calls in
    flight.

    **Loom**: ``parent_node_id`` attaches the assistant node
    under a specific tree node; ``n>1`` fans out N siblings serially.
    Each sibling produces its own ``started`` / token-stream / ``done``
    triplet, all tagged with the assistant node id.  ``tree_mutated``
    events ride the connection-level subscription
    in ``session_stream``; this handler only emits the per-sibling
    ``started`` / ``token`` / ``done`` frames.  ``authored`` (from a
    ``submit`` that also generates) is committed once inside the worker,
    under the same session lock as the whole fan.
    """
    n = msg.n
    parent_node_id = msg.parent_node_id
    submitted_parent_holder: list[str] = []

    # Per-sibling seed schedule: when n>1, derive deterministic per-
    # sibling seeds from the request seed (or fresh entropy).  Single
    # streams (n=1) use the user's seed verbatim.
    from drowse.core.loom import derive_seed_schedule
    base_seed = sampling.seed if sampling is not None else None
    seeds: list[int | None]
    seeds = [base_seed] if n == 1 else list(derive_seed_schedule(base_seed, n))

    # Acquire the session lock for the full N-way batch lifetime so
    # concurrent WS clients serialize FIFO instead of overlapping.
    # The session engine itself uses the threading ``_gen_lock``
    # to gate the actual generation, but the async-level lock is what
    # queues HTTP/WS endpoints fairly.  Bounded to SESSION_LOCK_TIMEOUT_SECONDS
    # (300 s) so a long-running generation doesn't pin the lock forever;
    # a timeout surfaces as a WS error frame (no HTTP in the WS context).
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            await send_json(_error_frame(
                "session locked — try again when the current generation finishes",
                code="SessionLocked", status=503,
            ))
            return
        for sibling_idx, seed_i in enumerate(seeds):
            generation_id = uuid.uuid4().hex[:12]
            tree_rev_before_generation = int(session.tree.rev)

            # Per-sibling sampling override carrying the derived seed.
            if n == 1 and seed_i is None:
                per_sibling_sampling = sampling
            else:
                from dataclasses import replace as _dc_replace
                base_sc = sampling if sampling is not None else SamplingConfig()
                per_sibling_sampling = _dc_replace(base_sc, seed=seed_i)

            token_queue = _OutboundQueue[_TokenQueueItem](
                lambda: incoming.close(1013, "client is not consuming events"),
            )
            cancelled = threading.Event()
            # The tree assigns the assistant node id at ``begin_assistant``
            # time inside ``_generate_core``; we don't know it before the
            # gen starts.  The on_token callback reads the live active
            # node off the tree (which is set to the streaming assistant
            # node for the lifetime of the gen).
            current_node_holder: list[str | None] = [None]

            def _on_token(
                text: str,
                is_thinking: bool,
                tid: int | None,
                lp: float | None,
                top_alts: list[TokenAlt] | None,
                perplexity: float | None = None,
                _node_holder: list[str | None] = current_node_holder,
                _token_queue: _OutboundQueue[_TokenQueueItem] = token_queue,
            ) -> None:
                event = build_token_event(
                    session,
                    _node_holder,
                    text=text,
                    is_thinking=is_thinking,
                    tid=tid,
                    lp=lp,
                    top_alts=top_alts,
                    perplexity=perplexity,
                )
                _token_queue.put(_TokenFrame(cast(JSONObject, event)))
            consumer = TokenConsumer(
                _on_token,
                TokenConsumerOptions(
                    live_scores=True,
                    per_layer_scores=True,
                    lens_readout=True,
                    sae_readout=True,
                    perplexity=True,
                ),
            )
            # Live J-lens workspace readout: computed only when the session's
            # live lens is enabled (POST .../lens/live) AND the tap consumer
            # declares interest through ``TokenConsumerOptions`` (the same
            # typed capability object used by ``generate_stream``), so an enabled lens
            # streams per-step top-k on the ``token`` frame's ``lens_readout``.

            result_holder: list[GenerationResult] = []
            error_holder: list[BaseException] = []

            # Recipe-override (phase 5): accept either a mode string or a
            # partial-recipe expression.  We pass it through ``generate``
            # so the engine resolves the overlay against the parent's
            # recipe; ``session.regen_with_modifier`` is the matching
            # higher-level wrapper but the WS path already has the
            # required context.
            recipe_override = msg.recipe_override

            def _worker(
                _sampling: SamplingConfig | None = per_sibling_sampling,
                _on_token: TokenConsumer = consumer,
                _result_holder: list[GenerationResult] = result_holder,
                _error_holder: list[BaseException] = error_holder,
                _token_queue: _OutboundQueue[_TokenQueueItem] = token_queue,
                _recipe_override: str | None = recipe_override,
            ) -> None:
                try:
                    with session.generation_state.cancellation_scope(cancelled):
                        effective_parent = parent_node_id
                        if authored is not None:
                            if not submitted_parent_holder:
                                commit_label = None
                                if not msg.raw and msg.sampling is not None:
                                    commit_label = (
                                        msg.sampling.user_role
                                        if authored.role == "user"
                                        else msg.sampling.assistant_role
                                    ) or None
                                committed_id = session.append_turn(
                                    parent_node_id,
                                    authored.text,
                                    role=authored.role,
                                    raw=msg.raw,
                                    role_label=commit_label,
                                    thinking=authored.thinking,
                                )
                                submitted_parent_holder.append(committed_id)
                            effective_parent = submitted_parent_holder[0]
                        if msg.fork_node_id is not None:
                            # Fork: recipe / sampling / parent all come from
                            # the source node inside ``fork_from_token``; the
                            # WS-level steering/sampling/n fields are ignored.
                            result = session.fork_from_token(
                                msg.fork_node_id,
                                int(msg.fork_raw_index),  # pyright: ignore[reportArgumentType]  # guarded non-None by is_fork check above; int() accepts int|None only at runtime with None already excluded
                                alt_token_id=msg.fork_alt_token_id,
                                replacement_text=msg.fork_replacement_text,
                                **({"seed": msg.fork_seed} if msg.fork_seed is not None else {}),
                                on_token=_on_token,
                            )
                        elif msg.prefill_node_id is not None:
                            # Prefill: anchor / parent come from the user node
                            # inside ``prefill_assistant``; ``input`` is
                            # ignored.  ``steering`` / ``sampling`` ride through
                            # like a normal generate; ``thinking`` is forced
                            # off (the prefill is an answer, not a thought).
                            result = session.prefill_assistant(
                                msg.prefill_node_id,
                                str(msg.prefill_text),
                                steering=steering,
                                sampling=_sampling,
                                on_token=_on_token,
                            )
                        else:
                            gen_kwargs: dict[str, Any] = {
                                "steering": steering,
                                "sampling": _sampling,
                                "stateless": msg.stateless,
                                "raw": msg.raw,
                                "thinking": msg.thinking,
                                "on_token": _on_token,
                                "parent_node_id": effective_parent,
                            }
                            if _recipe_override is not None:
                                gen_kwargs["recipe_override"] = _recipe_override
                            if msg.generate_seat is not None:
                                gen_kwargs["gen_seat"] = msg.generate_seat
                            # Weave exploration preserves the source even for one
                            # candidate; ordinary chat keeps its coalescing default.
                            if n > 1 or not msg.append_same_role:
                                gen_kwargs["append_same_role"] = False
                            result = session.generate(
                                build_input(msg.input), **gen_kwargs,
                            ).first
                        _result_holder.append(result)
                except BaseException as e:
                    _error_holder.append(e)
                finally:
                    _token_queue.put(_TokenDone())

            await send_json({
                "type": "started",
                "generation_id": generation_id,
                # ``node_id`` is filled in lazily by the first token
                # event (the assistant node is created inside
                # ``_generate_core``); ``started`` includes the request-
                # level context the client needs to allocate state.
                "node_id": None,
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            worker_task = asyncio.create_task(asyncio.to_thread(_worker))

            # Race two queue reads — token frames from the worker and
            # client frames from the connection's perpetual reader.
            # Neither side ever calls ``websocket.receive_json()``
            # directly, so the underlying ``recv_in_progress`` flag is
            # owned by the reader task alone for the connection's
            # lifetime.
            done = False
            stop_signaled = False
            token_get = asyncio.create_task(token_queue.get())
            client_get = asyncio.create_task(incoming.get(control_only=True))
            try:
                while not done:
                    finished, _pending = await asyncio.wait(
                        {token_get, client_get}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    if client_get in finished:
                        incoming_msg = client_get.result()
                        if isinstance(incoming_msg, _Disconnect):
                            raise WebSocketDisconnect(code=incoming_msg.code)
                        cancelled.set()
                        stop_signaled = True
                        client_get = asyncio.create_task(incoming.get(control_only=True))
                    if token_get in finished:
                        item = token_get.result()
                        if isinstance(item, _TokenDone):
                            done = True
                        else:
                            await send_json(item.payload)
                            token_get = asyncio.create_task(token_queue.get())
            finally:
                cancelled.set()
                token_queue.close()
                with CancelScope(shield=True):
                    # A stop racing the final token must still abort the fan.
                    if client_get.done() and not client_get.cancelled():
                        stop_signaled = True
                    client_get.cancel()
                    token_get.cancel()
                    try:
                        await asyncio.gather(client_get, token_get, return_exceptions=True)
                    finally:
                        await finish_worker(worker_task)

            if error_holder and not result_holder:
                exc = error_holder[0]
                # The model worker runs in a thread, so by the time control
                # returns here there is no active exception for
                # ``logger.exception`` to capture.  Preserve its original
                # traceback explicitly: native WebSocket failures used to be
                # surfaced only to the browser, leaving operators with no
                # actionable server-side evidence for backend/device bugs.
                _logger.error(
                    "native WebSocket generation failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                if isinstance(exc, DrowseError):
                    status, message = exc.user_message()
                else:
                    status = 500
                    message = "Generation failed. Check the server log for details."
                await send_json(_error_frame(
                    message,
                    code=type(exc).__name__,
                    status=status,
                    node_id=current_node_holder[0],
                    sibling_index=sibling_idx,
                ))
                # On error inside a sibling, abort the remaining fan-out
                # rather than continuing with stale state.
                return

            if not result_holder:
                raise RuntimeError("generation completed without a result")
            result = result_holder[0]
            result_json = result_to_json(result)
            # The engine deliberately uses the OpenAI-compatible ``stop``
            # finish reason for EOS, stop sequences, and an external stop
            # request.  This native WebSocket knows which case happened:
            # surface a distinct UI-only reason so the dashboard does not
            # present a user-cancelled 659/1024-token turn as ordinary
            # completion.  Protocol adapters and the stored loom recipe keep
            # the engine's canonical reason unchanged.
            if stop_signaled and result_json.get("finish_reason") == "stop":
                result_json["finish_reason"] = "cancelled"
            # The settled per-probe aggregate rides the ``done`` event in the
            # 5.x measurement envelope and nowhere else — the same clean break
            # the ``token`` frame already made.  Geometry / lens / SAE readings
            # are split by family inside ``instruments``; a client merges them
            # exactly as it merges the token frame's.  The flat pre-5.x
            # ``probe_readings`` block is gone from this frame (it survives on
            # the OpenAI / Ollama ``x-drowse-probe-readings`` extension, which
            # is a real external contract).  Result-parameterized so each n>1
            # sibling reports its own result.
            # Built once by the engine at finalize (``GenerationResult.
            # measurements``), so the server does not re-split readings by
            # family — and the lens/SAE channels keep their native
            # ``ScalarReading`` shape instead of a reprojection.
            if result is not None and result.measurements:
                result_json["measurements"] = result.measurements
            # Phase 1 logit pass: stamp the per-turn logprob rollup on the
            # ``done`` event so subscribers (loom sidebar's sort-by-surprise,
            # webui chat-header summary) don't need to re-fetch the node.
            # Source of truth is the finalized loom node, populated by
            # :meth:`LoomTree.finalize_assistant` upstream of this branch.
            # Stateless gens / pre-logit-pass replays land with ``None``
            # which the wire layer passes through transparently.
            mean_logprob_out: float | None = None
            mean_surprise_out: float | None = None
            finalized_node_id = current_node_holder[0]
            if finalized_node_id is not None:
                node = session.tree.get(finalized_node_id)
                mean_logprob_out = node.mean_logprob
                mean_surprise_out = node.mean_surprise
            result_json["mean_logprob"] = mean_logprob_out
            result_json["mean_surprise"] = mean_surprise_out
            if (
                finalized_node_id is not None
                and int(session.tree.rev) > tree_rev_before_generation
            ):
                await wait_for_tree_finalization(finalized_node_id)
            await send_json({
                "type": "done",
                "result": result_json,
                "node_id": current_node_holder[0],
                "sibling_index": sibling_idx,
                "sibling_count": n,
            })

            # Mid-batch stop honors the batch plan: cancel the currently
            # streaming sibling and skip every remaining queued sibling.
            if stop_signaled:
                break
