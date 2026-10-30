"""The fake sessions the HTTP/WS route tests drive.

:func:`mock_session` returns a ``MagicMock`` wired with the session surface
the OpenAI, Ollama and native routes read on every request: model identity
and info, a sampling config, an idle loom tree, the three instrument faces
(idle, no probes), the Monitor, the generation state and a working event bus.
It is an autospec of ``DrowseSession``: a name the class does not define
reads as ``AttributeError`` and method calls are signature-checked.  Keyword
overrides are applied last; :func:`script_generate` makes its
``generate`` stream scripted tokens.  :class:`StubSession` is a hand-written
session over a real :class:`LoomTree` with scripted generation, for the loom
routes and the WS request lifecycle.  Route tests that need real behavior
build a ``tiny_session`` and the ``app_client`` fixture instead.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, create_autospec

from drowse.events import EventBus
from drowse.generation.results import GenerationResult, RunSet
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree
from drowse.readout.types import GeometryLiveState, LensLiveState, SaeLiveState
from drowse.session.session import DrowseSession
from tests.support.tokens import emit_token

__all__ = ["StubSession", "mock_session", "script_generate"]


def _event_bus() -> Any:
    subscribers: list[Any] = []

    def _subscribe(cb: Any) -> Any:
        subscribers.append(cb)

        def _unsub() -> None:
            try:
                subscribers.remove(cb)
            except ValueError:
                pass

        return _unsub

    def _emit(event: Any) -> None:
        for cb in list(subscribers):
            try:
                cb(event)
            except Exception:
                pass

    events = MagicMock()
    events.subscribe = _subscribe
    events.emit = _emit
    return events


def mock_session(**overrides: Any) -> Any:
    """An autospec of ``DrowseSession`` shaped like an idle session.

    Reading a name the class does not define raises ``AttributeError`` and
    every method call is checked against the real signature, so a route that
    calls a removed or renamed session method fails here instead of passing
    against a permissive mock.
    """
    session = create_autospec(DrowseSession, instance=True)
    session.model_id = "test/model"
    session.model_info = {
        "model_type": "gemma2",
        "num_layers": 26,
        "hidden_dim": 2304,
        "vram_used_gb": 5.2,
        "param_count": 2_614_000_000,
        "device": "cpu",
        "dtype": "torch.bfloat16",
    }
    session.model = MagicMock()
    session.renderer.model_type = "gemma2"
    session.layers = [MagicMock() for _ in range(26)]

    session.config = MagicMock()
    session.config.temperature = 1.0
    session.config.top_p = 0.9
    session.config.top_k = None
    session.config.max_new_tokens = 1024
    session.config.system_prompt = None
    session.config.thinking = None

    session.profiles = {}
    session.probes = {}
    session.manifolds = {}
    session.tree = MagicMock()
    session.tree.messages_for.return_value = []
    session.tree.active_node_id = "test-assistant"
    session.tree.get.return_value.mean_logprob = None
    session.tree.get.return_value.mean_surprise = None
    session.is_base_model = False
    session.scene_grammar = None
    session.joint_logprob_cache = {}
    session.chat_traits = SimpleNamespace(
        supports_thinking=False, thinking_optional=False,
    )

    # The three instrument faces the routes and session_info read: idle, no
    # probes, no compatible lens.
    session.lens.has_compatible.return_value = False
    session.geometry.names = []
    session.geometry.specs.return_value = {}
    session.geometry.active_source = None
    session.geometry.live_state = GeometryLiveState(enabled=True)
    session.lens.names = []
    session.lens.specs.return_value = {}
    session.lens.active_source = None
    session.lens.live_state = LensLiveState(enabled=False)
    session.sae.names = []
    session.sae.specs.return_value = {}
    session.sae.active_source = None
    session.sae.live_state = SaeLiveState(enabled=False)
    session.instruments = {
        "geometry": session.geometry,
        "lens": session.lens,
        "sae": session.sae,
    }

    # The unified monitor: ``probe_names`` (wire/gate surface) and
    # ``attached_probes()`` (the serializer source).
    monitor = MagicMock()
    monitor.probe_names = []
    monitor.profiles = {}
    monitor.attached_probes.return_value = {}
    session.monitor = monitor

    session.tokenizer = MagicMock()
    session.tokenizer.decode.side_effect = lambda ids: f"<{ids[0]}>" if ids else ""
    session.last_per_token_scores = None
    session.last_result = None
    gen_state = MagicMock()
    gen_state.finish_reason = "stop"
    gen_state.emit_map = []
    session.generation_state = gen_state
    session.events = _event_bus()

    for key, value in overrides.items():
        setattr(session, key, value)
    return session


def script_generate(session: Any, tokens: Any) -> None:
    """Make ``session.generate`` stream ``tokens`` through ``on_token`` and
    finish with one ``stop`` result (per-token ``happy`` scores recorded)."""

    def _gen(input: Any, *, steering: Any = None, sampling: Any = None,
             stateless: Any = False, raw: Any = False, thinking: Any = None,
             on_token: Any = None, parent_node_id: Any = None, n: Any = 1,
             append_same_role: Any = True) -> Any:
        for i, tok in enumerate(tokens):
            emit_token(on_token, tok, 1000 + i, index=i)
            time.sleep(0.001)
        result = GenerationResult(
            text="".join(tokens), tokens=list(range(1000, 1000 + len(tokens))),
            token_count=len(tokens), tok_per_sec=50.0, elapsed=0.05,
            finish_reason="stop",
        )
        session.last_result = result
        session.last_per_token_scores = {
            "happy": [0.1 * (i + 1) for i in range(len(tokens))],
        }
        return RunSet([result])

    session.generate.side_effect = _gen


class StubSession:
    """Minimal DrowseSession-shaped object backing the routes.

    Wraps a real :class:`LoomTree` so the routes operate on actual tree
    semantics (rev bumps, event emission, conflict-check on mutators).
    ``generate`` is the only generation entry point: it walks the
    sibling loop, calls ``tree.begin_assistant`` + ``tree.append_token``
    + ``tree.finalize_assistant`` per sibling, and emits tokens through
    the supplied ``on_token`` callback.
    """

    def __init__(self) -> None:
        self.model_id = "test/model"
        self.model_info = {
            "model_type": "gemma2",
            "num_layers": 4,
            "hidden_dim": 16,
            "device": "cpu",
            "dtype": "torch.float32",
        }
        self._device = "cpu"
        self._dtype = "torch.float32"
        self._created_ts = 1_700_000_000

        self.events = EventBus()
        self.tree = LoomTree(events=self.events, model_id=self.model_id)
        self.scene_grammar = None

        cfg = MagicMock()
        cfg.temperature = 1.0
        cfg.top_p = 0.9
        cfg.top_k = None
        cfg.max_new_tokens = 64
        cfg.system_prompt = "You are a stub."
        self.config = cfg

        self.profiles: dict[str, Any] = {}
        self.probes: dict[str, Any] = {}

        monitor = MagicMock()
        monitor.probe_names = []
        monitor.profiles = {}
        self.monitor = monitor
        self._monitor = monitor
        self._joint_logprob_cache: dict[Any, Any] = {}
        # Public instrument surface — ``probe_measurements_aggregate`` reads
        # both the attached roster and the live source/layer binding off
        # these faces.
        self.lens = SimpleNamespace(live=None, names=[])
        self.sae = SimpleNamespace(live=None, names=[])
        # End-of-generation readings the stub stamps onto every result, so a
        # test can pin the ``done`` frame's aggregate channel.
        self.stub_probe_readings: dict[str, Any] = {}
        self.stub_lens_readings: dict[str, Any] = {}
        self._tokenizer = MagicMock()
        self._tokenizer.encode.side_effect = lambda text, **_: [
            3000 + i for i, _ch in enumerate(str(text))
        ]
        self._layers = []
        capture = MagicMock()
        capture._per_layer = {}
        self._capture = capture
        self._last_per_token_scores = None
        self._last_result = None
        self.last_per_token_scores = None
        self.last_result = None

        from drowse.generation.decode import GenerationState
        self.generation_state = GenerationState()

        self._next_token_stream: list[str] = ["hi"]
        self._fail_next: bool = False
        self._block_until_stop = False
        self._stop_event = threading.Event()

    def clear_history(self) -> None:
        self.tree.reset()

    def restore_tree(self, data: dict[str, Any]) -> LoomTree:
        previous = self.tree
        restored = LoomTree.replacing(previous, data, model_id=self.model_id)
        self.tree = restored
        restored.announce_replacement(previous)
        return restored

    def rewind(self) -> None:
        self.tree.rewind()

    def probe_hashes(self) -> dict[str, str]:
        return {}

    def stop(self) -> None:
        self._stop_event.set()
        self.generation_state.request_stop()

    # ----- generation entry point --------------------------------------
    def generate(self, input: Any, *, steering: Any = None, sampling: Any = None,
                 stateless: bool = False, raw: bool = False, thinking: Any = None,
                 on_token: Any = None, parent_node_id: Any = None, n: int = 1,
                 gen_seat: str = "assistant", recipe_override: Any = None,
                 append_same_role: bool = True):
        """Stub generate.

        Routes through the tree the same way DrowseSession does for
        phase-2's WS plumbing to see the right LoomMutated events fire.
        Each sibling emits one synthetic token, finalizes, and produces
        a :class:`GenerationResult`.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        from drowse.loom.recipe import derive_seed_schedule
        base_seed = sampling.seed if sampling is not None else None
        schedule = derive_seed_schedule(base_seed, n) if n > 1 else [base_seed]

        results = []
        for sibling_idx, seed_i in enumerate(schedule):
            # Text input retains the compatibility generate contract (a user
            # commit); ``None`` is the native bare continuation used by
            # seat-neutral submit after it commits explicitly.
            if input is None:
                anchor_id = parent_node_id or self.tree.active_node_id
            else:
                anchor_id = self.tree.add_user_turn(
                    str(input), parent_id=parent_node_id,
                )
            self.tree.reserve(anchor_id)
            role_label = None
            if sampling is not None and not raw:
                role_label = (
                    sampling.user_role
                    if gen_seat == "user"
                    else sampling.assistant_role
                )
            anchor = self.tree.get(anchor_id)
            can_continue = bool(
                append_same_role
                and n == 1
                and input is None
                and anchor.role == gen_seat
                and anchor.role_label == role_label
            )
            prefix = anchor.text if can_continue else ""
            if can_continue:
                assistant_id = self.tree.begin_continuation(
                    anchor_id, Recipe(),
                )
            else:
                assistant_id = self.tree.begin_assistant(
                    anchor_id, recipe=Recipe(), role_label=role_label,
                    seat=gen_seat,
                )
            try:
                token_text = f"tok{sibling_idx}"
                if on_token is not None:
                    emit_token(on_token, token_text, 1000 + sibling_idx)
                self.tree.append_token(
                    assistant_id, {"token_id": 1000 + sibling_idx, "text": token_text},
                )
                if self._block_until_stop:
                    import time
                    deadline = time.monotonic() + 5
                    while not self.generation_state.is_stop_requested():
                        assert time.monotonic() < deadline
                        self._stop_event.wait(timeout=0.01)
                full_text = prefix + token_text
                from drowse.readout.measurements import build_measurements

                result = GenerationResult(
                    text=full_text, tokens=[1000 + sibling_idx],
                    token_count=1, tok_per_sec=10.0, elapsed=0.1,
                    finish_reason="stop",
                    probe_readings=dict(self.stub_probe_readings),
                    # The engine builds this once at finalize; the ``done``
                    # frame just forwards it.
                    measurements=cast(dict[str, Any] | None, build_measurements(
                        scope="aggregate",
                        geometry_readings=(
                            dict(self.stub_probe_readings) or None
                        ),
                        lens_readings=dict(self.stub_lens_readings) or None,
                        lens_source=(
                            (self.lens.live or {}).get("source")
                            if self.stub_lens_readings else None
                        ),
                    )),
                )
                self.tree.finalize_assistant(
                    assistant_id,
                    text=full_text,
                    aggregate_readings={},
                    applied_steering=None,
                    finish_reason="stop",
                )
                results.append(result)
                self._last_result = result
                self.last_result = result
            finally:
                self.tree.reserve(None)
        return RunSet(results)

    # ----- answer-prefill entry point ----------------------------------
    def prefill_assistant(self, node_id: Any, text: Any, *, steering: Any = None,
                          sampling: Any = None, on_token: Any = None):
        """Stub answer-prefill.

        Mirrors ``DrowseSession.prefill_assistant``'s tree shape: anchor
        at the user node's parent so ``add_user_turn`` dedup re-uses the
        existing user turn, then land an assistant child whose text opens
        with ``text`` (the seeded prefix) followed by a synthetic
        continuation token.
        """
        from drowse.loom.errors import InvalidNodeOperationError
        node = self.tree.get(node_id)
        if node.role != "user":
            raise InvalidNodeOperationError(
                f"prefill_assistant: {node_id!r} is a {node.role} node"
            )
        user_id = self.tree.add_user_turn(node.text, parent_id=node.parent_id)
        self.tree.reserve(user_id)
        assistant_id = self.tree.begin_assistant(user_id)
        try:
            if on_token is not None:
                emit_token(on_token, text, 2000)
                emit_token(on_token, "-cont", 2001, index=1)
            self.tree.append_token(
                assistant_id, {"token_id": 2000, "text": text},
            )
            self.tree.append_token(
                assistant_id, {"token_id": 2001, "text": "-cont"},
            )
            full = text + "-cont"
            result = GenerationResult(
                text=full, tokens=[2000, 2001], token_count=2,
                tok_per_sec=10.0, elapsed=0.1, finish_reason="stop",
            )
            self.tree.finalize_assistant(
                assistant_id, text=full, aggregate_readings={},
                applied_steering=None, finish_reason="stop",
            )
            self._last_result = result
            self.last_result = result
        finally:
            self.tree.reserve(None)
        return result

    # ----- commit entry point (Ctrl+Enter on either surface) -----------
    def append_turn(
        self, parent_node_id: Any, text: Any, *, role: Any,
        raw: bool = False, role_label: Any = None, thinking: Any = None,
    ):
        from drowse.session.session import DrowseSession
        return DrowseSession.append_turn(
            self,  # pyright: ignore[reportArgumentType]
            parent_node_id, text, role=role, raw=raw,
            role_label=role_label, thinking=thinking,
        )

    def set_cast_member(self, label: Any, **kwargs: Any) -> Any:
        from drowse.loom.recipe import CastMember

        member = CastMember.from_parts(**kwargs)
        self.tree.set_cast_member(label, member)
        return member

    def remove_cast_member(self, label: Any) -> None:
        self.tree.remove_cast_member(label)
