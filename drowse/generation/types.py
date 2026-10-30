"""Generation lifecycle types: the phase enum, the resolved call, the stream.

These are the value types the session's generation engine
(:class:`drowse.session.engine.GenerationEngine`) passes between its stages and
hands back to callers.  None of them reference a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol

from drowse.generation.decode import GenerationConfig
from drowse.generation.results import GenerationResult, TokenEvent
from drowse.generation.sampling import SamplingConfig
from drowse.steering.terms import Steering


class GenState(IntEnum):
    """Lifecycle phase of the session's (single) generation.

    A typed phase rather than a bare in-flight flag, so the teardown order is
    self-documenting.  The engine's transaction owns the transitions:

    - ``IDLE`` → ``PREAMBLE``: lock acquired, re-entry guard passed.  The call
      is resolved, the loom nodes are created and the steering scope is pushed.
    - ``PREAMBLE`` → ``RUNNING``: capture attached and the steering
      :class:`~drowse.steering.triggers.TriggerContext` reset; the decode loop
      runs.  Token callbacks fire in this phase.
    - ``RUNNING`` → ``FINALIZING``: capture detached and the steering scope
      popped; the result is being built.
    - ``FINALIZING`` → ``IDLE``: the transaction released the lock.

    The reentrant lock stays alongside this enum: the lock serializes threads,
    the phase rejects same-thread re-entry from a token callback.
    """

    IDLE = 0
    PREAMBLE = 1
    RUNNING = 2
    FINALIZING = 3


class GenerationStream(Protocol):
    """The streaming iterator :meth:`DrowseSession.generate_stream` returns."""

    def __iter__(self) -> "GenerationStream": ...

    def __next__(self) -> TokenEvent: ...

    def close(self) -> None: ...

    @property
    def result(self) -> GenerationResult | None: ...


@dataclass(frozen=True, slots=True)
class CallSpec:
    """One generation call, fully resolved before any model work.

    Built once by the engine's ``resolve_call`` from the call's kwargs, the
    cast roster, a regen's recipe override and the same-role continuation
    rule, then read by every later stage.  ``sampling`` is the composed
    per-call box exactly as the node recipe records it (``None`` when the call
    passed none); every value derived from it is resolved here so no stage
    re-derives one.
    """

    input: Any
    steering_obj: Steering | None
    sampling: SamplingConfig | None
    #: The resolved system prompt (the session default already substituted).
    system_prompt: str | None
    stateless: bool
    raw: bool
    gen_seat: str
    parent_node_id: str | None
    #: The loom path the prompt renders against (``None`` = active path).
    history_anchor: str | None
    #: The thinking channel the call asked for (before the family gate).
    use_thinking_req: bool
    gen_config: GenerationConfig
    #: Top-alternative width the sampler captures (``None`` = no logprobs).
    lp_count: int | None
    seed: int | None
    stop_list: list[str] | None
    logit_bias: dict[int, float] | None
    presence_penalty: float
    frequency_penalty: float
    #: ``sampling.logprobs`` was set: the result carries per-token logprobs.
    collect_logprobs: bool
    readout_top_k: int
    user_role: str | None
    assistant_role: str | None
    return_probe_readings: bool
    want_hidden: bool
    persist_per_layer_scores: bool
    persist_subspace_coords: bool
    forced_prefix: list[int] | None
    #: An existing same-role node this call continues in place.
    continuation_node_id: str | None = None


class GenerationTransaction:
    """The mutable state one generation transaction owns.

    ``steering_cm`` is whatever internal steering scope the body entered and
    has not popped itself; the transaction pops the remainder on the way out.
    ``failed`` records that the body is unwinding, which decides whether a
    teardown ``Exception`` may mask the original failure.
    """

    __slots__ = ("failed", "steering_cm")

    def __init__(self) -> None:
        self.steering_cm: Any = None
        self.failed: bool = False


__all__ = [
    "CallSpec",
    "GenState",
    "GenerationStream",
    "GenerationTransaction",
]
