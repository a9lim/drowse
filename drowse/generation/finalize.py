"""Generation result finalization: one finished decode → its result.

:func:`finalize_generation` decodes the text, assembles the
:class:`~drowse.generation.results.GenerationResult` and its aggregate
measurement envelope from the generation's :class:`FinalReads`, publishes the
result, emits ``ProbeScored`` and finalizes the loom node.  The reads arrive
already scored (the session read plane's ``finalize``, or one batch row's
aggregates), so this module never touches the capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from drowse.events import EventBus, ProbeScored
from drowse.readout.types import (
    ScalarReading,
    as_probe_reading,
    reading_axis0,
)
from drowse.readout.measurements import build_measurements
from drowse.generation.results import GenerationResult, ProbeReading

if TYPE_CHECKING:
    from drowse.generation.decode import GenerationState

#: Below this many seconds a generation reports ``tok_per_sec = 0`` rather
#: than a rate dominated by timer resolution.
MIN_ELAPSED_FOR_RATE = 0.1


@dataclass(frozen=True)
class FinalReads:
    """One generation's end-of-generation reads, already scored.

    Each family keeps its native reading type (geometry ``ProbeReading``,
    lens/SAE ``ScalarReading``).  ``per_token`` is the geometry axis-0 coord
    stream (``None`` when nothing kept one), ``hidden_states`` the
    ``return_hidden`` stack, and the three source fields the live sources the
    aggregate envelope reports beside the lens/SAE readings.
    """

    geometry: dict[str, ProbeReading] = field(default_factory=dict)
    lens: dict[str, ScalarReading] = field(default_factory=dict)
    sae: dict[str, ScalarReading] = field(default_factory=dict)
    per_token: dict[str, list[float]] | None = None
    hidden_states: dict[int, torch.Tensor] | None = None
    lens_source: str | None = None
    sae_source: str | None = None
    sae_layer: int | None = None


def finalize_generation(
    generated_ids: list[int],
    elapsed: float,
    vector_snapshot: dict[str, float],
    prompt_tokens: int = 0,
    stateless: bool = False,
    logprobs_list: list[tuple[int, float | None, list[Any]]] | None = None,
    applied_steering: str | None = None,
    *,
    state: "GenerationState",
    reads: FinalReads,
    tokenizer: Any,
    events: EventBus,
    record: Callable[[GenerationResult, dict[str, list[float]] | None], None],
    tree: Any = None,
    scene_grammar: Callable[[], Any] = lambda: None,
    assistant_node_id: str | None = None,
    mean_logprob: float | None = None,
    mean_surprise: float | None = None,
) -> GenerationResult:
    """Decode, build the result from ``reads``, and finalize the loom node.

    ``state`` is the decode loop's state for this generation.  ``record``
    publishes the result and its per-token scores (the engine's
    ``last_result``) once the result exists, before the loom node is
    finalized.  ``tree`` (the loom tree) and ``scene_grammar`` (the lazily
    validated turn grammar) are read only for a stateful generation with an
    ``assistant_node_id``.
    """
    token_count = len(generated_ids)
    tok_per_sec = (
        token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
    )
    response_ids = generated_ids[state.thinking_end_idx:]
    if state.finish_reason == "stop_sequence" and state.response_text is not None:
        text: str = state.response_text
    else:
        decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
        text = decoded if isinstance(decoded, str) else decoded[0]

    geometry_aggregates = reads.geometry
    lens_aggregates = reads.lens
    sae_aggregates = reads.sae
    manifold_aggregates: dict[str, Any] = {
        **geometry_aggregates, **lens_aggregates, **sae_aggregates,
    }
    measurement_envelope = build_measurements(
        scope="aggregate",
        geometry_readings=geometry_aggregates or None,
        lens_readings=lens_aggregates or None,
        sae_readings=sae_aggregates or None,
        lens_source=reads.lens_source,
        sae_source=reads.sae_source,
        sae_layer=reads.sae_layer,
        steering=applied_steering,
    )
    # ``Measurements`` is a TypedDict describing the wire shape, while the
    # established ``GenerationResult`` compatibility field is a plain mapping.
    # Materialize the latter at this boundary without changing the envelope.
    measurements = (
        dict(measurement_envelope)
        if measurement_envelope is not None else None
    )

    result = GenerationResult(
        text=text, tokens=list(generated_ids), token_count=token_count,
        tok_per_sec=tok_per_sec, elapsed=elapsed,
        steering_alphas=vector_snapshot,
        prompt_tokens=prompt_tokens,
        finish_reason=state.finish_reason,
        logprobs=logprobs_list,
        applied_steering=applied_steering,
        hidden_states=reads.hidden_states,
        # The compatibility dict: one reading type across families, for the
        # vendor extension and cross-family callers.
        probe_readings={
            name: as_probe_reading(reading)
            for name, reading in manifold_aggregates.items()
        },
        measurements=measurements,
    )
    record(result, reads.per_token)

    if manifold_aggregates:
        events.emit(ProbeScored(readings={
            name: reading_axis0(reading)
            for name, reading in manifold_aggregates.items()
        }))

    if not stateless and assistant_node_id is not None:
        # Decoded thinking-channel text, joined from the streamed token
        # rows — stamped so history re-renders can carry the block
        # through the family's think delimiters (the stitcher applies
        # the family's history policy; strip families render it only
        # while the turn is last).  Stamped ONLY when the scene grammar
        # can actually re-render it (``think_open`` set): on a family
        # whose thinking isn't delimiter-shaped (gpt-oss channels) or
        # that fell back to template rendering, a stamped block would
        # make every later render of this path raise.
        thinking_text: str | None = None
        grammar = scene_grammar()
        if grammar is not None and grammar.think_open is not None:
            node = tree.nodes.get(assistant_node_id)
            if node is not None and node.thinking_tokens:
                joined = "".join(
                    str(t.get("text", "")) for t in node.thinking_tokens
                )
                thinking_text = joined or None
        tree.finalize_assistant(
            assistant_node_id,
            text=text,
            aggregate_readings={
                name: reading_axis0(reading)
                for name, reading in manifold_aggregates.items()
            },
            applied_steering=applied_steering,
            finish_reason=state.finish_reason,
            mean_logprob=mean_logprob,
            mean_surprise=mean_surprise,
            raw_token_ids=generated_ids,
            thinking_text=thinking_text,
        )

    return result
