"""Drowse — local activation steering + trait monitoring for HuggingFace causal LMs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "5.3.1"

_EXPORTS: dict[str, tuple[str, str]] = {
    "DrowseSession": ("drowse.session.session", "DrowseSession"),
    "DrowseError": ("drowse.errors", "DrowseError"),
    "GenState": ("drowse.generation.types", "GenState"),
    "LayerWhitener": ("drowse.geometry.whitener", "LayerWhitener"),
    "WhitenerError": ("drowse.errors", "WhitenerError"),
    "Profile": ("drowse.geometry.profile", "Profile"),
    "ProfileError": ("drowse.geometry.profile", "ProfileError"),
    "SamplingConfig": ("drowse.generation.sampling", "SamplingConfig"),
    "Steering": ("drowse.steering.terms", "Steering"),
    "Trigger": ("drowse.steering.triggers", "Trigger"),
    "EventBus": ("drowse.events", "EventBus"),
    "ManifoldExtracted": ("drowse.events", "ManifoldExtracted"),
    "SteeringApplied": ("drowse.events", "SteeringApplied"),
    "SteeringCleared": ("drowse.events", "SteeringCleared"),
    "ProbeScored": ("drowse.events", "ProbeScored"),
    "GenerationStarted": ("drowse.events", "GenerationStarted"),
    "GenerationFinished": ("drowse.events", "GenerationFinished"),
    "GenerationResult": ("drowse.generation.results", "GenerationResult"),
    "RunSet": ("drowse.generation.results", "RunSet"),
    "TokenAlt": ("drowse.generation.results", "TokenAlt"),
    "TokenEvent": ("drowse.generation.results", "TokenEvent"),
    "ProbeReadings": ("drowse.generation.results", "ProbeReadings"),
    "Manifold": ("drowse.geometry.manifold", "Manifold"),
    "ProbeReading": ("drowse.generation.results", "ProbeReading"),
    "ResultCollector": ("drowse.generation.results", "ResultCollector"),
    "LoomTree": ("drowse.loom.tree", "LoomTree"),
    "LoomNode": ("drowse.loom.tree", "LoomNode"),
    "LoomMutated": ("drowse.events", "LoomMutated"),
    "Recipe": ("drowse.loom.recipe", "Recipe"),
    "CastMember": ("drowse.loom.recipe", "CastMember"),
    "LoomTreeError": ("drowse.loom.errors", "LoomTreeError"),
    "UnknownNodeError": ("drowse.loom.errors", "UnknownNodeError"),
    "InvalidNodeOperationError": ("drowse.loom.errors", "InvalidNodeOperationError"),
    "MutationDuringGenerationError": ("drowse.loom.errors", "MutationDuringGenerationError"),
    "derive_seed_schedule": ("drowse.loom.recipe", "derive_seed_schedule"),
    "FilterClause": ("drowse.loom.filter", "FilterClause"),
    "FilterParseError": ("drowse.loom.filter", "FilterParseError"),
    "parse_filter": ("drowse.loom.filter", "parse_filter"),
    "DiffSpan": ("drowse.loom.diff", "DiffSpan"),
    "NodeDiff": ("drowse.loom.diff", "NodeDiff"),
    "ReadingDelta": ("drowse.loom.diff", "ReadingDelta"),
    "TokenDeltaSpan": ("drowse.loom.diff", "TokenDeltaSpan"),
    "per_token_diff": ("drowse.loom.diff", "per_token_diff"),
    "readings_diff": ("drowse.loom.diff", "readings_diff"),
    "steering_delta": ("drowse.loom.diff", "steering_delta"),
    "text_diff": ("drowse.loom.diff", "text_diff"),
    "ProbeRef": ("drowse.loom.transcript", "ProbeRef"),
    "Transcript": ("drowse.loom.transcript", "Transcript"),
    "TranscriptError": ("drowse.loom.transcript", "TranscriptError"),
    "TranscriptFormatError": ("drowse.loom.transcript", "TranscriptFormatError"),
    "TranscriptModelMismatch": ("drowse.loom.transcript", "TranscriptModelMismatch"),
    "TranscriptProbeDriftError": ("drowse.loom.transcript", "TranscriptProbeDriftError"),
    "TranscriptTurn": ("drowse.loom.transcript", "Turn"),
    # T2.3 — public types returned/raised by public methods
    "ChoiceScores": ("drowse.generation.scoring", "ChoiceScores"),
    "ChoiceScore": ("drowse.generation.scoring", "ChoiceScore"),
    "parse_expr": ("drowse.steering.grammar", "parse_expr"),
    "format_expr": ("drowse.steering.grammar", "format_expr"),
    "ManifoldTerm": ("drowse.steering.grammar", "ManifoldTerm"),
    "ProjectedTerm": ("drowse.steering.grammar", "ProjectedTerm"),
    "AblationTerm": ("drowse.steering.grammar", "AblationTerm"),
    "SelectorError": ("drowse.store.selectors", "SelectorError"),
    "AmbiguousSelectorError": ("drowse.store.selectors", "AmbiguousSelectorError"),
    "ManifoldNotRegisteredError": ("drowse.session.session", "ManifoldNotRegisteredError"),
    "ProfileNotRegisteredError": ("drowse.session.session", "ProfileNotRegisteredError"),
    # Read-side instrument facades: the contract, the three families, and
    # the per-family live-state records ``set_live``/``live_state`` return.
    "Instrument": ("drowse.readout.protocol", "Instrument"),
    "InstrumentRun": ("drowse.readout.protocol", "InstrumentRun"),
    "GeometryInstrument": ("drowse.readout.geometry", "GeometryInstrument"),
    "LensInstrument": ("drowse.lens.instrument", "LensInstrument"),
    "SaeInstrument": ("drowse.sae.instrument", "SaeInstrument"),
    "LiveState": ("drowse.readout.types", "LiveState"),
    "GeometryLiveState": ("drowse.readout.types", "GeometryLiveState"),
    "LensLiveState": ("drowse.readout.types", "LensLiveState"),
    "SaeLiveState": ("drowse.readout.types", "SaeLiveState"),
    "ScalarReading": ("drowse.readout.types", "ScalarReading"),
    "UnsupportedProbeChannelError": ("drowse.errors", "UnsupportedProbeChannelError"),
    # Jacobian lens (the verbalizable-workspace readout)
    "JacobianLens": ("drowse.lens.jlens", "JacobianLens"),
    "SparseDecomposition": ("drowse.lens.decompose", "SparseDecomposition"),
    "LensKey": ("drowse.lens.key", "LensKey"),
    "JacobianLensError": ("drowse.lens.jlens", "JacobianLensError"),
    "LensNotFittedError": ("drowse.lens.jlens", "LensNotFittedError"),
    "MultiTokenWordError": ("drowse.lens.jlens", "MultiTokenWordError"),
    "InvalidTokenIdError": ("drowse.lens.jlens", "InvalidTokenIdError"),
    "RelpUnsupportedError": ("drowse.errors", "RelpUnsupportedError"),
}

__all__ = [
    "DrowseSession",
    "DrowseError",
    "GenState",
    "LayerWhitener",
    "WhitenerError",
    "Profile",
    "ProfileError",
    "SamplingConfig",
    "Steering",
    "Trigger",
    "EventBus",
    "ManifoldExtracted",
    "SteeringApplied",
    "SteeringCleared",
    "ProbeScored",
    "GenerationStarted",
    "GenerationFinished",
    "GenerationResult",
    "RunSet",
    "TokenAlt",
    "TokenEvent",
    "ProbeReadings",
    "Manifold",
    "ProbeReading",
    "ResultCollector",
    "LoomTree",
    "LoomNode",
    "LoomMutated",
    "Recipe",
    "CastMember",
    "LoomTreeError",
    "UnknownNodeError",
    "InvalidNodeOperationError",
    "MutationDuringGenerationError",
    "derive_seed_schedule",
    "FilterClause",
    "FilterParseError",
    "parse_filter",
    "DiffSpan",
    "NodeDiff",
    "ReadingDelta",
    "TokenDeltaSpan",
    "per_token_diff",
    "readings_diff",
    "steering_delta",
    "text_diff",
    "ProbeRef",
    "Transcript",
    "TranscriptError",
    "TranscriptFormatError",
    "TranscriptModelMismatch",
    "TranscriptProbeDriftError",
    "TranscriptTurn",
    # T2.3 — public types returned/raised by public methods
    "ChoiceScores",
    "ChoiceScore",
    "parse_expr",
    "format_expr",
    "ManifoldTerm",
    "ProjectedTerm",
    "AblationTerm",
    "SelectorError",
    "AmbiguousSelectorError",
    "ManifoldNotRegisteredError",
    "ProfileNotRegisteredError",
    # Read-side instrument facades
    "Instrument",
    "InstrumentRun",
    "GeometryInstrument",
    "LensInstrument",
    "SaeInstrument",
    "LiveState",
    "GeometryLiveState",
    "LensLiveState",
    "SaeLiveState",
    "ScalarReading",
    "UnsupportedProbeChannelError",
    # Jacobian lens (the verbalizable-workspace readout)
    "JacobianLens",
    "SparseDecomposition",
    "LensKey",
    "JacobianLensError",
    "LensNotFittedError",
    "MultiTokenWordError",
    "InvalidTokenIdError",
    "RelpUnsupportedError",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(f"module 'drowse' has no attribute {name!r}") from e
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])


if TYPE_CHECKING:
    from drowse.errors import (
        RelpUnsupportedError as RelpUnsupportedError,
        DrowseError as DrowseError,
    )
    from drowse.events import (
        EventBus as EventBus,
        GenerationFinished as GenerationFinished,
        GenerationStarted as GenerationStarted,
        ProbeScored as ProbeScored,
        SteeringApplied as SteeringApplied,
        SteeringCleared as SteeringCleared,
        ManifoldExtracted as ManifoldExtracted,
    )
    from drowse.events import LoomMutated as LoomMutated
    from drowse.loom.errors import (
        InvalidNodeOperationError as InvalidNodeOperationError,
        LoomTreeError as LoomTreeError,
        MutationDuringGenerationError as MutationDuringGenerationError,
        UnknownNodeError as UnknownNodeError,
    )
    from drowse.loom.recipe import (
        CastMember as CastMember,
        Recipe as Recipe,
        derive_seed_schedule as derive_seed_schedule,
    )
    from drowse.loom.tree import LoomNode as LoomNode, LoomTree as LoomTree
    from drowse.loom.diff import (
        DiffSpan as DiffSpan,
        NodeDiff as NodeDiff,
        ReadingDelta as ReadingDelta,
        TokenDeltaSpan as TokenDeltaSpan,
        per_token_diff as per_token_diff,
        readings_diff as readings_diff,
        steering_delta as steering_delta,
        text_diff as text_diff,
    )
    from drowse.geometry.whitener import LayerWhitener as LayerWhitener
    from drowse.errors import WhitenerError as WhitenerError
    from drowse.geometry.manifold import Manifold as Manifold
    from drowse.geometry.profile import Profile as Profile, ProfileError as ProfileError
    from drowse.generation.results import (
        GenerationResult as GenerationResult,
        ProbeReading as ProbeReading,
        ProbeReadings as ProbeReadings,
        ResultCollector as ResultCollector,
        RunSet as RunSet,
        TokenAlt as TokenAlt,
        TokenEvent as TokenEvent,
    )
    from drowse.generation.sampling import SamplingConfig as SamplingConfig
    from drowse.generation.types import GenState as GenState
    from drowse.session.session import DrowseSession as DrowseSession
    from drowse.steering.terms import Steering as Steering
    from drowse.loom.transcript import (
        ProbeRef as ProbeRef,
        Transcript as Transcript,
        TranscriptError as TranscriptError,
        TranscriptFormatError as TranscriptFormatError,
        TranscriptModelMismatch as TranscriptModelMismatch,
        TranscriptProbeDriftError as TranscriptProbeDriftError,
        Turn as TranscriptTurn,  # noqa: F401
    )
    from drowse.loom.filter import (
        FilterClause as FilterClause,
        FilterParseError as FilterParseError,
        parse_filter as parse_filter,
    )
    from drowse.steering.triggers import Trigger as Trigger
    # T2.3 — public types returned/raised by public methods
    from drowse.generation.scoring import ChoiceScore as ChoiceScore, ChoiceScores as ChoiceScores
    from drowse.steering.grammar import (
        AblationTerm as AblationTerm,
        ManifoldTerm as ManifoldTerm,
        ProjectedTerm as ProjectedTerm,
        format_expr as format_expr,
        parse_expr as parse_expr,
    )
    from drowse.store.selectors import (
        AmbiguousSelectorError as AmbiguousSelectorError,
        SelectorError as SelectorError,
    )
    from drowse.session.session import (
        ManifoldNotRegisteredError as ManifoldNotRegisteredError,
        ProfileNotRegisteredError as ProfileNotRegisteredError,
    )
    from drowse.lens.decompose import (
        SparseDecomposition as SparseDecomposition,
    )
    from drowse.lens.jlens import (
        JacobianLens as JacobianLens,
        JacobianLensError as JacobianLensError,
        LensNotFittedError as LensNotFittedError,
        MultiTokenWordError as MultiTokenWordError,
        InvalidTokenIdError as InvalidTokenIdError,
    )
    from drowse.lens.key import LensKey as LensKey
    from drowse.errors import (
        UnsupportedProbeChannelError as UnsupportedProbeChannelError,
    )
    from drowse.readout.geometry import (
        GeometryInstrument as GeometryInstrument,
    )
    from drowse.lens.instrument import (
        LensInstrument as LensInstrument,
    )
    from drowse.readout.protocol import (
        Instrument as Instrument,
        InstrumentRun as InstrumentRun,
    )
    from drowse.sae.instrument import (
        SaeInstrument as SaeInstrument,
    )
    from drowse.readout.types import (
        GeometryLiveState as GeometryLiveState,
        LensLiveState as LensLiveState,
        LiveState as LiveState,
        SaeLiveState as SaeLiveState,
        ScalarReading as ScalarReading,
    )
