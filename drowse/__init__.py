"""Drowse — local activation steering + trait monitoring for HuggingFace causal LMs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "5.3.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    "DrowseSession": ("drowse.core.session", "DrowseSession"),
    "DrowseError": ("drowse.core.errors", "DrowseError"),
    "GenState": ("drowse.core.session", "GenState"),
    "LayerWhitener": ("drowse.core.mahalanobis", "LayerWhitener"),
    "WhitenerError": ("drowse.core.mahalanobis", "WhitenerError"),
    "Profile": ("drowse.core.profile", "Profile"),
    "ProfileError": ("drowse.core.profile", "ProfileError"),
    "SamplingConfig": ("drowse.core.sampling", "SamplingConfig"),
    "Steering": ("drowse.core.steering", "Steering"),
    "Trigger": ("drowse.core.triggers", "Trigger"),
    "EventBus": ("drowse.core.events", "EventBus"),
    "ManifoldExtracted": ("drowse.core.events", "ManifoldExtracted"),
    "SteeringApplied": ("drowse.core.events", "SteeringApplied"),
    "SteeringCleared": ("drowse.core.events", "SteeringCleared"),
    "ProbeScored": ("drowse.core.events", "ProbeScored"),
    "GenerationStarted": ("drowse.core.events", "GenerationStarted"),
    "GenerationFinished": ("drowse.core.events", "GenerationFinished"),
    "GenerationResult": ("drowse.core.results", "GenerationResult"),
    "RunSet": ("drowse.core.results", "RunSet"),
    "TokenAlt": ("drowse.core.results", "TokenAlt"),
    "TokenEvent": ("drowse.core.results", "TokenEvent"),
    "ProbeReadings": ("drowse.core.results", "ProbeReadings"),
    "Manifold": ("drowse.core.manifold", "Manifold"),
    "ProbeReading": ("drowse.core.results", "ProbeReading"),
    "ResultCollector": ("drowse.core.results", "ResultCollector"),
    "LoomTree": ("drowse.core.loom", "LoomTree"),
    "LoomNode": ("drowse.core.loom", "LoomNode"),
    "LoomMutated": ("drowse.core.loom", "LoomMutated"),
    "Recipe": ("drowse.core.loom", "Recipe"),
    "CastMember": ("drowse.core.loom", "CastMember"),
    "LoomTreeError": ("drowse.core.loom", "LoomTreeError"),
    "UnknownNodeError": ("drowse.core.loom", "UnknownNodeError"),
    "InvalidNodeOperationError": ("drowse.core.loom", "InvalidNodeOperationError"),
    "MutationDuringGenerationError": ("drowse.core.loom", "MutationDuringGenerationError"),
    "derive_seed_schedule": ("drowse.core.loom", "derive_seed_schedule"),
    "FilterClause": ("drowse.core.tree_filter", "FilterClause"),
    "FilterParseError": ("drowse.core.tree_filter", "FilterParseError"),
    "parse_filter": ("drowse.core.tree_filter", "parse_filter"),
    "DiffSpan": ("drowse.core.loom_diff", "DiffSpan"),
    "NodeDiff": ("drowse.core.loom_diff", "NodeDiff"),
    "ReadingDelta": ("drowse.core.loom_diff", "ReadingDelta"),
    "TokenDeltaSpan": ("drowse.core.loom_diff", "TokenDeltaSpan"),
    "per_token_diff": ("drowse.core.loom_diff", "per_token_diff"),
    "readings_diff": ("drowse.core.loom_diff", "readings_diff"),
    "steering_delta": ("drowse.core.loom_diff", "steering_delta"),
    "text_diff": ("drowse.core.loom_diff", "text_diff"),
    "ProbeRef": ("drowse.core.transcript", "ProbeRef"),
    "Transcript": ("drowse.core.transcript", "Transcript"),
    "TranscriptError": ("drowse.core.transcript", "TranscriptError"),
    "TranscriptFormatError": ("drowse.core.transcript", "TranscriptFormatError"),
    "TranscriptModelMismatch": ("drowse.core.transcript", "TranscriptModelMismatch"),
    "TranscriptProbeDriftError": ("drowse.core.transcript", "TranscriptProbeDriftError"),
    "TranscriptTurn": ("drowse.core.transcript", "Turn"),
    # T2.3 — public types returned/raised by public methods
    "ChoiceScores": ("drowse.core.scoring", "ChoiceScores"),
    "ChoiceScore": ("drowse.core.scoring", "ChoiceScore"),
    "parse_expr": ("drowse.core.steering_expr", "parse_expr"),
    "format_expr": ("drowse.core.steering_expr", "format_expr"),
    "ManifoldTerm": ("drowse.core.steering_expr", "ManifoldTerm"),
    "ProjectedTerm": ("drowse.core.steering_expr", "ProjectedTerm"),
    "AblationTerm": ("drowse.core.steering_expr", "AblationTerm"),
    "SelectorError": ("drowse.io.selectors", "SelectorError"),
    "AmbiguousSelectorError": ("drowse.io.selectors", "AmbiguousSelectorError"),
    "ManifoldNotRegisteredError": ("drowse.core.session", "ManifoldNotRegisteredError"),
    "ProfileNotRegisteredError": ("drowse.core.session", "ProfileNotRegisteredError"),
    # Read-side instrument facades: the contract, the three families, and
    # the per-family live-state records ``set_live``/``live_state`` return.
    "Instrument": ("drowse.core.instruments.protocol", "Instrument"),
    "InstrumentRun": ("drowse.core.instruments.protocol", "InstrumentRun"),
    "GeometryInstrument": ("drowse.core.instruments.geometry", "GeometryInstrument"),
    "LensInstrument": ("drowse.core.instruments.lens", "LensInstrument"),
    "SaeInstrument": ("drowse.core.instruments.sae", "SaeInstrument"),
    "LiveState": ("drowse.core.instruments.types", "LiveState"),
    "GeometryLiveState": ("drowse.core.instruments.types", "GeometryLiveState"),
    "LensLiveState": ("drowse.core.instruments.types", "LensLiveState"),
    "SaeLiveState": ("drowse.core.instruments.types", "SaeLiveState"),
    "ScalarReading": ("drowse.core.instruments.types", "ScalarReading"),
    "UnsupportedProbeChannelError": ("drowse.core.errors", "UnsupportedProbeChannelError"),
    # Jacobian lens (the verbalizable-workspace readout)
    "JacobianLens": ("drowse.core.jlens", "JacobianLens"),
    "JSpaceDecomposition": ("drowse.core.jlens", "JSpaceDecomposition"),
    "JacobianLensError": ("drowse.core.jlens", "JacobianLensError"),
    "LensNotFittedError": ("drowse.core.jlens", "LensNotFittedError"),
    "MultiTokenWordError": ("drowse.core.jlens", "MultiTokenWordError"),
    "RelpUnsupportedError": ("drowse.core.errors", "RelpUnsupportedError"),
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
    "JSpaceDecomposition",
    "JacobianLensError",
    "LensNotFittedError",
    "MultiTokenWordError",
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
    from drowse.core.errors import (
        RelpUnsupportedError as RelpUnsupportedError,
        DrowseError as DrowseError,
    )
    from drowse.core.events import (
        EventBus as EventBus,
        GenerationFinished as GenerationFinished,
        GenerationStarted as GenerationStarted,
        ProbeScored as ProbeScored,
        SteeringApplied as SteeringApplied,
        SteeringCleared as SteeringCleared,
        ManifoldExtracted as ManifoldExtracted,
    )
    from drowse.core.loom import (
        CastMember as CastMember,
        InvalidNodeOperationError as InvalidNodeOperationError,
        LoomMutated as LoomMutated,
        LoomNode as LoomNode,
        LoomTree as LoomTree,
        LoomTreeError as LoomTreeError,
        MutationDuringGenerationError as MutationDuringGenerationError,
        Recipe as Recipe,
        UnknownNodeError as UnknownNodeError,
        derive_seed_schedule as derive_seed_schedule,
    )
    from drowse.core.loom_diff import (
        DiffSpan as DiffSpan,
        NodeDiff as NodeDiff,
        ReadingDelta as ReadingDelta,
        TokenDeltaSpan as TokenDeltaSpan,
        per_token_diff as per_token_diff,
        readings_diff as readings_diff,
        steering_delta as steering_delta,
        text_diff as text_diff,
    )
    from drowse.core.mahalanobis import LayerWhitener as LayerWhitener, WhitenerError as WhitenerError
    from drowse.core.manifold import Manifold as Manifold
    from drowse.core.profile import Profile as Profile, ProfileError as ProfileError
    from drowse.core.results import (
        GenerationResult as GenerationResult,
        ProbeReading as ProbeReading,
        ProbeReadings as ProbeReadings,
        ResultCollector as ResultCollector,
        RunSet as RunSet,
        TokenAlt as TokenAlt,
        TokenEvent as TokenEvent,
    )
    from drowse.core.sampling import SamplingConfig as SamplingConfig
    from drowse.core.session import GenState as GenState, DrowseSession as DrowseSession
    from drowse.core.steering import Steering as Steering
    from drowse.core.transcript import (
        ProbeRef as ProbeRef,
        Transcript as Transcript,
        TranscriptError as TranscriptError,
        TranscriptFormatError as TranscriptFormatError,
        TranscriptModelMismatch as TranscriptModelMismatch,
        TranscriptProbeDriftError as TranscriptProbeDriftError,
        Turn as TranscriptTurn,  # noqa: F401
    )
    from drowse.core.tree_filter import (
        FilterClause as FilterClause,
        FilterParseError as FilterParseError,
        parse_filter as parse_filter,
    )
    from drowse.core.triggers import Trigger as Trigger
    # T2.3 — public types returned/raised by public methods
    from drowse.core.scoring import ChoiceScore as ChoiceScore, ChoiceScores as ChoiceScores
    from drowse.core.steering_expr import (
        AblationTerm as AblationTerm,
        ManifoldTerm as ManifoldTerm,
        ProjectedTerm as ProjectedTerm,
        format_expr as format_expr,
        parse_expr as parse_expr,
    )
    from drowse.io.selectors import (
        AmbiguousSelectorError as AmbiguousSelectorError,
        SelectorError as SelectorError,
    )
    from drowse.core.session import (
        ManifoldNotRegisteredError as ManifoldNotRegisteredError,
        ProfileNotRegisteredError as ProfileNotRegisteredError,
    )
    from drowse.core.jlens import (
        JacobianLens as JacobianLens,
        JacobianLensError as JacobianLensError,
        JSpaceDecomposition as JSpaceDecomposition,
        LensNotFittedError as LensNotFittedError,
        MultiTokenWordError as MultiTokenWordError,
    )
    from drowse.core.errors import (
        UnsupportedProbeChannelError as UnsupportedProbeChannelError,
    )
    from drowse.core.instruments.geometry import (
        GeometryInstrument as GeometryInstrument,
    )
    from drowse.core.instruments.lens import (
        LensInstrument as LensInstrument,
    )
    from drowse.core.instruments.protocol import (
        Instrument as Instrument,
        InstrumentRun as InstrumentRun,
    )
    from drowse.core.instruments.sae import (
        SaeInstrument as SaeInstrument,
    )
    from drowse.core.instruments.types import (
        GeometryLiveState as GeometryLiveState,
        LensLiveState as LensLiveState,
        LiveState as LiveState,
        SaeLiveState as SaeLiveState,
        ScalarReading as ScalarReading,
    )
