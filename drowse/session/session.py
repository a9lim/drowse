"""DrowseSession — unified backend for drowse's programmatic and server APIs."""
from __future__ import annotations
import logging
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Protocol, cast, overload

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from drowse.fit.capture import CaptureMode, CaptureState, _capture_all_hidden_states
from drowse.fit.corpus import RESPONSE_MAX_TOKENS, CorpusGenerator
from drowse.session.extraction import Extraction
from drowse.errors import DrowseError
from drowse.events import (
    EventBus,
    GenerationFinished,
    GenerationStarted,
    ProbeScored,
)
from drowse.generation.prefix_cache import PrefixCacheEntry
from drowse.generation.decode import (
    GenerationConfig,
    GenerationState,
    build_chat_input,
    generate_steered,
    hf_generation_config,
    hf_unneutralizable_processors,
)
from drowse.model.chat import ChatTraits
from drowse.readout.capture_buffer import HiddenCapture
from drowse.steering.hooks import SteeringManager
from drowse.steering.composer import GateSources, SteeringComposer, SteeringScope
from drowse.session.registry import ArtifactRegistry, ModelBinding
from drowse.session.neutral import NeutralBaseline
from drowse.errors import (
    ConcurrentExtractionError,
    ConcurrentGenerationError,
    ManifoldNotRegisteredError,
)
from drowse.generation.token_callback import (
    StepTokenCallback,
    TokenCallback,
    TokenConsumer,
    TokenConsumerOptions,
    consumer_options,
)
from drowse.model.scene import TurnGrammar
from drowse.generation.render import PromptRenderer
from drowse.session.authored_capture import AuthoredPromptCapture, plan_authored_capture
from drowse.events import LoomMutated
from drowse.loom.errors import InvalidNodeOperationError
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, InheritSystemPrompt
from drowse.loom.recipe import (
    CastMember,
    ComposedCall,
    Recipe,
    compose_call,
    derive_seed_schedule,
)
from drowse.loom.tree import LoomTree
from drowse.loom.diff import node_diff
from drowse.loom.ops import (
    append_turn,
    begin_generation_turn,
    plan_fork,
    plan_prefill,
)
from drowse.model.arch import get_layers, get_model_info
from drowse.model.load import load_model
from drowse.readout.types import (
    AGG_TAIL_DEPTH,
    InstrumentFamily,
    InstrumentPlan,
    ReadRequest,
    ScalarReading,
    as_probe_reading,
    reading_axis0,
)
from drowse.readout.measurements import build_measurements
from drowse.readout.monitor import Monitor
from drowse.readout.geometry import GeometryInstrument
from drowse.lens.instrument import LensInstrument
from drowse.sae.instrument import SaeInstrument
from drowse.sae.runtime import SaeRuntime
from drowse.session.replay import replay_token
from drowse.store.bundled_probes import bootstrap_probe_roster
from drowse.geometry.profile import Profile
from drowse.generation.results import (
    GenerationResult,
    ProbeReading,
    RunSet,
    TokenAlt,
    TokenEvent,
)
from drowse.generation.sampling import SamplingConfig
from drowse.steering.terms import Steering
from drowse.geometry.manifold import Manifold

if TYPE_CHECKING:
    from drowse.readout.protocol import Instrument
    from drowse.generation.scoring import ChoiceScores
    from drowse.generation.token_payloads import TokenProbePayload
    from drowse.store.templates import TemplateFolder
from drowse.steering.triggers import Trigger

_log = logging.getLogger(__name__)

_DEFAULT_READOUT_TOP_K = 8


class GenerationStream(Protocol):
    """Current streaming iterator contract returned by :meth:`generate_stream`."""

    def __iter__(self) -> "GenerationStream": ...

    def __next__(self) -> TokenEvent: ...

    def close(self) -> None: ...

    @property
    def result(self) -> GenerationResult | None: ...

# Tail-ring depth for aggregate-only capture (probes attached, no per-token
# consumer): how many trailing hidden slices to retain so finalize can pool the
# last *content* token after walking back past trailing special tokens (EOS,
# end-of-turn). 8 comfortably covers any chat template's trailing-marker run.
# Canonical value lives in the instrument-plan vocabulary so families can
# declare the ring depth they demand.
_AGG_TAIL_DEPTH = AGG_TAIL_DEPTH

# Floor on the shared token prefix ``generate_batch`` will KV-cache.  Below
# this the saved prefill is in the noise and the double-render + cache-management
# overhead isn't worth it; the prefix must also leave a non-empty suffix per row
# (``_try_prefix_cache_hit`` requires it), so the cached length is capped at
# ``min_row_len - 1``.
_PREFIX_CACHE_MIN_TOKENS = 16


@dataclass(slots=True)
class _SerialGenerationJob:
    input: Any
    steering: Any = None
    sampling: SamplingConfig | None = None
    raw: bool = False
    thinking: bool | None = None
    on_token: TokenCallback | None = None
    parent_node_id: str | None = None
    recipe_override: Recipe | str | None = None
    system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT
    grid: dict[str, Any] | None = None
    gen_seat: str = "assistant"


class _SessionStopCriteria(StoppingCriteria):
    """Bridge ``DrowseSession.stop()`` into ``transformers.generate``."""

    def __init__(self, state: GenerationState) -> None:
        self._state = state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | tuple[torch.FloatTensor, ...] | None,
        **kwargs: Any,
    ) -> Any:
        del scores, kwargs
        stop = 1 if self._state.is_stop_requested() else 0
        return input_ids.new_full((int(input_ids.shape[0]),), stop).bool()


def _run_serial_generation_jobs(
    session: Any,
    jobs: list[_SerialGenerationJob],
    *,
    stateless: bool,
    kind: str,
    on_result: Callable[[int, GenerationResult, dict[str, Any]], None] | None = None,
    stop_between_jobs: bool = False,
) -> RunSet:
    """Run generation jobs through the shared serial executor.

    This is still a one-row-at-a-time runner; centralizing it gives the
    eventual true batched decode path one place to branch from, instead of
    keeping separate loops in fan, batch, and sweep surfaces.
    """
    results: list[GenerationResult] = []
    node_ids: list[str | None] = []
    grid_rows: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        result = session._generate_core(
            job.input,
            steering=job.steering,
            sampling=job.sampling,
            stateless=stateless,
            raw=job.raw,
            thinking=job.thinking,
            on_token=job.on_token,
            parent_node_id=job.parent_node_id,
            recipe_override=job.recipe_override,
            system_prompt=job.system_prompt,
            gen_seat=job.gen_seat,
        )
        row = dict(job.grid or {})
        results.append(result)
        node_ids.append(session.tree.active_node_id if not stateless else None)
        grid_rows.append(row)
        if on_result is not None:
            on_result(idx, result, row)
        if stop_between_jobs and session._gen_state.is_stop_requested():
            break
    return RunSet(results, node_ids=node_ids, grid=grid_rows, kind=kind)


PROBE_CATEGORIES = [
    "epistemic",
    "alignment",
    "register",
    "cultural",
]
MIN_ELAPSED_FOR_RATE = 0.1


class GenState(IntEnum):
    """Lifecycle phases of a single generation call.

    A typed phase rather than a bare in-flight flag, so the five-handle
    teardown (lock, steering scope CM, capture, monitor live, threading lock)
    is self-documenting.

    Transitions live in :meth:`DrowseSession._generation_transaction` and
    :meth:`DrowseSession._generate_core`:

    - ``IDLE`` → ``PREAMBLE``: lock acquired, re-entry guard passed.
    - ``PREAMBLE`` → ``RUNNING``: capture attached, monitor ``begin_live``,
      steering :class:`TriggerContext` reset; ``generate_steered`` enters.
    - ``RUNNING`` → ``FINALIZING``: inner ``finally`` ran — capture detached,
      steering scope exited; monitor ``end_live`` / lock release pending.
    - ``FINALIZING`` → ``IDLE``: outer ``finally`` ran.

    The threading ``_gen_lock`` primitive stays alongside this enum — the
    enum makes the state field typed and self-documenting; the lock still
    enforces single-flight at the Python level.
    """

    IDLE = 0
    PREAMBLE = 1
    RUNNING = 2
    FINALIZING = 3


@dataclass(frozen=True, slots=True)
class PreparedCall:
    """The per-call generation controls, normalized before any model work.

    Produced once by :meth:`DrowseSession._prepare_generation_call` and read by
    both the ordinary and the batched entry points.  Named fields rather than a
    positional tuple: several members share a type (two ``int | None``, two
    ``float``), so a reordering would rebind silently at every unpack site.
    """

    steering_obj: "Steering | None"
    use_thinking_req: bool
    gen_config: GenerationConfig
    lp_count: int | None
    seed: int | None
    stop_list: list[str] | None
    logit_bias: dict[int, float] | None
    presence_penalty: float
    frequency_penalty: float
    logprobs_list: list[Any] | None


@dataclass(frozen=True, slots=True)
class ReadDemand:
    """What this generation's consumers need from the read plane.

    Resolved once per generation by
    :meth:`DrowseSession._resolve_read_demand` and consumed by
    :meth:`DrowseSession._begin_capture`, which chooses the physical capture
    retention from it.  Every derived predicate — notably
    ``per_token_full_consumer`` — is computed in the resolver and only read
    here, so the caller and the capture planner cannot form different opinions
    about the same generation.  The defaults are the "everything, nothing
    gated" demand a bare capture attach wants.

    - ``need_per_token`` — anything consumes a per-token reading (a probe
      gate, a loom token row, a live-scores client).  False
      means aggregate-only: a bounded tail ring, no per-token scoring, one
      pooled read at finalize.
    - ``per_token_full_consumer`` — a per-token consumer wants the FULL
      roster, not just a gate's scalars.  False with ``need_per_token`` true
      is the gating-only case.
    - ``gating_only_probes`` — the gated probe names when gating is the sole
      per-token consumer; the step sink then scores just that subset and the
      big-K roster's per-token nearest-distance work is skipped.  When
      ``final_probe_aggregate`` is set the tail ring is kept alongside, so
      finalize still pools the full roster once.
    - ``lean_per_token`` — the live consumers read only axis-0 coords, so
      per-token scoring drops nearest / assignment / per-layer work.
    - ``capture_prompt`` — authored prompt rows are scored this generation,
      which forces transient capture hooks over the persistent buffers.
    """

    need_per_token: bool = True
    per_token_full_consumer: bool = True
    gating_only_probes: "set[str] | None" = None
    gating_probe_keys: "set[str] | None" = None
    lens_gating_probe_keys: "set[str] | None" = None
    sae_gating_probe_keys: "set[str] | None" = None
    lean_per_token: bool = False
    final_probe_aggregate: bool = True
    live_lens_active: bool = True
    live_sae_active: bool = True
    capture_prompt: bool = False


class _GenerationTransaction:
    """The mutable state :meth:`DrowseSession._generation_transaction` owns.

    ``steering_cm`` is whatever internal steering scope the body entered and
    has not popped itself; the transaction pops the remainder on the way out.
    ``failed`` records that the body is unwinding, which decides whether a
    teardown ``Exception`` may mask the original failure.
    """

    __slots__ = ("failed", "steering_cm")

    def __init__(self) -> None:
        self.steering_cm: Any = None
        self.failed: bool = False



class DrowseSession:
    """Unified backend for activation steering, monitoring, and generation.

    Steering resources are registered or resolved lazily, then applied through
    the per-call ``steering`` expression on ``generate``/``generate_stream``.
    Eager sessions install transient steering/capture hooks for one transaction;
    compiled sessions may retain branchless offset/capture hooks whose buffers
    are zeroed or repopulated at generation boundaries.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        dtype: torch.dtype | str | None = None,
        quantize: str | None = None,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        dls: bool = True,
        compile: bool = False,
        compile_mode: str | None = None,
        cuda_graphs: bool = False,
        trust_remote_code: bool | None = None,
        return_top_k: int = 0,
        on_progress: Callable[[str], None] | None = None,
    ) -> "DrowseSession":
        """Load a HF model + tokenizer and return a fully initialized session.

        This is the primary entry point for library users; it owns all the
        HF-loading heavy lifting. To wrap an already-loaded model use the
        plain ``__init__(model, tokenizer, ...)`` form.

        ``~`` / ``|`` projection terms always materialize through the
        closed-form LEACE projector against the per-model whitener (it
        provably erases linearly-decodable information along ``onto`` from
        ``base``); there is no Euclidean path and no metric knob.

        ``dls`` controls centered discriminative layer selection at
        extraction time. When enabled, layers whose fitted poles do not
        straddle the neutral baseline are omitted.

        ``compile`` (default ``False``) opts in to ``torch.compile`` on
        CUDA or MPS. Persistent offset and capture hooks keep unsteered and
        static-affine generations, including live probes, on the compiled
        path; dynamic or transient-hook work falls back to eager. CPU skips
        compilation.

        ``cuda_graphs`` (default ``False``) opts in to
        ``transformers.StaticCache`` + ``torch.compile(mode=
        "reduce-overhead")`` on CUDA-supported architectures.  Static
        K/V buffers across decode steps mean inductor can capture CUDA
        graphs internally — an additional 1.5–2.5× decode tok/s on top
        of the kernel-fusion win from ``compile=True``.  Off by default
        for the same reason as ``compile``; setting this without
        ``compile=True`` has no effect.  Auto-skipped on MPS/CPU and on
        architectures whose StaticCache construction fails.

        ``compile_mode`` (default ``None`` → auto-select) overrides the
        torch.compile mode.  When None, the session picks
        ``"reduce-overhead"`` if ``cuda_graphs`` is on (paired with
        StaticCache for full graph capture) and ``"default"`` otherwise
        (kernel fusion only).  Pass an explicit value to force a
        specific mode regardless of the cuda_graphs decision.

        ``on_progress`` receives one line per construction step — the
        model-load status lines, the neutral/whitener forward pass, and each
        bundled concept fit the default probe roster still needs.  Unset (the
        library default) constructs silently; the CLI passes a printing
        callback so a first ``drowse serve`` narrates instead of hanging.

        ``trust_remote_code`` defaults off unless DROWSE_TRUST_REMOTE_CODE is
        enabled. Explicit True permits repository Python; explicit False
        overrides the environment. Only opt in for code you trust.
        """
        # Load WITHOUT compile so the StaticCache probe runs against the
        # bare nn.Module (probing through the OptimizedModule wrapper
        # forwards correctly via __getattr__, but avoiding the wrapper
        # at probe time keeps the failure mode "StaticCache constructor
        # raised" rather than "compile + probe interaction").  We then
        # decide ``compile_mode`` based on the probe outcome and apply
        # ``torch.compile`` manually below.  Ordering matters: committing
        # to ``"reduce-overhead"`` from the *requested* ``cuda_graphs=True``
        # before the probe can veto leaves an arch-failed session running
        # DynamicCache under a graph-capture compile mode.
        model, tokenizer = load_model(
            model_id,
            quantize=quantize,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            on_progress=on_progress,
        )

        cg_supported = False
        _cg_reason: str | None = None
        device_obj = next(model.parameters()).device
        if cuda_graphs:
            from drowse.generation.static_cache import is_cuda_graphs_supported
            cg_supported, _cg_reason = is_cuda_graphs_supported(
                model, device_obj,
            )

        # Resolve compile_mode now that the probe outcome is known.
        # ``"reduce-overhead"`` captures CUDA graphs internally for
        # fixed-shape inference regions and only pays off paired with
        # StaticCache; ``"default"`` is the kernel-fusion-only fallback
        # that composes cleanly with DynamicCache.  An explicit
        # ``compile_mode`` arg overrides regardless of probe outcome —
        # power users (benching, debugging) can force a mismatch on
        # purpose.
        effective_compile_mode = compile_mode
        if effective_compile_mode is None:
            effective_compile_mode = (
                "reduce-overhead" if cg_supported else "default"
            )

        # Apply compile manually with the resolved mode.  Routes through
        # ``compile_with_probe`` so an architecture-specific inductor /
        # Triton bug surfaces at load time as a caught warning + eager
        # fallback, rather than as a segfault on the user's first token.
        #
        # MPS is now eligible (not just CUDA): inductor's MPS backend fuses the
        # per-layer kernels and amortizes dispatch, which is the dominant decode
        # cost on Metal — measured ~1.7x on gemma-3-4b paired with StaticCache.
        # CUDA-graph capture (``reduce-overhead``) stays CUDA-only, so on MPS
        # ``effective_compile_mode`` is ``"default"`` (``cg_supported`` is False
        # there) — fusion without graph capture, which composes with the
        # transient steering hooks and StaticCache the decode loop installs.
        offset_buffers: dict[int, Any] = {}
        offset_handles: list[Any] = []
        capture_buffers: dict[int, Any] = {}
        capture_handles: list[Any] = []
        if compile and device_obj.type in ("cuda", "mps"):
            from drowse.readout.capture_buffer import (
                install_persistent_capture_hooks,
            )
            from drowse.steering.hooks import install_persistent_offset_hooks
            from drowse.model.compile import compile_with_probe
            # Attach the persistent branchless steering hooks BEFORE
            # compile so they ride inside the captured graph (post-compile hook
            # changes are not retraced — ``skip_nnmodule_hook_guards``).  The
            # offset hooks (``add_(offset)``) carry static-affine steering and
            # update their buffers in place per gen, so the compiled fast path
            # sees a stable hook topology and never recompiles. Persistent
            # capture hooks (``copy_`` the last-token slice) are installed only
            # when the construction-time probe roster is non-empty/default; an
            # explicit ``probes=[]`` session takes a no-capture compiled mode and
            # later ad-hoc probes fall back to transient capture. ``hidden_size``
            # falls back to the text sub-config for multimodal wrappers
            # (Gemma-3/4).
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None and hasattr(model.config, "get_text_config"):
                hidden_size = model.config.get_text_config().hidden_size
            if hidden_size is not None:
                model_dtype = next(model.parameters()).dtype
                layers_ml = get_layers(model)
                offset_buffers, offset_handles = install_persistent_offset_hooks(
                    layers_ml, int(hidden_size), device_obj, model_dtype,
                )
                if probes is None or bool(probes):
                    capture_buffers, capture_handles = (
                        install_persistent_capture_hooks(
                            layers_ml, int(hidden_size), device_obj, model_dtype,
                        )
                    )
            if on_progress is not None:
                on_progress(
                    f"Compiling model with torch.compile"
                    f"(mode={effective_compile_mode!r})..."
                )
            model = compile_with_probe(
                model, tokenizer, device_obj,
                mode=effective_compile_mode,
            )
        elif compile:
            _log.info(
                "compile=True but device=%s — skipping torch.compile "
                "(supported only on CUDA / MPS)",
                device_obj.type,
            )
            if on_progress is not None:
                on_progress(
                    f"compile=True but device={device_obj.type} — skipping "
                    f"torch.compile (supported only on CUDA / MPS)"
                )

        # ``__init__`` consults the same probe helper, but the helper now
        # caches by the underlying module id (survives the torch.compile
        # wrapper via ``_orig_mod``), so this second call is a dictionary
        # lookup rather than another StaticCache(max_cache_len=1)
        # allocation.
        session = cls(
            model,  # pyright: ignore[reportArgumentType]  # may be torch.compile OptimizedModule wrapping PreTrainedModel
            tokenizer,
            probes=probes,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            dls=dls,
            cuda_graphs=cuda_graphs,
            return_top_k=return_top_k,
            on_progress=on_progress,
        )
        # Adopt the persistent offset + capture hooks iff compile actually stuck;
        # if it fell back to eager, drop them (their add_(0) / copy_ would be pure
        # overhead with no compiled graph to ride).
        if offset_handles or capture_handles:
            if session._compiled:
                session._steering.adopt_compiled_offsets(
                    offset_buffers, offset_handles,
                )
                session._capture_buffers = capture_buffers
                session._capture_handles = capture_handles
            else:
                for h in offset_handles:
                    h.remove()
                for h in capture_handles:
                    h.remove()
        return session

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        dls: bool = True,
        cuda_graphs: bool = False,
        return_top_k: int = 0,
        on_progress: Callable[[str], None] | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        # The tokenizer's chat facts, read once: stop ids (adopted as the
        # model's generation default so HF ``generate`` stops where the decode
        # loop does), thinking delimiters, the render cache.
        self._chat = ChatTraits(tokenizer, model)
        self._chat.adopt_eos(model)
        self._layers = get_layers(self._model)
        self._model_info = get_model_info(self._model, self._tokenizer)

        first_param = next(self._model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype

        self.config = GenerationConfig(
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # Session-level default for SamplingConfig.return_top_k.
        # Per-call value > 0 wins; per-call K=0 (the
        # SamplingConfig default) inherits this stored value via the
        # composition in ``_generate_core``. The sampler bounds this by
        # the actual candidate pool when capturing alternatives.
        if return_top_k < 0:
            return_top_k = 0
        self._default_return_top_k: int = int(return_top_k)
        self._steering = SteeringManager()
        # CUDA-graphs / StaticCache routing.  Probe
        # support once at construction so the per-generation hot path
        # only consults a boolean.  Off when (a) user opted out, (b)
        # device != cuda, or (c) the model's StaticCache constructor
        # raises (architecture-specific quirks).  The fallback reason
        # is logged once via :func:`drowse.generation.static_cache.warn_once`
        # which dedupes on ``id(model)``, so when ``from_pretrained``
        # already probed for its compile_mode decision and we re-probe
        # here, the user only sees one message.
        cuda_graphs_requested = bool(cuda_graphs)
        self._cuda_graphs_active: bool = False
        if cuda_graphs_requested:
            from drowse.generation.static_cache import (
                is_cuda_graphs_supported, warn_once,
            )
            supported, reason = is_cuda_graphs_supported(
                self._model, self._device,
            )
            if supported:
                self._cuda_graphs_active = True
            elif reason is not None:
                warn_once(self._model, reason)
        # ``torch.compile`` left an ``_orig_mod`` wrapper iff it stuck (the
        # probe in ``from_pretrained`` falls back to the eager model on failure).
        # The compiled fast path wants StaticCache so inductor traces one fixed
        # decode shape instead of re-specializing as a DynamicCache grows;
        # ``_static_cache_active`` gates that, device-agnostically (CUDA *or*
        # MPS), distinct from the CUDA-only ``_cuda_graphs_active`` graph-capture
        # signal.  Either one (or a future StaticCache-only opt-in) enables the
        # preallocated K/V path for fast-path-eligible steering.
        self._compiled: bool = hasattr(self._model, "_orig_mod")
        # CUDA graph trees are thread-local in PyTorch. ``from_pretrained``
        # warms them on this construction thread; synchronous ``generate`` can
        # replay them here, while ``generate_stream`` must route its worker to
        # eager execution. Non-graph compiled modes are cross-thread safe.
        self._compile_owner_thread: int = threading.get_ident()
        self._static_cache_active: bool = False
        if self._compiled or self._cuda_graphs_active:
            from drowse.generation.static_cache import (
                is_static_cache_supported, warn_once,
            )
            sc_supported, sc_reason = is_static_cache_supported(
                self._model, self._device,
            )
            if sc_supported:
                self._static_cache_active = True
            elif sc_reason is not None:
                warn_once(self._model, sc_reason)

        # Persistent compile-clean *capture* buffers + hook handles, adopted from
        # ``from_pretrained`` when compile stuck.  The always-on hooks
        # ``copy_`` each layer's last-token slice into ``_capture_buffers[L]``
        # every forward (fused into the compiled graph); a probed gen on the
        # compiled path reads them post-forward via ``HiddenCapture.ingest_persistent``
        # instead of registering transient capture hooks that would graph-break.
        # Empty unless a compiled CUDA/MPS session adopted them.
        self._capture_buffers: dict[int, torch.Tensor] = {}
        self._capture_handles: list[Any] = []
        # Per-gen flags: ``_compiled_clean_eligible`` (set early in
        # ``_generate_core``) means this gen *can* take the compiled clean path —
        # compiled CUDA/MPS, static cache, capture buffers present, not return_hidden —
        # provided steering also lowers to offsets.  ``_capture_state.persistent``
        # (set in ``_begin_capture``) means the active capture rides the persistent
        # buffers, so the decode loop wires ``ingest_persistent`` as its step
        # callback and the routing keeps the compiled module.
        self._compiled_clean_eligible: bool = False

        # One session-resident StaticCache for ordinary (non-prefix-cache)
        # generations.  CUDA graphs guard on the K/V tensors' identities, so a
        # fresh cache per request forces a full Dynamo recompile even when every
        # shape is unchanged.  Grow this cache only when needed, then reset and
        # reuse it in place across serialized session generations.
        self._generation_static_cache: object | None = None
        self._generation_static_cache_len: int = 0

        # Prompt rendering: model type, the lazily validated scene grammar,
        # system-prompt resolution, the loom-path render, seat stop strings.
        # The steering role is read lazily from the composer (built below).
        self._renderer = PromptRenderer(
            tokenizer=tokenizer,
            model=model,
            traits=self._chat,
            device=self._device,
            history=lambda: self.tree,
            default_system_prompt=lambda: self.config.system_prompt,
            steering_role=lambda: self._steering_composer.active_role,
        )

        # Synchronous event bus.  Emits on extraction, steering enter/exit,
        # probe scoring, generation start/finish.  Subscribers run on the
        # emit thread — async consumers must hop via call_soon_threadsafe
        # inside their callback.
        self.events: EventBus = EventBus()

        # Transient per-token hidden-state capture — attached around
        # generate_steered when probes are active so scoring happens
        # without a second forward pass.
        self._capture = HiddenCapture()
        # Per-gen capture configuration — one :class:`CaptureMode` plus the
        # orthogonal ``persistent`` flag (and the gated subset / keys for
        # ``GATING_SUBSET``).  One enum rather than a set of correlated
        # booleans, so illegal combinations are unrepresentable.  Set wholesale
        # in ``_begin_capture`` and reset on teardown; the ``_score_*``
        # dispatch keys off ``mode``.  Each mode and its memory/scoring
        # trade-off is documented on :class:`CaptureMode`.
        self._capture_state: CaptureState = CaptureState()
        # Per-token reading buffers the modes append into (the *data*, distinct
        # from the *config* above).  ``_incremental_readings`` holds the full /
        # lean per-token ``ProbeReading`` rows (INCREMENTAL / LEAN_INCREMENTAL);
        # ``_incremental_gate_scores`` holds the gated-subset scalar dicts
        # (GATING_SUBSET).  ``_finalize_generation`` builds (aggregate, per_token)
        # from these instead of rescoring captured hidden states.  Reset at each
        # ``_begin_capture`` and on teardown.
        self._incremental_readings: list[dict[str, ProbeReading]] = []
        self._incremental_gate_scores: list[dict[str, float]] = []

        # Reentrant — ``_generate_core`` acquires it for the whole gen,
        # then enters an internal ``self.steering(...)`` scope whose push
        # re-acquires the lock from the same thread.  The single-in-flight invariant is enforced
        # by the ``_gen_phase`` state check, not by the lock owner
        # count, so RLock is correct: cross-thread re-entry blocks
        # (which is the property fix #4 wants), same-thread re-entry
        # passes (which keeps the internal steering scope from
        # deadlocking against itself).
        self._gen_lock = threading.RLock()
        # ``_gen_lock`` doubles as the process-wide **exclusive-GPU** lock:
        # PyTorch's MPS backend serializes every device op through one global,
        # NOT-thread-safe command buffer, so two threads committing it at once
        # aborts the process ("commit an already committed command buffer").
        # Every model op (fit / extract / generate) already holds it for its
        # duration; read-side GPU work (the analytics CPU snapshots,
        # ``add_probe``) acquires it non-blocking and defers rather than race.
        self._gen_state = GenerationState()
        # Typed lifecycle phase of the current generation (or ``IDLE`` between
        # gens).  Re-entry guard between preamble and finalize: prevents a
        # pending-action dispatch from double-attaching capture/steering
        # hooks and leaking them.  See :class:`GenState` for transitions.
        # Distinct from ``_gen_state`` (the per-call ``GenerationState``
        # holding token queue, finish_reason, etc.) — the names are close
        # because the enum field is the *session*'s view of state, while
        # ``_gen_state`` is the *generator's*.
        self._gen_phase: GenState = GenState.IDLE

        # Conversation state lives in a :class:`LoomTree`. The active path
        # through the tree is what the model sees as context. Generation routes
        # through ``tree.add_user_turn``
        # / ``tree.begin_assistant`` / ``tree.finalize_assistant``.  The
        # tree is in-memory only — there is no automatic cross-session
        # persistence. ``LoomTree.save`` / ``LoomTree.load`` are the explicit
        # save/restore path.
        self.tree = LoomTree(
            events=self.events,
            model_id=self._model_info["model_id"],
        )
        self._joint_logprob_cache: dict[tuple[str, str], Any] = {}

        def _invalidate_tree_analysis_caches(event: Any) -> None:
            if isinstance(event, LoomMutated) and event.op in {
                "edit",
                "delete",
                "reset",
                "finalize_assistant",
            }:
                self._joint_logprob_cache.clear()

        self.events.subscribe(_invalidate_tree_analysis_caches)

        self._last_result: GenerationResult | None = None
        self._last_per_token_scores: dict[str, list[float]] | None = None
        self._last_token_probe_payload: dict[str, Any] | None = None
        # The merged per-family :class:`ProbeReading` dict for the latest token
        # (the union of all three families' slots).  The measurement envelope
        # on ``_last_token_probe_payload`` carries the serialized readings; this
        # keeps the live objects so ``generate_stream`` can populate
        # ``TokenEvent.probe_readings`` (the compat vendor-extension / traits
        # channel) without reconstructing them from the envelope.
        # The token tap's payload, retained UN-PROJECTED: the cross-family
        # ``ProbeReading`` dict (``all_readings``) is the compat channel for
        # ``TokenEvent.probe_readings``, so it is built at the stream
        # consumer, not per token in the tap.
        self._last_token_payload: "TokenProbePayload | None" = None

        # Probe content-hash cache for transcript export / replay.
        # Keyed by probe name → sha256 hex of the baked tensor
        # bytes (concatenated layer order).  Invalidated by
        # :meth:`add_probe` / :meth:`remove_probe`; rebuilt lazily by
        # :meth:`_probe_hash`.
        self._probe_hash_cache: dict[str, str] = {}

        # Ensure bundled concepts are materialized in the user cache and the
        # selector index reflects them.  The probe-roster bootstrap does
        # this transitively via ``load_default_manifolds``, but is skipped
        # entirely when ``probes=[]`` — leaving freshly-added bundled concepts
        # (e.g. via updating package-data bundled manifolds) invisible to the
        # selector layer for the rest of the session.  Calling explicitly here
        # keeps the invariant intact regardless of probe-loading config; the
        # call is cheap when up-to-date (per-home process guard).  The io
        # entry point owns the templates-before-manifolds ordering (a bundled
        # manifold may ``template_ref`` a bundled ``default/<name>`` template,
        # and its fit resolves that ref) and drops the stale selector index
        # itself, so the bare-name resolver sees every bundled node label.
        from drowse.store.bundled import materialize_bundled_artifacts

        materialize_bundled_artifacts()

        # Bootstrap probes
        probe_categories = PROBE_CATEGORIES if probes is None else probes

        # Prefix KV cache (opt-in, off by default).  Populated by
        # :meth:`cache_prefix`; consumed by :meth:`_generate_core` when the
        # incoming ``input_ids`` start with the cached prefix.  Shape:
        # :class:`~drowse.generation.prefix_cache.PrefixCacheEntry`, which owns
        # the live HF cache the prefix-prefill forward produced and rewinds it
        # to ``prefix_len`` after each reuse; StaticCache entries additionally
        # carry capacity so oversized generations miss safely instead of
        # writing past the preallocated buffers.  Invalidated
        # on any state change that would alter the cached prefix's hidden-state
        # semantics: prefill-steering push/pop, probe install/remove,
        # registry mutation.
        self._prefix_cache: PrefixCacheEntry | None = None

        # -- collaborators --------------------------------------------------
        # Artifact registry: profiles + loaded manifolds; its mutations drop
        # the prefix cache and refresh monitor probes on a manifold swap.
        self._registry = ArtifactRegistry(ModelBinding(
            model=self._model, tokenizer=self._tokenizer,
            model_id=self.model_id, device=self._device, dtype=self._dtype,
        ))
        self._registry.on_change(lambda: self._invalidate_prefix_cache())
        self._registry.on_swap(self._refresh_swapped_probes)
        self._registry.set_probe_folder(self._fold_profile_probe)
        self._registry.set_atom_resolver("jlens", lambda atom: self._lens_instrument.register_direction(atom))
        self._registry.set_atom_resolver("sae", lambda atom: self._sae_instrument.register_direction(atom))
        # Neutral means + whitener, built lazily from one neutral load.
        self._neutral = NeutralBaseline(
            model=self._model, tokenizer=self._tokenizer, layers=self._layers,
            model_id=self.model_id, exclusive=self._model_exclusive,
            steered=lambda: self._steering.plan is not None,
        )
        self._neutral.on_whitener(self._adopt_whitener)
        # Corpus generation and the extract/fit/bake glue.  Wired before the
        # probe bootstrap below, which fits stale bundled axes through ``fit``;
        # ``dls`` gates every flat fit's straddle-pruning.
        self._corpus = CorpusGenerator(
            lambda *args, **kwargs: self._model.generate(*args, **kwargs),  # pyright: ignore[reportCallIssue]  # transformers stubs don't expose generate
            self._tokenizer, device=self._device,
            traits=self._chat, scene=lambda: self.scene_grammar,
            model_type=self._resolved_model_type(),
        )
        self._extraction = Extraction(
            self, self.events, self._corpus,
            exclusive=self._model_exclusive,
            assert_unsteered=self._assert_unsteered_artifact_operation,
            adopt=self._registry.adopt_fitted,
            evict=self._registry.evict_artifact,
            register_profile=self._registry.put_profile,
            dls=dls,
        )
        self._steering_composer = SteeringComposer(
            artifacts=self._registry, neutral=self._neutral,
            manager=self._steering, layers=self._layers,
            device=self._device, dtype=self._dtype,
            lock=self._gen_lock, events=self.events,
            generation_phase=self._mid_generation_phase,
            on_prefill_steering_change=lambda: self._invalidate_prefix_cache(),
            gates=GateSources(
                monitor=lambda: self._monitor,
                capture=lambda: self._capture,
                capture_state=lambda: self._capture_state,
                incremental_readings=lambda: self._incremental_readings,
                incremental_gate_scores=lambda: self._incremental_gate_scores,
                instruments=lambda: self.instruments,
                lens_gate_scalars=lambda keys, *, step_id: (
                    self._lens_instrument.current_run.gate_scalars(
                        step_id, None, keys,
                    )
                ),
                sae_gate_scalars=lambda keys, *, step_id: (
                    self._sae_instrument.current_run.gate_scalars(
                        step_id, None, keys,
                    )
                ),
            ),
        )
        # One unified Monitor for every probe shape (flat axes, flat discover
        # fits, curved manifolds), then the three read instruments over it.
        self._monitor = Monitor(n_layers=len(self._layers))
        self._geometry_instrument = GeometryInstrument(self)
        # The J-lens family: resident lens, fit lifecycle, probes, live
        # readout.  Its dependencies are explicit; ``replay_host`` serves
        # only the token replay's prompt rebuild.
        self._lens_instrument = LensInstrument(
            model=self._model,
            tokenizer=self._tokenizer,
            model_id=self.model_id,
            device=self._device,
            n_layers=len(self._layers),
            capture=self._capture,
            exclusive=self._model_exclusive,
            capture_hidden=partial(
                _capture_all_hidden_states, self._model, self._layers,
            ),
            pooled_slice=self._pooled_aggregate_slice,
            registry=self._registry,
            on_change=self._on_lens_change,
            resolve_profile=self.ensure_profile_registered,
            assert_unsteered=self._assert_unsteered_artifact_operation,
            replay_host=self,
        )
        # The SAE family: resident dictionary + Neuronpedia metadata (the
        # runtime), probes, live discovery and token replay.  ``sae/<id>``
        # directions are registry profiles; the session reference serves only
        # the capture plane.
        self._sae_instrument = SaeInstrument(
            self,
            SaeRuntime(
                self,
                exclusive=self._model_exclusive,
                assert_unsteered=self._assert_unsteered_artifact_operation,
            ),
            replay=partial(replay_token, self),
            put_direction=self._registry.put_profile,
            evict_directions=lambda: self._registry.evict_prefix("sae/"),
            forget_probe_hashes=self._forget_probe_hashes,
        )

        if probe_categories:
            # The default probe roster: fit-or-load each tagged concept axis
            # (and, for the default ``probes=None``, attach the already-fitted
            # bundled multi-node manifolds).  The neutral baseline comes first:
            # fitting needs the whitener.  An explicit ``probes=[]`` skips both.
            self._neutral.ensure(on_progress=on_progress)
            roster = bootstrap_probe_roster(
                probe_categories, model_id=self.model_id,
                load=self._registry.ensure_manifold_loaded,
                fit=lambda folder, progress: self.fit(folder, on_progress=progress),
                include_fitted_defaults=probes is None,
                on_progress=on_progress,
            )
            for name, manifold in roster.items():
                self._monitor.add_probe(name, manifold)

        self._sae_instrument.restore_active()

    # -- ModelHandle protocol surface (consumed by ManifoldExtractionPipeline) --

    @property
    def model(self) -> torch.nn.Module:
        """Live HF model.  Part of the :class:`~drowse.fit.pipeline.ModelHandle` protocol."""
        return self._model

    @property
    def tokenizer(self):
        """Live HF tokenizer.  Part of the :class:`~drowse.fit.pipeline.ModelHandle` protocol."""
        return self._tokenizer

    @property
    def chat_traits(self) -> ChatTraits:
        """The tokenizer's chat facts (stop ids, thinking delimiters, renders)."""
        try:
            return self._chat
        except AttributeError:  # a session assembled without ``__init__``
            self._chat = ChatTraits(self._tokenizer, self._model)
            return self._chat

    @property
    def device(self) -> torch.device:
        """Model device.  Part of the :class:`~drowse.fit.pipeline.ModelHandle` protocol."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Model dtype.  Part of the :class:`~drowse.fit.pipeline.ModelHandle` protocol."""
        return self._dtype

    @property
    def layers(self):
        """Layer-list accessor.  Part of the :class:`~drowse.fit.pipeline.ModelHandle` protocol.

        Returns whatever ``get_layers`` produced — typically an
        ``nn.ModuleList``, list-like enough for the downstream
        consumers (the extraction pipeline and hooks).
        """
        return self._layers

    # -- State queries --

    @property
    def model_info(self) -> dict[str, Any]:
        return dict(self._model_info)

    @property
    def model_id(self) -> str:
        return self._model_info.get("model_id", "unknown")

    def analytics_names(self) -> list[str]:
        """Names available to read-side analytics: registered steering
        vectors plus attached probes (a vector wins a same-named probe,
        matching the correlation pool's dedup).  Pure metadata — no GPU.

        A multi-node / curved probe (a rank-``R`` fit like ``personas`` or
        ``emotions``) is excluded: the direction-cosine analytics fold every
        name to a single steering direction, which only a folded affine
        ``R = 1`` manifold has."""
        from drowse.geometry.fold import is_foldable_vector_manifold

        names = set(self._registry.profiles)  # registered vectors are always R=1
        # One locked roster snapshot — this serves un-locked analytics
        # polling, where a raw per-name ``manifolds`` rebuild tears under
        # a concurrent detach.
        for pname, manifold in self._geometry_instrument.manifolds().items():
            if pname in names:
                continue  # a registered vector wins the same-named probe
            if is_foldable_vector_manifold(manifold):
                names.add(pname)
        return sorted(names)

    def _live_direction_tensors(self, name: str) -> "dict[int, torch.Tensor] | None":
        """Live per-layer direction tensors for *name* (may be device-
        resident — callers must hold ``_gen_lock``).  A registered steering
        vector wins over a same-named probe; otherwise the folded view of an
        attached probe's manifold.  ``None`` for a multi-node / curved probe,
        which has no single direction to fold."""
        prof = self._registry.profiles.get(name)
        if prof is not None:
            return dict(prof)
        manifold = self._monitor.manifolds.get(name)
        if manifold is not None:
            from drowse.geometry.fold import (
                folded_directions,
                is_foldable_vector_manifold,
            )
            if is_foldable_vector_manifold(manifold):
                return folded_directions(manifold)
        return None

    def analytics_profile(self, name: str) -> "Profile | None":
        """CPU-resident snapshot of a steering vector / probe direction for
        read-side analytics (correlation, pairwise), cached.

        The snapshot is built once under the exclusive-GPU ``_gen_lock``
        (acquired non-blocking) so the polling path never copies off an MPS
        device while a model op runs.  ``None`` when *name* isn't registered,
        or when a model op holds the GPU and no snapshot is cached yet; a
        later poll builds it once the GPU is free.
        """
        return self._registry.analytics_snapshot(
            name, live=lambda: self._live_direction_tensors(name),
            lock=self._gen_lock,
        )

    @property
    def probes(self) -> dict[str, dict[str, Any]]:
        # All three families via the instruments' LOCKED snapshots: this
        # property serves the un-locked session-info route, and a raw
        # registry iteration tears under a concurrent detach (round-7 for
        # lens/SAE; the geometry state_lock closed the Monitor sibling).
        out: dict[str, dict[str, Any]] = {
            name: {"manifold": m}
            for name, m in self._geometry_instrument.manifolds().items()
        }
        out.update(
            {name: {"lens": spec}
             for name, spec in self._lens_instrument.specs().items()}
        )
        out.update(
            {name: {"sae": spec}
             for name, spec in self._sae_instrument.specs().items()}
        )
        return out

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    @property
    def last_per_token_scores(self) -> dict[str, list[float]] | None:
        return self._last_per_token_scores

    @property
    def monitor(self) -> Monitor:
        """The unified read-side :class:`Monitor` (probe roster + scoring).

        Frontends read ``monitor.probe_names`` / ``monitor.manifolds`` to
        enumerate the attached probes without reaching past the public API.
        """
        return self._monitor

    @property
    def registry(self) -> ArtifactRegistry:
        """The in-memory artifact registry (profiles + loaded manifolds).

        Register an in-memory steering vector with
        ``registry.put_profile(name, profile)``; drop one with ``pop_profile``.
        """
        return self._registry

    @property
    def manifolds(self) -> Mapping[str, Manifold]:
        """Read-only view of the loaded manifolds: registry key
        (``[ns/]name[:variant]``) -> :class:`Manifold`."""
        return self._registry.manifolds

    @property
    def profiles(self) -> Mapping[str, dict[int, torch.Tensor]]:
        """Read-only view of the steering-profile registry: name -> per-layer
        tensors.  Mutate through :attr:`registry`."""
        return self._registry.profiles

    def evict_artifact(
        self, namespace: str, name: str, *, feature: str | None = None,
    ) -> None:
        """Drop every in-memory consumer of the manifold folder ``namespace/name``.

        Call after the folder changes on disk (deleted, edited, replaced,
        regenerated): loaded manifolds, folded profiles, attached geometry
        probes that read them, the prefix cache and the analytics snapshots
        all go, so the next steer or attach reloads from disk.  ``feature``
        (``"raw"`` / ``"sae-<release>"``) narrows it to one tensor family.
        Refuses (``ConcurrentExtractionError``) while a model op runs.
        """
        with self._model_exclusive(
            "evict_artifact called while another model use is in flight",
            phase_msg="evict_artifact called while a generation is in flight",
        ):
            self._registry.evict_artifact(namespace, name, feature=feature)

    def _refresh_swapped_probes(self, swaps: Mapping[int, Manifold | None]) -> None:
        """Move attached geometry probes off replaced manifold objects.

        ``swaps`` maps a replaced object's ``id`` to its replacement: each
        probe reading one is re-attached on the replacement (same name and
        nearest-node roster size), or detached when it maps to ``None`` (the
        artifact was evicted).  Lands under the geometry-state lock so the
        un-locked roster readers see the remove+add pair atomically.
        """
        monitor = self._monitor
        with self._geometry_instrument.state_lock:
            for probe_name, attached in list(monitor.attached_probes().items()):
                if id(attached.manifold) not in swaps:
                    continue
                replacement = swaps[id(attached.manifold)]
                top_n = attached.top_n
                monitor.remove_probe(probe_name)
                if replacement is not None:
                    monitor.add_probe(probe_name, replacement, top_n=top_n)
                self._probe_hash_cache.pop(probe_name, None)

    @property
    def generation_state(self) -> GenerationState:
        """The current per-call :class:`GenerationState` (finish reason, emit
        map, thinking spans, stop event).

        Distinct from :attr:`gen_state`, which reports the coarse lifecycle
        *phase* (``IDLE``/``RUNNING``/…); this is the live mutable streaming
        state the server reads while a generation is in flight.
        """
        return self._gen_state

    @property
    def token_probe_payload(self) -> dict[str, Any]:
        """Latest token-tap payload for live frontend serialization.

        The token tap is the sole producer. Frontends consume this current
        step snapshot instead of reconstructing it from persisted loom rows.
        """
        return self._last_token_probe_payload or {}

    @property
    def gen_lock(self) -> "threading.RLock":
        """The exclusive-GPU re-entry lock.  Callers that need to probe
        whether a generation is in flight without blocking use
        ``gen_lock.acquire(blocking=False)`` / ``release()``."""
        return self._gen_lock

    @property
    def joint_logprob_cache(self) -> dict[tuple[str, str], Any]:
        """Cross-branch joint-logprob cache (loom NodeCompareDrawer).

        The native API keeps this cache bounded and invalidates it after tree
        edits."""
        return self._joint_logprob_cache

    def ensure_manifold_loaded(self, key: str) -> Manifold:
        """The manifold artifact for registry key ``key``, loading it if absent.

        ``key`` is the grammar's ``[ns/]name[:variant]`` registry key.
        Raises :class:`ManifoldNotRegisteredError` on a miss."""
        return self._registry.ensure_manifold_loaded(key)

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Resolve and register a steering direction for ``name``, returning
        its per-layer tensors.

        Uses the manifold-first resolution chain: in-memory bake, then a fitted
        2-node ``pca`` manifold."""
        return self._registry.ensure_profile_registered(name, role=role)

    @property
    def gen_state(self) -> GenState:
        """Lifecycle phase of the current generation (``IDLE`` between gens).

        Read-only window into the session's typed re-entry guard — see
        :class:`GenState` for transitions. Surfaces to any external
        introspector that wants to ask "is a gen running right
        now?" without reaching past the public API.
        """
        return self._gen_phase

    @property
    def is_generating(self) -> bool:
        """``True`` whenever :attr:`gen_state` is not ``GenState.IDLE``."""
        return self._gen_phase is not GenState.IDLE

    @property
    def is_base_model(self) -> bool:
        """``True`` when the loaded model has no chat template (a base model).

        Frontends branch on this to switch to a raw-completion UI — no
        roles, no chat bubbles, the whole buffer is a prefill.
        """
        return self.chat_traits.is_base

    # -- Neutral baseline --

    @property
    def layer_means(self) -> dict[int, torch.Tensor]:
        """Per-layer neutral means, built on first use.

        Derived from the single fp32 ``neutral_activations`` cache (the same
        load builds the whitener).  Raises :class:`DrowseError` when it cannot
        be built, and :class:`ConcurrentExtractionError` when a build is
        needed while a model op, a generation, or a steering scope is active.
        """
        return self._neutral.layer_means()

    @property
    def whitener(self) -> "Any":
        """Per-layer Mahalanobis :class:`~drowse.geometry.whitener.LayerWhitener`,
        built on first use; ``None`` when it cannot be built (the Mahalanobis
        consumers then raise ``WhitenerError``).  Raises
        :class:`ConcurrentExtractionError` when a build is needed while a model
        op, a generation, or a steering scope is active."""
        return self._neutral.whitener()

    def _adopt_whitener(self, whitener: Any) -> None:
        """Keep the Monitor's read metric in lock-step with a freshly built
        whitener (a roster mutation, so under the geometry-state boundary)."""
        with self._geometry_instrument.state_lock:
            self._monitor.set_whitener(whitener)

    def _mid_generation_phase(self) -> str | None:
        """The phase name while a generation is mid-step, else ``None``."""
        phase = self._gen_phase
        if phase in (GenState.RUNNING, GenState.FINALIZING):
            return phase.name
        return None

    # -- The three read instruments --
    # Each family owns its own state; there is ONE addressing scheme for
    # it (``self._lens_instrument.<field>`` in-session, ``session.lens``
    # / ``.sae`` / ``.geometry`` outside).

    def _on_lens_change(self, probe_names: "Iterable[str]") -> None:
        """The lens instrument's change hook: registered ``jlens/``
        directions or the resident lens changed, so the prefix cache, the
        analytics cache and the lens probes' hashes are stale."""
        self._forget_probe_hashes(probe_names)
        self._invalidate_prefix_cache()
        self._registry.invalidate_analytics()

    def _forget_probe_hashes(self, names: "Iterable[str]") -> None:
        """Drop the cached transcript digests of the probes ``names``."""
        for name in names:
            self._probe_hash_cache.pop(name, None)

    @property
    def instruments(self) -> "dict[InstrumentFamily, Instrument]":
        """THE read-family registry — ``geometry`` (the Monitor adapter),
        ``lens``, ``sae``.

        This mapping is the family *enumeration*, not a convenience view:
        the probe-hash roster, the composer's gate preflight, and the
        server's ``{family}`` validation + per-family dispatch all derive
        from it, so a fourth read family is one registry entry plus its
        :class:`~drowse.readout.protocol.Instrument`
        implementation.
        """
        return {
            # The concrete families deliberately retain family-specific run
            # subclasses; expose only their shared read-family surface here.
            # ``Instrument.current_run`` is mutable, so Protocol structural
            # matching cannot widen those subclasses automatically.
            "geometry": cast("Instrument", self._geometry_instrument),
            "lens": cast("Instrument", self._lens_instrument),
            "sae": cast("Instrument", self._sae_instrument),
        }

    def _close_instrument_runs(self) -> None:
        """Close every family's per-generation run (idempotent teardown).

        Every capture transaction that binds runs must reach this in its
        ``finally`` — the ordinary generation, the batch fast path, and
        the joint-logprob replay.  A bound run leaking past its
        transaction pins a stale lens (suppressing disk refresh between
        generations) and freezes SAE units past their generation."""
        self._geometry_instrument.close_run()
        self._lens_instrument.close_run()
        self._sae_instrument.close_run()

    @property
    def lens(self) -> "LensInstrument":
        """The J-lens instrument — the public face of the lens family
        (``session.instruments["lens"]``): the resident lens and its source
        (``resident`` / ``select_source``), ``fit``, the offline reads
        (``readout`` / ``decompose`` / ``register_direction`` /
        ``token_readout``), probes, and the live readout."""
        return self._lens_instrument

    @property
    def sae(self) -> "SaeInstrument":
        """The SAE instrument — the public face of the SAE family
        (``session.instruments["sae"]``): the resident dictionary's lifecycle
        (``load`` / ``unload`` / ``train`` / ``info``), feature validation and
        Neuronpedia metadata, ``sae/<id>`` directions, probes, live discovery
        and token replay."""
        return self._sae_instrument

    @property
    def geometry(self) -> "GeometryInstrument":
        """The geometry instrument — the typed public face of the whitened
        Monitor read family (``session.instruments["geometry"]``): probe
        attach/detach, the CAA live toggle, the roster snapshots.  Manifold
        artifact lifecycle (``fit`` / ``extract`` / ``ensure_manifold_
        loaded``) stays on the session."""
        return self._geometry_instrument

    def geometry_token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Whitened subspace-probe readings at the forward that produced a
        loom token — the geometry family's token replay.

        Rebuilds the exact token stream that produced the clicked token
        (the same render :meth:`fork_from_token` and the lens/SAE replays
        use), runs one capture forward at the attached probes' layer
        union, and scores the full Monitor roster at the final position
        via :meth:`Monitor.score_single_token` — so the reading shape
        (coords / fraction / nearest / residual / assignment / membership
        + per-layer traces) matches the live per-token read at that
        position.  Post-hoc by construction: it reads generations that ran
        aggregate-only (live probe scores off) and probes attached after
        the fact, and ``apply_steering=False`` reads the unsteered
        counterfactual.  Same steering-fidelity class as the fork replay:
        an always-active affine term reproduces the original decode
        injections exactly (the slide is position-independent);
        phase-split and probe-gated terms fire as the trigger context's
        rest state decides.  Curved probes score with a cold foot solve
        (warm-start is the sequential live path's optimization; a stale
        foot from a prior generation must not seed a one-shot replay).

        Returns ``{node_id, raw_index, token_id, token_text, steering,
        readings: {probe_name: ProbeReading}}``.  Raises ``ValueError``
        with no geometry probe attached, :class:`UnknownNodeError` /
        :class:`InvalidNodeOperationError` on a bad target (mirrors
        :meth:`fork_from_token`), and ``WhitenerError`` when the whitener
        doesn't cover the probed layers.
        """
        from drowse.fit.capture import _capture_all_hidden_states

        if not self._monitor.probe_names:
            raise ValueError(
                "geometry_token_readout: no geometry probes attached — "
                "attach one with session.add_probe(selector) first"
            )
        node = self.tree.get(node_id)
        if node.role not in ("assistant", "user"):
            raise InvalidNodeOperationError(
                f"geometry_token_readout: {node_id!r} is a {node.role} "
                f"node — only a turn with a decode record can be read out"
            )
        raw_ids = node.raw_token_ids
        if not raw_ids:
            raise InvalidNodeOperationError(
                f"geometry_token_readout: {node_id!r} has no raw token "
                f"record (transcript-loaded node)"
            )
        if not 0 <= raw_index < len(raw_ids):
            raise InvalidNodeOperationError(
                f"geometry_token_readout: raw_index {raw_index} out of "
                f"range [0, {len(raw_ids)}) for {node_id!r}"
            )
        recipe = node.recipe
        steering_expr = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )
        # The steering scope opens OUTSIDE the exclusive-GPU guard — the
        # composer's push/pop take ``_gen_lock`` blocking, so nesting it
        # inside ``_model_exclusive`` would self-deadlock (same ordering
        # as the lens/SAE replays and ``score_choices``).
        scope = self.steering(steering_expr) if steering_expr else nullcontext()
        with scope:
            thinking_req = recipe.thinking if recipe is not None else None
            use_thinking = (
                self.chat_traits.supports_thinking
                if thinking_req is None
                else bool(thinking_req) and self.chat_traits.supports_thinking
            )
            if raw:
                prompt_ids = self._renderer.prepare_input(
                    "", raw=True, parent_node_id=node.parent_id,
                )
            else:
                # Continue-mode rebuild — seat-general, mirrors
                # :func:`drowse.session.replay.replay_token`.
                prompt_ids = self._renderer.prepare_input(
                    None,
                    thinking=use_thinking,
                    parent_node_id=node.parent_id,
                    system_prompt=recipe.system_prompt if recipe is not None else INHERIT_SYSTEM_PROMPT,
                    user_role=(
                        node.role_label if node.role == "user" else None
                    ),
                    assistant_role=(
                        node.role_label if node.role == "assistant" else None
                    ),
                    gen_seat=node.role,
                )
            if raw_index > 0:
                prefix = torch.tensor(
                    [[int(token) for token in raw_ids[:raw_index]]],
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )
                ids = torch.cat([prompt_ids, prefix], dim=1)
            else:
                ids = prompt_ids
            with self._model_exclusive(
                "session.geometry_token_readout called while another model "
                "use is in flight",
                phase_msg="session.geometry_token_readout called while a "
                "generation is in flight",
            ):
                # Repeat the roster check under the exclusive section: a
                # detach winning the gap since the entry check would
                # otherwise surface as a capture-layer implementation error
                # instead of the intended caller-facing message.
                if not self._monitor.probe_names:
                    raise ValueError(
                        "geometry_token_readout: no geometry probes "
                        "attached — attach one with "
                        "session.add_probe(selector) first"
                    )
                layer_idxs = sorted(self._monitor.probe_layers())
                hidden = _capture_all_hidden_states(
                    self._model, self._layers, ids,
                    layer_indices=layer_idxs, pool_index=ids.shape[1] - 1,
                )
                self._monitor.enable_curved_warm(False)
                readings = self._monitor.score_single_token(hidden)
        return {
            "node_id": node_id,
            "raw_index": int(raw_index),
            "token_id": int(raw_ids[raw_index]),
            "token_text": str(self._tokenizer.decode([int(raw_ids[raw_index])])),
            "steering": steering_expr,
            "readings": readings,
        }

    # -- Extraction --

    def generate_responses(
        self,
        concepts: list[str],
        kinds: list[str | None],
        *,
        roles: dict[str, str | None] | None = None,
        custom_system: str | None = None,
        samples_per_prompt: int = 1,
        max_new_tokens: int = RESPONSE_MAX_TOKENS,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]:
        """Each concept's in-character corpus; see :meth:`CorpusGenerator.responses`."""
        return self._corpus.responses(
            concepts, kinds, roles=roles, custom_system=custom_system,
            samples_per_prompt=samples_per_prompt, max_new_tokens=max_new_tokens,
            on_progress=on_progress,
        )

    def generate_neutral_responses(
        self,
        *,
        samples_per_prompt: int = 1,
        max_new_tokens: int = RESPONSE_MAX_TOKENS,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[str]:
        """The neutral baseline corpus; see :meth:`CorpusGenerator.neutral_responses`."""
        return self._corpus.neutral_responses(
            samples_per_prompt=samples_per_prompt, max_new_tokens=max_new_tokens,
            on_progress=on_progress,
        )

    @contextmanager
    def _model_exclusive(
        self,
        busy_msg: str,
        *,
        phase_msg: str | None = None,
        exc: type[Exception] = ConcurrentExtractionError,
    ) -> Iterator[None]:
        """Hold the exclusive-GPU ``_gen_lock`` for a model op.

        Acquires non-blocking — a cross-thread model op in flight raises
        ``exc(busy_msg)`` rather than races PyTorch's single global MPS
        command buffer (which would abort the process).  When ``phase_msg``
        is given, also guards ``_gen_phase is IDLE`` (raises ``exc(phase_msg)``
        on a generation in flight); ``add_probe`` passes ``None`` to skip the
        phase guard.  Releases in ``finally``.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise exc(busy_msg)
        try:
            if phase_msg is not None and self._gen_phase is not GenState.IDLE:
                raise exc(phase_msg)
            yield
        finally:
            self._gen_lock.release()

    def _assert_unsteered_artifact_operation(self) -> None:
        """Refuse persisted neutral/fit work under active steering hooks."""
        if self._steering_composer.active:
            raise ConcurrentExtractionError(
                "artifact-producing model operations cannot run inside an "
                "active steering scope; exit session.steering() first"
            )

    def extract(
        self,
        concept: str,
        baseline: str | None = None,
        *,
        kind: str = "abstract",
        custom_system: str | None = None,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
        sae: str | None = None,
        sae_revision: str | None = None,
        namespace: str | None = None,
        role: str | None = None,
    ) -> tuple[str, Profile]:
        """Author and fit a steering vector; see :meth:`Extraction.extract`."""
        return self._extraction.extract(
            concept, baseline, kind=kind, custom_system=custom_system, force=force,
            on_progress=on_progress, sae=sae, sae_revision=sae_revision,
            namespace=namespace, role=role,
        )

    def extract_from_corpora(
        self,
        name: str,
        positive: list[str],
        negative: list[str],
        *,
        kind: str = "abstract",
        namespace: str | None = None,
        role: str | None = None,
        sae: str | None = None,
        sae_revision: str | None = None,
        on_progress: Callable[[str], None] | None = None,
        force: bool = False,
        description: str = "",
    ) -> tuple[str, Profile]:
        """Fit a vector from given pole corpora; see :meth:`Extraction.extract_from_corpora`."""
        return self._extraction.extract_from_corpora(
            name, positive, negative, kind=kind, namespace=namespace, role=role,
            sae=sae, sae_revision=sae_revision, on_progress=on_progress,
            force=force, description=description,
        )

    def fit(
        self,
        folder: Any,
        *,
        sae: str | None = None,
        sae_revision: str | None = None,
        layers: "Sequence[int] | str | None" = None,
        fit_mode: str | None = None,
        hyperparams: Mapping[str, object] | None = None,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> Manifold:
        """Fit a manifold folder for this model; see :meth:`Extraction.fit`."""
        return self._extraction.fit(
            folder, sae=sae, sae_revision=sae_revision, layers=layers,
            fit_mode=fit_mode, hyperparams=hyperparams, force=force,
            on_progress=on_progress,
        )

    def bake(
        self,
        name: str,
        expression: str,
        *,
        force: bool = False,
        strict: bool = False,
    ) -> tuple[str, Profile]:
        """Bake additive terms into a steerable manifold; see :meth:`Extraction.bake`."""
        return self._extraction.bake(name, expression, force=force, strict=strict)

    def steering(self, value: "str | Steering") -> SteeringScope:
        """Context manager applying steering for the duration of a with-block.

        ``value`` is a steering expression string (parsed through the shared
        grammar in :mod:`drowse.steering.grammar`) or a pre-built
        :class:`Steering`; dicts are not accepted.  Every artifact resolves
        and loads here: an unknown profile name raises
        ``ProfileNotRegisteredError`` and an ambiguous pole name
        ``AmbiguousSelectorError``.  Nesting flattens — an inner scope's
        entries override the outer ones for the inner block and the outer
        entries come back on exit — with one hook installation per active
        layer regardless of depth.  Role-augmented terms select the
        assistant-role label of the scope (see
        :func:`~drowse.steering.composer.resolve_scope_role`).
        """
        return self._steering_composer.open_scope(value, stacklevel=2)

    @contextmanager
    def _generation_transaction(self) -> "Iterator[_GenerationTransaction]":
        """The generation boundary: single-flight guard in, hard teardown out.

        Every capture-owning generation path runs inside this — the ordinary
        decode loop and the batched ``model.generate`` fast path — so the
        order ARCHITECTURE.md §8.1/§8.3 describes exists once.  On entry: the
        non-reentrant lock, the ``IDLE`` re-entry guard, and the phase
        transition to ``PREAMBLE``.  On exit, whatever raised: pop any
        steering scope the body left open, detach capture, drop the
        per-generation session slots, close the instrument runs, reset capture
        state, zero the persistent compile-clean steering offsets, and release
        the phase and the lock.

        The body owns the steering scope through ``txn.steering_cm``: it
        assigns the scope when it enters one and clears the slot again if it
        pops the scope itself (the decode path pops before finalizing, so the
        finalize reads happen unsteered).  A scope still present here is
        popped with ``swallow=txn.failed`` — a teardown ``Exception`` must not
        mask the failure already propagating.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise ConcurrentGenerationError("Generation already in progress")
        if self._gen_phase is not GenState.IDLE:
            self._gen_lock.release()
            raise ConcurrentGenerationError("session generation already in flight")
        self._gen_phase = GenState.PREAMBLE
        txn = _GenerationTransaction()
        try:
            yield txn
        except BaseException:
            txn.failed = True
            raise
        finally:
            try:
                try:
                    if txn.steering_cm is not None:
                        txn.steering_cm.close(internal=True, swallow=txn.failed)
                finally:
                    try:
                        # Defense-in-depth: even if the body's own teardown
                        # never ran (a BaseException between the transaction
                        # entry and ``_begin_capture``), any hooks that did
                        # get attached must come off.  Idempotent.
                        self._end_capture()
                        # The probe-inspector subspace-coords post-pass is
                        # per-generation; clear it so it never leaks into a
                        # later gen that didn't opt in.
                        self._monitor.set_subspace_coords(False)
                        # Release the loom-tree reservation in the same scope
                        # as the gen-lock release.  Even if finalize raised,
                        # mutators (edit / delete on this subtree) need to be
                        # free again now that the streaming target is no
                        # longer live.
                        self.tree.reserve(None)
                        self._last_token_probe_payload = None
                        self._last_token_payload = None
                    finally:
                        # Run closure and capture-state resets must survive a
                        # teardown failure above (a raising hook detach must
                        # not leave a bound run pinning a stale lens).
                        self._close_instrument_runs()
                        # Reset capture state to the default (FULL,
                        # non-persistent) so the next gen starts clean —
                        # finalize has already consumed the rows, and
                        # ``_begin_capture`` resets it at gen start too.
                        self._capture_state = CaptureState()
                        self._compiled_clean_eligible = False
                        self._incremental_readings = []
                        self._incremental_gate_scores = []
                        # Steering state is not touched here: the installed
                        # plan (offset buffers included) is a function of the
                        # steering stack, restored by the scope pop above.
            finally:
                # Unconditional: an earlier teardown exception must not leave
                # the session phase-wedged or the gen lock held.
                self._gen_phase = GenState.IDLE
                self._gen_lock.release()

    def _bind_instrument_runs(
        self,
        geometry_request: ReadRequest,
        lens_request: ReadRequest,
        sae_request: ReadRequest,
    ) -> "tuple[InstrumentPlan, InstrumentPlan, InstrumentPlan]":
        """Run the uniform instrument transaction: close → prepare → plan → bind.

        The defensive close comes first (the transaction teardown is the
        ordinary one) — a stale pin would short-circuit the lens disk refresh
        inside ``prepare``, and a stale binding would keep serving frozen specs
        to idle reads.  ``prepare`` is the source-boundary step: the lens
        family refreshes its resident lens from disk under pin demand there
        and snapshots specs + live config against that identity;
        ``plan``/``bind`` consume the prep only, so an interleaved adoption
        (the un-locked ``lens.has_compatible``) cannot desynchronize them
        (``LensInstrument.prepare`` carries the ordering + snapshot rationale).

        ``bind`` freezes each family's probe specs into an immutable
        ``InstrumentBinding`` (a concurrent SAE metadata backfill can no longer
        change what this generation measures), starts fresh stashes and step
        memos, and hands the lens run the pin its prep took — so every token,
        gate, and final aggregate consumes the same resident lens even if
        another process switches the active source mid-decode.

        Returns the three plans; the caller unions their declared layers and
        chooses the physical retention.
        """
        self._close_instrument_runs()
        geometry_prep = self._geometry_instrument.prepare(geometry_request)
        lens_prep = self._lens_instrument.prepare(lens_request)
        sae_prep = self._sae_instrument.prepare(sae_request)
        geometry_plan = self._geometry_instrument.plan(geometry_prep)
        lens_plan = self._lens_instrument.plan(lens_prep)
        sae_plan = self._sae_instrument.plan(sae_prep)
        self._geometry_instrument.bind(geometry_plan, geometry_prep)
        self._lens_instrument.bind(lens_plan, lens_prep)
        self._sae_instrument.bind(sae_plan, sae_prep)
        return geometry_plan, lens_plan, sae_plan

    def _resolve_read_demand(
        self,
        *,
        want_hidden: bool,
        live_scores_on: bool,
        has_monitor_probes: bool,
        loom_attached: bool,
        wants_live_token_scores: bool,
        persists_layer_scores: bool,
        persists_probe_row: bool,
        persists_subspace_coords: bool,
        live_lens_active: bool,
        live_sae_active: bool,
        final_probe_aggregate: bool,
        capture_prompt: bool,
    ) -> ReadDemand:
        """Fold this generation's consumers into one :class:`ReadDemand`.

        The single place the interdependent read-demand predicates are
        derived, so :meth:`_begin_capture` re-derives none of them.
        """
        composer = self._steering_composer
        needs_gating = composer.steering_needs_probe_gating()
        gating_probe_keys = composer.gated_probe_keys() if needs_gating else None
        lens_gating_probe_keys = (
            composer.gated_lens_probe_keys() if needs_gating else None
        )
        sae_gating_probe_keys = (
            composer.gated_sae_probe_keys() if needs_gating else None
        )
        # The four UI / loom / persist consumers each need a per-token
        # reading for the FULL roster.  When NONE of them is live but a probe
        # gate is, gating is the SOLE per-token consumer: the step sink can
        # score just the gated probes per token and leave the big-K roster to
        # the one-shot full aggregate at finalize.
        per_token_full_consumer = bool(
            (live_scores_on and loom_attached)
            or wants_live_token_scores
            or persists_layer_scores
            or persists_probe_row
            or persists_subspace_coords
        )
        # J-lens token probes gate on the lens path (the gating score callback
        # computes their scalars from the latest capture slices), so a
        # lens-only gate doesn't force per-token MONITOR scoring —
        # ``need_per_token`` keys on monitor-attached gate keys only.
        needs_monitor_gating = bool(gating_probe_keys)
        need_per_token = bool(needs_monitor_gating or per_token_full_consumer)
        gating_only_probes = (
            composer.gated_probe_names()
            if (needs_monitor_gating and not per_token_full_consumer)
            else None
        )
        # Lean-incremental: the live consumers present read ONLY the axis-0
        # coord (the loom probe row, the loom node
        # text/aggregate) — none of the consumers that genuinely need a richer
        # per-token reading is live, and there is no probe gate.  Then
        # per-token scoring drops the nearest / assignment / per-layer work
        # (the full aggregate is re-scored once from the tail ring at
        # finalize), which is the common loom monitoring path.
        full_reading_consumer = bool(
            wants_live_token_scores       # full ProbeReading in TokenEvent
            or persists_layer_scores      # per-layer heatmap row
            or persists_subspace_coords   # inspector whitened coords
        )
        lean_per_token = bool(
            need_per_token
            and not want_hidden
            and not needs_monitor_gating
            and not full_reading_consumer
            and has_monitor_probes
        )
        return ReadDemand(
            need_per_token=need_per_token,
            per_token_full_consumer=per_token_full_consumer,
            gating_only_probes=gating_only_probes,
            gating_probe_keys=gating_probe_keys,
            lens_gating_probe_keys=lens_gating_probe_keys,
            sae_gating_probe_keys=sae_gating_probe_keys,
            lean_per_token=lean_per_token,
            final_probe_aggregate=final_probe_aggregate,
            live_lens_active=live_lens_active,
            live_sae_active=live_sae_active,
            capture_prompt=capture_prompt,
        )

    def _begin_capture(
        self, demand: "ReadDemand | None" = None, *, widen: bool = False,
    ) -> bool:
        """Attach hidden-state capture. Returns True if attached.

        ``demand`` is this generation's resolved :class:`ReadDemand` — every
        predicate it carries was computed once by
        :meth:`_resolve_read_demand`, and nothing is re-derived here, so the
        capture cannot disagree with the caller about what the generation
        needs.  ``None`` means the all-defaults demand: full per-token scoring,
        a final aggregate, both live readouts on, no gates.

        The capture layer set is **plan-driven**: each instrument family
        prepares its source snapshot (``Instrument.prepare(ReadRequest)``),
        declares its demand via ``Instrument.plan(prep) ->
        InstrumentPlan``, and this planner unions the declared layers.
        Retention (incremental vs tail ring vs full stack) remains the
        planner's decision below — cross-instrument resource sharing the
        plans deliberately do not decide (``protocol.py``).

        ``widen`` is the one axis that isn't consumer demand but a retention
        request, so it stays a parameter.  ``widen=False`` (default) covers
        the union of the layers the instrument plans declared.  ``widen=True``
        covers every model layer — the caller asked for
        ``SamplingConfig.return_hidden=True``, so the monitor still reads its
        probe subset but the full dict lands on
        ``GenerationResult.hidden_states``.
        """
        demand = ReadDemand() if demand is None else demand
        # ---- Per-family capture demand (Instrument.plan) -------------------
        # The planner hands each family what the session knows about this
        # generation's read demand; each family answers with its declared
        # ``InstrumentPlan``, and the union of those declared layers is
        # what capture covers.  RETENTION (incremental
        # vs tail ring vs full stack) stays session logic below — the
        # ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
        # resource sharing, which plans deliberately do not decide
        # (``protocol.py`` division of labor).
        geometry_plan, lens_plan, sae_plan = self._bind_instrument_runs(
            ReadRequest(
                gate_keys=frozenset(demand.gating_probe_keys or ()),
                per_token_consumers=demand.per_token_full_consumer,
                final_aggregate=demand.final_probe_aggregate,
            ),
            ReadRequest(
                gate_keys=frozenset(demand.lens_gating_probe_keys or ()),
                live=demand.live_lens_active,
                per_token_consumers=demand.per_token_full_consumer,
                final_aggregate=demand.final_probe_aggregate,
            ),
            ReadRequest(
                gate_keys=frozenset(demand.sae_gating_probe_keys or ()),
                live=demand.live_sae_active,
                per_token_consumers=demand.per_token_full_consumer,
                final_aggregate=demand.final_probe_aggregate,
            ),
        )
        has_lens_gate = bool(lens_plan.gate_keys)
        has_sae_gate = bool(sae_plan.gate_keys)
        if widen:
            layer_idxs = list(range(len(self._layers)))
        else:
            union: set[int] = set(
                geometry_plan.latest_layers
                | lens_plan.latest_layers | lens_plan.tail_layers
                | sae_plan.latest_layers | sae_plan.tail_layers
            )
            if not union:
                # No probes ⇒ the degenerate FULL (capture-disabled) state.
                self._capture_state = CaptureState(
                    gating_keys=set(demand.gating_probe_keys or ())
                    if demand.gating_probe_keys else None,
                    final_probe_aggregate=demand.final_probe_aggregate,
                )
                self._incremental_readings = []
                return False
            layer_idxs = sorted(union)
        # Persistent compile-clean capture: when this gen is eligible
        # for the compiled clean path AND steering lowered to offsets (or is
        # unsteered), capture rides the always-on persistent buffers — the decode
        # loop ``copy_``s each slice in inside the compiled graph, and
        # ``ingest_persistent`` (wired as the step callback) does the per-token
        # accumulation + scoring post-forward.  No transient ``register_forward_hook``
        # is installed, so the routing keeps the compiled module + StaticCache.
        # Anything that forced the eager path (curved / gated / phased steering,
        # ``return_hidden``) falls back to transient capture hooks.
        use_persistent = bool(
            not widen
            and not demand.capture_prompt
            # Persistent capture buffers are installed for every layer before
            # compile; layer_idxs selects the subset this generation consumes.
            # Live lens layers can therefore ride the same compile-clean path
            # instead of forcing transient hooks.
            and self._compiled_clean_eligible
            and self._capture_buffers
            and (
                self._steering.uses_compiled_offsets
                or self._steering.all_fast_path()
            )
        )
        self._capture.clear()
        if use_persistent:
            self._capture.attach_persistent(layer_idxs, self._capture_buffers)
        else:
            self._capture.attach(self._layers, layer_idxs)
        # Default this gen to FULL (append / full-retention); the mode branches
        # below replace it.  ``persistent`` is orthogonal to the mode — set here
        # from the compiled-clean routing decision and carried through unchanged.
        self._capture_state = CaptureState(
            persistent=use_persistent,
            gating_keys=(
                set(demand.gating_probe_keys or ())
                if demand.gating_probe_keys else None
            ),
            final_probe_aggregate=demand.final_probe_aggregate,
        )
        self._incremental_readings = []
        self._incremental_gate_scores = []
        # Gating-only-subset path: per-token scoring is needed solely to feed
        # probe gates, so the step sink scores only the gated probes (the gate
        # consumes only those scalars), skipping the big-K roster's per-token
        # nearest-distance work.  A bounded tail ring is kept alongside the sink
        # so finalize still pools the FULL roster once via the aggregate path —
        # the gated subset's per-token rows are NOT the source of the final
        # readings, the one-shot full aggregate is, so no probe drops.
        gating_subset = (
            demand.gating_only_probes
            if (demand.gating_only_probes and demand.need_per_token and not widen
                and self._monitor.probe_names)
            else None
        )
        if (
            not widen and self._monitor.probe_names
            and demand.need_per_token and gating_subset
        ):
            subset = set(gating_subset)
            gate_keys = set(demand.gating_probe_keys or ())
            gate_plan = self._monitor.plan_gate_scalars(
                gate_keys, probe_names=subset,
            )

            def _score_step_subset(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # Score ONLY the exact gate scalar keys each token (the gate's
                # sole consumer); append even an empty dict to keep row indices
                # aligned with decode forwards.  The full roster is pooled once at
                # finalize from the tail ring (``_score_aggregate_only``).
                del step_id
                self._incremental_gate_scores.append(
                    self._monitor.score_planned_gate_scalars(latest, gate_plan)
                    if latest and gate_plan else {}
                )

            if demand.final_probe_aggregate:
                # Deep tail ring (full-roster finalize via ``tail_slice_at``) PLUS a
                # per-token step sink (gated subset, for the gate) — armed together by
                # ``set_tail_with_sink`` (neither single setter gives both).
                self._capture.set_tail_with_sink(_AGG_TAIL_DEPTH, _score_step_subset)
            else:
                # Pure control/gating call: no final aggregate requested, so a
                # length-1 incremental buffer is enough and capture has already been
                # narrowed to the gated probes' layer union.
                self._capture.set_incremental(_score_step_subset)
            self._capture_state.mode = CaptureMode.GATING_SUBSET
            self._capture_state.gating_subset = subset
            self._capture_state.gating_keys = gate_keys
            self._capture_state.final_probe_aggregate = demand.final_probe_aggregate
            # The subset is scored one token per step in order, so curved gated
            # probes can warm-start their foot; cold-start the feet first.
            self._monitor.reset_curved_feet()
            self._monitor.enable_curved_warm(True)
        elif (
            not widen and self._monitor.probe_names
            and demand.need_per_token and demand.lean_per_token
        ):
            # Lean-incremental: the only per-token consumers read just
            # the axis-0 coord (the loom probe row) — no nearest /
            # assignment / per-layer trace, no probe gate.  Score each token
            # ``coords_only`` (skips the big-K nearest norm + assignment softmax +
            # per-layer host reconstruction), store the lean rows for the per-token
            # stream, and keep a bounded tail ring so finalize re-scores the FULL
            # aggregate once via ``_score_lean_incremental``.
            def _score_step_lean(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                del step_id
                self._incremental_readings.append(
                    self._monitor.score_single_token(latest, coords_only=True)
                    if latest else {}
                )

            # Deep tail ring (full-roster aggregate at finalize) PLUS the lean
            # per-token step sink — same dual-arming as the gating-subset path.
            self._capture.set_tail_with_sink(_AGG_TAIL_DEPTH, _score_step_lean)
            self._capture_state.mode = CaptureMode.LEAN_INCREMENTAL
            # Lean rows are scored one token per step in order, so curved probes
            # can warm-start their foot for the per-token coord stream.
            self._monitor.reset_curved_feet()
            self._monitor.enable_curved_warm(True)
        elif not widen and self._monitor.probe_names and demand.need_per_token:

            def _score_step(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # Score once while the just-produced hidden slice is still the
                # only retained device payload.  Append even an empty dict so
                # row indices remain aligned with decode forwards; finalization
                # trims any terminal EOS-only overcapture to generated_ids.
                # The gate callback and the token tap read this row.
                del step_id
                self._incremental_readings.append(
                    self._monitor.score_single_token(latest) if latest else {}
                )

            if (
                lens_plan.final_aggregate or has_lens_gate
                or sae_plan.final_aggregate or has_sae_gate
            ):
                # Lens probes pool their finalize aggregate from the tail
                # ring; ``set_incremental``'s length-1 buffers can't walk back
                # past trailing specials, so arm the ring alongside the sink.
                # When final probe aggregates are disabled, pinned J-lens/SAE
                # probes still need the latest slice for gates/live consumers,
                # but do not need the 8-deep EOS walk-back ring.  The ring's
                # layer span is the readout families' declared finalize-
                # pooling demand (empty when final readings are off).
                readout_tail_layers = set(
                    lens_plan.tail_layers | sae_plan.tail_layers
                )
                self._capture.set_tail_with_sink(
                    _AGG_TAIL_DEPTH if demand.final_probe_aggregate else 1,
                    _score_step,
                    tail_layers=readout_tail_layers or None,
                )
            else:
                self._capture.set_incremental(_score_step)
            self._capture_state.mode = CaptureMode.INCREMENTAL
            # The step sink scores one token per decode step in order, so curved
            # probes can warm-start their nearest-point foot from the previous
            # token. Cold-start the feet for this generation first.
            self._monitor.reset_curved_feet()
            self._monitor.enable_curved_warm(True)
        elif not widen and self._monitor.probe_names:
            # Aggregate-only: probes attached but no per-token consumer. Keep a
            # bounded tail ring, score nothing per token; finalize pools the
            # last content token once. Curved warm-start is a sequential-live
            # optimization, so leave it off for the one-shot aggregate read.
            self._capture.set_aggregate_tail(_AGG_TAIL_DEPTH)
            self._capture_state.mode = CaptureMode.AGGREGATE_ONLY
            self._monitor.enable_curved_warm(False)
        elif not widen and (
            lens_plan.latest_layers or lens_plan.tail_layers
            or sae_plan.latest_layers or sae_plan.tail_layers
        ):
            # Lens/SAE-only capture (live readout on and/or readout probes
            # pinned, no monitor probes): a bounded tail ring keeps the
            # reader's latest slice fresh without FULL retention growing over
            # the generation. No step sink — live display runs at the token tap,
            # gates score from ``latest_per_layer`` in the gate callback, and
            # final aggregates pool from the ring only when requested. If final
            # readings are disabled, keep a length-1 latest buffer instead of
            # cloning/popping the 8-deep EOS walk-back tail every token.
            self._capture.set_aggregate_tail(
                _AGG_TAIL_DEPTH if demand.final_probe_aggregate else 1
            )
            # The armed retention IS an aggregate tail, so record it: leaving
            # the FULL default here would make ``CaptureState`` describe a
            # retention the capture never installed.
            self._capture_state.mode = CaptureMode.AGGREGATE_ONLY
            self._monitor.enable_curved_warm(False)
        else:
            # FULL retention (the ``CaptureState`` default): return_hidden full
            # stack, or no per-token consumer on a non-tail read.  Scored out of
            # order / one-shot, so keep curved probes on the cold foot solve for
            # reproducibility.
            self._monitor.enable_curved_warm(False)
        return True

    def _end_capture(self) -> None:
        self._capture.detach()

    # -- Score entry points --

    def score_choices(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        assistant_prefix: str = "",
        labels: list[str] | None = None,
        steering: "str | Steering | None" = None,
        system_prompt: str | None = None,
    ) -> "ChoiceScores":
        """Restricted-choice logprob distribution over ``choices``.

        Scores each candidate completion against the raw model distribution
        given ``messages`` (+ optional ``assistant_prefix`` before the slot) and
        returns the set with per-candidate ``sum``/``mean`` logprobs and their
        softmax probabilities. ``steering=`` runs the scoring forward under a
        steering expression — the distributional before/after read. See
        :func:`drowse.generation.scoring.score_choices`.
        """
        from drowse.generation.scoring import score_choices as _score_choices

        return _score_choices(
            self, messages, choices,
            assistant_prefix=assistant_prefix, labels=labels,
            steering=steering, system_prompt=system_prompt,
        )

    def score_template(
        self,
        template: "str | TemplateFolder",
        *,
        steering: "str | Steering | None" = None,
        system_prompt: str | None = None,
    ) -> list["ChoiceScores"]:
        """Score a template's values against each of its contexts.

        ``template`` is a :class:`~drowse.store.templates.TemplateFolder` or a
        selector string (``<name>`` / ``<ns>/<name>``). Returns one
        :class:`~drowse.generation.scoring.ChoiceScores` per context.
        """
        from drowse.generation.scoring import score_template as _score_template
        from drowse.store.templates import TemplateFolder, resolve_template

        tmpl = (
            template if isinstance(template, TemplateFolder)
            else resolve_template(template)
        )
        return _score_template(
            self, tmpl, steering=steering, system_prompt=system_prompt,
        )

    def _aggregate_forward_index(self, generated_ids: list[int]) -> int | None:
        """Forward index to use for the end-of-generation aggregate read."""
        idx = getattr(self._gen_state, "response_aggregate_index", None)
        if idx is not None and 0 <= int(idx) < len(generated_ids):
            return int(idx)
        if (
            self._gen_state.finish_reason == "stop_sequence"
            and getattr(self._gen_state, "response_text", None) is not None
        ):
            return None
        from drowse.fit.capture import last_content_index
        return last_content_index(generated_ids, self._tokenizer)

    def _pooled_aggregate_slice(
        self, generated_ids: list[int],
    ) -> dict[int, torch.Tensor]:
        """Per-layer ``[D]`` slice at the end-of-generation aggregate position.

        The ONE finalize-pooling resolution, shared by all three families
        (geometry's ``_score_aggregate_only`` and the lens/SAE instruments'
        ``score_aggregate``).  Empty when there is no aggregate position or
        nothing was captured.

        Retention decides how the position is reached.  Every bounded-ring
        mode counts decode forwards, so :meth:`HiddenCapture.tail_slice_at`
        maps an absolute forward index into the ring.  FULL retention keeps
        every forward but never advances that counter, so the ring mapping
        would clamp to the last slice; there the stack is indexed directly,
        which is what makes the aggregate land on the last *content* token in
        every mode rather than only in the ring modes.
        """
        agg_fwd = self._aggregate_forward_index(generated_ids)
        if agg_fwd is None:
            return {}
        if self._capture_state.mode is CaptureMode.FULL:
            return {
                layer: rows[agg_fwd]
                for layer, rows in self._capture.stacked().items()
                if rows.shape[0] > agg_fwd
            }
        return self._capture.tail_slice_at(agg_fwd)

    def _empty_readings(self, names: list[str]) -> dict[str, "ProbeReading"]:
        """A zero ``ProbeReading`` per probe — the no-tokens / missing-pool
        fallback shared by the incremental and aggregate-only score paths."""
        return {
            name: ProbeReading(fraction=0.0, nearest=[], coords=())
            for name in names
        }

    def _extract_coord_stream(
        self, rows: list[dict[str, "ProbeReading"]], n: int, names: list[str],
    ) -> dict[str, list[float]]:
        """Per-probe axis-0 coord stream from per-token reading rows.

        Reads ``reading.coords[0]`` (0.0 when a row is empty) into a length-``n``
        per-probe list — the loop shared by the (full) incremental and the lean
        incremental score paths.
        """
        per_token: dict[str, list[float]] = {name: [0.0] * n for name in names}
        for i, readings in enumerate(rows):
            for name, reading in readings.items():
                if name in per_token:
                    per_token[name][i] = (
                        reading.coords[0] if reading.coords else 0.0
                    )
        return per_token

    def _score_incremental(
        self, generated_ids: list[int],
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Aggregate readings + per-token coord stream from live-scored rows."""
        names = list(self._monitor.probe_names)
        n = len(generated_ids)
        if n == 0 or not self._incremental_readings:
            return self._empty_readings(names), {name: [] for name in names}

        rows = self._incremental_readings[:n]
        per_token = self._extract_coord_stream(rows, n, names)

        empty_agg = self._empty_readings(names)
        agg_idx = self._aggregate_forward_index(generated_ids)
        agg_row = rows[agg_idx] if agg_idx is not None and agg_idx < len(rows) else {}
        agg_vals = {
            name: agg_row.get(name, empty_agg[name]) for name in names
        }
        return agg_vals, per_token

    def _score_aggregate_only(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, "ProbeReading"]:
        """Score *only* the end-of-gen aggregate from the bounded tail ring.

        The aggregate-only capture path (``CaptureMode.AGGREGATE_ONLY``, shared by
        ``GATING_SUBSET`` whose per-token rows feed only the gate) scored nothing
        of the full roster per token; here we pool the last content token's slice
        (:meth:`_pooled_aggregate_slice`) and run one
        :meth:`Monitor.score_aggregate`.  ``generated_ids`` token ``k`` was
        produced by decode forward ``k``, so the last content token's forward
        index is ``last_content_index(generated_ids)``.
        """
        names = list(self._monitor.probe_names)
        empty = self._empty_readings(names)
        if not generated_ids:
            return empty
        if pooled is None:
            pooled = self._pooled_aggregate_slice(generated_ids)
        if not pooled:
            return empty
        # One-shot pooled read over the full roster — keep curved probes on the
        # cold foot solve so the aggregate is reproducible (it pools the last
        # *content* token, which may differ from the live loop's last forward,
        # so a warm foot seeded during gating-only subset scoring would not be
        # the right warm start here).  Routed through the geometry run's
        # ``observe_aggregate`` (the run protocol face — the last finalize
        # read that bypassed its run); the cold-foot reset above stays
        # session-side.
        self._monitor.enable_curved_warm(False)
        agg_vals = self._geometry_instrument.current_run.observe_aggregate(
            pooled,
        )
        # Fill any probe the pool missed (e.g. a layer absent from the ring).
        return {name: agg_vals.get(name, empty[name]) for name in names}

    def _score_lean_incremental(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Lean per-token coord stream + full aggregate from the tail ring.

        The lean-incremental capture path scored each token ``coords_only`` (just
        the cross-layer axis-0 coord / fraction, no nearest / assignment /
        per-layer trace), so the per-token coordinate stream comes straight from
        those stored rows.  The end-of-gen aggregate, which DOES carry the full
        reading, is re-scored once from the bounded tail ring via
        :meth:`_score_aggregate_only` — so no field is lost despite the lean
        per-token rows.
        """
        names = list(self._monitor.probe_names)
        n = len(generated_ids)
        rows = self._incremental_readings[:n]
        per_token = self._extract_coord_stream(rows, n, names)
        agg_vals = self._score_aggregate_only(
            generated_ids, pooled=pooled,
        )
        return agg_vals, per_token

    @overload
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: Literal[False] = False,
    ) -> dict[str, "ProbeReading"]: ...
    @overload
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: Literal[True],
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]: ...
    def score_hidden(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        per_token: bool = False,
    ) -> (
        dict[str, "ProbeReading"]
        | tuple[dict[str, "ProbeReading"], dict[str, list[float]]]
    ):
        """Score registered probes against a pre-captured hidden-state dict.

        Accepts any ``{layer_idx: Tensor}`` mapping — e.g. the
        ``GenerationResult.hidden_states`` dict from a prior
        ``generate(..., sampling=SamplingConfig(return_hidden=True))``
        call, or hidden states the caller captured externally.

        Shape rules:
        - Each value ``[D]``          → single-state aggregate.
          Returns ``dict[probe, ProbeReading]``.
        - Each value ``[T, D]``       → per-token stack.
          ``per_token=False`` (default) returns the aggregate pooled from
          row ``T-1``; ``per_token=True`` returns
          ``(aggregate, per_token_scores)`` where ``per_token_scores`` is
          the per-probe axis-0 coordinate stream.

        Mixed shapes (``[D]`` alongside ``[T, D]``) or uneven ``T`` across
        layers raise :class:`DrowseError`. Empty dict raises.
        """
        if not hidden:
            raise DrowseError("score_hidden: no layers provided")

        # Classify shapes up-front.
        shapes = [v.ndim for v in hidden.values()]
        if len(set(shapes)) > 1:
            by_ndim: dict[int, list[int]] = {}
            for layer_idx, t in hidden.items():
                by_ndim.setdefault(t.ndim, []).append(layer_idx)
            detail = ", ".join(
                f"ndim={n} at layers {ls}" for n, ls in sorted(by_ndim.items())
            )
            raise DrowseError(
                "score_hidden: mixed shapes in input; expected either all "
                f"[D] or all [T, D] across layers ({detail})",
            )
        if shapes[0] not in (1, 2):
            raise DrowseError(
                f"score_hidden: expected [D] or [T, D] tensors, got ndim={shapes[0]}",
            )

        # Dim pre-flight: each input tensor's last dim must match any
        # probe that covers that layer. Without this, a shape mismatch
        # would leak a raw torch RuntimeError at the scoring matmul,
        # violating the "all public errors are DrowseError" invariant.
        probe_manifolds = self._geometry_instrument.manifolds()
        for layer_idx, t in hidden.items():
            actual_dim = t.shape[-1]
            for probe_name, manifold in probe_manifolds.items():
                sub = manifold.layers.get(layer_idx)
                if sub is None:
                    continue
                expected_dim = sub.mean.shape[-1]
                if expected_dim != actual_dim:
                    raise DrowseError(
                        f"score_hidden: dim mismatch at layer {layer_idx} — "
                        f"got {actual_dim}, probe '{probe_name}' expects "
                        f"{expected_dim}",
                    )
                break  # one covering probe settles the expected dim

        if shapes[0] == 1:
            if per_token:
                # [D] input + per_token is meaningless.
                raise DrowseError(
                    "score_hidden: per_token=True requires [T, D] input; "
                    "got [D] (single state per layer)",
                )
            # Fall through to the monitor's single-state path.
            return self._monitor.score_single_token(hidden)

        # [T, D] path — delegate to monitor.score_stack. Wrap its
        # ValueError (uneven T across layers is the only path that
        # can reach here after the shape checks above) so callers
        # catching DrowseError get a uniform exception surface.
        try:
            agg, per_tok = self._monitor.score_stack(hidden)
        except ValueError as exc:
            raise DrowseError(f"score_hidden: {exc}") from exc
        return (agg, per_tok) if per_token else agg

    # -- Monitoring --

    def add_probe(
        self,
        selector: str,
        *,
        as_name: str | None = None,
        top_n: int = 3,
    ) -> str:
        """Attach a read-side probe — any shape — and return its name.

        One attach for vector and manifold probes alike: a vector probe is the
        rank-1 case of the unified subspace readout.  ``selector`` rides the
        same ``[ns/]name[:variant]`` shape the steering grammar uses for ``%``
        operands, so a probe shares the lazy-load cache with a manifold steered
        through ``<selector>%<position>``.  ``as_name`` overrides the registered
        name (default ``selector``); ``top_n`` sets the nearest-node list
        length.  Resolution order is in :meth:`_resolve_probe_manifold`.
        """
        # Reserved J-lens namespace: a ``jlens/<word>`` probe is NOT a linear
        # probe — it reads the readout channel (per-layer softmax probability
        # of the token under ``softmax(W_U · norm(J_l h))``, mean-banded),
        # the paper-native "how disposed is the model to say this
        # word" quantity, not a whitened coordinate along ``W_U[v] @ J_l``.
        # Routed to the session lens-probe registry; the Monitor never sees it
        # (no whitener requirement — the softmax is the calibration).
        if selector.startswith("jlens/"):
            name = self._lens_instrument.attach(selector, as_name=as_name)
        elif selector.startswith("sae/"):
            name = self._sae_instrument.attach(selector, as_name=as_name)
        else:
            # Geometry (Monitor) probes attach through their instrument like
            # the other families; the whitener build + manifold resolution +
            # monitor registration run under the exclusive-GPU section.
            name = self._geometry_instrument.attach(
                selector, as_name=as_name, top_n=top_n,
            )
        # New probe → _begin_capture attaches a different layer set than was
        # live when the prefix was prefilled; drop the cache so the next gen
        # re-prefills with the fresh capture-attach layout in place.
        self._invalidate_prefix_cache()
        # Transcript probe-hash cache is keyed by name; any change to the
        # registered probes invalidates the relevant entry.
        self._probe_hash_cache.pop(name, None)
        self._registry.invalidate_analytics()
        return name

    def remove_probe(self, name: str) -> None:
        """Detach a previously-attached probe (any shape).

        Lens/SAE detaches are safe mid-generation — a bound run's frozen
        binding keeps measuring the bind-time roster, so the removal
        applies at the next generation boundary.  A geometry detach
        mutates the live Monitor roster in-flight scoring walks, so it
        routes through the instrument's exclusive section and rejects
        while a generation is running (retry when idle).  The lens
        removal is the instrument's atomic ``try_detach`` — a bare
        membership check + direct registry delete bypassed the
        lens-state lock and could land inside a ``prepare`` snapshot or
        an adoption.
        """
        if self._lens_instrument.try_detach(name):
            pass
        elif self._sae_instrument.try_detach(name):
            pass
        else:
            self._geometry_instrument.detach(name)
        self._invalidate_prefix_cache()
        self._probe_hash_cache.pop(name, None)
        self._registry.invalidate_analytics()

    def _resolve_probe_manifold(self, selector: str) -> "Manifold":
        """Resolve a probe selector to a loaded :class:`Manifold`.

        Mirrors the steering resolver, in order: (1) a direction profile
        already in the registry (an ``extract`` / ``bake`` result, a
        registered ``Profile``, a fold cached by an earlier steer) → folded
        into a 1-node neutral-anchored ray; (2) a fitted manifold on disk
        (``[ns/]name[:variant]``) → loaded directly, so a 2-node ``pca`` reads
        rank-1 and a discover / curved fit reads rank-R; (3) a bare concept
        with neither → extracted, then folded.  Folds go through
        :meth:`ArtifactRegistry.fold_probe`, which tracks them, so evicting
        or re-fitting the artifact reaches these probes too.

        ``jlens/<word>`` selectors never reach this resolver — ``add_probe``
        routes them to the lens-probe registry (the readout channel), so the
        reserved namespace can't fall through to ``extract()`` and author a
        nonsense manifold named "jlens/<word>".
        """
        profile = self._registry.profiles.get(selector)
        if profile is None:
            try:
                return self._registry.ensure_manifold_loaded(selector)
            except ManifoldNotRegisteredError:
                pass
            _, profile = self.extract(selector)
        return self._registry.fold_probe(selector, profile)

    def _fold_profile_probe(self, name: str, profile: Any) -> "Manifold":
        """Fold a baked per-layer ``Profile`` into a 1-node neutral-anchored ray.

        A ``Profile`` is a per-layer baked direction; the monitor reads probes
        as flat manifolds (the rank-1 coordinate case), so fold it via the
        same primitive ``extract`` uses for a monopolar concept.  The registry
        calls this through :meth:`ArtifactRegistry.fold_probe`.
        """
        from drowse.geometry.fold import fold_directions_to_subspace
        return fold_directions_to_subspace(
            name, dict(profile), self.layer_means, whitener=self.whitener,
        )

    def _probe_hash(self, name: str) -> str | None:
        """Return sha256 hex of the baked tensor bytes for ``name``.

        Stamps :class:`drowse.loom.recipe.Recipe.probe_hashes` so transcript
        replay can detect probe drift between save and load. Cached on the
        session — adding
        or removing a probe invalidates the relevant cache entry.

        Returns ``None`` when the probe isn't registered.  Hashing is
        deterministic across machines: layers iterated in sorted order,
        each tensor's CPU bytes hashed (fp32 cast to keep dtype neutral
        across mixed-precision storage).
        """
        if name in self._probe_hash_cache:
            return self._probe_hash_cache[name]
        lens_digest = self._lens_instrument.probe_hash(name)
        if lens_digest is not None:
            # A lens probe has no baked tensor — the instrument hashes the
            # readout-channel identity (model, token, band, channel version)
            # so transcript drift detection works across a semantics change.
            self._probe_hash_cache[name] = lens_digest
            return lens_digest
        sae_digest = self._sae_instrument.probe_hash(name)
        if sae_digest is not None:
            # The instrument hashes the readout-channel identity; ``max_act``
            # is part of it (it sets the unit), so drift detection catches a
            # unit change.
            self._probe_hash_cache[name] = sae_digest
            return sae_digest
        geometry_digest = self._geometry_instrument.probe_hash(name)
        if geometry_digest is None:
            return None
        self._probe_hash_cache[name] = geometry_digest
        return geometry_digest

    def probe_hashes(self) -> dict[str, str]:
        """Return ``{probe_name: sha256_hex}`` for every registered probe.

        Roster names come from the instruments' LOCKED snapshots (raw
        registry expansion tears under a concurrent detach; round-7);
        the per-name hash lookup already tolerates a probe vanishing
        between enumeration and hashing.
        """
        out: dict[str, str] = {}
        for name in (
            *self._geometry_instrument.names,
            *self._lens_instrument.names,
            *self._sae_instrument.names,
        ):
            d = self._probe_hash(name)
            if d is not None:
                out[name] = d
        return out

    # -- Conversation (loom) --

    def diff_nodes(self, a_id: str, b_id: str) -> Any:
        """Return a :class:`drowse.loom.diff.NodeDiff` between two nodes."""
        return node_diff(self.tree, a_id, b_id)

    def fork_from_token(
        self,
        node_id: str,
        raw_index: int,
        alt_token_id: int | None = None,
        *,
        replacement_text: str | None = None,
        seed: int | None = None,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        """Regenerate ``node_id`` as a sibling with a replacement at one token.

        Forces ``alt_token_id`` (or the tokenization of ``replacement_text``)
        at ``raw_index`` of the node's raw decode sequence and samples the
        rest under the node's recipe; see :func:`drowse.loom.ops.plan_fork`.
        """
        turn = plan_fork(
            self.tree, self._tokenizer, node_id, raw_index, alt_token_id,
            replacement_text=replacement_text, seed=seed,
            default_max_tokens=self.config.max_new_tokens,
        )
        return self._generate_core(
            turn.input, on_token=on_token, **turn.generate_kwargs(),
        )

    def prefill_assistant(
        self,
        node_id: str,
        text: str,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        """Generate an assistant reply to user node ``node_id`` that opens with ``text``.

        See :func:`drowse.loom.ops.plan_prefill`.
        """
        turn = plan_prefill(
            self.tree, self._tokenizer, node_id, text,
            steering=steering, sampling=sampling,
            default_max_tokens=self.config.max_new_tokens,
        )
        return self._generate_core(
            turn.input, on_token=on_token, **turn.generate_kwargs(),
        )

    def append_turn(
        self,
        parent_node_id: str | None,
        text: str,
        *,
        role: Literal["user", "assistant"],
        raw: bool = False,
        role_label: str | None = None,
        thinking: str | None = None,
    ) -> str:
        """Commit one authored span in seat ``role`` without generating.

        ``raw`` is flat (base-model) mode, where seating is free; see
        :func:`drowse.loom.ops.append_turn`.
        """
        return append_turn(
            self.tree, self._tokenizer, self.scene_grammar, parent_node_id,
            text, seat=role, flat=raw, role_label=role_label, thinking=thinking,
        )

    def set_cast_member(
        self,
        label: str,
        *,
        steering: str | None = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        seed: int | None = None,
        notes: str = "",
    ) -> CastMember:
        """Create or replace the cast member under ``label`` (steering validated now)."""
        member = CastMember.from_parts(
            steering=steering, sampling=sampling, thinking=thinking,
            seed=seed, notes=notes,
        )
        self.tree.set_cast_member(label, member)
        return member

    def remove_cast_member(self, label: str) -> None:
        """Drop the cast member under ``label`` (no-op when absent)."""
        self.tree.remove_cast_member(label)

    def rewind(self) -> None:
        """Walk the active node back one submission (non-destructive)."""
        self.tree.rewind()

    def clear_history(self) -> None:
        """Reset the tree to a fresh root (destructive; refused mid-generation)."""
        self.tree.reset()

    def restore_tree(self, data: dict[str, Any]) -> LoomTree:
        """Atomically replace the live conversation with a validated tree.

        The caller must guarantee no generation is in flight (the server
        holds its session lock).  A tree saved under another model is
        refused; see :meth:`LoomTree.replacing`.
        """
        previous = self.tree
        restored = LoomTree.replacing(previous, data, model_id=self.model_id)
        self.tree = restored
        self._joint_logprob_cache.clear()
        restored.announce_replacement(previous)
        return restored

    # -- Prefix KV cache --

    def cache_prefix(
        self,
        messages: "list[dict[str, str]] | torch.Tensor | None",
        *,
        max_new_tokens: int | None = None,
        prefer_static: bool = False,
    ) -> int:
        """Pre-prefill an identical chat prefix so subsequent ``generate()``
        calls forward only the suffix.

        Useful for batch workloads that re-issue the same chat-template
        head (system + leading user-instruction) hundreds of times — the
        The motivating workload does 800 stateless
        generations with the same kaomoji instruction prefix tokens.
        Per-call savings scale with prefix_len / total_input_len.

        Accepts:
        - ``list[dict[str, str]]``: encoded via ``build_chat_input`` with
          ``add_generation_prompt=False`` (the prefix should be the
          turn(s) that PRECEDE the assistant turn — generation-prompt
          tokens for the assistant turn are part of the per-call suffix).
        - ``torch.Tensor`` of shape ``[seq_len]`` or ``[1, seq_len]``:
          stored verbatim. Use this when the natural common prefix sits
          mid-content and isn't a clean message-list boundary (e.g. a
          fixed instruction concatenated into the user message before
          the variable prompt body).
        - ``None``: clear the cache. Equivalent to ``cache_prefix()``.

        ``prefer_static=True`` builds a StaticCache-backed prefix entry when the
        session supports StaticCache. ``max_new_tokens`` sizes the decode
        headroom; callers that pass a larger generation budget later will miss
        this static entry and fall back to the regular full-prefill path. When
        StaticCache construction or prefill fails, the method falls back to the
        ordinary DynamicCache prefix entry.

        Returns the cached prefix length in tokens (0 when clearing).

        Caveats (DOCUMENTED INLINE because they're easy to step on):
        1. Only safe to call OUTSIDE any active ``session.steering()``
           scope.  The prefix is prefilled with whatever steering hooks
           are live at call time — if those differ from the steering
           regime active at consume time, the cached hidden states are
           stale.  We invalidate the cache automatically on push/pop,
           but the call itself errors if a scope is open.
        2. Hidden-state capture is suspended for the duration of the
           prefill so the cached prefix doesn't pollute later score
           buckets.  No guarantees about what the monitor sees mid-call;
           callers shouldn't rely on probe state during cache_prefix.
        3. The cached tokens MUST appear as a byte-prefix of every
           ``generate()`` call's ``input_ids``.  If the caller's
           messages drift, the cache silently MISSES (cheap — full
           prefill on miss) but never silently MIS-HITS — the
           ``input_ids[..., :prefix_len].equal(prefix_tokens)`` check
           is exact.
        """
        # Clear path.
        if messages is None:
            self._invalidate_prefix_cache()
            return 0

        if self._steering_composer.active:
            raise DrowseError(
                "cache_prefix called inside an active session.steering() "
                "scope; prefill must run with the neutral baseline so the "
                "cached prefix is consume-regime independent. Cache before "
                "entering any steering scope."
            )
        if self.is_generating:
            raise ConcurrentGenerationError(
                "cache_prefix called while a generation is in flight"
            )

        # Build prefix tokens.
        if isinstance(messages, torch.Tensor):
            prefix_ids = cast(torch.Tensor, messages)
            if prefix_ids.dim() == 1:
                prefix_ids = prefix_ids.unsqueeze(0)
            prefix_ids = prefix_ids.to(device=self._device, dtype=torch.long)
        else:
            # List of message dicts.  add_generation_prompt=False so we
            # don't bake the assistant-turn opener into the prefix —
            # the per-call suffix carries that, ensuring the same
            # cached prefix can serve multi-turn variants.
            prefix_ids = build_chat_input(
                self._tokenizer, list(messages),
                self.config.system_prompt,
                thinking=False,
                add_generation_prompt=False,
                scene=self.scene_grammar,
                traits=self.chat_traits,
            ).to(self._device)

        prefix_len = int(prefix_ids.shape[1])
        if prefix_len == 0:
            self._invalidate_prefix_cache()
            return 0

        # Replace any prior cache entry; old past_key_values goes out of
        # scope and is GC'd when no longer referenced.
        self._prefix_cache = None

        # Suspend capture for the prefill so the prefix tokens don't fill
        # the per-layer buckets the next generation's scoring code reads.
        # _begin_capture/_end_capture are idempotent re no-op no-probe
        # configurations, so the bracketing here is always safe.
        self._end_capture()

        use_static = bool(prefer_static and self._static_cache_active)
        static_max_cache_len: int | None = None
        past_key_values: object | None = None
        cache_position: torch.Tensor | None = None
        if use_static:
            try:
                from drowse.generation.static_cache import make_static_cache

                model_dtype = next(self._model.parameters()).dtype
                headroom = int(
                    max_new_tokens
                    if max_new_tokens is not None
                    else self.config.max_new_tokens
                )
                static_max_cache_len = prefix_len + max(headroom, 1)
                past_key_values = make_static_cache(
                    self._model,
                    max_cache_len=static_max_cache_len,
                    device=self._device,
                    dtype=model_dtype,
                )
                cache_position = torch.arange(
                    prefix_len, device=self._device, dtype=torch.long,
                )
            except Exception as exc:
                _log.debug(
                    "cache_prefix: StaticCache prefix skipped (%s)", exc,
                )
                use_static = False
                past_key_values = None
                cache_position = None
                static_max_cache_len = None

        try:
            with torch.inference_mode():
                if use_static and past_key_values is not None and cache_position is not None:
                    outputs = self._model(
                        input_ids=prefix_ids,
                        attention_mask=torch.ones_like(prefix_ids),
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                else:
                    outputs = self._model(
                        input_ids=prefix_ids,
                        attention_mask=torch.ones_like(prefix_ids),
                        use_cache=True,
                    )
        except Exception:
            if not use_static:
                raise
            _log.debug(
                "cache_prefix: StaticCache prefill failed; retrying DynamicCache",
                exc_info=True,
            )
            use_static = False
            past_key_values = None
            cache_position = None
            static_max_cache_len = None
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids),
                    use_cache=True,
                )

        if use_static and past_key_values is not None:
            # StaticCache mutates in place. Some modeling files return the same
            # object, others may return ``None``; keep the preallocated cache in
            # both cases.
            returned_pkv = outputs.past_key_values
            past_key_values = (
                returned_pkv if returned_pkv is not None else past_key_values
            )
        else:
            past_key_values = outputs.past_key_values
        if past_key_values is None:
            # Model doesn't expose KV cache (custom modeling that ignores
            # use_cache).  Nothing to cache; drop the prefix.
            return 0

        self._prefix_cache = PrefixCacheEntry.from_prefill(
            prefix_ids,
            past_key_values,
            static=use_static,
            max_cache_len=static_max_cache_len,
        )
        return prefix_len

    def _invalidate_prefix_cache(self) -> None:
        """Drop the prefix KV cache.

        Called on every state change that affects what a fresh prefill
        of the cached prefix would produce: steering push/pop, steer /
        unsteer, probe install / remove, profile autoload.  Cheap — just
        a reference drop; HF's cache objects are GC'd when their refcount
        falls to zero.
        """
        self._prefix_cache = None

    def _try_prefix_cache_hit(
        self,
        input_ids: torch.Tensor,
        *,
        static_eligible: bool = False,
        required_max_new_tokens: int | None = None,
    ) -> tuple[torch.Tensor, object, int, bool] | None:
        """``(suffix_ids, past_key_values, prefix_len, static)`` on a hit.

        The hit rules live in :meth:`PrefixCacheEntry.suffix_for`.
        """
        entry = self._prefix_cache
        if entry is None:
            return None
        suffix_ids = entry.suffix_for(
            input_ids,
            static_eligible=static_eligible,
            required_max_new_tokens=required_max_new_tokens,
        )
        if suffix_ids is None:
            return None
        return suffix_ids, entry.past_key_values, entry.prefix_len, entry.static

    # -- Generation helpers --

    @property
    def renderer(self) -> PromptRenderer:
        """The session's :class:`~drowse.generation.render.PromptRenderer`."""
        return self._renderer

    @property
    def scene_grammar(self) -> TurnGrammar | None:
        """The validated turn grammar for the loaded chat template, or None."""
        return self._renderer.scene_grammar

    def _finalize_generation(
        self, generated_ids: list[int], elapsed: float,
        vector_snapshot: dict[str, float], prompt_tokens: int = 0,
        stateless: bool = False,
        logprobs_list: list[tuple[int, float | None, list[Any]]] | None = None,
        applied_steering: str | None = None,
        *,
        return_hidden: bool = False,
        return_probe_readings: bool = True,
        assistant_node_id: str | None = None,
        mean_logprob: float | None = None,
        mean_surprise: float | None = None,
    ) -> GenerationResult:
        """Finalize result state through the generation finalizer collaborator."""
        from drowse.generation.finalize import finalize_generation

        return finalize_generation(
            self, generated_ids, elapsed, vector_snapshot,
            prompt_tokens=prompt_tokens,
            stateless=stateless,
            logprobs_list=logprobs_list,
            applied_steering=applied_steering,
            return_hidden=return_hidden,
            return_probe_readings=return_probe_readings,
            assistant_node_id=assistant_node_id,
            mean_logprob=mean_logprob,
            mean_surprise=mean_surprise,
            min_elapsed_for_rate=MIN_ELAPSED_FOR_RATE,
        )

    def _generation_preamble(self, input: Any, raw: bool, thinking: bool, stateless: bool = False,
                             parent_node_id: str | None = None,
                             user_role: str | None = None,
                             assistant_role: str | None = None,
                             gen_seat: str = "assistant",
                             system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT):
        """Shared input prep + gen-state reset.

        Steering is NOT installed here — the caller is expected to hold a
        ``session.steering()`` scope open across the generation.
        ``parent_node_id`` selects which loom-tree path the input is
        anchored against (default: the active path).  ``user_role`` /
        ``assistant_role`` are this send's per-message role labels (the
        roleplay scaffold): the new user turn renders under ``user_role``
        and the generation prompt under ``assistant_role`` (a role-bearing
        steering scope overrides the latter inside the renderer).
        """
        use_thinking = thinking and self.chat_traits.supports_thinking
        input_ids = self._renderer.prepare_input(
            input, raw=raw, thinking=use_thinking, stateless=stateless,
            parent_node_id=parent_node_id,
            user_role=user_role, assistant_role=assistant_role,
            gen_seat=gen_seat,
            system_prompt=system_prompt,
        )
        self._gen_state.reset()
        return input_ids, use_thinking, int(input_ids.shape[1])

    def _compose_gen_config(
        self, sampling: SamplingConfig | None,
    ) -> "GenerationConfig":
        """Build a per-call GenerationConfig from session defaults + sampling.

        Does NOT mutate ``self.config`` — returns a new frozen instance the
        generation worker holds for its lifetime.  ``None`` fields in
        ``sampling`` fall through to the session default.
        """
        from dataclasses import replace as _replace

        if sampling is None:
            return self.config
        overrides: dict[str, Any] = {}
        if sampling.temperature is not None:
            overrides["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            overrides["top_p"] = sampling.top_p
        if sampling.top_k is not None:
            overrides["top_k"] = sampling.top_k
        if sampling.max_tokens is not None:
            overrides["max_new_tokens"] = sampling.max_tokens
        if not overrides:
            return self.config
        return _replace(self.config, **overrides)

    # -- Generation: core --

    def _compose_generation_call(
        self,
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        system_prompt: str | None | InheritSystemPrompt,
        raw: bool,
        gen_seat: str,
        parent_node_id: str | None,
        recipe_override: "Recipe | str | None",
    ) -> ComposedCall:
        """Compose a generation's effective kwargs in one :func:`compose_call`.

        Tiers, weakest first: the gen label's cast member, the call's own
        kwargs, and for a regen the recipe of the node it reproduces
        (:meth:`LoomTree.anchor_recipe`) plus the modifier
        ``recipe_override`` derives from it — a :class:`Recipe` partial, a
        built-in mode, or a comma-separated partial-recipe string
        (:meth:`Recipe.compose_modifier`).  All four go through the one rule
        together, so the seed order it documents (modifier > call > anchor >
        cast) holds across them.

        The cast label rides the call's sampling box (``user_role`` for a
        user-seat gen, ``assistant_role`` otherwise), falling back to the
        structural seat name, so a bare continuation of a cast turn doesn't
        re-trigger the recipe.  Raw / flat mode renders no labels, so the
        roster doesn't apply there.
        """
        cast_recipe: Recipe | None = None
        if not raw:
            label = None
            if sampling is not None:
                label = (
                    sampling.user_role if gen_seat == "user"
                    else sampling.assistant_role
                )
            member = self.tree.cast.get(label or gen_seat)
            cast_recipe = member.recipe if member is not None else None
        anchor: Recipe | None = None
        modifier: Recipe | None = None
        if recipe_override is not None:
            anchor = self.tree.anchor_recipe(parent_node_id)
            modifier = (anchor or Recipe()).compose_modifier(recipe_override)
        if cast_recipe is None and modifier is None:
            return ComposedCall(steering, sampling, thinking, system_prompt)
        return compose_call(
            steering=steering, sampling=sampling, thinking=thinking,
            system_prompt=system_prompt,
            cast=cast_recipe, anchor=anchor, modifier=modifier,
        )

    def _effective_return_top_k(self, sampling: SamplingConfig | None) -> int:
        """Resolve the generation's logit-alternative width.

        Per-call ``return_top_k`` wins when positive; zero inherits the
        session default.
        """
        top_k = sampling.return_top_k if sampling is not None else 0
        return int(top_k or self._default_return_top_k)

    def _effective_readout_top_k(self, sampling: SamplingConfig | None) -> int:
        """Resolve the shared logit/J-lens/SAE top-k width.

        Read-side instruments keep the canonical eight-wide view when logit
        alternatives are disabled; otherwise they use that exact ``alts``
        width.
        """
        return self._effective_return_top_k(sampling) or _DEFAULT_READOUT_TOP_K

    def _prepare_generation_call(
        self,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
    ) -> "PreparedCall":
        """Normalize per-call generation controls before model work."""
        steering_obj = Steering.from_value(
            steering, profile_names=set(self._registry.profiles),
        )
        if thinking is None:
            config_thinking = getattr(self.config, "thinking", None)
            if steering_obj is not None and steering_obj.thinking is not None:
                use_thinking_req = bool(steering_obj.thinking)
            elif config_thinking is not None:
                use_thinking_req = bool(config_thinking)
            else:
                use_thinking_req = self.chat_traits.supports_thinking
        else:
            use_thinking_req = bool(thinking)

        gen_config = self._compose_gen_config(sampling)
        raw_lp = sampling.logprobs if sampling is not None else None
        raw_top_k = self._effective_return_top_k(sampling)
        if raw_top_k > 0:
            lp_count: int | None = (
                max(raw_top_k, raw_lp) if raw_lp is not None else raw_top_k
            )
        else:
            lp_count = raw_lp

        seed = sampling.seed if sampling is not None else None
        stop_tuple = sampling.stop if sampling is not None else None
        stop_list = list(stop_tuple) if stop_tuple else None
        logit_bias = sampling.logit_bias if sampling is not None else None
        presence_penalty = sampling.presence_penalty if sampling is not None else 0.0
        frequency_penalty = (
            sampling.frequency_penalty if sampling is not None else 0.0
        )
        logprobs_list: list[Any] | None = [] if raw_lp is not None else None
        return PreparedCall(
            steering_obj=steering_obj,
            use_thinking_req=use_thinking_req,
            gen_config=gen_config,
            lp_count=lp_count,
            seed=seed,
            stop_list=stop_list,
            logit_bias=logit_bias,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logprobs_list=logprobs_list,
        )

    def _snapshot_steering_alphas(self) -> dict[str, float]:
        """Flatten the active steering stack for result receipts."""
        return self._steering_composer.snapshot_steering_alphas()

    def _start_loom_assistant(
        self,
        input: Any,
        *,
        stateless: bool,
        raw: bool,
        parent_node_id: str | None,
        sampling: SamplingConfig | None,
        steering_obj: Steering | None,
        use_thinking_req: bool,
        gen_seat: str = "assistant",
        continuation_node_id: str | None = None,
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> str | None:
        """Open the loom node a stateful generation streams into.

        Stamps the generation receipt (steering, sampling, thinking, the
        resolved system prompt and the probe roster with its hashes) and
        delegates the node rules to :func:`drowse.loom.ops.begin_generation_turn`.
        """
        if stateless:
            return None
        recipe = Recipe(
            steering=str(steering_obj) if steering_obj is not None else None,
            sampling=sampling,
            thinking=use_thinking_req,
            system_prompt=(
                None if raw else self._renderer.resolve_system_prompt(system_prompt)
            ),
            probes=list(self._monitor.probe_names),
        ).with_probe_hashes(self._probe_hash)
        return begin_generation_turn(
            self.tree, input, raw=raw, parent_node_id=parent_node_id,
            sampling=sampling, recipe=recipe, gen_seat=gen_seat,
            continuation_node_id=continuation_node_id,
        )

    def _compiled_graph_thread_safe(self) -> bool:
        """Whether this thread owns PyTorch's warmed CUDA graph tree."""
        if not getattr(self, "_cuda_graphs_active", False):
            return True
        return (
            threading.get_ident()
            == getattr(self, "_compile_owner_thread", threading.get_ident())
        )

    def _acquire_generation_static_cache(
        self,
        model: PreTrainedModel,
        *,
        max_cache_len: int,
    ) -> object:
        """Return a reset, identity-stable StaticCache with enough capacity."""
        cached = self._generation_static_cache
        if cached is not None and self._generation_static_cache_len >= max_cache_len:
            reset = getattr(cached, "reset", None)
            if callable(reset):
                reset()
                return cached
            self._generation_static_cache = None
            self._generation_static_cache_len = 0

        from drowse.generation.static_cache import make_static_cache

        model_dtype = next(model.parameters()).dtype
        cache = make_static_cache(
            model,
            max_cache_len=max_cache_len,
            device=self._device,
            dtype=model_dtype,
        )
        self._generation_static_cache = cache
        self._generation_static_cache_len = max_cache_len
        return cache

    def _run_generation_loop(
        self,
        input_ids: torch.Tensor,
        gen_config: GenerationConfig,
        *,
        use_thinking: bool,
        want_hidden: bool,
        effective_tap: "StepTokenCallback | None",
        seed: int | None,
        stop_list: list[str] | None,
        logit_bias: dict[int, float] | None,
        presence_penalty: float,
        frequency_penalty: float,
        lp_count: int | None,
        forced_prefix: list[int] | None = None,
        want_perplexity: bool = True,
        authored_capture: AuthoredPromptCapture | None = None,
    ) -> tuple[list[int], float]:
        """Run the decode loop once capture and steering are installed."""
        cached_pkv = None
        cache_position_offset = 0
        cached_static = False
        effective_input_ids = input_ids
        static_cache_eligible = (
            self._static_cache_active
            and self._compiled_graph_thread_safe()
            and (
                self._steering.all_fast_path()
                or self._steering.static_steerable()
            )
        )
        # The cached prefix KV is unsteered.  It's safe to reuse whenever the
        # active steering doesn't touch the prefill region — ``@response`` /
        # ``@generated`` / probe-gated steering leaves the prompt KV identical
        # to unsteered, so a steered (response-phase) batch can still share one
        # prefill.  ``want_hidden`` (needs every prefix hidden state) and
        # thinking (different decode path) still force a full re-prefill.
        if (
            not want_hidden
            and not use_thinking
            and not self._steering_composer.steering_active_in_prefill()
            and self._prefix_cache is not None
        ):
            hit = self._try_prefix_cache_hit(
                input_ids,
                static_eligible=static_cache_eligible,
                required_max_new_tokens=gen_config.max_new_tokens,
            )
            if hit is not None:
                suffix_ids, cached_pkv, cache_position_offset, cached_static = hit
                effective_input_ids = suffix_ids

        prompt_capture_pending = False
        if authored_capture is not None:
            self._capture.set_prompt_positions(authored_capture.match(
                input_ids, cache_position_offset=cache_position_offset,
            ))
            prompt_capture_pending = bool(authored_capture.matches)

        base_step_callback = (
            self._capture.ingest_persistent
            if self._capture_state.persistent
            else self._capture.fire_step_sink
        )

        def _step_callback(step_id: int) -> None:
            nonlocal prompt_capture_pending
            # The first callback follows the full/suffix prefill.  Score the
            # visible authored positions in prompt order before the ordinary
            # final-prompt sink advances curved monitor feet to the state that
            # produces generated token 0.
            if prompt_capture_pending and authored_capture is not None:
                prompt_capture_pending = False
                authored_capture.persist(self._capture.prompt_stacked())
            base_step_callback(step_id)

        start = time.monotonic()
        composer = self._steering_composer
        gating_callback = (
            composer.build_gating_score_callback()
            if composer.steering_needs_probe_gating()
            else None
        )
        # StaticCache (+ compile fusion on MPS, + CUDA-graph capture on CUDA) is
        # eligible when generation is unsteered (``all_fast_path``) OR every
        # steered layer is the static single-affine fast path
        # (``static_steerable`` — Trigger.BOTH affine slide, no ctx, no foot, no
        # gate).  StaticCache never bypasses the forward hooks, so the steering
        # still applies; only the KV buffers are preallocated and the decode
        # shape stays fixed (what lets the compiled graph stay specialized).
        # ``_static_cache_active`` is device-agnostic (set when compile stuck or
        # CUDA graphs are on).  Curved / gated / phased steering keeps the eager
        # DynamicCache path.
        use_static_cache = static_cache_eligible and (
            cached_pkv is None or cached_static
        )
        # Compiled-model routing (CUDA/MPS). The graph was traced with ONLY the
        # persistent branchless offset + capture hooks present (attached
        # pre-compile), so the compiled module is correct exactly when the live
        # hook topology matches that: no *transient* per-token capture hooks
        # (``self._capture.is_transient()`` false — true for the persistent-capture
        # and no-probe paths) AND either unsteered or a static-affine steering that
        # lowered to the persistent offset buffers (``SteeringManager.uses_compiled_offsets``
        # — the push rides the offset, no transient hook).  A probed gen on the
        # persistent-capture path satisfies both, so it keeps the
        # compiled graph.  Everything else (transient ctx-consulting steering
        # hooks, transient capture hooks for curved/gated/return_hidden) graph-breaks
        # / recompiles, so route it to the eager original (``_orig_mod``) +
        # DynamicCache — where the same persistent hooks still apply any offsets
        # and capture, so correctness holds and ``compile=True`` never regresses
        # the hooked path. On CUDA this boundary also avoids Dynamo retracing on
        # every per-generation hook topology.
        gen_model = self._model
        steering_uses_compiled_offsets = bool(
            self._steering.uses_compiled_offsets
        )
        if self._compiled and self._device.type in {"cuda", "mps"}:
            compiled_clean = not self._capture.is_transient() and (
                steering_uses_compiled_offsets
                or self._steering.all_fast_path()
            )
            if compiled_clean and self._compiled_graph_thread_safe():
                use_static_cache = self._static_cache_active and (
                    cached_pkv is None or cached_static
                )
            else:
                gen_model = getattr(self._model, "_orig_mod", self._model)
                use_static_cache = False
        generation_pkv = cached_pkv
        if use_static_cache and generation_pkv is None:
            # Reserve at least the session default decode window and bucket the
            # total length.  Most interactive requests then keep one cache
            # allocation and one set of Dynamo guards even when their per-call
            # max_tokens differ.  The session lock serializes access, so reset +
            # reuse cannot race another generation.
            reserve_tokens = max(
                int(gen_config.max_new_tokens),
                int(self.config.max_new_tokens),
                1,
            )
            required_len = (
                cache_position_offset
                + int(effective_input_ids.shape[1])
                + reserve_tokens
            )
            cache_len = ((required_len + 255) // 256) * 256
            try:
                generation_pkv = self._acquire_generation_static_cache(
                    gen_model,
                    max_cache_len=cache_len,
                )
            except Exception as exc:
                _log.warning(
                    "Reusable StaticCache allocation failed (%s: %s); "
                    "falling back to DynamicCache",
                    type(exc).__name__,
                    exc,
                )
                use_static_cache = False
        generated_ids = generate_steered(
            gen_model, self._tokenizer, effective_input_ids,
            gen_config, self._gen_state, thinking=use_thinking,
            on_token=effective_tap,
            seed=seed, stop=stop_list, logit_bias=logit_bias,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logprobs=lp_count,
            trigger_ctx=self._steering.ctx,
            past_key_values=generation_pkv,
            cache_position_offset=cache_position_offset,
            score_callback=gating_callback,
            # Per-token probe scoring fires post-forward, not inside the
            # capture hook.  On the persistent-capture path the step callback is
            # ``ingest_persistent`` (accumulate from the persistent buffers +
            # fire the step sink); otherwise ``fire_step_sink`` (the transient
            # hooks already accumulated in-forward).  Both no-op when no per-token
            # sink is installed (aggregate-only / full-retention / no-probe).
            step_callback=_step_callback,
            use_static_cache=use_static_cache,
            forced_prefix=forced_prefix,
            steering_active=bool(
                self._steering.hooks or steering_uses_compiled_offsets
            ),
            want_perplexity=want_perplexity,
            traits=self.chat_traits,
        )
        elapsed = time.monotonic() - start

        entry = self._prefix_cache
        if (
            cached_pkv is not None
            and entry is not None
            and entry.past_key_values is cached_pkv
            and not entry.rollback()
        ):
            self._invalidate_prefix_cache()
        return generated_ids, elapsed

    def _generate_core(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenCallback | None = None,
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        forced_prefix: list[int] | None = None,
        gen_seat: str = "assistant",
        append_same_role: bool = False,
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> GenerationResult:
        """Shared generation implementation.

        Runs inside :meth:`_generation_transaction` (the lock, the re-entry
        guard, and the exception-hard teardown), composes a per-call
        GenerationConfig, opens an internal steering scope (if any), runs
        ``generate_steered`` with capture attached, and finalizes the result.
        ``generate`` and ``generate_stream`` are thin wrappers around this.
        """
        with self._generation_transaction() as txn:

            # Cast roster (weakest), the call's kwargs, and a regen's
            # anchor + modifier compose in one pass, before the Steering
            # object is built.
            steering, sampling, thinking, system_prompt = (
                self._compose_generation_call(
                    steering=steering,
                    sampling=sampling,
                    thinking=thinking,
                    system_prompt=system_prompt,
                    raw=raw,
                    gen_seat=gen_seat,
                    parent_node_id=parent_node_id,
                    recipe_override=recipe_override,
                )
            )
            # ``""`` is the composition tiers' explicit "unsteered" (it stops
            # a cast default from filling the slot, and is what the
            # ``unsteered`` / ``inverted`` modifiers produce); once composed
            # it means no steering, which the grammar spells ``None``.
            if isinstance(steering, str) and not steering.strip():
                steering = None

            # Ordinary one-shot continuation is message-oriented: if the
            # selected leaf already occupies the same structural seat under
            # the same effective role label, regenerate that node in place
            # with its current text as the forced prefix.  Specialist fan,
            # fork, and prefill paths leave this flag off and retain their
            # explicit sibling semantics.
            continuation_node_id: str | None = None
            continuation_prefix: list[int] | None = None
            continuation_parent_id: str | None = None
            if (
                append_same_role
                and not stateless
                and forced_prefix is None
                and (input is None or (raw and input == ""))
            ):
                candidate_id = parent_node_id or self.tree.active_node_id
                candidate = self.tree.get(candidate_id)
                candidate_seat = (
                    "user" if gen_seat == "user" else "assistant"
                )
                generated_label = None
                if sampling is not None and not raw:
                    generated_label = (
                        sampling.user_role
                        if gen_seat == "user"
                        else sampling.assistant_role
                    )
                if (
                    candidate.role == candidate_seat
                    and candidate.role_label == generated_label
                ):
                    continuation_node_id = candidate.id
                    continuation_parent_id = candidate.parent_id
                    continuation_prefix = list(
                        self._tokenizer.encode(
                            candidate.text, add_special_tokens=False,
                        )
                    )
                    if continuation_prefix:
                        from dataclasses import replace as _replace
                        base_sampling = (
                            sampling if sampling is not None else SamplingConfig()
                        )
                        base_max = (
                            base_sampling.max_tokens
                            or self.config.max_new_tokens
                        )
                        sampling = _replace(
                            base_sampling,
                            max_tokens=len(continuation_prefix) + base_max,
                        )
                    # Same-message continuation is an answer/body prefill,
                    # not a request to open a fresh thinking channel.
                    thinking = False

            # Bound out by name (not unpacked positionally) because the body
            # below reads all ten deep in the transaction and reassigns
            # ``stop_list`` after the seat augmentation.
            prepared = self._prepare_generation_call(steering, sampling, thinking)
            steering_obj = prepared.steering_obj
            use_thinking_req = prepared.use_thinking_req
            gen_config = prepared.gen_config
            lp_count = prepared.lp_count
            seed = prepared.seed
            stop_list = prepared.stop_list
            logit_bias = prepared.logit_bias
            presence_penalty = prepared.presence_penalty
            frequency_penalty = prepared.frequency_penalty
            logprobs_list = prepared.logprobs_list
            readout_top_k = self._effective_readout_top_k(sampling)
            stop_list = self._renderer.seat_stop_strings(
                stop_list, gen_seat=gen_seat, raw=raw,
            )
            return_probe_readings = bool(
                sampling is None or sampling.return_probe_readings
            )
            # ``mean_logprob_accum`` averages chosen-token logprobs over the
            # non-thinking response span — surfaced on ``LoomNode.mean_logprob``
            # at finalize-assistant time so the loom sidebar can sort siblings
            # by surprise.  Populates whenever the engine captures chosen
            # logprob (any ``on_token`` consumer with ``lp_count is not None``
            # — i.e. the loom path or an explicit logprobs request).
            mean_logprob_sum: float = 0.0
            mean_logprob_count: int = 0
            mean_logprob_missing = False
            # CAA live toggle: when off, every per-token monitor consumer is
            # masked at the source — generations run aggregate-only capture
            # (probes still report the end-of-gen aggregate) and only probe
            # gates can force a per-token subset.
            _live_scores_on = self._geometry_instrument.is_live
            _callback_options = consumer_options(on_token)
            _wants_live_token_scores = bool(
                _live_scores_on
                and on_token is not None
                and _callback_options.live_scores
            )
            _persists_layer_scores = bool(
                _live_scores_on
                and (
                    (sampling is not None and sampling.persist_per_layer_scores)
                    or (
                        on_token is not None and _callback_options.per_layer_scores
                    )
                )
            )
            # Probe-inspector live point + fading trail: stamp each token's probe
            # reading with per-layer whitened subspace coords.  Gates the monitor
            # post-pass (set here, reset in the teardown ``finally``) and forces
            # per-token scoring so the per-token reading actually exists to carry it.
            _persists_subspace_coords = bool(
                _live_scores_on
                and sampling is not None
                and sampling.persist_subspace_coords
            )
            self._monitor.set_subspace_coords(_persists_subspace_coords)
            from drowse.generation.token_payloads import (
                TokenProbePayload,
                build_token_probe_payload,
            )

            _has_monitor_probes = bool(self._monitor.probe_names)

            def _token_tap(
                text: str, is_thinking: bool, tid: int | None, lp: float | None,
                top_alts: list[TokenAlt] | None, perplexity: float | None = None,
                step_id: int = -1,
            ) -> None:
                # ``step_id`` is the decode loop's forward index (the internal
                # StepTokenCallback contract) — the SAME value this forward's
                # gate callback received, so the lens/SAE display steps below
                # can reuse the gate's step-keyed stash rows.  User callbacks keep
                # the public six-argument TokenCallback shape (invoked below).
                nonlocal mean_logprob_sum, mean_logprob_count, mean_logprob_missing
                self._last_token_probe_payload = None
                self._last_token_payload = None
                if logprobs_list is not None and tid is not None and tid >= 0 and not is_thinking:
                    logprobs_list.append((tid, lp, top_alts or []))
                if lp is None and tid is not None and tid >= 0 and not is_thinking:
                    mean_logprob_missing = True
                if lp is not None and tid is not None and tid >= 0 and not is_thinking:
                    mean_logprob_sum += lp
                    mean_logprob_count += 1
                needs_scores = bool(
                    _live_scores_on
                    and _has_monitor_probes
                    and (
                        assistant_node_id is not None
                        or _wants_live_token_scores
                        or _persists_subspace_coords
                    )
                )
                payload = (
                    build_token_probe_payload(
                        capture=self._capture,
                        capture_state=self._capture_state,
                        incremental_readings=self._incremental_readings,
                        geometry_run=self._geometry_instrument.current_run,
                        step_id=step_id,
                        needs_scores=needs_scores,
                        persists_layer_scores=_persists_layer_scores,
                        assistant_node_id=assistant_node_id,
                    )
                    if needs_scores
                    else None
                )
                # Live J-lens readout (None when off): the step's top-k
                # lens tokens per selected layer + the layer-aggregated
                # chip list.  ``step_id`` lets the display reuse the gate
                # callback's stash rows iff they came from THIS forward
                # (step identity replaced the old freshness handshake).
                lens_step = (
                    self._lens_instrument.live_readout_step(
                        top_k=readout_top_k, step_id=step_id,
                    )
                    if _has_lens_consumer
                    else None
                )
                sae_step = (
                    self._sae_instrument.live_readout_step(
                        top_k=readout_top_k, step_id=step_id,
                    )
                    if _has_sae_consumer
                    else None
                )
                # Pinned lens probes tick on the same step (readings extracted
                # from the display logits inside the readout step) — merge them
                # into every populated probe channel so the loom row, trait
                # stream, and WS frames carry them uniformly.
                if lens_step is not None and self._lens_instrument.last_step_readings:
                    if payload is None:
                        payload = TokenProbePayload()
                    payload.merge_readings(
                        self._lens_instrument.last_step_readings,
                        family="lens",
                        per_layer=(
                            assistant_node_id is not None
                            and _persists_layer_scores
                        ),
                    )
                if sae_step is not None and self._sae_instrument.last_step_readings:
                    if payload is None:
                        payload = TokenProbePayload()
                    payload.merge_readings(
                        self._sae_instrument.last_step_readings,
                        family="sae",
                        per_layer=(
                            assistant_node_id is not None
                            and _persists_layer_scores
                        ),
                    )
                scores = payload.scores if payload is not None else None
                per_layer_payload = (
                    payload.per_layer_scores if payload is not None else None
                )
                lens_payload = lens_step[0] if lens_step is not None else None
                lens_aggregate_payload = (
                    lens_step[1] if lens_step is not None else None
                )
                lens_token_ids = (
                    lens_step[2]
                    if lens_step is not None and len(lens_step) > 2
                    else None
                )
                if (
                    lens_payload
                    or lens_aggregate_payload
                    or sae_step
                    or (
                        payload is not None
                        and (
                            payload.scores
                            or payload.per_layer_scores
                            or payload.has_readings
                        )
                    )
                ):
                    if payload is None:
                        payload = TokenProbePayload()
                    self._last_token_payload = payload
                    recipe_steering = None
                    if assistant_node_id is not None:
                        node = self.tree.nodes.get(assistant_node_id)
                        if node is not None and node.recipe is not None:
                            recipe_steering = node.recipe.steering
                    self._last_token_probe_payload = payload.to_token_payload(
                        lens=lens_payload,
                        lens_aggregate=lens_aggregate_payload,
                        lens_token_ids=lens_token_ids,
                        lens_source=(
                            self._lens_instrument.live.get("source")
                            if self._lens_instrument.live is not None
                            else None
                        ),
                        sae=sae_step,
                        sae_source=(
                            self._sae_instrument.live.get("source")
                            if self._sae_instrument.live is not None
                            else None
                        ),
                        sae_layer=(
                            int(self._sae_instrument.live["layer"])
                            if self._sae_instrument.live is not None
                            else None
                        ),
                        steering=recipe_steering,
                    )
                if assistant_node_id is not None and tid is not None:
                    token_row: dict[str, Any] = {
                        "token_id": int(tid),
                        "text": text,
                        "logprob": float(lp) if lp is not None else None,
                        "perplexity": (
                            float(perplexity) if perplexity is not None else None
                        ),
                    }
                    # The decode loop records this emission's
                    # ``generated_ids`` index just before the tap runs: the
                    # join key a logit fork uses to slice ``raw_token_ids``.
                    # Stamped here so the row is complete (and never
                    # mutated again) once it lands in the tree.
                    emit_map = self._gen_state.emit_map
                    if emit_map:
                        token_row["raw_index"] = int(emit_map[-1][0])
                    if top_alts:
                        token_row["top_alts"] = [
                            {
                                "id": int(a.id),
                                "text": a.text,
                                "logprob": float(a.logprob),
                            }
                            for a in top_alts
                        ]
                    if scores:
                        token_row["probes"] = {
                            p: round(float(v), 6) for p, v in scores.items()
                        }
                    if per_layer_payload:
                        token_row["per_layer_scores"] = per_layer_payload
                    measurements = (
                        self._last_token_probe_payload.get("measurements")
                        if self._last_token_probe_payload is not None
                        else None
                    )
                    if measurements:
                        token_row["measurements"] = measurements
                    self.tree.append_token(
                        assistant_node_id,
                        token_row,
                        thinking=bool(is_thinking),
                    )
                if on_token is not None:
                    on_token(text, is_thinking, tid, lp, top_alts, perplexity)

            # Pass _token_tap into generate_steered only when at least one of its
            # branches is live: caller-supplied on_token, logprobs collection, or
            # the loom probe row.  When all are inactive, _token_tap
            # would be a no-op called once per generated token, AND its presence
            # forces generate_steered to compute the unconditional fp32
            # log_softmax + entropy sync per step (gate at generation.py:571).
            # Skipping it here trims that cost from the stateless prefill
            # workload (800 back-to-back gens of ~16 tokens each, no logprobs,
            # no streaming).  Stop-sequence behavior is preserved by
            # not wiring _token_tap=None when stop_list is set — the tokenizer-
            # decode + stop-match in generation.py only runs under on_token.
            # The tap also writes per-token ``probes`` / ``per_layer_scores``
            # onto the loom row when probes are loaded and the gen is loom-
            # attached — required so a webui refresh can rehydrate highlight
            # tints and the token-drilldown heatmap from the server tree.
            _persists_probe_row = bool(
                _live_scores_on
                and return_probe_readings
                and not stateless
                and _has_monitor_probes
            )
            _need_tap = (
                on_token is not None
                or logprobs_list is not None
                or _persists_probe_row
                or stop_list is not None
            )
            _has_lens_consumer = bool(
                self._lens_instrument.live is not None
                and on_token is not None
                and _callback_options.lens_readout
            )
            _has_sae_consumer = bool(
                self._sae_instrument.live is not None
                and on_token is not None
                and _callback_options.sae_readout
            )
            _effective_tap = _token_tap if _need_tap else None

            # Compiled-clean eligibility for this gen: the compiled CUDA/MPS
            # graph + static cache are live, the persistent capture buffers were
            # adopted, and the caller didn't ask for the full per-step hidden
            # stack (``return_hidden`` keeps the transient full-retention
            # capture).  Capture then rides the persistent buffers; a static
            # steering plan rides the offsets — both compile-clean.
            self._compiled_clean_eligible = bool(
                self._compiled
                and self._device.type in {"cuda", "mps"}
                and self._static_cache_active
                and self._capture_buffers
                and not (sampling and sampling.return_hidden)
            )

            if steering_obj is not None and steering_obj.alphas:
                txn.steering_cm = self.steering(steering_obj)

            # Snapshot the chat-history anchor BEFORE _start_loom_assistant
            # mutates the tree.  Otherwise the new (empty) assistant node it
            # creates becomes the active leaf, and ``messages_for(None)`` later
            # walks INCLUDING that empty assistant — producing a trailing
            # empty-content assistant message AND a duplicated trailing user
            # turn after the render re-appends ``input``.  Strict chat
            # templates (Mistral 3) reject the empty assistant outright with
            # ``Assistant message must have a string or a list of chunks ...``;
            # permissive templates render the duplicate user silently and
            # degrade prompt quality.
            chat_history_anchor = (
                continuation_parent_id
                if continuation_node_id is not None
                else parent_node_id
            )
            if (
                not stateless
                and (isinstance(input, str) or input is None)
                and chat_history_anchor is None
            ):
                chat_history_anchor = self.tree.active_node_id

            assistant_node_id = self._start_loom_assistant(
                input,
                stateless=stateless,
                raw=raw,
                parent_node_id=parent_node_id,
                sampling=sampling,
                steering_obj=steering_obj,
                use_thinking_req=use_thinking_req,
                gen_seat=gen_seat,
                system_prompt=system_prompt,
                continuation_node_id=continuation_node_id,
            )

            if txn.steering_cm is not None:
                txn.steering_cm.__enter__()
            input_ids, use_thinking, prompt_tokens = self._generation_preamble(
                input, raw, use_thinking_req, stateless=stateless,
                parent_node_id=chat_history_anchor,
                user_role=sampling.user_role if sampling is not None else None,
                assistant_role=(
                    sampling.assistant_role if sampling is not None else None
                ),
                gen_seat=gen_seat,
                system_prompt=system_prompt,
            )
            # Refresh snapshot now that steering is pushed (first-scope case).
            vector_snapshot: dict[str, float] = self._snapshot_steering_alphas()

            want_hidden = bool(sampling and sampling.return_hidden)
            self._lens_instrument.active_for_generation = _has_lens_consumer
            self._sae_instrument.active_for_generation = _has_sae_consumer
            # F11: nothing is tokenized unless a channel will measure the rows.
            authored_capture = plan_authored_capture(
                self.tree, self._tokenizer, assistant_node_id, raw=raw,
                monitor=self._monitor,
                lens=self._lens_instrument,
                sae=self._sae_instrument,
                monitor_active=bool(_live_scores_on and _has_monitor_probes),
                lens_active=_has_lens_consumer,
                sae_active=_has_sae_consumer,
                readout_top_k=readout_top_k,
                persist_per_layer_scores=_persists_layer_scores,
                content_end=lambda: int(self._renderer.prepare_input(
                    input,
                    raw=False,
                    thinking=use_thinking,
                    stateless=stateless,
                    parent_node_id=chat_history_anchor,
                    user_role=(
                        sampling.user_role if sampling is not None else None
                    ),
                    assistant_role=(
                        sampling.assistant_role if sampling is not None else None
                    ),
                    to_device=False,
                    gen_seat=gen_seat,
                    system_prompt=system_prompt,
                    add_generation_prompt=False,
                ).shape[1]),
            )
            # Fold every consumer of this generation into one read demand.
            # Per-token scoring is only needed when something consumes a
            # per-token reading: a probe gate, a loom token row,
            # a live-scores client, or a per-layer-heatmap persist.
            # Otherwise (probes attached but only the aggregate wanted, e.g. a
            # stateless server gen) the capture skips per-token scoring
            # entirely and pools the aggregate once at finalize.
            read_demand = self._resolve_read_demand(
                want_hidden=want_hidden,
                live_scores_on=_live_scores_on,
                has_monitor_probes=_has_monitor_probes,
                loom_attached=assistant_node_id is not None,
                wants_live_token_scores=_wants_live_token_scores,
                persists_layer_scores=_persists_layer_scores,
                persists_probe_row=_persists_probe_row,
                persists_subspace_coords=_persists_subspace_coords,
                live_lens_active=_has_lens_consumer,
                live_sae_active=_has_sae_consumer,
                final_probe_aggregate=return_probe_readings,
                capture_prompt=authored_capture is not None,
            )
            self.events.emit(GenerationStarted(input=input, stateless=stateless))
            try:
                # Capture attach + monitor live + ctx.reset live INSIDE the
                # inner try so a BaseException (KeyboardInterrupt, etc.)
                # between any pair of these still hits the cleanup finally.
                # ``_end_capture`` and ``end_live`` are idempotent.
                self._begin_capture(read_demand, widen=want_hidden)
                # Reset the steering manager's TriggerContext for this gen;
                # ``generate_steered`` mutates it at lifecycle boundaries.
                self._steering.ctx.reset()
                # Cold-start every manifold foot-follower so this gen re-seeds
                # at the origin instead of inheriting the prior run's foot.
                self._steering.reset_manifold_feet()
                self._gen_phase = GenState.RUNNING

                generated_ids, elapsed = self._run_generation_loop(
                    input_ids,
                    gen_config,
                    use_thinking=use_thinking,
                    want_hidden=want_hidden,
                    effective_tap=_effective_tap,
                    seed=seed,
                    stop_list=stop_list,
                    logit_bias=logit_bias,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    lp_count=lp_count,
                    forced_prefix=(
                        continuation_prefix
                        if continuation_node_id is not None
                        else forced_prefix
                    ),
                    # Per-token perplexity costs one host sync/token.  Compute
                    # it only when something surfaces it: a loom-attached gen
                    # persists it on the token row, an interactive (non-
                    # stateless) gen may show it, or a caller opts its on_token
                    # in.  Stateless server streaming (OpenAI/Ollama deltas, the
                    # native WS tap) never reads per-token ppl, so it skips the
                    # sync.  Callers can force it via ``on_token
                    # requests perplexity through ``TokenConsumerOptions``.
                    want_perplexity=(
                        not stateless
                        or assistant_node_id is not None
                        or (
                            on_token is not None and _callback_options.perplexity
                        )
                    ),
                    authored_capture=authored_capture,
                )
            finally:
                self._gen_state.stop_requested.set()
                self._end_capture()
                if txn.steering_cm is not None:
                    # The generation's own scope teardown passes the
                    # mid-generation guard; ``swallow=False`` so a teardown
                    # failure surfaces here.  Clearing the slot hands the
                    # transaction an already-popped scope, so its own teardown
                    # has nothing left to do.
                    txn.steering_cm.close(internal=True)
                    txn.steering_cm = None
                self._gen_phase = GenState.FINALIZING

            applied_steering = (
                str(steering_obj) if steering_obj is not None else None
            )
            # Convert the in-loop logprob accumulator
            # into per-turn rollups before handing finalize the slot.
            # ``mean_logprob_count == 0`` covers both "no captures because
            # gen was empty" and "no captures because no on_token consumer
            # was wired" — both produce ``None`` so the wire/tree carry a
            # current empty/no-capture shape.
            _mean_logprob_out: float | None = None
            _mean_surprise_out: float | None = None
            if mean_logprob_count > 0 and not mean_logprob_missing:
                _mean_logprob_out = mean_logprob_sum / mean_logprob_count
                _mean_surprise_out = -_mean_logprob_out
            result = self._finalize_generation(
                generated_ids, elapsed, vector_snapshot,
                prompt_tokens=prompt_tokens, stateless=stateless,
                logprobs_list=logprobs_list,
                applied_steering=applied_steering,
                return_hidden=want_hidden,
                return_probe_readings=return_probe_readings,
                assistant_node_id=assistant_node_id,
                mean_logprob=_mean_logprob_out,
                mean_surprise=_mean_surprise_out,
            )
            self.events.emit(GenerationFinished(result=result))
            return result

    def _generate_runset(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenCallback | None = None,
        parent_node_id: str | None = None,
        n: int = 1,
        recipe_override: "Recipe | str | None" = None,
        gen_seat: str = "assistant",
        append_same_role: bool = True,
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> RunSet:
        """Run one or more sibling generations and return a ``RunSet``."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n == 1:
            result = self._generate_core(
                input,
                steering=steering,
                sampling=sampling,
                stateless=stateless,
                raw=raw,
                thinking=thinking,
                on_token=on_token,
                parent_node_id=parent_node_id,
                recipe_override=recipe_override,
                gen_seat=gen_seat,
                system_prompt=system_prompt,
                append_same_role=append_same_role,
            )
            node_id = self.tree.active_node_id if not stateless else None
            return RunSet([result], node_ids=[node_id], kind="generation")

        # The batched fast fan renders through its own prefix path — route
        # non-assistant seats through the serial runner instead.
        fast_fan = None if gen_seat != "assistant" or not isinstance(system_prompt, InheritSystemPrompt) else self._generate_fan_fast(
            input,
            steering=steering,
            sampling=sampling,
            stateless=stateless,
            raw=raw,
            thinking=thinking,
            on_token=on_token,
            parent_node_id=parent_node_id,
            n=n,
            recipe_override=recipe_override,
        )
        if fast_fan is not None:
            return fast_fan

        # N-way regen: derive per-sibling seeds from the supplied base seed
        # (or a fresh entropy-derived one). Each iteration runs
        # ``_generate_core`` independently; ``add_user_turn`` dedups so all
        # siblings share the same user-parent.
        base_seed = sampling.seed if sampling is not None else None
        schedule = derive_seed_schedule(base_seed, n)
        jobs: list[_SerialGenerationJob] = []
        for seed_i in schedule:
            from dataclasses import replace as _replace
            si = sampling if sampling is not None else SamplingConfig()
            si = _replace(si, seed=seed_i)
            jobs.append(_SerialGenerationJob(
                input=input,
                steering=steering,
                sampling=si,
                raw=raw,
                thinking=thinking,
                on_token=on_token,
                parent_node_id=parent_node_id,
                recipe_override=recipe_override,
                gen_seat=gen_seat,
                system_prompt=system_prompt,
            ))
        return _run_serial_generation_jobs(
            self,
            jobs,
            stateless=stateless,
            kind="fan",
            stop_between_jobs=True,
        )

    # -- Generation: blocking --

    def _generate_fan_fast(
        self,
        input: Any,
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        stateless: bool,
        raw: bool,
        thinking: bool | None,
        on_token: TokenCallback | None,
        parent_node_id: str | None,
        n: int,
        recipe_override: "Recipe | str | None",
    ) -> RunSet | None:
        """Batch deterministic fan-out without changing seed semantics."""
        del parent_node_id
        if (
            n <= 1
            or not stateless
            or on_token is not None
            or recipe_override is not None
        ):
            return None
        if self._compose_gen_config(sampling).temperature > 0:
            return None
        fast = self._generate_batch_fast(
            [input for _ in range(n)],
            steering=steering,
            sampling=sampling,
            thinking=thinking,
            stateless=stateless,
            raw=raw,
            on_result=None,
        )
        if fast is None:
            return None
        return RunSet(
            fast,
            node_ids=[None] * len(fast),
            grid=[{} for _ in fast],
            kind="fan",
        )

    def generate(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenCallback | None = None,
        parent_node_id: str | None = None,
        n: int = 1,
        recipe_override: "Recipe | str | None" = None,
        gen_seat: str = "assistant",
        append_same_role: bool = True,
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> RunSet:
        """Blocking generation.

        Args:
            input: prompt string or list of message dicts.
            steering: expression string (e.g. ``"0.5 honest + 0.3 warm"``)
                or a pre-built :class:`Steering`.  Pole aliases resolve at
                parse time via ``store.selectors.resolve_bare_atom``.  ``None`` =
                no steering.
            sampling: per-call ``SamplingConfig``.  ``None`` fields fall
                through to the session's ``GenerationConfig`` defaults.
                The session's config is never mutated by this call.
            stateless: do not mutate session history.
            system_prompt: per-call system instruction, stamped in the node recipe.
                Omission inherits the session; explicit None clears it. Raw
                completion ignores this field. Replay uses the captured value.
            raw: skip chat template, tokenize input string directly.
            thinking: per-call thinking override.  ``None`` = auto-detect
                via ``supports_thinking`` (or ``steering.thinking`` if set).
            on_token: optional callback ``(text, is_thinking, token_id,
                logprob, top_alts, perplexity)`` called on each emitted
                token.  ``top_alts`` is ``list[TokenAlt]`` (decoded
                ``(id, text, logprob)`` triples) when ``sampling.logprobs > 0``
                or ``sampling.return_top_k > 0``; ``None`` otherwise.
                ``perplexity`` is ``exp(entropy_nats)`` of the
                sampler distribution after temperature, top-k, and
                top-p renormalization.
            parent_node_id: loom-tree node id to anchor the new turn
                under.  ``None`` = active node (today's behavior).
            n: fan-out count.  ``n=1`` (default) returns a one-result
                :class:`RunSet`; ``n>1`` runs the same prompt
                ``n`` times under deterministically-derived per-sibling
                seeds (see :func:`~drowse.loom.recipe.derive_seed_schedule`)
                and returns a multi-result ``RunSet`` in sibling order.
            gen_seat: which seat the generated turn occupies (the cast
                model).  ``"assistant"`` (default) is the classic flow;
                ``"user"`` renders the generation prompt as a user-seat
                header (labeled by ``sampling.user_role``) and the new
                node lands with ``role="user"`` + a stamped recipe —
                generated is provenance, not a seat.  A non-assistant
                seat needs a validated scene grammar
                (:attr:`scene_grammar`). ``input=None`` skips an authored
                span. For a one-shot generation, ``append_same_role=True``
                reuses a matching leaf and treats its existing text as a
                forced prefix; fan-out retains sibling branches.
            append_same_role: coalesce a one-shot bare generation into a leaf
                with the same structural role and role label. Disable for
                specialist operations whose contract requires a new sibling.

        Returns:
            :class:`RunSet` in every case.  It is list-like for fan-out
            and exposes ``.first`` for the common single-result case.
        """
        return self._generate_runset(
            input,
            steering=steering,
            sampling=sampling,
            stateless=stateless,
            raw=raw,
            thinking=thinking,
            on_token=on_token,
            parent_node_id=parent_node_id,
            n=n,
            recipe_override=recipe_override,
            gen_seat=gen_seat,
            append_same_role=append_same_role,
            system_prompt=system_prompt,
        )

    # -- Generation: streaming --

    def generate_stream(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        parent_node_id: str | None = None,
        recipe_override: "Recipe | str | None" = None,
        live_scores: bool = True,
        live_readouts: bool = True,
        gen_seat: str = "assistant",
        append_same_role: bool = True,
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> GenerationStream:
        """Streaming generation.  See :meth:`generate` for kwargs.

        Yields ``TokenEvent`` per token.  On iterator close (normal
        exhaustion, ``GeneratorExit``, or an exception raised through
        ``yield``) the worker is signaled to stop and joined, and the
        underlying ``_generate_core`` cleanup runs — probes detached,
        steering scope popped, lock released.

        ``sampling=SamplingConfig(return_hidden=True)`` works here too:
        per-token events do not carry hidden states (that would break
        the allocation-free hot path), but after iteration completes
        the populated ``session.last_result.hidden_states`` is
        available for round-tripping through :meth:`score_hidden`.

        ``live_scores=False`` skips inline per-token probe scoring for this
        stream. Final aggregate/per-token readings are still computed during
        generation finalization from the captured hidden states.

        ``live_readouts=False`` skips live J-LENS/SAE token readouts for
        stream formats that never serialize those richer cards.
        """
        q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        done = object()
        result_holder: list[GenerationResult] = []
        exc_holder: list[BaseException] = []
        idx_counter = [0]
        cancelled = threading.Event()

        def _push(
            text: str, is_thinking: bool, tid: int | None, lp: float | None,
            top_alts: list[TokenAlt] | None, perplexity: float | None = None,
        ) -> None:
            payload = self._last_token_probe_payload or {}
            probe_readings: dict[str, "ProbeReading"] | None = None
            # Monitor probes AND pinned lens/SAE probes (readout channels) all
            # land in the merged per-family readings — a lens-only roster still
            # carries per-token readings while the live lens is on.  This is
            # the ``TokenEvent.probe_readings`` compat channel (the
            # OpenAI/Ollama vendor extension), kept as live
            # :class:`ProbeReading` objects; the serialized envelope rides
            # ``measurements`` with each family's NATIVE reading shape.
            #
            # The single-axis families' ``ScalarReading -> ProbeReading``
            # projection happens HERE, per stream consumer — never in the
            # token tap, so a generation nobody streams pays nothing for a
            # compatibility shape nobody reads.
            if live_scores and (
                self._monitor.probe_names
                or self._lens_instrument.probes
                or self._sae_instrument.probes
            ):
                last_payload = self._last_token_payload
                if last_payload is not None and last_payload.has_readings:
                    probe_readings = last_payload.all_readings
            event = TokenEvent(
                text=text, token_id=tid if tid is not None else -1, index=idx_counter[0],
                thinking=is_thinking, logprob=lp, top_alts=top_alts,
                probe_readings=probe_readings, perplexity=perplexity,
                measurements=payload.get("measurements"),
            )
            idx_counter[0] += 1
            q.put(event)
        consumer = TokenConsumer(
            _push,
            TokenConsumerOptions(
                live_scores=bool(live_scores),
                lens_readout=bool(live_readouts),
                sae_readout=bool(live_readouts),
            ),
        )

        def _worker():
            try:
                with self._gen_state.cancellation_scope(cancelled):
                    result = self._generate_core(
                        input,
                        steering=steering,
                        sampling=sampling,
                        stateless=stateless,
                        raw=raw,
                        thinking=thinking,
                        on_token=consumer,
                        parent_node_id=parent_node_id,
                        recipe_override=recipe_override,
                        gen_seat=gen_seat,
                        append_same_role=append_same_role,
                        system_prompt=system_prompt,
                    )
                result_holder.append(result)
            except BaseException as e:
                exc_holder.append(e)
            finally:
                q.put(done)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        finished = False

        def _finish() -> None:
            nonlocal finished
            if finished:
                return
            cancelled.set()
            worker.join()
            while not q.empty():
                q.get_nowait()
            finished = True
            if exc_holder and not result_holder:
                raise exc_holder[0]

        def _events() -> Iterator[TokenEvent]:
            try:
                while True:
                    item = q.get()
                    if item is done:
                        break
                    yield item
            finally:
                _finish()

        class _GenerationStream:
            def __init__(self, iterator: Iterator[TokenEvent]) -> None:
                self._iterator = iterator

            def __iter__(self) -> "_GenerationStream":
                return self

            def __next__(self) -> TokenEvent:
                return next(self._iterator)

            def close(self) -> None:
                close = getattr(self._iterator, "close", None)
                try:
                    if callable(close):
                        close()
                finally:
                    _finish()

            @property
            def result(self) -> GenerationResult | None:
                return result_holder[0] if result_holder else None

        return _GenerationStream(_events())

    # -- Generation: batch + sweep --

    def _generate_batch_fast(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        stateless: bool,
        raw: bool,
        on_result: Callable[[int, GenerationResult], None] | None,
    ) -> RunSet | None:
        """Run a compatible stateless batch through one HF ``generate`` call.

        ``None`` means "fall back to the serial Drowse decode loop."  The fast
        path is intentionally narrow: no per-token consumers/logprobs/stop
        strings, no stateful loom writes, and only always-active steering.  When
        probes are attached it uses a batched aggregate-only capture ring and
        scores one final reading per row after decode; per-token probe gates and
        streams stay on the serial Drowse loop.
        """
        if len(prompts) < 2 or not stateless:
            return None
        if self._batch_fast_sampling_blocked(sampling):
            return None
        return_probe_readings = bool(
            sampling is None or sampling.return_probe_readings
        )
        has_sae_probes = bool(return_probe_readings and self._sae_instrument.probes)
        if has_sae_probes and self._sae_instrument.runtime.layer is None:
            return None

        prepared = self._prepare_generation_call(steering, sampling, thinking)
        if (
            prepared.use_thinking_req
            or prepared.gen_config.max_new_tokens < 1
            or prepared.lp_count is not None
            or (
                prepared.seed is not None
                and prepared.gen_config.temperature > 0
            )
            or prepared.stop_list is not None
            or prepared.logit_bias is not None
            or prepared.presence_penalty != 0.0
            or prepared.frequency_penalty != 0.0
            or prepared.logprobs_list is not None
            or not self._batch_fast_steering_is_always_on(prepared.steering_obj)
        ):
            return None

        model = self._batch_generate_model()
        if model is None or hf_unneutralizable_processors(model):
            return None

        return self._run_generate_batch_fast(
            prompts,
            model=model,
            steering_obj=prepared.steering_obj,
            sampling=sampling,
            gen_config=prepared.gen_config,
            stateless=stateless,
            raw=raw,
            on_result=on_result,
        )

    def _finalize_batch_probe_result(
        self,
        *,
        generated_ids: list[int],
        elapsed: float,
        vector_snapshot: dict[str, float],
        prompt_tokens: int,
        finish_reason: str,
        applied_steering: str | None,
        geometry_readings: dict[str, ProbeReading],
        lens_readings: "dict[str, ScalarReading]",
        sae_readings: "dict[str, ScalarReading]",
    ) -> GenerationResult:
        """Build a stateless batch result with pre-scored probe aggregates.

        Per-family in, per-family out: the aggregate envelope keeps each
        family's native reading shape (the serial finalize does the same),
        and ``probe_readings`` is the cross-family compatibility projection.
        """
        token_count = len(generated_ids)
        tok_per_sec = (
            token_count / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
        )
        decoded = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = decoded if isinstance(decoded, str) else decoded[0]
        merged: dict[str, Any] = {
            **geometry_readings, **lens_readings, **sae_readings,
        }
        live_lens = self._lens_instrument.live
        live_sae = self._sae_instrument.live
        result = GenerationResult(
            text=text,
            tokens=list(generated_ids),
            token_count=token_count,
            tok_per_sec=tok_per_sec,
            elapsed=elapsed,
            steering_alphas=vector_snapshot,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
            logprobs=None,
            applied_steering=applied_steering,
            hidden_states=None,
            probe_readings={
                name: as_probe_reading(reading)
                for name, reading in merged.items()
            },
            measurements=(
                dict(measurement_envelope)
                if (measurement_envelope := build_measurements(
                    scope="aggregate",
                    geometry_readings=geometry_readings or None,
                    lens_readings=lens_readings or None,
                    sae_readings=sae_readings or None,
                    lens_source=(
                        live_lens.get("source")
                        if lens_readings and isinstance(live_lens, dict) else None
                    ),
                    sae_source=(
                        live_sae.get("source")
                        if sae_readings and isinstance(live_sae, dict) else None
                    ),
                    sae_layer=(
                        live_sae.get("layer")
                        if sae_readings and isinstance(live_sae, dict) else None
                    ),
                    steering=applied_steering,
                )) is not None else None
            ),
        )
        self._last_result = result
        self._last_per_token_scores = None
        if merged:
            self.events.emit(ProbeScored(readings={
                name: reading_axis0(reading)
                for name, reading in merged.items()
            }))
        return result

    def _batch_probe_aggregate_for_row(
        self,
        row_index: int,
        generated_ids: list[int],
        probe_names: list[str],
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, ProbeReading]:
        """Score the final aggregate probe reading for one batched row."""
        empty = self._empty_readings(probe_names)
        if not generated_ids:
            return empty
        if pooled is None:
            pooled = self._batch_pooled_aggregate_for_row(row_index, generated_ids)
        if not pooled:
            return empty
        # Routed through the geometry run like the serial finalize
        # (``_score_aggregate_only``); the cold-foot reset stays here.
        self._monitor.enable_curved_warm(False)
        agg_vals = self._geometry_instrument.current_run.observe_aggregate(
            pooled,
        )
        return {name: agg_vals.get(name, empty[name]) for name in probe_names}

    def _batch_pooled_aggregate_for_row(
        self,
        row_index: int,
        generated_ids: list[int],
    ) -> dict[int, torch.Tensor]:
        """Shared final-token pooled slice for one batched generation row."""
        if not generated_ids:
            return {}
        from drowse.fit.capture import last_content_index
        agg_fwd = last_content_index(generated_ids, self._tokenizer)
        return self._capture.batch_tail_slice_at(row_index, agg_fwd)

    def _batch_readout_probe_aggregate_for_row(
        self,
        pooled: dict[int, torch.Tensor],
    ) -> "tuple[dict[str, ScalarReading], dict[str, ScalarReading]]":
        """Pinned J-lens/SAE readout probe aggregates from a batched tail
        slice, kept per family (``(lens, sae)``).

        The runs resolve through each family's bound state (frozen spec
        snapshot + pinned lens), so every row of the batch measures against
        the same bind-time binding — the guards consult the binding too, so
        an ``on_result`` callback detaching a probe cannot change later
        rows' roster."""
        if not pooled:
            return {}, {}
        lens_readings: dict[str, ScalarReading] = {}
        sae_readings: dict[str, ScalarReading] = {}
        if self._lens_instrument._measurement_specs():
            lens_readings = (
                self._lens_instrument.current_run.observe_aggregate(pooled)
            )
        if self._sae_instrument._measurement_specs():
            sae_readings = (
                self._sae_instrument.current_run.observe_aggregate(pooled)
            )
        return lens_readings, sae_readings

    @staticmethod
    def _batch_fast_sampling_blocked(sampling: SamplingConfig | None) -> bool:
        """True iff per-row custom-loop sampling features are requested."""
        if sampling is None:
            return False
        return bool(
            sampling.stop is not None
            or sampling.logit_bias is not None
            or sampling.presence_penalty != 0.0
            or sampling.frequency_penalty != 0.0
            or sampling.logprobs is not None
            or sampling.return_hidden
            or sampling.return_top_k != 0
            or sampling.persist_per_layer_scores
            or sampling.persist_subspace_coords
        )

    @staticmethod
    def _batch_fast_steering_is_always_on(
        steering_obj: Steering | None,
    ) -> bool:
        """HF ``generate`` can batch only trigger-free/``Trigger.BOTH`` terms."""
        return steering_obj is None or steering_obj.triggers() <= {Trigger.BOTH}

    def _batch_generate_model(self) -> Any | None:
        """Model object exposing ``generate``; unwrap compiled modules if needed."""
        model = self._model
        if callable(getattr(model, "generate", None)):
            return model
        orig = getattr(model, "_orig_mod", None)
        if orig is not None and callable(getattr(orig, "generate", None)):
            return orig
        return None

    def _batch_pad_token_id(self) -> int:
        """Pad id for left-padded generation batches."""
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(self._tokenizer, attr, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                if value:
                    return int(value[0])
            else:
                return int(value)
        return 0

    def _prepare_batch_input_ids(
        self,
        prompts: list[Any],
        *,
        sampling: SamplingConfig | None,
        raw: bool,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Render and left-pad a stateless prompt batch."""
        rendered: list[torch.Tensor] = []
        user_role = sampling.user_role if sampling is not None else None
        assistant_role = sampling.assistant_role if sampling is not None else None
        for prompt in prompts:
            ids = self._renderer.prepare_input(
                prompt,
                raw=raw,
                thinking=False,
                stateless=True,
                user_role=user_role,
                assistant_role=assistant_role,
                to_device=False,
            )
            if ids.ndim != 2 or ids.shape[0] != 1:
                raise RuntimeError(
                    "generate_batch fast path expected rendered prompt shape [1, T]"
                )
            rendered.append(ids[0].detach().to(device="cpu", dtype=torch.long))

        lengths = [int(ids.shape[0]) for ids in rendered]
        max_len = max(lengths)
        batch = torch.full(
            (len(rendered), max_len), pad_id, dtype=torch.long,
        )
        attention = torch.zeros_like(batch)
        for row, ids in enumerate(rendered):
            length = lengths[row]
            batch[row, max_len - length:] = ids
            attention[row, max_len - length:] = 1
        return (
            batch.to(self._device),
            attention.to(self._device),
            lengths,
        )

    @staticmethod
    def _batch_generated_tokens(
        row: torch.Tensor,
        *,
        eos_ids: set[int],
        pad_id: int,
        max_new_tokens: int,
    ) -> tuple[list[int], str]:
        """Trim HF batch output into Drowse' generated-token convention."""
        raw_ids = [int(tok) for tok in row.detach().to("cpu").tolist()]
        tokens: list[int] = []
        finish_reason = "length"
        for token_id in raw_ids:
            if token_id in eos_ids:
                finish_reason = "stop"
                break
            tokens.append(token_id)
        else:
            stripped_pad = False
            while tokens and tokens[-1] == pad_id:
                tokens.pop()
                stripped_pad = True
            if stripped_pad or len(tokens) < max_new_tokens:
                finish_reason = "stop"
        return tokens, finish_reason

    def _run_generate_batch_fast(
        self,
        prompts: list[Any],
        *,
        model: Any,
        steering_obj: Steering | None,
        sampling: SamplingConfig | None,
        gen_config: GenerationConfig,
        stateless: bool,
        raw: bool,
        on_result: Callable[[int, GenerationResult], None] | None,
    ) -> RunSet:
        """Implementation of the compatible batched ``model.generate`` path.

        Shares :meth:`_generation_transaction` with the ordinary decode path,
        so the boundary guard and the teardown are the same code; only the
        body differs (one batched ``model.generate`` instead of the decode
        loop, and no loom attachment).
        """
        with self._generation_transaction() as txn:
            if steering_obj is not None and steering_obj.alphas:
                txn.steering_cm = self.steering(steering_obj)
                txn.steering_cm.__enter__()
            vector_snapshot: dict[str, float] = (
                self._snapshot_steering_alphas()
                if self._steering_composer.active or txn.steering_cm is not None
                else {}
            )
            pad_id = self._batch_pad_token_id()
            input_ids, attention_mask, prompt_lengths = self._prepare_batch_input_ids(
                prompts,
                sampling=sampling,
                raw=raw,
                pad_id=pad_id,
            )
            return_probe_readings = bool(
                sampling is None or sampling.return_probe_readings
            )
            probe_names = (
                list(self._monitor.probe_names)
                if return_probe_readings else []
            )
            has_lens_probes = bool(return_probe_readings and self._lens_instrument.probes)
            has_sae_probes = bool(return_probe_readings and self._sae_instrument.probes)
            # Batched generation bypasses ``_begin_capture`` but binds the
            # instrument runs through the same helper.  One request serves all
            # three families here: the lens prepare takes the disk refresh +
            # pin and snapshots specs against that identity before the plans
            # and bindings consume them (its pin-demand formula reduces to
            # "probes attached and a final aggregate wanted" on a batch
            # request), so one pinned lens serves every row aggregate rather
            # than reopening all shards per batch item.
            batch_request = ReadRequest(
                final_aggregate=return_probe_readings,
            )
            geometry_plan, lens_plan, sae_plan = self._bind_instrument_runs(
                batch_request, batch_request, batch_request,
            )
            capture_probe_aggregates = bool(
                probe_names or has_lens_probes or has_sae_probes
            )
            if capture_probe_aggregates:
                layer_set: set[int] = set()
                for batch_plan in (geometry_plan, lens_plan, sae_plan):
                    layer_set |= batch_plan.latest_layers
                    layer_set |= batch_plan.tail_layers
                layer_idxs = sorted(layer_set)
                self._capture.clear()
                if layer_idxs:
                    self._capture.attach_batch_tail(
                        self._layers,
                        layer_idxs,
                        depth=_AGG_TAIL_DEPTH,
                    )
                self._capture_state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)
                if probe_names:
                    self._monitor.enable_curved_warm(False)
            self._gen_state.reset()
            self._steering.ctx.reset()
            self._steering.reset_manifold_feet()
            self._gen_phase = GenState.RUNNING
            for prompt in prompts:
                self.events.emit(GenerationStarted(input=prompt, stateless=stateless))

            # The full EOS set (generation_config + tokenizer + added
            # end-of-turn specials), matching the ordinary decode loop —
            # the tokenizer's single ``eos_token_id`` alone misses e.g.
            # Gemma's ``<end_of_turn>`` and would generate past it to the
            # token cap.  The explicit config pins the decode loop's sampler
            # so the checkpoint's own ``generation_config`` (Qwen's
            # ``top_k: 20`` / ``repetition_penalty: 1.1``) cannot leak in.
            eos_ids = set(self.chat_traits.eos_ids)
            text_cfg = getattr(model.config, "text_config", model.config)
            hf_config = hf_generation_config(
                gen_config,
                vocab=int(text_cfg.vocab_size),
                eos_ids=sorted(eos_ids),
                pad_id=pad_id,
                steering_active=bool(
                    self._steering.hooks or self._steering.uses_compiled_offsets
                ),
            )

            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=hf_config,
                    stopping_criteria=StoppingCriteriaList([
                        _SessionStopCriteria(self._gen_state),
                    ]),
                )
            elapsed = time.monotonic() - start
            if capture_probe_aggregates:
                self._end_capture()
            sequences = getattr(generated, "sequences", generated)
            if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                raise RuntimeError(
                    "generate_batch fast path expected model.generate to return "
                    "a [batch, sequence] tensor"
                )

            max_prompt_len = int(input_ids.shape[1])
            self._gen_phase = GenState.FINALIZING
            results: list[GenerationResult] = []
            node_ids: list[str | None] = []
            grid_rows: list[dict[str, Any]] = []
            applied_steering = (
                str(steering_obj) if steering_obj is not None else None
            )
            for idx, prompt in enumerate(prompts):
                generated_ids, finish_reason = self._batch_generated_tokens(
                    sequences[idx, max_prompt_len:],
                    eos_ids=eos_ids,
                    pad_id=pad_id,
                    max_new_tokens=int(gen_config.max_new_tokens),
                )
                self._gen_state.thinking_end_idx = 0
                self._gen_state.emit_map = []
                self._gen_state.response_text = None
                self._gen_state.response_aggregate_index = None
                self._gen_state.finish_reason = finish_reason
                if capture_probe_aggregates:
                    pooled = self._batch_pooled_aggregate_for_row(idx, generated_ids)
                    geometry_readings: dict[str, ProbeReading] = (
                        self._batch_probe_aggregate_for_row(
                            idx,
                            generated_ids,
                            probe_names,
                            pooled=pooled,
                        )
                        if probe_names
                        else {}
                    )
                    lens_readings, sae_readings = (
                        self._batch_readout_probe_aggregate_for_row(pooled)
                    )
                    result = self._finalize_batch_probe_result(
                        generated_ids=generated_ids,
                        elapsed=elapsed,
                        vector_snapshot=vector_snapshot,
                        prompt_tokens=prompt_lengths[idx],
                        finish_reason=finish_reason,
                        applied_steering=applied_steering,
                        geometry_readings=geometry_readings,
                        lens_readings=lens_readings,
                        sae_readings=sae_readings,
                    )
                else:
                    result = self._finalize_generation(
                        generated_ids,
                        elapsed,
                        vector_snapshot,
                        prompt_tokens=prompt_lengths[idx],
                        stateless=stateless,
                        logprobs_list=None,
                        applied_steering=applied_steering,
                        return_hidden=False,
                        return_probe_readings=return_probe_readings,
                        assistant_node_id=None,
                    )
                results.append(result)
                node_ids.append(None)
                row = {"prompt_index": idx}
                grid_rows.append(row)
                self.events.emit(GenerationFinished(result=result))
                if on_result is not None:
                    on_result(idx, result)
            self._gen_state.stop_requested.set()
            total_tokens = sum(r.token_count for r in results)
            batch_tok_per_sec = (
                total_tokens / elapsed if elapsed > MIN_ELAPSED_FOR_RATE else 0.0
            )
            return RunSet(
                results,
                node_ids=node_ids,
                grid=grid_rows,
                kind="batch",
                metrics={
                    "batch_elapsed": elapsed,
                    "batch_token_count": total_tokens,
                    "batch_tok_per_sec": batch_tok_per_sec,
                },
            )

    def generate_batch(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult], None] | None = None,
    ) -> RunSet:
        """Run N prompts under the same steering, return a ``RunSet``.

        Compatible stateless batches run through one ``transformers.generate``
        call, including aggregate-only probe reads via batched tail capture.
        Anything that needs Drowse' row-shaped custom loop (stateful history,
        live lens, hidden capture returns, logprobs, stop strings, seeded
        per-row sampling, penalties, or phased/gated triggers) falls back to the
        serial single-prompt path.  The session's threading lock keeps concurrent
        ``generate_batch`` calls from interleaving — they queue FIFO at the
        per-call level, same as today's ``generate``.

        Args mirror ``generate``.  ``stateless`` defaults to ``True``
        (batch generation is overwhelmingly used for sweeps and evals
        where conversational history would corrupt the comparison);
        pass ``stateless=False`` if you genuinely want each prompt to
        accumulate against the running history.

        **Prefix KV caching.**  When the rows share a token prefix (a common
        system prompt, few-shot preamble, or shared instruction) and the
        steering doesn't touch the prefill region (``None`` or ``@response`` /
        ``@generated`` / probe-gated), that shared prefix is prefilled once and
        its KV reused across every row — the per-call saving scales with
        ``shared_prefix_len / total_input_len``.  Skipped for ``stateless=False``
        (per-row history breaks the shared prefix), ``return_hidden`` (needs
        every prefix hidden state), thinking, or prefill-active steering (the
        prompt KV would differ per row).  Correctness is unaffected either way —
        a cache hit produces the identical token stream.

        ``on_result(idx, result)`` fires after each completion for local
        progress hooks.

        Returns:
            ``RunSet`` aligned with ``prompts``.  ``runset.grid`` records
            ``{"prompt_index": i}`` for each row.
        """
        prompts_in: Any = prompts
        if not isinstance(prompts_in, list) or not prompts_in:
            raise ValueError("generate_batch: prompts must be a non-empty list")

        fast = self._generate_batch_fast(
            prompts,
            steering=steering,
            sampling=sampling,
            thinking=thinking,
            stateless=stateless,
            raw=raw,
            on_result=on_result,
        )
        if fast is not None:
            return fast

        prefix_cached = self._maybe_cache_batch_prefix(
            prompts, steering=steering, sampling=sampling,
            thinking=thinking, stateless=stateless, raw=raw,
        )
        try:
            jobs = [
                _SerialGenerationJob(
                    input=prompt,
                    steering=steering,
                    sampling=sampling,
                    raw=raw,
                    thinking=thinking,
                    grid={"prompt_index": idx},
                )
                for idx, prompt in enumerate(prompts)
            ]
            runset = _run_serial_generation_jobs(
                self,
                jobs,
                stateless=stateless,
                kind="batch",
                on_result=(
                    (lambda idx, result, _row: on_result(idx, result))
                    if on_result is not None
                    else None
                ),
            )
        finally:
            if prefix_cached:
                self.cache_prefix(None)
        return runset

    def _maybe_cache_batch_prefix(
        self,
        prompts: list[Any],
        *,
        steering: "str | Steering | None",
        sampling: SamplingConfig | None,
        thinking: bool | None,
        stateless: bool,
        raw: bool,
    ) -> bool:
        """Prefill + cache the batch's shared token prefix when reuse is valid.

        Returns ``True`` iff a prefix was cached (the caller clears it after the
        batch).  Eligibility (all must hold): ``stateless`` (per-row history
        would break the shared prefix), the steering is prefill-inactive
        (``steering_value_prefill_inactive`` — else the prompt KV differs per
        row and the consume gate would refuse the hit anyway), no
        ``return_hidden`` and no thinking (both force a full re-prefill), at
        least two rows, and a shared prefix of at least
        :data:`_PREFIX_CACHE_MIN_TOKENS`. Ordinary ineligibility returns False;
        lifecycle, rendering, and initialization failures propagate.
        """
        if len(prompts) < 2 or not stateless:
            return False
        if thinking:
            return False
        if sampling is not None and getattr(sampling, "return_hidden", False):
            return False
        if not self._steering_composer.steering_value_prefill_inactive(steering):
            return False
        common = self._batch_common_prefix_ids(prompts, raw=raw)
        if common is None:
            return False
        max_new_tokens = (
            sampling.max_tokens
            if sampling is not None and sampling.max_tokens is not None
            else self.config.max_new_tokens
        )
        return (
            self.cache_prefix(
                common,
                max_new_tokens=max_new_tokens,
                prefer_static=bool(
                    self._static_cache_active and steering is None
                ),
            )
            >= _PREFIX_CACHE_MIN_TOKENS
        )

    def _batch_common_prefix_ids(
        self, prompts: list[Any], *, raw: bool,
    ) -> "torch.Tensor | None":
        """Longest shared leading token run across the batch's rendered inputs.

        Renders each prompt to ``input_ids`` in stateless mode (the same render
        ``_generate_core`` will reproduce, so the cached tokens are a true
        byte-prefix of every row) and returns the shared head as a 1-D tensor,
        or ``None`` when the shared run is shorter than
        :data:`_PREFIX_CACHE_MIN_TOKENS`.  The run is capped at ``min_row_len -
        1`` so every row keeps a non-empty suffix to sample from.
        """
        rendered: list[torch.Tensor] = []
        for prompt in prompts:
            ids = self._renderer.prepare_input(
                prompt, raw=raw, stateless=True, to_device=False,
            )
            # Prefix discovery is scalar-heavy Python work. Keep the token
            # comparison on CPU so MPS/CUDA do not pay one device sync per
            # ``int(t[i])`` while walking the common prefix.
            rendered.append(ids[0].detach().to("cpu"))
        min_len = min(int(t.shape[0]) for t in rendered)
        if min_len <= _PREFIX_CACHE_MIN_TOKENS:
            return None
        ref = rendered[0]
        common = 0
        for i in range(min_len):
            tok = int(ref[i])
            if all(int(t[i]) == tok for t in rendered):
                common += 1
            else:
                break
        common = min(common, min_len - 1)  # leave a >=1-token suffix per row
        if common < _PREFIX_CACHE_MIN_TOKENS:
            return None
        return ref[:common].clone()

    def _generate_sweep_fast(
        self,
        jobs: list[_SerialGenerationJob],
        *,
        stateless: bool,
        on_result: Callable[[int, GenerationResult, dict[str, Any]], None] | None,
    ) -> RunSet | None:
        """Fast path for exact degenerate sweeps.

        Different alpha rows intentionally install different hook coefficients,
        so they must stay on the serial path until the hook layer can carry
        per-row steering.  When every row resolves to the same steering
        expression, however, a stateless greedy sweep is equivalent to a batch
        of repeated prompts under one steering scope; route that through the
        shared ``generate_batch`` fast path.
        """
        if len(jobs) < 2 or not stateless:
            return None
        first = jobs[0]
        gen_config = self._compose_gen_config(first.sampling)
        if gen_config.temperature > 0:
            return None
        for job in jobs[1:]:
            if (
                job.input != first.input
                or job.steering != first.steering
                or job.raw != first.raw
                or job.thinking != first.thinking
                or job.recipe_override is not None
                or self._compose_gen_config(job.sampling).temperature > 0
            ):
                return None

        def _batch_on_result(
            idx: int,
            result: GenerationResult,
        ) -> None:
            if on_result is not None:
                on_result(idx, result, jobs[idx].grid or {})

        fast = self._generate_batch_fast(
            [job.input for job in jobs],
            steering=first.steering,
            sampling=first.sampling,
            thinking=first.thinking,
            stateless=True,
            raw=first.raw,
            on_result=_batch_on_result if on_result is not None else None,
        )
        if fast is None:
            return None
        return RunSet(
            list(fast),
            node_ids=[None] * len(fast),
            grid=[job.grid or {} for job in jobs],
            kind="fan",
        )

    def generate_sweep(
        self,
        prompt: Any,
        sweep: dict[str, list[float]],
        *,
        base_steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        thinking: bool | None = None,
        stateless: bool = True,
        raw: bool = False,
        on_result: Callable[[int, GenerationResult, dict[str, float]], None] | None = None,
        parent_node_id: str | None = None,
    ) -> RunSet:
        """Fan a single prompt across a Cartesian product of alpha values.

        ``sweep`` maps ``concept_name → [alpha_0, alpha_1, ...]``.  The
        function generates one result per element of the product across
        every concept's alpha list.  For a single-concept sweep
        (``{"honest": [0.0, 0.3, 0.6]}``) you get three results; for
        ``{"honest": [-0.4, 0.0, 0.4], "warm": [0.0, 0.3]}`` you get
        six (3 × 2).

        Each generation runs under the steering expression
        ``base_steering + " + ".join(f"{α} {name}")``.  ``base_steering``
        defaults to ``None`` so the swept alphas are the only steering;
        pass a string to layer a fixed-alpha context underneath.

        ``on_result(idx, result, alpha_values)`` fires per completion.
        ``alpha_values`` is the ``{concept: alpha}`` dict that produced
        this row — recorded on each result's ``applied_steering`` too,
        but exposed here so SSE consumers don't have to re-parse the
        expression.

        Returns:
            ``RunSet`` in product order.  ``runset.grid[i]`` is the
            alpha dict for row ``i``; ``runset.node_ids[i]`` is the
            assistant node id when ``stateless=False``.
        """
        sweep_in: Any = sweep
        if not isinstance(sweep_in, dict) or not sweep_in:
            raise ValueError("generate_sweep: sweep dict must be non-empty")
        for name, alphas in sweep.items():
            alphas_in: Any = alphas
            if not isinstance(alphas_in, (list, tuple)) or not alphas_in:
                raise ValueError(
                    f"generate_sweep: sweep['{name}'] must be a non-empty "
                    f"list of alpha values"
                )

        # Cartesian product across concepts.  ``itertools.product``
        # preserves the order of ``sweep.items()`` so the per-concept
        # alpha lists in the output are predictable.
        import itertools

        concept_names = list(sweep.keys())
        alpha_lists = [list(sweep[name]) for name in concept_names]
        total = 1
        for values in alpha_lists:
            total *= len(values)
        base_seed = sampling.seed if sampling is not None else None
        seed_schedule = derive_seed_schedule(base_seed, total)

        base_str: str | None
        if base_steering is None:
            base_str = None
        elif isinstance(base_steering, str):
            base_str = base_steering
        else:
            # Pre-built Steering — render through the canonical formatter
            # so we can compose with new alpha terms via string concat.
            base_str = str(base_steering)

        # Anchor every sibling under a shared user turn so
        # the surfaces render the sweep as siblings under a common
        # parent rather than a flat result list.  Stateless sweeps
        # skip the tree mutation entirely;
        # stateful sweeps land siblings under ``parent_node_id`` (or
        # the active node when None) and dedup on identical user text.
        anchor_user_id: str | None = None
        if not stateless and isinstance(prompt, str):
            # Stamp the label ``_generate_core`` will send so its label-aware
            # dedup lands every sibling on this anchor.
            anchor_user_id = self.tree.add_user_turn(
                prompt, parent_id=parent_node_id,
                role_label=(
                    sampling.user_role
                    if sampling is not None and not raw else None
                ),
            )
            # The anchor parent for ``_generate_core`` is the user
            # node's *parent* — generate's ``add_user_turn`` will dedup
            # against the user we just spawned, so every sibling gets
            # attached under it without spawning duplicate user turns.
            gen_parent_id = self.tree.nodes[anchor_user_id].parent_id
        else:
            gen_parent_id = parent_node_id

        jobs: list[_SerialGenerationJob] = []
        for idx, combo in enumerate(itertools.product(*alpha_lists)):
            alpha_values = dict(zip(concept_names, combo, strict=True))
            alpha_values = {k: float(v) for k, v in alpha_values.items()}
            terms = [f"{alpha} {name}" for name, alpha in alpha_values.items()]
            expr = " + ".join(terms)
            if base_str:
                expr = f"{base_str} + {expr}"

            from dataclasses import replace as _replace
            si = sampling if sampling is not None else SamplingConfig()
            si = _replace(si, seed=seed_schedule[idx])

            jobs.append(_SerialGenerationJob(
                input=prompt,
                steering=expr,
                sampling=si,
                raw=raw,
                thinking=thinking,
                parent_node_id=gen_parent_id,
                grid=alpha_values,
            ))

        fast_sweep = self._generate_sweep_fast(
            jobs,
            stateless=stateless,
            on_result=(
                (lambda idx, result, row: on_result(
                    idx, result, cast(dict[str, float], row),
                ))
                if on_result is not None
                else None
            ),
        )
        if fast_sweep is not None:
            return fast_sweep

        return _run_serial_generation_jobs(
            self,
            jobs,
            stateless=stateless,
            kind="fan",
            on_result=(
                (lambda idx, result, row: on_result(
                    idx, result, cast(dict[str, float], row),
                ))
                if on_result is not None
                else None
            ),
        )

    # -- Generation control --

    def stop(self) -> None:
        self._gen_state.request_stop()

    # -- Lifecycle --

    def close(self) -> None:
        """Detach everything this session attached to the model.

        ``from_pretrained`` installs two families of *permanent* forward hooks
        before ``torch.compile`` — the persistent steering offsets and the
        persistent capture buffers — and ``DrowseSession.__init__`` accepts a
        pre-loaded model the caller may keep using, so the model does not
        necessarily die with the session.  Both families come off here
        alongside the transient steering hooks, releasing their
        ``2 x n_layers x hidden_size`` of device buffers with them.
        """
        self._steering.clear_all()
        self._steering.detach_compiled_offsets()
        self.detach_persistent_capture()
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.detach()
            capture.clear()

        for instrument_name in (
            "_geometry_instrument", "_lens_instrument", "_sae_instrument",
        ):
            instrument = self.__dict__.get(instrument_name)
            if instrument is None:
                continue
            instrument.close_run()
            if hasattr(instrument, "probes"):
                instrument.probes.clear()
            if hasattr(instrument, "live"):
                instrument.live = None

        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            for name in list(monitor.probe_names):
                monitor.remove_probe(name)
            monitor.set_whitener(None)

        self._registry.clear()
        self._neutral.reset()
        for cache_name in (
            "_joint_logprob_cache",
            "_probe_hash_cache",
            "_incremental_readings",
            "_incremental_gate_scores",
        ):
            cache = getattr(self, cache_name, None)
            if cache is not None:
                cache.clear()
        self._prefix_cache = None
        self._generation_static_cache = None
        self._generation_static_cache_len = 0
        self._lens_instrument.close()
        self._sae_instrument.runtime.close()
        self._last_result = None
        self._last_per_token_scores = None
        self._last_token_probe_payload = None
        self._last_token_payload = None
        events = getattr(self, "events", None)
        if events is not None:
            events.clear()

    def detach_persistent_capture(self) -> None:
        """Remove the persistent capture hooks and drop their buffers.

        The capture-side sibling of
        :meth:`SteeringManager.detach_compiled_offsets`.  Idempotent.
        """
        for handle in self._capture_handles:
            handle.remove()
        self._capture_handles = []
        self._capture_buffers = {}

    def __enter__(self) -> "DrowseSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
    def __iter__(self) -> "GenerationStream": ...

    def __next__(self) -> TokenEvent: ...
