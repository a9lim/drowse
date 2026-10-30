"""DrowseSession — unified backend for drowse's programmatic and server APIs."""
from __future__ import annotations
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, cast, overload

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from drowse.fit.capture import CaptureMode, CaptureState, _capture_all_hidden_states
from drowse.fit.corpus import RESPONSE_MAX_TOKENS, CorpusGenerator
from drowse.session.extraction import Extraction
from drowse.errors import DrowseError, GenerationRequestError
from drowse.events import (
    EventBus,
)
from drowse.generation.compile_runtime import CompiledRuntime
from drowse.generation.types import CallSpec, GenerationStream, GenState
from drowse.session.engine import GenerationEngine
from drowse.generation.decode import (
    GenerationConfig,
    GenerationState,
)
from drowse.model.chat import ChatTraits
from drowse.readout.capture_buffer import HiddenCapture
from drowse.steering.hooks import SteeringManager
from drowse.steering.composer import GateSources, SteeringComposer, SteeringScope
from drowse.session.registry import ArtifactRegistry, ModelBinding
from drowse.session.neutral import NeutralBaseline
from drowse.errors import (
    ConcurrentExtractionError,
    ManifoldNotRegisteredError,
)
from drowse.generation.token_callback import TokenSink
from drowse.model.scene import TurnGrammar
from drowse.generation.render import PromptRenderer
from drowse.session.authored_capture import AuthoredPromptCapture, plan_authored_capture
from drowse.events import LoomMutated
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, InheritSystemPrompt
from drowse.loom.recipe import (
    CastMember,
    Recipe,
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
)
from drowse.readout.monitor import Monitor
from drowse.readout.geometry import GeometryInstrument
from drowse.lens.instrument import LensInstrument
from drowse.sae.instrument import SaeInstrument
from drowse.sae.runtime import SaeRuntime
from drowse.session.replay import TokenReplayer
from drowse.store.bundled_probes import bootstrap_probe_roster
from drowse.geometry.profile import Profile
from drowse.generation.results import (
    GenerationResult,
    ProbeReading,
    RunSet,
)
from drowse.generation.sampling import SamplingConfig
from drowse.session.batch import BatchRunner, FastBatchPath
from drowse.steering.terms import Steering
from drowse.geometry.manifold import Manifold

if TYPE_CHECKING:
    from drowse.readout.protocol import Instrument
    from drowse.generation.scoring import ChoiceScores
    from drowse.store.templates import TemplateFolder

_log = logging.getLogger(__name__)



# Tail-ring depth for aggregate-only capture (probes attached, no per-token
# consumer): how many trailing hidden slices to retain so finalize can pool the
# last *content* token after walking back past trailing special tokens (EOS,
# end-of-turn). 8 comfortably covers any chat template's trailing-marker run.
# Canonical value lives in the instrument-plan vocabulary so families can
# declare the ring depth they demand.
_AGG_TAIL_DEPTH = AGG_TAIL_DEPTH


PROBE_CATEGORIES = [
    "epistemic",
    "alignment",
    "register",
    "cultural",
]




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
        for the same reason as ``compile``; without ``compile=True`` it only
        routes eligible generations through an eager StaticCache.
        Auto-skipped on MPS/CPU and on architectures whose StaticCache
        construction fails.

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
        model, tokenizer = load_model(
            model_id,
            quantize=quantize,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            on_progress=on_progress,
        )
        model, runtime = CompiledRuntime.prepare(
            model,
            tokenizer,
            compile=compile,
            compile_mode=compile_mode,
            cuda_graphs=cuda_graphs,
            persistent_capture=probes is None or bool(probes),
            on_progress=on_progress,
        )
        return cls(
            model,  # pyright: ignore[reportArgumentType]  # may be torch.compile OptimizedModule wrapping PreTrainedModel
            tokenizer,
            probes=probes,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            dls=dls,
            runtime=runtime,
            return_top_k=return_top_k,
            on_progress=on_progress,
        )

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        probes: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        dls: bool = True,
        runtime: CompiledRuntime | None = None,
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

        self._steering = SteeringManager()
        # Compile / StaticCache / CUDA-graph routing: ``from_pretrained``
        # decides it; a model handed in directly gets ``for_model``'s routing.
        # The persistent offset hooks installed before compiling belong to the
        # steering manager from here on.
        self._runtime = (
            runtime if runtime is not None
            else CompiledRuntime.for_model(model, self._device)
        )
        offset_hooks = self._runtime.take_offset_hooks()
        if offset_hooks is not None:
            self._steering.adopt_compiled_offsets(*offset_hooks)

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
        # (GATING_SUBSET).  The engine's finalize builds (aggregate, per_token)
        # from these instead of rescoring captured hidden states.  Reset at each
        # ``_begin_capture`` and on teardown.
        self._incremental_readings: list[dict[str, ProbeReading]] = []
        self._incremental_gate_scores: list[dict[str, float]] = []

        # Reentrant — the engine's transaction acquires it for the whole gen,
        # then enters an internal ``self.steering(...)`` scope whose push
        # re-acquires the lock from the same thread.  The single-in-flight
        # invariant is enforced by the engine's ``phase`` check, not by the lock owner
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
        # The generation engine: phase, decode state, prefix cache, routing,
        # last result.  It shares the exclusive-GPU lock with every model op.
        self._engine = GenerationEngine(
            self,
            lock=self._gen_lock,
            runtime=self._runtime,
            steering=self._steering,
            capture=self._capture,
            release=self._release_generation,
            default_return_top_k=return_top_k,
        )

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


        # -- collaborators --------------------------------------------------
        # Artifact registry: profiles + loaded manifolds; its mutations drop
        # the prefix cache and refresh monitor probes on a manifold swap.
        self._registry = ArtifactRegistry(ModelBinding(
            model=self._model, tokenizer=self._tokenizer,
            model_id=self.model_id, device=self._device, dtype=self._dtype,
        ))
        self._registry.on_change(lambda: self._engine.prefix_cache.clear())
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
            model_type=self._renderer.model_type,
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
            on_prefill_steering_change=lambda: self._engine.prefix_cache.clear(),
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
        # The one loom token replay every family's ``token_readout`` reads
        # through (the producing forward's capture; the family adds its read).
        self._token_replay = TokenReplayer(
            tree=lambda: self.tree,
            renderer=self._renderer,
            traits=self._chat,
            tokenizer=self._tokenizer,
            open_scope=self._steering_composer.open_scope,
            exclusive=self._model_exclusive,
            capture=partial(_capture_all_hidden_states, self._model, self._layers),
        )
        self._geometry_instrument = GeometryInstrument(
            self, replay=self._token_replay,
        )
        # The J-lens family: resident lens, fit lifecycle, probes, live
        # readout and token replay, over explicit dependencies.
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
            replay=self._token_replay,
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
            replay=self._token_replay,
            put_direction=self._registry.put_profile,
            evict_directions=lambda: self._registry.evict_prefix("sae/"),
            forget_probe_hashes=self._forget_probe_hashes,
        )
        # Fans, batches and sweeps: the multi-row runner over the engine.
        # Built last among the collaborators: the batched ``model.generate``
        # path binds the three read instruments and the steering scope.
        self._batch = BatchRunner(
            generate_one=self._engine.generate_one,
            tree=lambda: self.tree,
            gen_state=self._engine.state,
            render=self._renderer.prepare_input,
            cache_prefix=self.cache_prefix,
            prefill_inactive=self._steering_composer.steering_value_prefill_inactive,
            static_cache_active=lambda: self._runtime.static_cache_active,
            default_max_tokens=lambda: self.config.max_new_tokens,
            fast=FastBatchPath(
                model=lambda: self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                chat_traits=self._chat,
                layers=self._layers,
                gen_state=self._engine.state,
                steering=self._steering,
                events=self.events,
                capture=self._capture,
                geometry=self._geometry_instrument,
                lens=self._lens_instrument,
                sae=self._sae_instrument,
                sae_layer=lambda: self._sae_instrument.runtime.layer,
                resolve_call=self._engine.resolve_call,
                render=self._renderer.prepare_input,
                transaction=self._engine.transaction,
                open_scope=self.steering,
                snapshot_alphas=self._steering_composer.snapshot_steering_alphas,
                phase_running=lambda: setattr(
                    self._engine, "phase", GenState.RUNNING,
                ),
                phase_finalizing=lambda: setattr(
                    self._engine, "phase", GenState.FINALIZING,
                ),
                bind_runs=self._bind_instrument_runs,
                score_geometry=self._score_aggregate_only,
                finalize=self._engine.finalize,
            ),
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
        return self._engine.last_result

    @property
    def last_per_token_scores(self) -> dict[str, list[float]] | None:
        return self._engine.last_per_token_scores

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
        return self._engine.state

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
        return self._engine.phase

    @property
    def is_generating(self) -> bool:
        """``True`` whenever :attr:`gen_state` is not ``GenState.IDLE``."""
        return self._engine.is_generating

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
        phase = self._engine.phase
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
        self._engine.prefix_cache.clear()
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
        attach/detach, the CAA live toggle, the roster snapshots and the
        token replay (``token_readout``).  Manifold
        artifact lifecycle (``fit`` / ``extract`` / ``ensure_manifold_
        loaded``) stays on the session."""
        return self._geometry_instrument

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
        is given, also guards that no generation is in flight (raises ``exc(phase_msg)``
        on a generation in flight); ``add_probe`` passes ``None`` to skip the
        phase guard.  Releases in ``finally``.
        """
        if not self._gen_lock.acquire(blocking=False):
            raise exc(busy_msg)
        try:
            if phase_msg is not None and self._engine.is_generating:
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
            and self._engine.compiled_clean_eligible
            and self._runtime.capture_buffers
            and (
                self._steering.uses_compiled_offsets
                or self._steering.all_fast_path()
            )
        )
        self._capture.clear()
        if use_persistent:
            self._capture.attach_persistent(layer_idxs, self._runtime.capture_buffers)
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
        state = self._engine.state
        idx = state.response_aggregate_index
        if idx is not None and 0 <= int(idx) < len(generated_ids):
            return int(idx)
        if state.finish_reason == "stop_sequence" and state.response_text is not None:
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
        self._engine.prefix_cache.clear()
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
        self._engine.prefix_cache.clear()
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
        on_token: TokenSink | None = None,
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
        return self._engine.generate_one(
            turn.input, on_token=on_token, **turn.generate_kwargs(),
        )

    def prefill_assistant(
        self,
        node_id: str,
        text: str,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        on_token: TokenSink | None = None,
    ) -> GenerationResult:
        """Generate an assistant reply to user node ``node_id`` that opens with ``text``.

        See :func:`drowse.loom.ops.plan_prefill`.
        """
        turn = plan_prefill(
            self.tree, self._tokenizer, node_id, text,
            steering=steering, sampling=sampling,
            default_max_tokens=self.config.max_new_tokens,
        )
        return self._engine.generate_one(
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
        """Pre-prefill a shared chat prefix so later ``generate()`` calls
        forward only their suffix.

        For workloads that re-issue the same chat-template head (system +
        leading instruction) many times; per-call savings scale with
        ``prefix_len / total_input_len``.

        ``messages`` is a message list (rendered with
        ``add_generation_prompt=False``: the prefix is the turns *before* the
        assistant turn, whose opener belongs to each call's suffix), a
        ``[seq_len]`` / ``[1, seq_len]`` token tensor stored verbatim (for a
        common prefix that isn't a message boundary), or ``None`` to clear.

        ``prefer_static=True`` builds a StaticCache-backed entry when the
        session supports StaticCache, with ``max_new_tokens`` of decode
        headroom (default: the session's); a later generation needing more
        misses it and prefills in full.  A failed StaticCache build falls back
        to an ordinary DynamicCache entry.

        Returns the cached prefix length in tokens (0 when clearing).

        Caveats:

        1. Refused inside an active ``session.steering()`` scope: the prefix
           is prefilled with the neutral baseline.  A later generation reuses
           it only when its steering leaves the prompt prefill untouched.
        2. Hidden-state capture is suspended for the prefill.
        3. The cached tokens must be an exact prefix of a generation's
           rendered ``input_ids`` to hit; a mismatch silently misses (full
           prefill), never mis-hits.
        """
        return self._engine.cache_prefix(
            messages, max_new_tokens=max_new_tokens, prefer_static=prefer_static,
        )

    # -- Generation helpers --

    @property
    def renderer(self) -> PromptRenderer:
        """The session's :class:`~drowse.generation.render.PromptRenderer`."""
        return self._renderer

    @property
    def scene_grammar(self) -> TurnGrammar | None:
        """The validated turn grammar for the loaded chat template, or None."""
        return self._renderer.scene_grammar

    def _plan_authored_prompt_capture(
        self,
        spec: "CallSpec",
        *,
        assistant_node_id: str | None,
        use_thinking: bool,
        monitor_active: bool,
        lens_active: bool,
        sae_active: bool,
        persist_per_layer_scores: bool,
    ) -> AuthoredPromptCapture | None:
        """This generation's authored-prefill capture policy, or ``None``.

        Nothing is tokenized unless some channel (monitor, lens, SAE) would
        record a measurement (F11).  The search bound is one render without
        the generation prompt, so a target never matches inside the
        generation header; see
        :func:`drowse.session.authored_capture.plan_authored_capture`.
        """
        return plan_authored_capture(
            self.tree, self._tokenizer, assistant_node_id, raw=spec.raw,
            monitor=self._monitor,
            lens=self._lens_instrument,
            sae=self._sae_instrument,
            monitor_active=monitor_active,
            lens_active=lens_active,
            sae_active=sae_active,
            readout_top_k=spec.readout_top_k,
            persist_per_layer_scores=persist_per_layer_scores,
            content_end=lambda: int(self._renderer.prepare_input(
                spec.input,
                raw=False,
                thinking=use_thinking,
                stateless=spec.stateless,
                parent_node_id=spec.history_anchor,
                user_role=spec.user_role,
                assistant_role=spec.assistant_role,
                to_device=False,
                gen_seat=spec.gen_seat,
                system_prompt=spec.system_prompt,
                add_generation_prompt=False,
            ).shape[1]),
        )

    # -- Generation: core --

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

    # -- Generation: blocking --

    def generate(
        self,
        input: Any,
        *,
        steering: "str | Steering | None" = None,
        sampling: SamplingConfig | None = None,
        stateless: bool = False,
        raw: bool = False,
        thinking: bool | None = None,
        on_token: TokenSink | None = None,
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
        if n < 1:
            raise GenerationRequestError(f"n must be >= 1, got {n}")
        if n != 1:
            return self._batch.fan(
                input, n=n, steering=steering, sampling=sampling,
                stateless=stateless, raw=raw, thinking=thinking,
                on_token=on_token, parent_node_id=parent_node_id,
                recipe_override=recipe_override, gen_seat=gen_seat,
                system_prompt=system_prompt,
            )
        result = self._engine.generate_one(
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
        node_id = None if stateless else self.tree.active_node_id
        return RunSet([result], node_ids=[node_id], kind="generation")

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
        generation's teardown runs — probes detached, steering scope popped,
        lock released.

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
        return self._engine.stream(
            input,
            live_scores=live_scores,
            live_readouts=live_readouts,
            steering=steering,
            sampling=sampling,
            stateless=stateless,
            raw=raw,
            thinking=thinking,
            parent_node_id=parent_node_id,
            recipe_override=recipe_override,
            gen_seat=gen_seat,
            append_same_role=append_same_role,
            system_prompt=system_prompt,
        )

    # -- Generation: batch + sweep --

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
        serial single-prompt path.  A cast member for the assistant seat
        applies to both paths.  The session's threading lock keeps concurrent
        ``generate_batch`` calls from interleaving.  :meth:`stop` ends the
        whole batch.

        Args mirror ``generate``.  ``stateless`` defaults to ``True``
        (batch generation is overwhelmingly used for sweeps and evals
        where conversational history would corrupt the comparison);
        pass ``stateless=False`` if you genuinely want each prompt to
        accumulate against the running history.

        **Prefix KV caching.**  When the rows share a token prefix (a common
        system prompt, few-shot preamble, or shared instruction) and the
        steering doesn't touch the prefill region (``None`` or ``@response`` /
        ``@generated`` / probe-gated), that shared prefix is prefilled once and
        its KV reused across every serial row.  Skipped for ``stateless=False``
        (per-row history breaks the shared prefix), ``return_hidden`` (needs
        every prefix hidden state), thinking, or prefill-active steering (the
        prompt KV would differ per row).

        ``on_result(idx, result)`` fires after each completion for local
        progress hooks.

        Returns:
            ``RunSet`` aligned with ``prompts``.  ``runset.grid`` records
            ``{"prompt_index": i}`` for each row.
        """
        return self._batch.batch(
            prompts, steering=steering, sampling=sampling, thinking=thinking,
            stateless=stateless, raw=raw, on_result=on_result,
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

        ``sweep`` maps ``concept_name -> [alpha_0, alpha_1, ...]``; one row per
        element of the product, in ``itertools.product`` order over
        ``sweep.items()`` (``{"honest": [-0.4, 0.0, 0.4], "warm": [0.0, 0.3]}``
        gives six rows).  Each row runs under ``base_steering + " + ".join(
        f"{alpha} {name}")`` with a derived per-row seed.  Stateful sweeps land
        every row as a sibling under one shared user turn.  :meth:`stop` ends
        the sweep.

        ``on_result(idx, result, alpha_values)`` fires per completion with the
        ``{concept: alpha}`` dict that produced the row.

        Returns:
            ``RunSet`` in product order.  ``runset.grid[i]`` is the
            alpha dict for row ``i``; ``runset.node_ids[i]`` is the
            assistant node id when ``stateless=False``.
        """
        return self._batch.sweep(
            prompt, sweep, base_steering=base_steering, sampling=sampling,
            thinking=thinking, stateless=stateless, raw=raw,
            on_result=on_result, parent_node_id=parent_node_id,
        )

    # -- Generation control --

    def stop(self) -> None:
        """Ask the in-flight generation to stop after its current step."""
        self._engine.stop()

    def _release_generation(self) -> None:
        """The session half of a generation's teardown (the engine calls it).

        Detaches capture, clears the per-generation probe-inspector pass and
        the loom reservation, closes the instrument runs and resets the read
        plane's per-generation state.  The run closure and the resets survive
        a raising detach, so a bound run can never pin a stale lens.
        """
        try:
            self._end_capture()
            self._monitor.set_subspace_coords(False)
        finally:
            # Mutators (edit / delete on this subtree) are free again once the
            # streaming target is no longer live, even if finalize raised.
            self.tree.reserve(None)
            try:
                self._close_instrument_runs()
            finally:
                # The next generation starts from the default (FULL,
                # non-persistent) capture state with empty reading rows.
                self._capture_state = CaptureState()
                self._incremental_readings = []
                self._incremental_gate_scores = []

    # -- Lifecycle --

    def close(self) -> None:
        """Detach everything this session attached to the model and drop its state.

        ``from_pretrained`` installs two families of *permanent* forward hooks
        before ``torch.compile`` — the persistent steering offsets and the
        persistent capture buffers — and ``DrowseSession.__init__`` accepts a
        pre-loaded model the caller may keep using, so the model does not
        necessarily die with the session.  Both families come off here
        alongside the transient steering hooks, releasing their
        ``2 x n_layers x hidden_size`` of device buffers with them.
        """
        # Hooks and device buffers.
        self._steering.clear_all()
        self._steering.detach_compiled_offsets()
        self._runtime.close()
        self._engine.close()
        self._capture.detach()
        self._capture.clear()
        self._chat.clear_caches()

        # Read families: runs, attached probes, live readouts, residency.
        self._geometry_instrument.close_run()
        self._lens_instrument.close()
        self._sae_instrument.close_run()
        self._sae_instrument.probes.clear()
        self._sae_instrument.live = None
        self._sae_instrument.runtime.close()
        for name in list(self._monitor.probe_names):
            self._monitor.remove_probe(name)
        self._monitor.set_whitener(None)
        self._incremental_readings = []
        self._incremental_gate_scores = []

        # The artifact registry, its derived caches, the neutral baseline.
        self._registry.clear()
        self._probe_hash_cache.clear()
        self._joint_logprob_cache.clear()
        self._neutral.reset()

        self.events.clear()

    def __enter__(self) -> "DrowseSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

