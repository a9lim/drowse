"""DrowseSession — unified backend for drowse's programmatic and server APIs."""
from __future__ import annotations
import logging
import threading
from contextlib import contextmanager
from functools import partial
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, cast, overload

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from drowse.fit.capture import _capture_all_hidden_states
from drowse.fit.corpus import RESPONSE_MAX_TOKENS, CorpusGenerator
from drowse.session.extraction import Extraction
from drowse.errors import GenerationRequestError
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
from drowse.steering.composer import SteeringComposer, SteeringScope
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
from drowse.readout.types import InstrumentFamily, ProbeReading
from drowse.readout.monitor import Monitor
from drowse.readout.roster import ProbeRoster
from drowse.session.read_plane import ReadPlane
from drowse.readout.geometry import GeometryInstrument
from drowse.lens.instrument import LensInstrument
from drowse.sae.instrument import SaeInstrument
from drowse.sae.runtime import SaeRuntime
from drowse.session.replay import TokenReplayer
from drowse.store.bundled_probes import bootstrap_probe_roster
from drowse.geometry.profile import Profile
from drowse.generation.results import (
    GenerationResult,
    RunSet,
)
from drowse.generation.sampling import SamplingConfig
from drowse.session.batch import BatchRunner, FastBatchPath
from drowse.steering.terms import Steering
from drowse.geometry.manifold import Manifold

if TYPE_CHECKING:
    from drowse.readout.protocol import Instrument
    from drowse.generation.scoring import ChoiceScores
    from drowse.store.manifold_folder import DiscoverGenerationPlan
    from drowse.store.templates import TemplateFolder

_log = logging.getLogger(__name__)



PROBE_CATEGORIES = [
    "epistemic",
    "alignment",
    "register",
    "cultural",
]




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

        # Per-token hidden-state capture, read inline with generation (no
        # second forward pass).  The read plane (built below) owns it; the
        # lens and SAE instruments read its latest slices.
        capture = HiddenCapture()

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
        # Artifact registry: profiles + loaded manifolds; a manifold swap
        # refreshes the monitor probes reading it.  Its mutations leave the
        # prefix cache alone: the cached prefix is unsteered, and reuse is
        # gated per generation by ``steering_active_in_prefill``.
        self._registry = ArtifactRegistry(ModelBinding(
            model=self._model, tokenizer=self._tokenizer,
            model_id=self.model_id, device=self._device, dtype=self._dtype,
        ))
        self._registry.on_swap(lambda swaps: self._roster.swap(swaps))
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
            capture=capture,
            exclusive=self._model_exclusive,
            capture_hidden=partial(
                _capture_all_hidden_states, self._model, self._layers,
            ),
            registry=self._registry,
            on_change=lambda names: self._roster.changed(names),
            resolve_profile=self.ensure_profile_registered,
            assert_unsteered=self._assert_unsteered_artifact_operation,
            replay=self._token_replay,
        )
        # The SAE family: resident dictionary + Neuronpedia metadata (the
        # runtime), probes, live discovery and token replay.  ``sae/<id>``
        # directions are registry profiles.
        self._sae_instrument = SaeInstrument(
            capture,
            SaeRuntime(
                self,
                exclusive=self._model_exclusive,
                assert_unsteered=self._assert_unsteered_artifact_operation,
            ),
            replay=self._token_replay,
            put_direction=self._registry.put_profile,
            evict_directions=lambda: self._registry.evict_prefix("sae/"),
            forget_probe_hashes=lambda names: self._roster.forget(names),
        )
        # The probe roster: attach/detach routing by namespace and the
        # transcript probe-hash caches; a roster change drops the prefix cache
        # (the next capture attaches a different layer set) and the analytics.
        self._roster = ProbeRoster(
            geometry=self._geometry_instrument,
            lens=self._lens_instrument,
            sae=self._sae_instrument,
            on_change=self._on_roster_change,
        )
        # The read plane: capture planning and buffers, the per-token reads,
        # the probe-gate scores, and the finalize reads.
        self._read_plane = ReadPlane(
            capture=capture,
            monitor=self._monitor,
            geometry=self._geometry_instrument,
            lens=self._lens_instrument,
            sae=self._sae_instrument,
            layers=self._layers,
            tokenizer=self._tokenizer,
            gate_refs=self._steering_composer.gate_refs,
        )
        # The generation engine: phase, decode state, prefix cache, routing,
        # last result.  It shares the exclusive-GPU lock with every model op.
        self._engine = GenerationEngine(
            self,
            lock=self._gen_lock,
            runtime=self._runtime,
            steering=self._steering,
            reads=self._read_plane,
            release=self._release_generation,
            default_return_top_k=return_top_k,
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
                reads=self._read_plane,
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
        """Every attached probe across the three families: ``name ->
        {"manifold": …}`` (geometry), ``{"lens": spec}`` or ``{"sae": spec}``,
        from each instrument's locked snapshot."""
        return self._roster.specs()

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

    def _on_roster_change(self) -> None:
        """A probe was attached, detached or changed: the next capture
        attaches a different layer set than the cached prefix was prefilled
        under, and the analytics roster moved."""
        self._engine.prefix_cache.clear()
        self._registry.invalidate_analytics()

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
        seed: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]:
        """Each concept's in-character corpus; see :meth:`CorpusGenerator.responses`."""
        return self._corpus.responses(
            concepts, kinds, roles=roles, custom_system=custom_system,
            samples_per_prompt=samples_per_prompt, max_new_tokens=max_new_tokens,
            seed=seed, on_progress=on_progress,
        )

    def generate_manifold(
        self,
        plan: "DiscoverGenerationPlan",
        *,
        custom_system: str | None = None,
        samples_per_prompt: int = 1,
        seed: int | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Fill a discover plan's pending node corpora; see :meth:`Extraction.generate_manifold`.

        ``plan`` comes from
        :func:`~drowse.store.manifold_authoring.plan_discover_generation`.
        """
        self._extraction.generate_manifold(
            plan, custom_system=custom_system,
            samples_per_prompt=samples_per_prompt, seed=seed,
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

    @overload
    def score_hidden(
        self, hidden: dict[int, torch.Tensor], *, per_token: Literal[False] = False,
    ) -> dict[str, "ProbeReading"]: ...
    @overload
    def score_hidden(
        self, hidden: dict[int, torch.Tensor], *, per_token: Literal[True],
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]: ...
    def score_hidden(
        self, hidden: dict[int, torch.Tensor], *, per_token: bool = False,
    ) -> (
        dict[str, "ProbeReading"]
        | tuple[dict[str, "ProbeReading"], dict[str, list[float]]]
    ):
        """Score the attached geometry probes against pre-captured hidden states.

        Accepts any ``{layer_idx: Tensor}`` mapping — e.g. the
        ``GenerationResult.hidden_states`` of a ``return_hidden=True``
        generation.  Each value ``[D]`` is one state (returns
        ``{probe: ProbeReading}``); each ``[T, D]`` a per-token stack (the
        aggregate pools row ``T-1``; ``per_token=True`` returns
        ``(aggregate, per_token_scores)`` with the per-probe axis-0 coordinate
        stream).  Mixed shapes, uneven ``T`` across layers, a dim mismatch or
        an empty dict raise :class:`DrowseError`.
        """
        if per_token:
            return self._read_plane.score_hidden(hidden, per_token=True)
        return self._read_plane.score_hidden(hidden)

    # -- Monitoring --

    def add_probe(
        self,
        selector: str,
        *,
        as_name: str | None = None,
        top_n: int = 3,
    ) -> str:
        """Attach a read-side probe — any shape — and return its name.

        ``jlens/<word>`` reads the J-lens readout channel and ``sae/<id>`` an
        SAE feature; any other selector is a geometry probe, one attach for
        vector and manifold probes alike (a vector is the rank-1 case of the
        unified subspace readout).  It rides the ``[ns/]name[:variant]`` shape
        the steering grammar uses for ``%`` operands, so a probe shares the
        lazy-load cache with a manifold steered through
        ``<selector>%<position>``.  ``as_name`` overrides the registered name
        (default ``selector``); ``top_n`` sets the nearest-node list length.
        Resolution order is in :meth:`_resolve_probe_manifold`.
        """
        return self._roster.attach(selector, as_name=as_name, top_n=top_n)

    def remove_probe(self, name: str) -> None:
        """Detach a previously-attached probe (any shape).

        Lens/SAE detaches are safe mid-generation — a bound run's frozen
        binding keeps measuring the bind-time roster, so the removal applies
        at the next generation boundary.  A geometry detach mutates the live
        Monitor roster in-flight scoring walks, so it rejects while a
        generation is running (retry when idle).
        """
        self._roster.detach(name)

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

    def probe_hashes(self) -> dict[str, str]:
        """Return ``{probe_name: sha256_hex}`` for every attached probe.

        Stamps transcript exports so a replay can detect probe drift: geometry
        probes hash their baked tensor bytes, lens/SAE probes their
        readout-channel identity.
        """
        return self._roster.probe_hashes()

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
        ).with_probe_hashes(self._roster.probe_hash)
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
            self._read_plane.end()
            self._monitor.set_subspace_coords(False)
        finally:
            # Mutators (edit / delete on this subtree) are free again once the
            # streaming target is no longer live, even if finalize raised.
            self.tree.reserve(None)
            try:
                self._read_plane.close_runs()
            finally:
                # The next generation starts from the default (FULL,
                # non-persistent) capture state with empty reading rows.
                self._read_plane.reset()

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
        self._read_plane.close()
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

        # The artifact registry, its derived caches, the neutral baseline.
        self._registry.clear()
        self._roster.clear()
        self._joint_logprob_cache.clear()
        self._neutral.reset()

        self.events.clear()

    def __enter__(self) -> "DrowseSession":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

