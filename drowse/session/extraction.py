"""Extraction — the session's extract / extract_from_corpora / fit / bake glue.

A steering vector is a flat ``pca`` manifold, so concept extraction authors a
1- or 2-node discover folder (generating its corpus through
:class:`~drowse.fit.corpus.CorpusGenerator` when needed) and fits it with the
one :class:`~drowse.fit.pipeline.ManifoldExtractionPipeline`.  ``fit`` runs the
same pipeline over any folder, and ``bake`` lands a corpus-less merge.

:class:`Extraction` holds explicit dependencies: the model handle the pipeline
reads (:class:`~drowse.fit.pipeline.ModelHandle`), the event bus, the corpus
generator, the exclusive-model guard, and the artifact-registry callbacks that
move every in-memory consumer onto a fresh fit.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drowse.base.paths import manifold_dir, tensor_filename
from drowse.events import EventBus, ManifoldExtracted
from drowse.fit.corpus import CorpusGenerator, split_composite_source
from drowse.fit.pipeline import ManifoldExtractionPipeline
from drowse.geometry.fold import folded_directions
from drowse.geometry.profile import Profile
from drowse.naming import _slug, canonical_concept_name

if TYPE_CHECKING:
    from drowse.fit.pipeline import ModelHandle
    from drowse.geometry.manifold import Manifold

# A 2-node vector folder's discover hyperparameters: one axis.
_VECTOR_HYPERPARAMS = {"max_dim": 1, "var_threshold": 0.7}


@dataclass(frozen=True)
class ExtractionTarget:
    """Where a concept folder is authored and fitted.

    ``upstream`` names the source tier (``"bundled"`` or ``"hf://..."``) of an
    installed folder that ``extract`` resolved by name; ``pack refresh``
    re-pulls such a folder, so its corpus must not be regenerated in place.
    It is ``None`` for an explicit namespace and for a local folder.
    """

    namespace: str
    upstream: str | None = None


def _upstream_source(namespace: str, folder: Path) -> str | None:
    """The refresh tier of an installed folder, or ``None`` for a local one.

    Mirrors :func:`drowse.store.manifold_lifecycle.refresh_manifold`: the
    ``default/`` namespace and ``source == "bundled"`` re-materialize from the
    package, and an ``hf://`` source re-pulls from the hub.
    """
    from drowse.store.manifold_folder import read_manifest

    if namespace == "default":
        return "bundled"
    try:
        source = read_manifest(folder).get("source")
    except (OSError, ValueError):
        return None
    if source == "bundled" or (isinstance(source, str) and source.startswith("hf://")):
        return source
    return None


def _describe_upstream(upstream: str) -> str:
    return "the drowse package" if upstream == "bundled" else upstream


def resolve_extraction_target(
    name: str, namespace: str | None, *, force: bool,
) -> ExtractionTarget:
    """The namespace ``extract`` targets for canonical ``name``.

    An explicit ``namespace`` wins.  Otherwise an installed manifold with this
    name is fitted where it lives, never shadowed by a second ``local/`` copy
    (that copy would make the bare name ambiguous for every later steering
    expression), and two installed copies are already ambiguous and raise.
    An installed folder with an upstream (bundled or hub-installed) is
    re-pulled by ``pack refresh``, so ``force`` there, which would regenerate
    its corpus in place only to be silently undone, requires an explicit
    namespace.
    """
    if namespace:
        return ExtractionTarget(namespace)
    from drowse.store.selectors import Selector, resolve

    matches = resolve(Selector(kind="name", value=name))
    if not matches:
        return ExtractionTarget("local")
    (match,) = matches
    upstream = _upstream_source(match.namespace, match.folder)
    if force and upstream is not None:
        raise ValueError(
            f"{name!r} is installed at {match.namespace}/{name} from "
            f"{_describe_upstream(upstream)}, which `pack refresh` restores its "
            "corpus from. Pass namespace='local' to regenerate a forked copy "
            "(bare references then need a namespace)."
        )
    return ExtractionTarget(match.namespace, upstream)


class Extraction:
    """Author and fit steering vectors and manifolds against one loaded model.

    Dependencies:

    - ``handle`` — the :class:`~drowse.fit.pipeline.ModelHandle` the pipeline
      reads (model, tokenizer, layers, whitener, neutral means);
    - ``events`` — where ``ManifoldExtracted`` lands;
    - ``corpus`` — generates node corpora for :meth:`extract`;
    - ``exclusive(busy_msg, *, phase_msg)`` — the exclusive-model guard every
      model-running operation holds;
    - ``assert_unsteered()`` — refuses artifact work under live steering;
    - ``adopt(folder, manifold)`` / ``evict(namespace, name, *, feature)`` —
      the :class:`~drowse.session.registry.ArtifactRegistry` hooks that move
      every in-memory consumer onto a fresh fit, or drop the consumers of one
      tensor family after a failed fit that already changed the manifest;
    - ``register_profile(name, profile)`` — registers a baked vector;
    - ``dls`` — Discriminative Layer Selection for flat fits.
    """

    def __init__(
        self,
        handle: "ModelHandle",
        events: EventBus,
        corpus: CorpusGenerator,
        *,
        exclusive: Callable[..., AbstractContextManager[Any]],
        assert_unsteered: Callable[[], None],
        adopt: Callable[[Any, "Manifold"], None],
        evict: Callable[..., None],
        register_profile: Callable[[str, Profile], None],
        dls: bool,
    ) -> None:
        self._handle = handle
        self._events = events
        self._corpus = corpus
        self._exclusive = exclusive
        self._assert_unsteered = assert_unsteered
        self._adopt = adopt
        self._evict = evict
        self._register_profile = register_profile
        self._dls = bool(dls)

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
        """Author and fit a steering vector by conversational extraction.

        A **bipolar** axis (``concept`` + ``baseline``, or a composite
        ``pos.neg`` concept) authors a 2-node ``pca`` folder, node 0 the
        positive pole.  A **monopolar** concept authors a 1-node folder that
        the fit folds against the model's neutral mean ν, the implicit negative
        pole, so ``0.5 <concept>`` steers neutral → concept.

        Each node's corpus is generated by the corpus generator only when the
        folder is (re-)authored: ``force`` or no folder yet.  ``kind`` picks the
        system template and elicitation role (``"custom"`` rides
        ``custom_system`` with no role swap); ``role`` is a persona baseline
        applied at generation and capture; ``sae`` fits in SAE feature space.
        Without ``namespace`` an installed concept is fitted in place (see
        :func:`resolve_extraction_target`).

        Returns ``(name, Profile)``: the name carries the variant tail
        (``:sae-<release>`` / ``:role-<slug>``) and the profile is the folded
        per-layer direction view.  Emits one ``ManifoldExtracted``.
        """
        with self._exclusive(
            "session.extract called while another model use is in flight",
            phase_msg="session.extract called while a generation is in flight",
        ):
            self._assert_unsteered()
            pos_raw, neg_raw = split_composite_source(concept, baseline)
            name = canonical_concept_name(concept, baseline)
            target = resolve_extraction_target(name, namespace, force=force)
            if neg_raw is None:
                # Monopolar: one node.  The fit folds ``concept − ν`` into a
                # neutral-anchored ray; ν is sourced per model, never stored.
                poles = [pos_raw]
                desc = f"Monopolar axis: {pos_raw} (+) vs neutral baseline (-)."
            else:
                poles = [pos_raw, neg_raw]
                desc = f"Bipolar axis: {pos_raw} (+) vs {neg_raw} (-)."
            labels = [_slug(pole) for pole in poles]

            def generate() -> dict[str, list[str]]:
                corpora = self._corpus.responses(
                    poles, [kind] * len(poles),
                    roles={pole: role for pole in poles} if role else None,
                    custom_system=custom_system, on_progress=on_progress,
                )
                return {
                    label: corpora[pole]
                    for label, pole in zip(labels, poles, strict=True)
                }

            return self._author_and_fit_vector(
                target, name, desc, corpora=generate,
                node_kinds={label: kind for label in labels},
                role=role, force=force, sae=sae, sae_revision=sae_revision,
                on_progress=on_progress,
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
        """Author and fit a 2-node vector from ready-made pole corpora.

        The corpus-in sibling of :meth:`extract`: each entry answers the shared
        baseline prompts positionally (``response[i] -> prompt[i % k]``), so a
        corpus length must be a multiple of the prompt set.  The namespace
        resolves as in :meth:`extract`.
        """
        with self._exclusive(
            "session.extract_from_corpora called while another "
            "model use is in flight",
            phase_msg="session.extract_from_corpora called while a "
            "generation is in flight",
        ):
            self._assert_unsteered()
            pos_raw, neg_raw = split_composite_source(name, None)
            canonical = canonical_concept_name(name)
            target = resolve_extraction_target(canonical, namespace, force=force)
            pos_label = _slug(pos_raw)
            neg_label = _slug(neg_raw) if neg_raw is not None else f"{pos_label}_neg"
            return self._author_and_fit_vector(
                target, canonical, description,
                corpora={pos_label: positive, neg_label: negative},
                node_kinds={pos_label: kind, neg_label: kind},
                role=role, force=force, sae=sae, sae_revision=sae_revision,
                on_progress=on_progress,
            )

    def _author_and_fit_vector(
        self,
        target: ExtractionTarget,
        name: str,
        description: str,
        *,
        corpora: "Mapping[str, list[str]] | Callable[[], dict[str, list[str]]]",
        node_kinds: dict[str, str | None],
        role: str | None,
        force: bool,
        sae: str | None,
        sae_revision: str | None,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[str, Profile]:
        """(Re-)author a vector folder when needed, then fit it.

        The author-or-reuse decision is made under the folder's manifest lock,
        so a concurrent author or ``rm`` cannot slip between the check and the
        write.  ``corpora`` is either ready-made or a generator; generation is
        the slow model step, so it runs outside the lock and the next pass
        re-decides with the corpus in hand (reusing a folder another writer
        authored meanwhile, after the same role check).
        """
        from drowse.store.manifold_authoring import create_discover_manifold_folder
        from drowse.store.manifold_folder import _locked_manifest

        if sae is not None and role is not None:
            raise ValueError(
                "SAE-backed and role-baselined extraction are mutually exclusive"
            )
        folder = manifold_dir(target.namespace, name)
        manifest = folder / "manifold.json"
        node_corpora = None if callable(corpora) else dict(corpora)
        while True:
            with _locked_manifest(folder):
                if manifest.exists() and not force:
                    self._check_role_baseline(folder, target, name, role)
                    break
                if node_corpora is not None:
                    create_discover_manifold_folder(
                        target.namespace, name, description, fit_mode="pca",
                        node_corpora=node_corpora,
                        hyperparams=dict(_VECTOR_HYPERPARAMS),
                        node_roles=(
                            {label: role for label in node_corpora} if role else None
                        ),
                        node_kinds=node_kinds, force=force,
                    )
                    break
            assert callable(corpora)
            node_corpora = corpora()
        return self._fit_vector(
            name, folder, sae=sae, sae_revision=sae_revision,
            role=role, on_progress=on_progress,
        )

    @staticmethod
    def _check_role_baseline(
        folder: Path, target: ExtractionTarget, name: str, role: str | None,
    ) -> None:
        """Refuse to reuse a folder whose role baseline is not uniformly ``role``."""
        from drowse.store.manifold_folder import ManifoldFolder

        existing = ManifoldFolder.load(folder, verify_manifest=False).roles
        if existing and all(value == role for value in existing):
            return
        where = f"{target.namespace}/{name}"
        if target.upstream is not None:
            # ``force`` on an upstream-backed folder is refused, so the only
            # remedy is a separate local copy.
            raise ValueError(
                f"manifold {where} (installed from "
                f"{_describe_upstream(target.upstream)}) carries "
                f"role baseline {existing!r}, not uniformly {role!r}; pass "
                "namespace='local' to author a separate copy (bare references "
                "then need a namespace)"
            )
        raise ValueError(
            f"manifold {where} already carries role baseline {existing!r}, not "
            f"uniformly {role!r}; pass force=True to replace its corpus"
        )

    def _fit_vector(
        self,
        name: str,
        folder: Path,
        *,
        sae: str | None,
        sae_revision: str | None,
        role: str | None,
        on_progress: Callable[[str], None] | None,
    ) -> tuple[str, Profile]:
        """Fit a vector folder (guard held) and return its folded profile.

        The pipeline's own ``ManifoldExtracted`` is suppressed so the vector
        path emits exactly one event, carrying the folded profile and the
        variant-tailed name the caller steers with.
        """
        manifold = ManifoldExtractionPipeline(self._handle, self._events).fit(
            folder, sae=sae, sae_revision=sae_revision, on_progress=on_progress,
            emit_event=False, dls=self._dls,
        )
        self._adopt(folder, manifold)
        if sae:
            ret_name = f"{name}:sae-{sae}"
        elif role:
            ret_name = f"{name}:role-{role}"
        else:
            ret_name = name
        profile = Profile(
            folded_directions(manifold),
            metadata={
                "method": "manifold_pca",
                "name": ret_name,
                "share_metric": manifold.metadata.get("share_metric"),
            },
        )
        self._events.emit(ManifoldExtracted(
            name=ret_name, manifold=manifold,
            metadata=dict(manifold.metadata), profile=profile,
        ))
        return ret_name, profile

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
    ) -> "Manifold":
        """Fit a manifold folder (authored or discover) for the loaded model.

        The pipeline owns corpus loading, pooling, the per-layer fit and the
        cache short-circuit; ``force`` bypasses that cache.  ``layers``
        restricts the fit to explicit indices or the ``"workspace"`` band.
        ``fit_mode``/``hyperparams`` are discover overrides merged into
        ``manifold.json`` under the fit's own manifest lock; when a fit fails
        after that rewrite, the in-memory consumers of the old fit are evicted.
        """
        with self._exclusive(
            "session.fit called while another model use is in flight",
            phase_msg="session.fit called while a generation is in flight",
        ):
            self._assert_unsteered()
            manifest = Path(folder) / "manifold.json"
            manifest_before: bytes | None = None
            if fit_mode is not None or hyperparams is not None:
                try:
                    manifest_before = manifest.read_bytes()
                except OSError:
                    pass
            try:
                manifold = ManifoldExtractionPipeline(self._handle, self._events).fit(
                    folder, sae=sae, sae_revision=sae_revision,
                    layer_indices=layers, fit_mode=fit_mode,
                    hyperparams=hyperparams, force=force,
                    on_progress=on_progress, dls=self._dls,
                )
                self._adopt(folder, manifold)
                return manifold
            except BaseException:
                if manifest_before is not None:
                    try:
                        changed = manifest.read_bytes() != manifest_before
                    except OSError:
                        changed = False
                    if changed:
                        path = Path(folder)
                        self._evict(
                            path.parent.name, path.name,
                            feature=f"sae-{sae}" if sae is not None else "raw",
                        )
                raise

    def bake(
        self, name: str, expression: str, *, force: bool, strict: bool,
    ) -> tuple[str, Profile]:
        """Bake additive steering terms into a corpus-less ``local/<name>``.

        Wraps :func:`drowse.fit.bake.merge_into_manifold` scoped to the loaded
        model (fingerprint-guarded), folds the baked tensor to a
        :class:`Profile`, and registers it so it is steerable at once.
        Dynamic terms and Mahalanobis projections are rejected by the offline
        merge, which carries no identity-matched whitener.
        """
        from drowse.fit.bake import MergeError, merge_into_manifold
        from drowse.model.fingerprint import loaded_model_fingerprint
        from drowse.store.manifold_tensors import load_manifold

        model_id = self._handle.model_id
        dst_folder = merge_into_manifold(
            name, expression, model_id, force=force, strict=strict,
            expected_model_fingerprint=loaded_model_fingerprint(
                self._handle.model, model_id,
            ),
        )
        tensor_path = dst_folder / tensor_filename(model_id)
        if not tensor_path.is_file():
            raise MergeError(f"bake produced no tensor for {model_id} at {tensor_path}")
        profile = Profile(folded_directions(load_manifold(str(tensor_path))))
        self._register_profile(name, profile)
        return name, profile
