"""Behavior-space naturalness eval for manifold steering.

Mirrors the second half of Goodfire's "Manifold Steering" (arXiv
2605.05115): fit a manifold over the model's *output* distributions in
Hellinger space, then measure how far a steered generation's behavioral
trajectory strays off it.

Both sides are read in the same regime: the unsteered model's next-token
distribution at every position of an assistant response, rendered through
the chat template under the capture framing (the shared length directive,
the user prompt the response answers).  A node's behavior centroid is the
mean of those distributions over its corpus; a trajectory is the
per-position sequence for one steered generation.  The behavior manifold
uses the manifold's own per-model layout — the authored coordinates, or the
discover-mode layout of the fitted manifold — so it shares the activation
manifold's chart.

:func:`evaluate` is the whole experiment (``drowse experiment naturalness``);
:func:`preflight` runs its model-free validation up front.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

import torch

from drowse.errors import DrowseError
from drowse.fit.capture import _load_baseline_prompts, _render_capture_text
from drowse.geometry.domains import ManifoldDomain, domain_from_spec
from drowse.geometry.invert import invert_parameterization
from drowse.model.tokens import last_content_index, special_token_ids
from drowse.geometry.subspace import (
    DEFAULT_N_COMPONENTS,
    LayerSubspace,
    fit_curved_subspace,
)

if TYPE_CHECKING:
    from drowse.geometry.manifold import Manifold
    from drowse.geometry.profile import Profile
    from drowse.generation.sampling import SamplingConfig
    from drowse.steering.grammar import ManifoldTerm
    from drowse.store.manifold_folder import ManifoldFolder


class ArtifactSink(Protocol):
    """Loads fitted manifolds and registers in-memory steering profiles."""

    def ensure_manifold_loaded(self, key: str) -> "Manifold": ...

    def put_profile(self, name: str, tensors: "Profile") -> None: ...


class NaturalnessHost(Protocol):
    """The loaded-model surface :func:`evaluate` drives (a ``DrowseSession``
    satisfies it): the model, manifold fits, steered generation, and the
    artifact registry (manifold loads, the linear-chord baseline profile)."""

    @property
    def model(self) -> Any: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def registry(self) -> ArtifactSink: ...

    def fit(self, folder: Any) -> "Manifold": ...

    def generate(
        self, input: Any, *, steering: str | None = ...,
        sampling: "SamplingConfig | None" = ...,
    ) -> Any: ...


class NaturalnessError(ValueError, DrowseError):
    """The manifold cannot carry a behavior-space naturalness eval."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def to_hellinger(p: torch.Tensor) -> torch.Tensor:
    """Map a probability distribution to Hellinger space: ``p -> sqrt(p)``.

    In Hellinger space the L2 distance between two mapped distributions
    is their Hellinger distance, and the inner product is the
    Bhattacharyya coefficient -- which linearizes the simplex enough for
    a Euclidean PCA + RBF fit.
    """
    return p.clamp(min=0.0).sqrt()


def bhattacharyya_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Bhattacharyya distance ``-ln sum sqrt(p*q)`` between distributions.

    ``p`` and ``q`` are probability distributions over the last axis.
    Returns the (non-negative) distance over any leading batch dims; the
    coefficient is clamped to ``[1e-12, 1]`` so fp32 roundoff on identical
    inputs cannot produce a negative distance.
    """
    bc = (p.clamp(min=0.0) * q.clamp(min=0.0)).sqrt().sum(dim=-1)
    return -torch.log(bc.clamp(min=1e-12, max=1.0))


def fit_behavior_manifold(
    centroid_dists: torch.Tensor,
    node_params: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> LayerSubspace:
    """Fit an RBF interpolant through per-node output-distribution centroids.

    ``centroid_dists`` is ``(K, V)`` -- one mean next-token distribution
    per manifold node -- and ``node_params`` is ``(K, m)`` the embedded
    domain coordinates.  The distributions are mapped to Hellinger space and
    fitted with Euclidean PCA + an exact ``r**3`` RBF (output-distribution
    space has no rogue activation dims, so no whitener applies).
    """
    return fit_curved_subspace(
        to_hellinger(centroid_dists), node_params, n_components=n_components,
    ).subspace


def trajectory_naturalness(
    traj_dists: torch.Tensor,
    behavior: LayerSubspace,
    domain: ManifoldDomain,
    node_coords: torch.Tensor,
) -> torch.Tensor:
    """Per-step Bhattacharyya distance from a trajectory to a behavior manifold.

    ``traj_dists`` is ``(T, V)``.  For each step the nearest point on the
    behavior manifold is found (in Hellinger space) and the Bhattacharyya
    distance to it is returned -- low means the step sits on the natural
    behavior manifold, high flags an off-manifold "teleportation" step.
    ``node_coords`` ``(K, n)`` seed the inverse map.  Returns ``(T,)``.
    """
    h = to_hellinger(traj_dists)  # (T, V)
    coords = (h - behavior.mean) @ behavior.basis.T  # (T, R)
    pos, _ = invert_parameterization(behavior, domain, coords, node_coords)
    curve_h = behavior.eval_at(domain.embed(pos))  # (T, V)
    # In Hellinger space ||h_a - h_b||^2 = 2 - 2*BC, so the Bhattacharyya
    # coefficient is BC = 1 - d^2/2 and the distance is -ln(BC).
    d2 = ((h - curve_h) ** 2).sum(dim=-1)
    bc = (1.0 - d2 / 2.0).clamp(min=1e-12, max=1.0)
    return -torch.log(bc)


def response_distributions(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: "str | list[dict[str, str]]",
    response: str,
    *,
    model_type: str | None = None,
) -> torch.Tensor:
    """Next-token distributions at every position of a chat-rendered response.

    Renders ``[system: directive, *prompt, assistant: response]`` exactly as
    activation capture does and returns the ``(T, V)`` fp32 CPU softmax of
    the logits that predict each response token.  Only those trailing
    positions are projected through the LM head (``logits_to_keep``).
    """
    full = _token_ids(tokenizer, _render_capture_text(
        tokenizer, prompt, response, model_type=model_type,
    ))
    empty = _token_ids(tokenizer, _render_capture_text(
        tokenizer, prompt, "", model_type=model_type,
    ))
    start = 0
    for a, b in zip(full, empty):
        if a != b:
            break
        start += 1
    start = max(start, 1)
    end = last_content_index(full, tokenizer, skip=special_token_ids(tokenizer))
    if end < start:
        raise NaturalnessError("response tokenized to zero content tokens")
    keep = len(full) - (start - 1)
    ids = torch.tensor([full], dtype=torch.long, device=device)
    with torch.inference_mode():
        try:
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits
        except TypeError:
            logits = model(input_ids=ids, use_cache=False).logits
    logits = logits[0, -keep:][: end - start + 1]
    return torch.softmax(logits.float(), dim=-1).to("cpu")


def node_behavior_centroid(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: "Sequence[tuple[str | list[dict[str, str]], str]]",
    *,
    model_type: str | None = None,
) -> torch.Tensor:
    """Mean response-position next-token distribution over a node's corpus.

    ``rows`` are ``(prompt, response)`` pairs.  Returns a ``(V,)`` fp32 CPU
    distribution: each response contributes the mean of its per-position
    distributions, so a long response does not outweigh a short one.
    """
    if not rows:
        raise NaturalnessError("manifold node has no corpus rows")
    is_mps = getattr(device, "type", None) == "mps"
    acc: torch.Tensor | None = None
    for prompt, response in rows:
        dist = response_distributions(
            model, tokenizer, device, prompt, response, model_type=model_type,
        ).mean(dim=0)
        acc = dist if acc is None else acc + dist
        if is_mps:
            torch.mps.empty_cache()
    assert acc is not None
    return acc / len(rows)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


@dataclass(frozen=True)
class NaturalnessScore:
    """One scored generation: its per-step distance to the behavior manifold."""

    label: str
    steering: str
    text: str
    per_step: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.per_step) / len(self.per_step)

    @property
    def max(self) -> float:
        return max(self.per_step)


@dataclass(frozen=True)
class NaturalnessReport:
    """The experiment result: the manifold-steered score, plus a chord baseline."""

    prompt: str
    scores: tuple[NaturalnessScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "results": [
                {
                    "label": s.label, "steering": s.steering,
                    "mean_bhattacharyya": s.mean, "max_bhattacharyya": s.max,
                }
                for s in self.scores
            ],
        }


def resolve_folder(manifold: "str | Path | ManifoldFolder") -> "ManifoldFolder":
    """Load the manifold folder named by a path or an installed selector."""
    from drowse.store.manifold_folder import ManifoldFolder
    from drowse.store.selectors import SelectorError, parse, resolve

    if isinstance(manifold, ManifoldFolder):
        return manifold
    path = Path(manifold)
    if not (path / "manifold.json").is_file():
        try:
            matches = resolve(parse(str(manifold)))
        except SelectorError:
            matches = []
        if len(matches) != 1:
            raise NaturalnessError(
                f"naturalness: {manifold!r} is neither a manifold folder nor "
                "a unique installed manifold"
            )
        path = matches[0].folder
    # Only the authoring geometry and the corpus are read here; the fitted
    # payloads' integrity is the session fit's concern.
    return ManifoldFolder.load(path, verify_manifest=False)


def preflight(
    manifold: "str | Path | ManifoldFolder",
    steering: str,
    compare_linear: bool = False,
) -> "ManifoldFolder":
    """Validate a naturalness eval without a model; return the resolved folder.

    Everything :func:`evaluate` can reject before it needs a session: the
    manifold must resolve to one folder with a node corpus (not ``baked``) and
    at least 3 nodes, and ``compare_linear`` needs ``steering`` to be a single
    ``%`` manifold term.  Raises :class:`NaturalnessError`; call it before an
    expensive model load to fail fast.
    """
    folder = resolve_folder(manifold)
    if folder.fit_mode == "baked":
        raise NaturalnessError(
            f"naturalness: {folder.name!r} is a baked manifold with no node corpus"
        )
    if len(folder.node_labels) < 3:
        raise NaturalnessError(
            f"naturalness: {folder.name!r} has {len(folder.node_labels)} node(s); "
            "a behavior manifold needs at least 3"
        )
    if compare_linear:
        _single_manifold_term(steering)
    return folder


def _single_manifold_term(steering: str) -> "ManifoldTerm":
    from drowse.steering.grammar import ManifoldTerm, parse_expr

    terms = list(parse_expr(steering).alphas.values())
    if len(terms) != 1 or not isinstance(terms[0], ManifoldTerm):
        raise NaturalnessError(
            "naturalness: compare_linear requires the steering expression "
            "to be a single manifold term"
        )
    return terms[0]


def _behavior_layout(
    session: NaturalnessHost, folder: "ManifoldFolder",
) -> tuple[ManifoldDomain, torch.Tensor]:
    """The chart the behavior manifold is fitted over: ``(domain, node_coords)``.

    Authored manifolds carry it in ``manifold.json``; discover manifolds only
    have it per model, so they come from the fitted manifold (a cache hit when
    the folder is already fitted for this model).
    """
    if folder.fit_mode == "authored":
        return (
            domain_from_spec(folder.domain),
            torch.tensor(folder.node_coords, dtype=torch.float32),
        )
    fitted = session.fit(folder.folder)
    if list(fitted.node_labels) != list(folder.node_labels):
        raise NaturalnessError(
            f"naturalness: the fitted {folder.name!r} nodes "
            f"{list(fitted.node_labels)} do not match the folder's corpus nodes "
            f"{list(folder.node_labels)}"
        )
    return (
        fitted.domain,
        fitted.node_coords.detach().to(device="cpu", dtype=torch.float32),
    )


def _corpus_rows(
    folder: "ManifoldFolder",
) -> list[tuple[str, list[tuple["str | list[dict[str, str]]", str]]]]:
    """Per-node ``(label, [(prompt, response), ...])`` aligned like capture."""
    if folder.template_ref is not None:
        from drowse.store.templates import resolve_template

        prompts: list[str | list[dict[str, str]]] = [
            context.messages()
            for context in resolve_template(folder.template_ref).contexts
        ]
    else:
        prompts = list(_load_baseline_prompts())
    k = len(prompts)
    return [
        (label, [(prompts[i % k], response) for i, response in enumerate(responses)])
        for label, responses in folder.node_groups()
    ]


def evaluate(
    session: NaturalnessHost,
    manifold: "str | Path | ManifoldFolder",
    prompt: str,
    steering: str,
    compare_linear: bool = False,
    *,
    max_tokens: int = 128,
    on_progress: Callable[[str], None] | None = None,
) -> NaturalnessReport:
    """Score how natural a steered generation is against a behavior manifold.

    ``manifold`` is a manifold folder path or an installed manifold selector;
    its nodes (≥ 3, with a corpus) define the behavior manifold over the
    manifold's own per-model layout.  ``steering`` is generated from
    ``prompt`` and scored per step.  ``compare_linear`` (``steering`` must be a
    single ``%`` manifold term) also scores the straight activation-space
    chord from the manifold's first node to the term's position.
    """
    from drowse.generation.sampling import SamplingConfig
    from drowse.geometry.profile import Profile

    def _say(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    folder = preflight(manifold, steering, compare_linear)
    linear_term = _single_manifold_term(steering) if compare_linear else None
    domain, node_coords = _behavior_layout(session, folder)
    model, tokenizer, device = session.model, session.tokenizer, session.device
    model_type = getattr(getattr(model, "config", None), "model_type", None)

    groups = _corpus_rows(folder)
    _say(f"fitting behavior manifold ({len(groups)} nodes)...")
    centroids = torch.stack([
        node_behavior_centroid(model, tokenizer, device, rows, model_type=model_type)
        for _label, rows in groups
    ])
    try:
        behavior = fit_behavior_manifold(centroids, domain.embed(node_coords))
    except ValueError as exc:
        raise NaturalnessError(
            f"naturalness: {folder.name!r} layout cannot carry a behavior "
            f"manifold ({exc})"
        ) from exc

    sampling = SamplingConfig(max_tokens=max_tokens, seed=0)

    def _score(label: str, steer: str, shown: str) -> NaturalnessScore:
        text = session.generate(prompt, steering=steer, sampling=sampling).first.text
        traj = response_distributions(
            model, tokenizer, device, prompt, text, model_type=model_type,
        )
        per_step = trajectory_naturalness(traj, behavior, domain, node_coords)
        return NaturalnessScore(label, shown, text, tuple(per_step.tolist()))

    scores = [_score("manifold", steering, steering)]
    if linear_term is not None:
        act = session.registry.ensure_manifold_loaded(linear_term.manifold)
        position = act.resolve_position(linear_term.position)
        start = act.node_coords[0]
        chord = {
            L: act.manifold_point(L, position) - act.manifold_point(L, start)
            for L in act.layer_indices
        }
        session.registry.put_profile("__manifold_linear_baseline__", Profile(chord))
        shown = (
            linear_term.position if isinstance(linear_term.position, str)
            else ",".join(f"{c:g}" for c in position)
        )
        scores.append(_score(
            "linear-chord",
            f"{linear_term.coeff:g} __manifold_linear_baseline__",
            f"chord@{shown}",
        ))
    return NaturalnessReport(prompt=prompt, scores=tuple(scores))


__all__ = [
    "NaturalnessError",
    "NaturalnessHost",
    "NaturalnessReport",
    "NaturalnessScore",
    "bhattacharyya_distance",
    "evaluate",
    "fit_behavior_manifold",
    "node_behavior_centroid",
    "preflight",
    "resolve_folder",
    "response_distributions",
    "to_hellinger",
    "trajectory_naturalness",
]
