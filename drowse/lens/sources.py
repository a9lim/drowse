"""The lens source family: local fits, provider bindings, per-side selection.

Drowse owns lenses it fits and stores those payloads below ``DROWSE_HOME``
(:mod:`drowse.lens.store`).  External publishers retain ownership of theirs: a
Hugging Face lens stays in the Hub cache and Drowse writes only a small,
commit-pinned :class:`LensBinding` that maps the provider's files to
:class:`~drowse.lens.key.LensKey` s.

Sources are addressed ``local:<kind>[-<tag>]`` (``local:r``, ``local:j``) or by
a provider binding's bare name (``neuronpedia``, ``workspace-r``,
``workspace-j``).  The selection is **per side**: one active output-side
(J/R) and one input-side (K/S) lens, so the two can be resident together;
every output-side read surface is estimator-agnostic within its side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from drowse.base.atomic import artifact_lock, write_json_atomic
# The three Hub indirections are imported (not redefined) so ``base.hf`` stays
# the single monkeypatchable HF seam; tests patch them on this module.
from drowse.base.hf import _hf_api, _hf_hub_download, _hf_snapshot_download
from drowse.base.integrity import hash_file
from drowse.base.source_registry import LOCAL, SourceFamily, SourceRef
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LENS_SIDES, LOCAL_LENS_NAME, LensKey, local_lens_kind

LENS_SOURCE_FORMAT_VERSION = 2
LENS_BINDING_FORMAT_VERSION = 2
HUGGINGFACE = "huggingface"
NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
WORKSPACE_REPO = "camilablank/workspace-lenses"


@dataclass(frozen=True)
class _Provider:
    """One fetchable source name: who publishes it and which lens it is."""

    provider: str
    repo: str
    key: LensKey
    arm: str | None = None        # workspace layout: <model>/<arm>/lens.pt
    estimator: str | None = None  # the arm's declared provenance estimator


#: Every fetchable source name.  The binding name is the public source token
#: (``lens fetch``/``use``/``show``/``rm`` all speak it), so both arms of a
#: matched workspace J/R pair coexist per model and switch with ``lens use``.
LENS_PROVIDERS: dict[str, _Provider] = {
    "neuronpedia": _Provider("neuronpedia", NEURONPEDIA_REPO, LensKey("J")),
    "workspace-r": _Provider(
        "workspace-lenses", WORKSPACE_REPO, LensKey("R"),
        arm="r-lens", estimator="relp",
    ),
    "workspace-j": _Provider(
        "workspace-lenses", WORKSPACE_REPO, LensKey("J"),
        arm="j-lens", estimator="standard",
    ),
}
_PROVENANCE_FIELDS = {
    "neuronpedia": {"config_file", "config_sha256", "dim_batch"},
    "workspace-lenses": {"estimator", "rules_json", "target_layer"},
}
#: Automatic attachment and source-picker order.  The provider R-lens is the
#: strongest default, then the official paper lens and the matched workspace
#: J-lens; Drowse-fitted local artifacts are the fallback tier (R before J).
DEFAULT_LENS_SOURCE_ORDER = ("workspace-r", "neuronpedia", "workspace-j")


def _lens_side(model_id: str, ref: SourceRef) -> str:
    del model_id
    if ref.local:
        return LensKey(local_lens_kind(ref.name)).side
    return LENS_PROVIDERS[ref.name].key.side


#: The lens source family: ``local:<kind>[-<tag>]`` artifacts and bare
#: provider bindings, one active source per side.
LENS_SOURCES = SourceFamily(
    label="lens",
    dirname="lens",
    format_version=LENS_SOURCE_FORMAT_VERSION,
    provider_kind=HUGGINGFACE,
    provider_prefix=None,
    local_name=LOCAL_LENS_NAME,
    provider_name_ok=lambda name: name in LENS_PROVIDERS,
    slots=LENS_SIDES,
    slot_of=_lens_side,
)


def lens_source_preference_key(source: str) -> tuple[int, int, str]:
    """Stable default ordering for prepared output-side lens sources.

    The provider tier first (:data:`DEFAULT_LENS_SOURCE_ORDER`), then local
    fits — ``local:r``, ``local:j``, then any tagged fit — and unknown names
    last.
    """
    try:
        return (DEFAULT_LENS_SOURCE_ORDER.index(source), 0, source)
    except ValueError:
        pass
    try:
        ref = LENS_SOURCES.parse(source)
    except ValueError:
        return (len(DEFAULT_LENS_SOURCE_ORDER) + 1, 0, source)
    if not ref.local:
        return (len(DEFAULT_LENS_SOURCE_ORDER) + 1, 0, source)
    rank = {"r": 0, "j": 1}.get(ref.name, 2)
    return (len(DEFAULT_LENS_SOURCE_ORDER), rank, ref.name)


# -- bindings ---------------------------------------------------------------

@dataclass(frozen=True)
class BoundLensFile:
    """One provider file holding one lens key's matrices."""

    path: str
    sha256: str
    layers: tuple[int, ...]


@dataclass(frozen=True)
class LensBinding:
    """A commit-pinned binding of provider files to lens keys.

    Common identity — publisher and base-model commits, dimensions, corpus
    recipe — is typed; ``provenance`` carries the provider's own fit record
    (a Neuronpedia config digest, a workspace arm's declared estimator and
    rules), validated per provider.
    """

    name: str
    model_id: str
    model_revision: str
    repo_id: str
    repo_revision: str
    files: dict[LensKey, BoundLensFile]
    d_model: int
    n_prompts: int
    corpus: str
    seq_len: int
    skip_first: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return LENS_PROVIDERS[self.name].provider

    @property
    def key(self) -> LensKey:
        """The bound output lens (every current provider binds exactly one)."""
        (key,) = self.files
        return key

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": LENS_BINDING_FORMAT_VERSION,
            "kind": HUGGINGFACE,
            "provider": self.provider,
            "name": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "files": {
                key.label: {
                    "path": bound.path,
                    "sha256": bound.sha256,
                    "layers": list(bound.layers),
                }
                for key, bound in self.files.items()
            },
            "d_model": self.d_model,
            "n_prompts": self.n_prompts,
            "corpus": self.corpus,
            "seq_len": self.seq_len,
            "skip_first": self.skip_first,
            "provenance": dict(self.provenance),
        }


_BINDING_FIELDS = {
    "format_version", "kind", "provider", "name", "model_id",
    "model_revision", "repo_id", "repo_revision", "files", "d_model",
    "n_prompts", "corpus", "seq_len", "skip_first", "provenance",
}


def _int_at_least(value: Any, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _layer_tuple(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list) or not value
        or any(not _int_at_least(layer, 0) for layer in value)
        or value != sorted(set(value))
    ):
        raise ValueError("lens binding has invalid layers")
    return tuple(value)


def _parse_binding(payload: Any, model_id: str, name: str) -> LensBinding:
    """Strictly validate one current-format binding payload."""
    provider = LENS_PROVIDERS.get(name)
    if (
        provider is None
        or not isinstance(payload, dict)
        or set(payload) != _BINDING_FIELDS
        or payload["format_version"] != LENS_BINDING_FORMAT_VERSION
        or payload["kind"] != HUGGINGFACE
        or payload["provider"] != provider.provider
        or payload["name"] != name
        or payload["model_id"] != model_id
    ):
        raise ValueError("lens binding has an invalid schema or identity")
    for key in ("model_revision", "repo_id", "repo_revision", "corpus"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"lens binding has invalid {key}")
    for key, minimum in (
        ("d_model", 1), ("n_prompts", 1), ("seq_len", 1), ("skip_first", 0),
    ):
        if not _int_at_least(payload[key], minimum):
            raise ValueError(f"lens binding has invalid {key}")
    raw_files = payload["files"]
    if not isinstance(raw_files, dict) or set(raw_files) != {provider.key.label}:
        raise ValueError(f"{name} must bind exactly {provider.key.label}")
    files: dict[LensKey, BoundLensFile] = {}
    for label, entry in raw_files.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "layers"}
            or not isinstance(entry["path"], str) or not entry["path"]
            or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
        ):
            raise ValueError(f"lens binding file {label} is invalid")
        files[LensKey.parse(label)] = BoundLensFile(
            entry["path"], entry["sha256"], _layer_tuple(entry["layers"]),
        )
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance) != _PROVENANCE_FIELDS[provider.provider]
    ):
        raise ValueError("lens binding has invalid provenance")
    if provider.provider == "neuronpedia":
        if (
            not isinstance(provenance["config_file"], str)
            or not provenance["config_file"]
            or not isinstance(provenance["config_sha256"], str)
            or len(provenance["config_sha256"]) != 64
            or not _int_at_least(provenance["dim_batch"], 1)
        ):
            raise ValueError("Neuronpedia lens binding has invalid provenance")
    elif (
        provenance["estimator"] != provider.estimator
        or not isinstance(provenance["rules_json"], str)
        or not _int_at_least(provenance["target_layer"], 0)
    ):
        raise ValueError("workspace lens binding has invalid provenance")
    return LensBinding(
        name=name,
        model_id=model_id,
        model_revision=payload["model_revision"],
        repo_id=payload["repo_id"],
        repo_revision=payload["repo_revision"],
        files=files,
        d_model=payload["d_model"],
        n_prompts=payload["n_prompts"],
        corpus=payload["corpus"],
        seq_len=payload["seq_len"],
        skip_first=payload["skip_first"],
        provenance=dict(provenance),
    )


def load_external_lens_binding(model_id: str, name: str) -> LensBinding | None:
    if name not in LENS_PROVIDERS:
        return None
    path = LENS_SOURCES.binding_path(model_id, name)
    if not path.exists():
        return None
    try:
        return _parse_binding(json.loads(path.read_text()), model_id, name)
    except (OSError, ValueError, TypeError, KeyError):
        return None


# -- provider checkpoints ---------------------------------------------------

def _load_provider_checkpoint(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not {"J", "n_prompts", "source_layers", "d_model"} <= set(payload)
        or set(payload) - {"J", "n_prompts", "source_layers", "d_model", "provenance"}
    ):
        raise ValueError("external Jacobian-lens checkpoint has an invalid schema")
    return payload


def _validated_external_lens(
    payload: dict[str, Any],
    key: LensKey,
    *,
    expected_d_model: int | None = None,
    expected_layers: int | None = None,
) -> JacobianLens:
    jacobians = payload.get("J")
    layers = payload.get("source_layers")
    d_model = payload.get("d_model")
    n_prompts = payload.get("n_prompts")
    if (
        not isinstance(jacobians, dict)
        or not isinstance(layers, list)
        or not layers
        or layers != sorted(set(layers))
        or set(jacobians) != set(layers)
        or not _int_at_least(d_model, 1)
        or not _int_at_least(n_prompts, 1)
        or (expected_d_model is not None and d_model != expected_d_model)
        or (
            expected_layers is not None
            and any(
                not isinstance(layer, int) or not 0 <= layer < expected_layers
                for layer in layers
            )
        )
    ):
        raise ValueError("external Jacobian-lens checkpoint metadata is incompatible")
    clean: dict[int, torch.Tensor] = {}
    for layer in layers:
        value = jacobians[layer]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (d_model, d_model)
            or not value.dtype.is_floating_point
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"external Jacobian-lens layer {layer} is invalid")
        clean[int(layer)] = value
    return JacobianLens(clean, n_prompts=n_prompts, d_model=d_model, key=key)


def _config_dimensions(model_id: str, revision: str) -> tuple[int, int]:
    """``(d_model, n_layers)`` from the config pinned to ``revision``."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=False, revision=revision,
    )
    text = getattr(config, "text_config", config)
    hidden = getattr(text, "hidden_size", getattr(text, "n_embd", None))
    layers = getattr(text, "num_hidden_layers", getattr(text, "n_layer", None))
    if not _int_at_least(hidden, 1) or not _int_at_least(layers, 1):
        raise ValueError(f"{model_id} config declares no hidden size / layer count")
    return int(hidden), int(layers)


def _resolve_model_for_fetch(model_id: str) -> tuple[str, int, int]:
    """The immutable model revision and dimensions a fetch validates against.

    The commit is resolved first and the config read pinned to it, so the
    dimensions provably belong to the revision the binding records.
    """
    from drowse.model.loading import resolve_model_revision

    model_revision = resolve_model_revision(model_id)
    if model_revision is None:
        raise ValueError(
            f"could not resolve immutable Hugging Face model revision for {model_id}"
        )
    try:
        d_model, n_layers = _config_dimensions(model_id, model_revision)
    except ValueError as exc:
        raise ValueError(f"could not resolve model dimensions for {model_id}") from exc
    return model_revision, d_model, n_layers


def _repo_commit(repo_id: str, revision: str) -> str:
    sha = getattr(_hf_api().model_info(repo_id, revision=revision), "sha", None)
    if not isinstance(sha, str) or not sha:
        raise ValueError(f"could not resolve immutable revision for {repo_id}")
    return sha


def _match_official_config(
    root: Path, model_id: str, dataset: str,
) -> tuple[Path, dict[str, Any], bytes]:
    matches: list[tuple[Path, dict[str, Any], bytes]] = []
    for path in sorted(root.glob(f"*/jlens/{dataset}/config.yaml")):
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("hf_model_name"), str)
            and payload["hf_model_name"].casefold() == model_id.casefold()
        ):
            matches.append((path, payload, raw))
    if not matches:
        raise FileNotFoundError(f"Neuronpedia has no official Jacobian lens for {model_id!r}")
    if len(matches) != 1:
        raise ValueError(f"multiple Neuronpedia lens configs match {model_id!r}")
    return matches[0]


def _select_checkpoint(files: list[str], config_file: str) -> str:
    parent = str(Path(config_file).parent)
    candidates = sorted(
        filename
        for filename in files
        if str(Path(filename).parent) == parent
        and Path(filename).suffix == ".pt"
        and "jacobian_lens" in Path(filename).name
    )
    if not candidates:
        raise FileNotFoundError(f"no Jacobian-lens checkpoint beside {config_file}")
    n1000 = [path for path in candidates if "_n1000.pt" in path]
    plain = [path for path in candidates if path.endswith("_jacobian_lens.pt")]
    return (n1000 or plain or candidates)[0]


def _fetch_neuronpedia(
    model_id: str, *, repo_id: str, revision: str, dataset: str = "Salesforce-wikitext",
) -> LensBinding:
    """Bind the official paper lens for ``model_id``."""
    provider = LENS_PROVIDERS["neuronpedia"]
    repo_revision = _repo_commit(repo_id, revision)
    model_revision, d_model, n_layers = _resolve_model_for_fetch(model_id)
    root = Path(_hf_snapshot_download(
        repo_id, revision=repo_revision,
        allow_patterns=[f"*/jlens/{dataset}/config.yaml"],
    ))
    config_path, fit_config, config_raw = _match_official_config(root, model_id, dataset)
    config_file = config_path.relative_to(root).as_posix()
    files = list(_hf_api().list_repo_files(repo_id, revision=repo_revision))
    checkpoint = _select_checkpoint(files, config_file)
    checkpoint_path = _hf_hub_download(repo_id, checkpoint, revision=repo_revision)
    lens = _validated_external_lens(
        _load_provider_checkpoint(checkpoint_path), provider.key,
        expected_d_model=d_model, expected_layers=n_layers,
    )
    fit_raw = fit_config.get("fit")
    fit: Mapping[str, Any] = fit_raw if isinstance(fit_raw, Mapping) else {}
    dataset_raw = fit_config.get("dataset")
    dataset_cfg: Mapping[str, Any] = dataset_raw if isinstance(dataset_raw, Mapping) else {}
    return LensBinding(
        name="neuronpedia",
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        files={provider.key: BoundLensFile(
            checkpoint, hash_file(Path(checkpoint_path)),
            tuple(lens.source_layers),
        )},
        d_model=lens.d_model,
        n_prompts=lens.n_prompts,
        corpus=":".join(
            str(dataset_cfg.get(key, "")) for key in ("name", "config", "split")
        ),
        seq_len=int(fit.get("max_seq_len", 128)),
        skip_first=0,
        provenance={
            "config_file": config_file,
            "config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "dim_batch": int(fit.get("dim_batch", 1)),
        },
    )


def _workspace_provenance(
    payload: dict[str, Any], model_id: str, provider: _Provider,
) -> dict[str, Any]:
    """Validate the embedded fit recipe against the requested model and arm."""
    prov = payload.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("workspace lens checkpoint carries no provenance dict")
    prov_model = prov.get("model_id")
    if not isinstance(prov_model, str) or prov_model.casefold() != model_id.casefold():
        raise ValueError(
            "workspace lens provenance names "
            f"{prov_model!r}, not the requested model {model_id!r}"
        )
    try:
        config = json.loads(prov.get("config_json", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("workspace lens provenance config is unreadable") from exc
    estimator = config.get("estimator") if isinstance(config, dict) else None
    if estimator != provider.estimator:
        raise ValueError(
            f"workspace {provider.arm} arm declares estimator {estimator!r}; "
            f"expected {provider.estimator!r}"
        )
    rules = config.get("rules", {})
    corpus = prov.get("dataset_id")
    if not isinstance(corpus, str) or not corpus:
        raise ValueError("workspace lens provenance has invalid dataset_id")
    fields: dict[str, int] = {}
    for key, minimum in (("t_max", 1), ("skip_first", 0), ("target_layer", 0)):
        value = prov.get(key)
        if not _int_at_least(value, minimum):
            raise ValueError(f"workspace lens provenance has invalid {key}")
        fields[key] = int(value)
    return {
        "corpus": corpus,
        "seq_len": fields["t_max"],
        "skip_first": fields["skip_first"],
        "provenance": {
            "estimator": estimator,
            "rules_json": json.dumps(
                rules if isinstance(rules, dict) else {}, sort_keys=True,
            ),
            "target_layer": fields["target_layer"],
        },
    }


def _fetch_workspace(
    model_id: str, name: str, *, repo_id: str, revision: str,
) -> LensBinding:
    """Bind one arm of a matched workspace J/R lens pair (``<m>/<arm>/lens.pt``)."""
    provider = LENS_PROVIDERS[name]
    repo_revision = _repo_commit(repo_id, revision)
    model_revision, d_model, n_layers = _resolve_model_for_fetch(model_id)
    checkpoint = f"{model_id.rsplit('/', 1)[-1].casefold()}/{provider.arm}/lens.pt"
    files = list(_hf_api().list_repo_files(repo_id, revision=repo_revision))
    if checkpoint not in files:
        published = sorted(
            {str(Path(f).parts[0]) for f in files if f.endswith("/lens.pt")}
        )
        raise FileNotFoundError(
            f"{repo_id} has no {provider.arm} for {model_id!r} "
            f"(expected {checkpoint}; published models: {', '.join(published)})"
        )
    checkpoint_path = _hf_hub_download(repo_id, checkpoint, revision=repo_revision)
    payload = _load_provider_checkpoint(checkpoint_path)
    lens = _validated_external_lens(
        payload, provider.key, expected_d_model=d_model, expected_layers=n_layers,
    )
    recipe = _workspace_provenance(payload, model_id, provider)
    return LensBinding(
        name=name,
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        files={provider.key: BoundLensFile(
            checkpoint, hash_file(Path(checkpoint_path)),
            tuple(lens.source_layers),
        )},
        d_model=lens.d_model,
        n_prompts=lens.n_prompts,
        corpus=recipe["corpus"],
        seq_len=recipe["seq_len"],
        skip_first=recipe["skip_first"],
        provenance=recipe["provenance"],
    )


def fetch_lens_source(
    model_id: str,
    source: str,
    *,
    repo_id: str | None = None,
    revision: str = "main",
    force: bool = False,
    activate: bool = True,
) -> LensBinding:
    """Fetch a named external lens into the HF cache and bind it.

    The CLI/server dispatch point.  ``workspace-r`` is the RelP lens (an
    LRP-modified backward graph, faithful on early layers), ``workspace-j``
    the standard lens fitted on the same forward passes, ``neuronpedia`` the
    official paper lens.  No payload is copied into ``DROWSE_HOME``; the
    binding pins the publisher and base-model repositories to commits.
    """
    provider = LENS_PROVIDERS.get(source)
    if provider is None:
        raise ValueError(
            f"unknown lens fetch source {source!r}; expected "
            + ", ".join(sorted(LENS_PROVIDERS))
        )
    repo = repo_id or provider.repo
    binding = (
        _fetch_neuronpedia(model_id, repo_id=repo, revision=revision)
        if provider.arm is None
        else _fetch_workspace(model_id, source, repo_id=repo, revision=revision)
    )
    path = LENS_SOURCES.binding_path(model_id, binding.name)
    with artifact_lock(path):
        current = load_external_lens_binding(model_id, binding.name)
        if force or current != binding:
            write_json_atomic(path, binding.to_json())
    if activate:
        LENS_SOURCES.select(model_id, SourceRef(HUGGINGFACE, binding.name))
    return binding


def load_external_lens(
    model_id: str, name: str,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load one pinned external lens from the provider cache, offline-only."""
    binding = load_external_lens_binding(model_id, name)
    if binding is None:
        return None
    key = binding.key
    bound = binding.files[key]
    try:
        config_file = binding.provenance.get("config_file")
        if isinstance(config_file, str):
            config_path = _hf_hub_download(
                binding.repo_id, config_file,
                revision=binding.repo_revision, local_files_only=True,
            )
            raw = Path(config_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != binding.provenance["config_sha256"]:
                raise ValueError("cached Neuronpedia config digest changed")
        checkpoint_path = _hf_hub_download(
            binding.repo_id, bound.path,
            revision=binding.repo_revision, local_files_only=True,
        )
        lens = _validated_external_lens(
            _load_provider_checkpoint(checkpoint_path), key,
            expected_d_model=binding.d_model,
        )
        if (
            lens.n_prompts != binding.n_prompts
            or tuple(lens.source_layers) != bound.layers
        ):
            raise ValueError("cached external lens does not match its binding")
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
    return lens, external_lens_sidecar(binding)


def external_lens_sidecar(binding: LensBinding) -> dict[str, Any]:
    """Runtime metadata for a binding, without loading provider weights.

    Shaped like a local manifest where the two agree (``kind``, ``sites``,
    dimensions, corpus, model identity) so one identity/compatibility check
    serves both; ``_source`` carries the provider record.
    """
    key = binding.key
    return {
        "kind": key.kind,
        "sites": {key.site: {"layers": list(binding.files[key].layers)}},
        "n_prompts": binding.n_prompts,
        "d_model": binding.d_model,
        "dtype": "external",
        "corpus_spec": binding.corpus,
        "seq_len": binding.seq_len,
        "skip_first_positions": binding.skip_first,
        "model_fingerprint": None,
        "model_source_fingerprint": binding.model_revision,
        "_source": {
            "kind": HUGGINGFACE,
            "name": binding.name,
            "provider": binding.provider,
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "model_id": binding.model_id,
            "model_revision": binding.model_revision,
            "files": {
                bound_key.label: {"path": bound.path, "sha256": bound.sha256}
                for bound_key, bound in binding.files.items()
            },
            **binding.provenance,
        },
    }


def load_external_lens_sidecar(model_id: str, name: str) -> dict[str, Any] | None:
    binding = load_external_lens_binding(model_id, name)
    return None if binding is None else external_lens_sidecar(binding)


# -- lifecycle --------------------------------------------------------------

def load_lens_source_sidecar(model_id: str, ref: SourceRef) -> dict[str, Any] | None:
    """A source's metadata: a local manifest or a binding's descriptor."""
    if ref.local:
        from drowse.lens.store import load_local_lens_sidecar

        return load_local_lens_sidecar(model_id, ref.name)
    return load_external_lens_sidecar(model_id, ref.name)


def list_lens_sources(model_id: str) -> list[dict[str, Any]]:
    """Every prepared source, in default preference order."""
    actives = {LENS_SOURCES.active(model_id, side) for side in LENS_SIDES}
    rows: list[dict[str, Any]] = []
    for name in LENS_SOURCES.local_names(model_id):
        ref = SourceRef(LOCAL, name)
        rows.append({
            "source": LENS_SOURCES.format(ref),
            "kind": LOCAL,
            "name": name,
            "lens": local_lens_kind(name),
            "active": ref in actives,
            "path": str(LENS_SOURCES.path_of(model_id, ref)),
        })
    for name in sorted(LENS_PROVIDERS):
        binding = load_external_lens_binding(model_id, name)
        if binding is None:
            continue
        ref = SourceRef(HUGGINGFACE, name)
        bound = binding.files[binding.key]
        rows.append({
            "source": name,
            "kind": HUGGINGFACE,
            "name": name,
            "lens": binding.key.kind,
            "provider": binding.provider,
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "checkpoint": bound.path,
            "active": ref in actives,
            "path": str(LENS_SOURCES.path_of(model_id, ref)),
        })
    rows.sort(key=lambda row: lens_source_preference_key(str(row["source"])))
    return rows


def use_lens_source(model_id: str, source: str) -> SourceRef:
    """Select a prepared source from the public grammar (into its side)."""
    ref = LENS_SOURCES.parse(source)
    if not LENS_SOURCES.exists(model_id, ref):
        raise FileNotFoundError(
            f"lens source {LENS_SOURCES.format(ref)} is not prepared for "
            f"{model_id} — fit it (local:…) or fetch it "
            f"({', '.join(sorted(LENS_PROVIDERS))})"
        )
    LENS_SOURCES.select(model_id, ref)
    return ref


def remove_lens_source(model_id: str, ref: SourceRef) -> bool:
    """Delete a local lens or forget a binding (provider cache untouched).

    Either way the selection is unpublished only when it named exactly this
    source.
    """
    if ref.local:
        from drowse.lens.store import remove_lens

        return remove_lens(model_id, ref.name)
    path = LENS_SOURCES.binding_path(model_id, ref.name)
    with artifact_lock(path):
        removed = path.exists()
        path.unlink(missing_ok=True)
    LENS_SOURCES.clear_if_active(model_id, ref)
    return removed

