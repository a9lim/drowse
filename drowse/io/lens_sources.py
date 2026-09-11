"""Source registry for local and externally managed Jacobian lenses.

Drowse owns lenses it fits and stores those payloads below ``DROWSE_HOME``.
External publishers retain ownership of their payloads: a Hugging Face lens
stays in the Hub cache and Drowse writes only a small, commit-pinned binding
plus the per-model active-source selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml

from drowse.core.jlens import JacobianLens
from drowse.io.atomic import artifact_lock, write_json_atomic
# The three Hub indirections are imported (not redefined) so ``io.hf`` stays
# the single monkeypatchable HF seam; tests patch them on this module.
from drowse.io.hf import _hf_api, _hf_hub_download, _hf_snapshot_download
from drowse.io.integrity import NAME_REGEX, hash_file
from drowse.io.paths import ensure_within, model_dir
from drowse.io.source_registry import ActiveSourceRegistry

LENS_SOURCE_FORMAT_VERSION = 1
NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
NEURONPEDIA_BINDING = "neuronpedia"
WORKSPACE_REPO = "camilablank/workspace-lenses"
WORKSPACE_PROVIDER = "workspace-lenses"
#: Binding name -> repository lens arm.  The binding name is the public source
#: token (``lens fetch``/``use``/``show``/``rm`` all speak it), so both arms of
#: a matched J/R pair coexist per model and switch with ``lens use``.
WORKSPACE_ARMS = {"workspace-r": "r-lens", "workspace-j": "j-lens"}
#: Automatic attachment and source-picker order.  The provider R-lens is the
#: strongest default, followed by the official paper lens and the matched
#: workspace J-lens; Drowse-fitted local artifacts are the fallback tier.
DEFAULT_LENS_SOURCE_ORDER = (
    "workspace-r",
    NEURONPEDIA_BINDING,
    "workspace-j",
)
#: The estimator each arm must declare in its embedded provenance config —
#: a mismatch means the repository layout changed under us.
_WORKSPACE_ARM_ESTIMATORS = {"r-lens": "relp", "j-lens": "standard"}
_LOCAL_NAME_RE = NAME_REGEX
# The local-source grammar prefix, spelled once (the external tier addresses
# its single binding by bare name).
LOCAL_SOURCE_PREFIX = "local:"


def lens_source_preference_key(source: str) -> tuple[int, int, str]:
    """Stable default ordering for prepared J-lens sources.

    Local RelP wins within the final local tier, then the historical standard
    fit, then any other named local fit. Unknown external binding names sort
    after the four supported fallback tiers.
    """
    try:
        return (DEFAULT_LENS_SOURCE_ORDER.index(source), 0, source)
    except ValueError:
        pass
    if source.startswith(LOCAL_SOURCE_PREFIX):
        local_name = source[len(LOCAL_SOURCE_PREFIX):]
        local_rank = (
            0 if local_name == "relp"
            else 1 if local_name == "default"
            else 2
        )
        return (len(DEFAULT_LENS_SOURCE_ORDER), local_rank, local_name)
    return (len(DEFAULT_LENS_SOURCE_ORDER) + 1, 0, source)


def lens_root(model_id: str) -> Path:
    return model_dir(model_id) / "jlens"


def local_lens_dir(model_id: str, name: str = "default") -> Path:
    if _LOCAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid local J-lens name {name!r}")
    return ensure_within(lens_root(model_id) / "local", name)


def lens_bindings_dir(model_id: str) -> Path:
    return lens_root(model_id) / "bindings"


def lens_binding_path(model_id: str, name: str) -> Path:
    if _LOCAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid J-lens binding name {name!r}")
    return ensure_within(lens_bindings_dir(model_id), f"{name}.json")


def lens_active_path(model_id: str) -> Path:
    return lens_root(model_id) / "active.json"


def _validate_source_name(kind: str, name: str) -> None:
    """Both J-lens kinds are addressed by a Drowse artifact-name slug."""
    del kind
    if _LOCAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid J-lens source name {name!r}")


def _source_exists(root: Path, kind: str, name: str) -> bool:
    """Whether the named source is on disk, so a selection can never dangle."""
    if kind == "local":
        return (root / "local" / name / "manifest.json").exists()
    return (root / "bindings" / f"{name}.json").exists()


#: The active-source selection, shared implementation with the SAE family.
LENS_SOURCES = ActiveSourceRegistry(
    label="J-lens",
    format_version=LENS_SOURCE_FORMAT_VERSION,
    kinds=("local", "huggingface"),
    validate_name=_validate_source_name,
    exists=_source_exists,
)


def _active_payload(model_id: str, kind: str, name: str) -> dict[str, Any]:
    return LENS_SOURCES.payload(model_id, kind, name)


def set_active_lens_source(model_id: str, kind: str, name: str) -> Path:
    """Select the runtime J-lens source (locked, validated, existing)."""
    return LENS_SOURCES.write(lens_root(model_id), model_id, kind, name)


def set_active_local_lens(model_id: str, name: str = "default") -> Path:
    """Make a local lens active after its manifest has been published."""
    return set_active_lens_source(model_id, "local", name)


def load_active_lens_source(model_id: str) -> dict[str, Any] | None:
    """The validated ``{format_version, model_id, kind, name}`` selection."""
    return LENS_SOURCES.read(lens_root(model_id), model_id)


def _model_commit(config: Any) -> str | None:
    commit = getattr(config, "_commit_hash", None) or getattr(
        getattr(config, "text_config", None), "_commit_hash", None
    )
    return str(commit) if isinstance(commit, str) and commit else None


def _config_dimensions(config: Any) -> tuple[int | None, int | None]:
    text = getattr(config, "text_config", config)
    hidden = getattr(text, "hidden_size", getattr(text, "n_embd", None))
    layers = getattr(
        text,
        "num_hidden_layers",
        getattr(text, "n_layer", None),
    )
    return (
        int(hidden) if isinstance(hidden, int) and not isinstance(hidden, bool) else None,
        int(layers) if isinstance(layers, int) and not isinstance(layers, bool) else None,
    )


@dataclass(frozen=True)
class ExternalLensBinding:
    name: str
    model_id: str
    model_revision: str
    repo_id: str
    repo_revision: str
    checkpoint: str
    config_file: str
    config_sha256: str
    corpus: str
    n_prompts: int
    seq_len: int
    dim_batch: int
    d_model: int
    source_layers: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": LENS_SOURCE_FORMAT_VERSION,
            "kind": "huggingface",
            "provider": "neuronpedia",
            "name": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "checkpoint": self.checkpoint,
            "config_file": self.config_file,
            "config_sha256": self.config_sha256,
            "corpus": self.corpus,
            "n_prompts": self.n_prompts,
            "seq_len": self.seq_len,
            "dim_batch": self.dim_batch,
            "d_model": self.d_model,
            "source_layers": list(self.source_layers),
        }


_BINDING_FIELDS = {
    "format_version",
    "kind",
    "provider",
    "name",
    "model_id",
    "model_revision",
    "repo_id",
    "repo_revision",
    "checkpoint",
    "config_file",
    "config_sha256",
    "corpus",
    "n_prompts",
    "seq_len",
    "dim_batch",
    "d_model",
    "source_layers",
}


@dataclass(frozen=True)
class WorkspaceLensBinding:
    """Commit-pinned binding for a stacked-Jacobian ``lens.pt`` arm.

    The workspace-lenses provider publishes matched J-lens / R-lens (RelP)
    arms per model as ``<model_dir>/<arm>/lens.pt`` with the fit recipe
    embedded in the payload's ``provenance`` dict — there is no small
    per-model config file to pin, so the binding records the provenance
    fields validated at fetch time plus the checkpoint's own sha256.
    """

    name: str
    model_id: str
    model_revision: str
    repo_id: str
    repo_revision: str
    checkpoint: str
    checkpoint_sha256: str
    arm: str
    estimator: str
    rules_json: str
    corpus: str
    n_prompts: int
    seq_len: int
    skip_first: int
    target_layer: int
    d_model: int
    source_layers: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": LENS_SOURCE_FORMAT_VERSION,
            "kind": "huggingface",
            "provider": WORKSPACE_PROVIDER,
            "name": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "repo_id": self.repo_id,
            "repo_revision": self.repo_revision,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "arm": self.arm,
            "estimator": self.estimator,
            "rules_json": self.rules_json,
            "corpus": self.corpus,
            "n_prompts": self.n_prompts,
            "seq_len": self.seq_len,
            "skip_first": self.skip_first,
            "target_layer": self.target_layer,
            "d_model": self.d_model,
            "source_layers": list(self.source_layers),
        }


_WORKSPACE_BINDING_FIELDS = {
    "format_version",
    "kind",
    "provider",
    "name",
    "model_id",
    "model_revision",
    "repo_id",
    "repo_revision",
    "checkpoint",
    "checkpoint_sha256",
    "arm",
    "estimator",
    "rules_json",
    "corpus",
    "n_prompts",
    "seq_len",
    "skip_first",
    "target_layer",
    "d_model",
    "source_layers",
}

AnyLensBinding = ExternalLensBinding | WorkspaceLensBinding
_LensBindingT = TypeVar(
    "_LensBindingT", ExternalLensBinding, WorkspaceLensBinding,
)


def _parse_binding(payload: Any, model_id: str, name: str) -> AnyLensBinding:
    if not isinstance(payload, dict):
        raise ValueError("external J-lens binding has an invalid schema")
    provider = payload.get("provider")
    if provider == WORKSPACE_PROVIDER:
        return _parse_workspace_binding(payload, model_id, name)
    return _parse_neuronpedia_binding(payload, model_id, name)


def _parse_workspace_binding(
    payload: dict[str, Any], model_id: str, name: str,
) -> WorkspaceLensBinding:
    if set(payload) != _WORKSPACE_BINDING_FIELDS:
        raise ValueError("workspace J-lens binding has an invalid schema")
    layers = payload.get("source_layers")
    arm = payload.get("arm")
    if not isinstance(arm, str):
        raise ValueError("workspace J-lens binding has invalid identity metadata")
    if (
        payload.get("format_version") != LENS_SOURCE_FORMAT_VERSION
        or payload.get("kind") != "huggingface"
        or payload.get("name") != name
        or payload.get("model_id") != model_id
        or WORKSPACE_ARMS.get(name) != arm
        or payload.get("estimator") != _WORKSPACE_ARM_ESTIMATORS.get(arm)
        or not isinstance(layers, list)
        or not layers
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("workspace J-lens binding has invalid identity metadata")
    for key in (
        "model_revision",
        "repo_id",
        "repo_revision",
        "checkpoint",
        "checkpoint_sha256",
        "rules_json",
        "corpus",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"workspace J-lens binding has invalid {key}")
    for key, minimum in (
        ("n_prompts", 1),
        ("seq_len", 1),
        ("skip_first", 0),
        ("target_layer", 0),
        ("d_model", 1),
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"workspace J-lens binding has invalid {key}")
    if len(payload["checkpoint_sha256"]) != 64:
        raise ValueError("workspace J-lens binding has invalid checkpoint digest")
    return WorkspaceLensBinding(
        name=name,
        model_id=model_id,
        model_revision=payload["model_revision"],
        repo_id=payload["repo_id"],
        repo_revision=payload["repo_revision"],
        checkpoint=payload["checkpoint"],
        checkpoint_sha256=payload["checkpoint_sha256"],
        arm=arm,
        estimator=payload["estimator"],
        rules_json=payload["rules_json"],
        corpus=payload["corpus"],
        n_prompts=payload["n_prompts"],
        seq_len=payload["seq_len"],
        skip_first=payload["skip_first"],
        target_layer=payload["target_layer"],
        d_model=payload["d_model"],
        source_layers=tuple(layers),
    )


def _parse_neuronpedia_binding(
    payload: dict[str, Any], model_id: str, name: str,
) -> ExternalLensBinding:
    if set(payload) != _BINDING_FIELDS:
        raise ValueError("external J-lens binding has an invalid schema")
    layers = payload.get("source_layers")
    if (
        payload.get("format_version") != LENS_SOURCE_FORMAT_VERSION
        or payload.get("kind") != "huggingface"
        or payload.get("provider") != "neuronpedia"
        or payload.get("name") != name
        or payload.get("model_id") != model_id
        or not isinstance(layers, list)
        or not layers
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("external J-lens binding has invalid identity metadata")
    for key in (
        "model_revision",
        "repo_id",
        "repo_revision",
        "checkpoint",
        "config_file",
        "config_sha256",
        "corpus",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"external J-lens binding has invalid {key}")
    for key in ("n_prompts", "seq_len", "dim_batch", "d_model"):
        if isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) or payload[key] <= 0:
            raise ValueError(f"external J-lens binding has invalid {key}")
    if len(payload["config_sha256"]) != 64:
        raise ValueError("external J-lens binding has invalid config digest")
    return ExternalLensBinding(
        name=name,
        model_id=model_id,
        model_revision=payload["model_revision"],
        repo_id=payload["repo_id"],
        repo_revision=payload["repo_revision"],
        checkpoint=payload["checkpoint"],
        config_file=payload["config_file"],
        config_sha256=payload["config_sha256"],
        corpus=payload["corpus"],
        n_prompts=payload["n_prompts"],
        seq_len=payload["seq_len"],
        dim_batch=payload["dim_batch"],
        d_model=payload["d_model"],
        source_layers=tuple(layers),
    )


def load_external_lens_binding(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> AnyLensBinding | None:
    path = lens_binding_path(model_id, name)
    if not path.exists():
        return None
    try:
        return _parse_binding(json.loads(path.read_text()), model_id, name)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _match_official_config(
    root: Path,
    model_id: str,
    dataset: str,
) -> tuple[Path, dict[str, Any], bytes]:
    configs = sorted(root.glob(f"*/jlens/{dataset}/config.yaml"))
    matches: list[tuple[Path, dict[str, Any], bytes]] = []
    for path in configs:
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


def _load_external_checkpoint(
    path: str, *, with_provenance: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"J", "n_prompts", "source_layers", "d_model"}
    if with_provenance:
        expected = expected | {"provenance"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("external Jacobian-lens checkpoint has an invalid schema")
    return payload


def _validated_external_lens(
    payload: dict[str, Any],
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
        or isinstance(d_model, bool)
        or not isinstance(d_model, int)
        or d_model <= 0
        or isinstance(n_prompts, bool)
        or not isinstance(n_prompts, int)
        or n_prompts <= 0
        or (expected_d_model is not None and d_model != expected_d_model)
        or (
            expected_layers is not None
            and any(not isinstance(layer, int) or not 0 <= layer < expected_layers for layer in layers)
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
    return JacobianLens(clean, n_prompts=n_prompts, d_model=d_model)


def fetch_neuronpedia_lens(
    model_id: str,
    *,
    repo_id: str = NEURONPEDIA_REPO,
    revision: str = "main",
    dataset: str = "Salesforce-wikitext",
    force: bool = False,
    activate: bool = True,
) -> ExternalLensBinding:
    """Fetch a supported official lens into the HF cache and bind it.

    No external payload is copied into ``DROWSE_HOME``. The binding pins both
    the publisher repository and base-model repository to immutable commits.
    """
    api = _hf_api()
    repo_info = api.model_info(repo_id, revision=revision)
    repo_revision = getattr(repo_info, "sha", None)
    if not isinstance(repo_revision, str) or not repo_revision:
        raise ValueError(f"could not resolve immutable revision for {repo_id}")
    model_revision, d_model, n_layers = _resolve_model_for_fetch(model_id)

    root = Path(
        _hf_snapshot_download(
            repo_id,
            revision=repo_revision,
            allow_patterns=[f"*/jlens/{dataset}/config.yaml"],
        )
    )
    config_path, fit_config, config_raw = _match_official_config(
        root,
        model_id,
        dataset,
    )
    config_file = config_path.relative_to(root).as_posix()
    files = list(api.list_repo_files(repo_id, revision=repo_revision))
    checkpoint = _select_checkpoint(files, config_file)
    checkpoint_path = _hf_hub_download(
        repo_id,
        checkpoint,
        revision=repo_revision,
    )
    lens = _validated_external_lens(
        _load_external_checkpoint(checkpoint_path),
        expected_d_model=d_model,
        expected_layers=n_layers,
    )
    config_model = fit_config.get("hf_model_name")
    if not isinstance(config_model, str) or config_model.casefold() != model_id.casefold():
        raise ValueError("Neuronpedia config model does not match requested model")
    fit_raw = fit_config.get("fit")
    fit: Mapping[str, Any] = fit_raw if isinstance(fit_raw, Mapping) else {}
    dataset_raw = fit_config.get("dataset")
    dataset_cfg: Mapping[str, Any] = dataset_raw if isinstance(dataset_raw, Mapping) else {}
    corpus = ":".join(str(dataset_cfg.get(key, "")) for key in ("name", "config", "split"))
    binding = ExternalLensBinding(
        name=NEURONPEDIA_BINDING,
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        checkpoint=checkpoint,
        config_file=config_file,
        config_sha256=hashlib.sha256(config_raw).hexdigest(),
        corpus=corpus,
        n_prompts=lens.n_prompts,
        seq_len=int(fit.get("max_seq_len", 128)),
        dim_batch=int(fit.get("dim_batch", 1)),
        d_model=lens.d_model,
        source_layers=tuple(lens.source_layers),
    )
    return _publish_external_binding(binding, force=force, activate=activate)


def _publish_external_binding(
    binding: _LensBindingT, *, force: bool, activate: bool,
) -> _LensBindingT:
    """Write a binding under its lock, keeping an identical one untouched."""
    path = lens_binding_path(binding.model_id, binding.name)
    with artifact_lock(path):
        if path.exists() and not force:
            current = load_external_lens_binding(binding.model_id, binding.name)
            if current == binding:
                if activate:
                    set_active_lens_source(
                        binding.model_id, "huggingface", binding.name,
                    )
                return binding
        write_json_atomic(path, binding.to_json())
    if activate:
        set_active_lens_source(binding.model_id, "huggingface", binding.name)
    return binding


def _resolve_model_for_fetch(model_id: str) -> tuple[str, int, int]:
    """The immutable model revision and dimensions a fetch validates against."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    model_revision = _model_commit(config)
    if model_revision is None:
        raise ValueError(
            f"could not resolve immutable Hugging Face model revision for {model_id}"
        )
    d_model, n_layers = _config_dimensions(config)
    if d_model is None or n_layers is None:
        raise ValueError(f"could not resolve model dimensions for {model_id}")
    return model_revision, d_model, n_layers


def _workspace_provenance(
    payload: dict[str, Any], model_id: str, arm: str,
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
    if estimator != _WORKSPACE_ARM_ESTIMATORS[arm]:
        raise ValueError(
            f"workspace {arm} arm declares estimator {estimator!r}; "
            f"expected {_WORKSPACE_ARM_ESTIMATORS[arm]!r}"
        )
    rules = config.get("rules", {})
    corpus = prov.get("dataset_id")
    if not isinstance(corpus, str) or not corpus:
        raise ValueError("workspace lens provenance has invalid dataset_id")

    def _int_field(key: str, minimum: int) -> int:
        value = prov.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"workspace lens provenance has invalid {key}")
        return int(value)

    return {
        "estimator": estimator,
        "rules_json": json.dumps(
            rules if isinstance(rules, dict) else {}, sort_keys=True,
        ),
        "corpus": corpus,
        "seq_len": _int_field("t_max", 1),
        "skip_first": _int_field("skip_first", 0),
        "target_layer": _int_field("target_layer", 0),
    }


def fetch_workspace_lens(
    model_id: str,
    source: str = "workspace-r",
    *,
    repo_id: str = WORKSPACE_REPO,
    revision: str = "main",
    force: bool = False,
    activate: bool = True,
) -> WorkspaceLensBinding:
    """Fetch one arm of a matched workspace J/R lens pair and bind it.

    ``source`` names the arm: ``workspace-r`` is the RelP lens (an
    LRP-modified backward graph, faithful on early layers), ``workspace-j``
    the standard Jacobian lens fitted on the same forward passes.  The
    payload stays in the Hugging Face cache; Drowse writes only the
    commit-pinned binding, exactly like the Neuronpedia path.
    """
    arm = WORKSPACE_ARMS.get(source)
    if arm is None:
        raise ValueError(
            "workspace lens source must be one of " + ", ".join(sorted(WORKSPACE_ARMS))
        )
    api = _hf_api()
    repo_info = api.model_info(repo_id, revision=revision)
    repo_revision = getattr(repo_info, "sha", None)
    if not isinstance(repo_revision, str) or not repo_revision:
        raise ValueError(f"could not resolve immutable revision for {repo_id}")
    model_revision, d_model, n_layers = _resolve_model_for_fetch(model_id)
    model_dir = model_id.rsplit("/", 1)[-1].casefold()
    checkpoint = f"{model_dir}/{arm}/lens.pt"
    files = list(api.list_repo_files(repo_id, revision=repo_revision))
    if checkpoint not in files:
        published = sorted(
            {str(Path(f).parts[0]) for f in files if f.endswith("/lens.pt")}
        )
        raise FileNotFoundError(
            f"{repo_id} has no {arm} for {model_id!r} "
            f"(expected {checkpoint}; published models: {', '.join(published)})"
        )
    checkpoint_path = _hf_hub_download(repo_id, checkpoint, revision=repo_revision)
    payload = _load_external_checkpoint(checkpoint_path, with_provenance=True)
    lens = _validated_external_lens(
        payload,
        expected_d_model=d_model,
        expected_layers=n_layers,
    )
    prov = _workspace_provenance(payload, model_id, arm)
    binding = WorkspaceLensBinding(
        name=source,
        model_id=model_id,
        model_revision=model_revision,
        repo_id=repo_id,
        repo_revision=repo_revision,
        checkpoint=checkpoint,
        checkpoint_sha256=hash_file(Path(checkpoint_path)),
        arm=arm,
        estimator=prov["estimator"],
        rules_json=prov["rules_json"],
        corpus=prov["corpus"],
        n_prompts=lens.n_prompts,
        seq_len=prov["seq_len"],
        skip_first=prov["skip_first"],
        target_layer=prov["target_layer"],
        d_model=lens.d_model,
        source_layers=tuple(lens.source_layers),
    )
    published_binding = _publish_external_binding(
        binding, force=force, activate=activate,
    )
    assert isinstance(published_binding, WorkspaceLensBinding)
    return published_binding


def fetch_lens_source(
    model_id: str,
    source: str,
    *,
    repo_id: str | None = None,
    revision: str = "main",
    force: bool = False,
    activate: bool = True,
) -> AnyLensBinding:
    """Fetch a named external lens source — the CLI/server dispatch point."""
    if source == NEURONPEDIA_BINDING:
        kwargs: dict[str, Any] = {} if repo_id is None else {"repo_id": repo_id}
        return fetch_neuronpedia_lens(
            model_id, revision=revision, force=force, activate=activate, **kwargs,
        )
    if source in WORKSPACE_ARMS:
        kwargs = {} if repo_id is None else {"repo_id": repo_id}
        return fetch_workspace_lens(
            model_id, source, revision=revision, force=force, activate=activate,
            **kwargs,
        )
    raise ValueError(
        f"unknown J-lens fetch source {source!r}; expected "
        + ", ".join([NEURONPEDIA_BINDING, *sorted(WORKSPACE_ARMS)])
    )


def load_external_lens(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load one pinned external lens from the provider cache, offline-only."""
    binding = load_external_lens_binding(model_id, name)
    if binding is None:
        return None
    try:
        if isinstance(binding, WorkspaceLensBinding):
            checkpoint_path = _hf_hub_download(
                binding.repo_id,
                binding.checkpoint,
                revision=binding.repo_revision,
                local_files_only=True,
            )
            lens = _validated_external_lens(
                _load_external_checkpoint(checkpoint_path, with_provenance=True),
                expected_d_model=binding.d_model,
            )
        else:
            config_path = _hf_hub_download(
                binding.repo_id,
                binding.config_file,
                revision=binding.repo_revision,
                local_files_only=True,
            )
            raw = Path(config_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != binding.config_sha256:
                raise ValueError("cached Neuronpedia config digest changed")
            checkpoint_path = _hf_hub_download(
                binding.repo_id,
                binding.checkpoint,
                revision=binding.repo_revision,
                local_files_only=True,
            )
            lens = _validated_external_lens(
                _load_external_checkpoint(checkpoint_path),
                expected_d_model=binding.d_model,
            )
        if lens.n_prompts != binding.n_prompts or tuple(lens.source_layers) != binding.source_layers:
            raise ValueError("cached external lens does not match its binding")
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
    return lens, external_lens_sidecar(binding)


def external_lens_sidecar(binding: AnyLensBinding) -> dict[str, Any]:
    """Return canonical runtime metadata without loading provider weights."""
    if isinstance(binding, WorkspaceLensBinding):
        return {
            "format_version": LENS_SOURCE_FORMAT_VERSION,
            "method": (
                "workspace_relp" if binding.estimator == "relp"
                else "workspace_jlens"
            ),
            "n_prompts": binding.n_prompts,
            "d_model": binding.d_model,
            "source_layers": list(binding.source_layers),
            "dtype": "external",
            "corpus_spec": binding.corpus,
            "corpus_sha256": binding.checkpoint_sha256,
            "seq_len": binding.seq_len,
            "dim_batch": 1,
            "skip_first_positions": binding.skip_first,
            "model_fingerprint": None,
            "model_source_fingerprint": binding.model_revision,
            "_source": {
                "kind": "huggingface",
                "name": binding.name,
                "provider": WORKSPACE_PROVIDER,
                "repo_id": binding.repo_id,
                "repo_revision": binding.repo_revision,
                "checkpoint": binding.checkpoint,
                "model_id": binding.model_id,
                "model_revision": binding.model_revision,
                "arm": binding.arm,
                "estimator": binding.estimator,
                "rules": binding.rules_json,
                "target_layer": binding.target_layer,
            },
        }
    return {
        "format_version": LENS_SOURCE_FORMAT_VERSION,
        "method": "anthropic_jlens",
        "n_prompts": binding.n_prompts,
        "d_model": binding.d_model,
        "source_layers": list(binding.source_layers),
        "dtype": "external",
        "corpus_spec": binding.corpus,
        "corpus_sha256": binding.config_sha256,
        "seq_len": binding.seq_len,
        "dim_batch": binding.dim_batch,
        "skip_first_positions": 0,
        "model_fingerprint": None,
        "model_source_fingerprint": binding.model_revision,
        "_source": {
            "kind": "huggingface",
            "name": binding.name,
            "provider": "neuronpedia",
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "checkpoint": binding.checkpoint,
            "model_id": binding.model_id,
            "model_revision": binding.model_revision,
        },
    }


def load_external_lens_sidecar(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> dict[str, Any] | None:
    binding = load_external_lens_binding(model_id, name)
    return None if binding is None else external_lens_sidecar(binding)


def list_lens_sources(model_id: str) -> list[dict[str, Any]]:
    active = load_active_lens_source(model_id)
    rows: list[dict[str, Any]] = []
    local_root = lens_root(model_id) / "local"
    for local_manifest in sorted(local_root.glob("*/manifest.json")):
        local_name = local_manifest.parent.name
        if _LOCAL_NAME_RE.fullmatch(local_name) is None:
            continue
        rows.append(
            {
                "source": f"{LOCAL_SOURCE_PREFIX}{local_name}",
                "kind": "local",
                "name": local_name,
                "active": active == _active_payload(model_id, "local", local_name),
                "path": str(local_manifest),
            }
        )
    for path in sorted(lens_bindings_dir(model_id).glob("*.json")):
        binding = load_external_lens_binding(model_id, path.stem)
        if binding is None:
            continue
        row = {
            "source": binding.name,
            "kind": "huggingface",
            "name": binding.name,
            "provider": (
                WORKSPACE_PROVIDER
                if isinstance(binding, WorkspaceLensBinding) else "neuronpedia"
            ),
            "repo_id": binding.repo_id,
            "repo_revision": binding.repo_revision,
            "checkpoint": binding.checkpoint,
            "active": active == _active_payload(model_id, "huggingface", binding.name),
            "path": str(path),
        }
        if isinstance(binding, WorkspaceLensBinding):
            row["estimator"] = binding.estimator
        rows.append(row)
    rows.sort(key=lambda row: lens_source_preference_key(str(row["source"])))
    return rows


def lens_source_label(active: dict[str, Any]) -> str:
    """Render a selection back into the public source grammar.

    ``local:<name>`` for a Drowse-fitted lens, the bare provider name for an
    external binding — the inverse of :func:`use_lens_source`, and the SAE
    family's :func:`drowse.io.sae.sae_source_release` twin.  This is the ONE
    place the lens prefix convention is applied.
    """
    if active["kind"] == "local":
        return f"{LOCAL_SOURCE_PREFIX}{active['name']}"
    return str(active["name"])


def use_lens_source(model_id: str, source: str) -> Path:
    """Select a source from the public ``local:NAME`` / binding-name grammar."""
    source = source.strip()
    if source.startswith(LOCAL_SOURCE_PREFIX):
        return set_active_lens_source(
            model_id, "local", source[len(LOCAL_SOURCE_PREFIX):],
        )
    if (
        _LOCAL_NAME_RE.fullmatch(source) is not None
        and lens_binding_path(model_id, source).exists()
    ):
        return set_active_lens_source(model_id, "huggingface", source)
    raise ValueError(
        "J-lens source must be local:NAME or a fetched external binding "
        f"({NEURONPEDIA_BINDING}, {', '.join(sorted(WORKSPACE_ARMS))})"
    )


def remove_external_lens_binding(
    model_id: str,
    name: str = NEURONPEDIA_BINDING,
) -> bool:
    """Forget an external binding without touching its provider cache."""
    path = lens_binding_path(model_id, name)
    with artifact_lock(path):
        removed = path.exists()
        path.unlink(missing_ok=True)
    # Unpublish after the binding is gone, so the selection can never outlive
    # what it points at.  The registry owns the payload shape.
    LENS_SOURCES.clear_if_active(
        lens_root(model_id), model_id, "huggingface", name,
    )
    return removed
