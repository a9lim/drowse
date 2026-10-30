"""Setup YAML (``~/.drowse/config.yaml`` and ``-c`` files).

The key set is the :class:`ConfigFile` dataclass fields; each key has one
validator in :data:`_VALIDATORS`.  ``vectors:`` is a steering expression
string (the grammar every surface speaks), validated at load time so a typo
fails at config load rather than at generation; it is the default steering
for the verbs that take ``-S`` (CLI ``-S`` wins).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Optional

from drowse.errors import DrowseError

log = logging.getLogger(__name__)


class ConfigFileError(ValueError, DrowseError):
    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


@dataclass
class ConfigFile:
    model: Optional[str] = None
    vectors: Optional[str] = None
    thinking: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    #: torch.compile opt-in (CUDA/MPS); unset = off.
    compile: Optional[bool] = None
    #: StaticCache + CUDA-graph capture opt-in; unset = off.
    cuda_graphs: Optional[bool] = None
    #: Session default for ``SamplingConfig.return_top_k`` (unset = 0).
    return_top_k: Optional[int] = None

    @classmethod
    def load_default(cls) -> Optional["ConfigFile"]:
        """Load ``~/.drowse/config.yaml`` if it exists, else ``None``."""
        from drowse.base.paths import drowse_home
        p = drowse_home() / "config.yaml"
        return cls.load(p) if p.exists() else None

    @classmethod
    def effective(
        cls,
        extra_paths: list[Path] | None = None,
        *,
        include_default: bool = True,
    ) -> "ConfigFile":
        """``~/.drowse/config.yaml`` (if present) then ``extra_paths`` in order,
        later entries overriding earlier ones."""
        chain: list[ConfigFile] = []
        if include_default:
            default = cls.load_default()
            if default is not None:
                chain.append(default)
        chain.extend(cls.load(Path(p)) for p in extra_paths or [])
        return compose(chain)

    def to_dict(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def to_yaml(self, *, header: Optional[str] = None) -> str:
        import yaml
        body = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        return f"{header}\n{body}" if header else body

    @classmethod
    def load(cls, path: Path) -> "ConfigFile":
        import yaml
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigFileError(f"{path}: YAML parse error: {e}") from e
        if not isinstance(data, dict):
            raise ConfigFileError(f"{path}: top-level must be a mapping")

        unknown = set(data) - set(_VALIDATORS)
        if unknown:
            names = ", ".join(repr(k) for k in sorted(unknown, key=str))
            raise ConfigFileError(f"{path}: unknown config key(s): {names}")
        values: dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue
            try:
                values[key] = _VALIDATORS[key](value)
            except ConfigFileError as e:
                raise ConfigFileError(f"{path}: {key} {e}") from e
        return cls(**values)


# ---------------------------------------------------------------------------
# Per-key validators
# ---------------------------------------------------------------------------

def _boolean(value: Any) -> bool:
    # ``bool("false") is True`` — refuse every non-bool so a quoted YAML
    # value never silently flips an opt-in.
    if not isinstance(value, bool):
        raise ConfigFileError(
            f"must be a boolean (got {type(value).__name__} {value!r})"
        )
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigFileError("must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigFileError("must be a finite number")
    return number


def _temperature(value: Any) -> float:
    number = _number(value)
    if number < 0:
        raise ConfigFileError("must be >= 0")
    return number


def _top_p(value: Any) -> float:
    number = _number(value)
    if not 0 < number <= 1:
        raise ConfigFileError("must be in (0, 1]")
    return number


def _integer(minimum: int) -> Callable[[Any], int]:
    def check(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            kind = "positive" if minimum > 0 else "nonnegative"
            raise ConfigFileError(f"must be a {kind} integer (got {value!r})")
        return value

    return check


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigFileError("must be a string")
    return value


def _model(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigFileError("must be a non-empty string")
    return value


def _vectors(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        raise ConfigFileError(
            f"must be a steering expression string (got "
            f"{type(value).__name__}). Example: "
            f'`vectors: "0.5 honest + 0.3 warm"`.'
        )
    text = value.strip()
    if not text:
        return None
    from drowse.steering.grammar import SteeringExprError, parse_expr
    try:
        parse_expr(text)
    except SteeringExprError as e:
        raise ConfigFileError(f"is not a valid steering expression: {e}") from e
    return text


_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "model": _model,
    "vectors": _vectors,
    "thinking": _boolean,
    "temperature": _temperature,
    "top_p": _top_p,
    "max_tokens": _integer(1),
    "system_prompt": _text,
    "compile": _boolean,
    "cuda_graphs": _boolean,
    "return_top_k": _integer(0),
}
assert set(_VALIDATORS) == {f.name for f in fields(ConfigFile)}


def compose(configs: list[ConfigFile]) -> ConfigFile:
    """Combine config files field by field; later entries override earlier.

    ``vectors`` overrides wholesale — a later file replaces the expression
    rather than concatenating.
    """
    out = ConfigFile()
    for c in configs:
        out = apply_flag_overrides(out, **c.to_dict())
    return out


def apply_flag_overrides(cfg_in: ConfigFile, **flags: Any) -> ConfigFile:
    """A new ConfigFile with every non-None ``flags`` value overriding ``cfg_in``."""
    return replace(cfg_in, **{k: v for k, v in flags.items() if v is not None})


# ---------------------------------------------------------------------------
# Steering references
# ---------------------------------------------------------------------------

def _bare_reference_resolves(concept: str) -> bool:
    """Whether a bare atom resolves through the steering read path's manifold
    tiers (a node label / pole, or a 2-node ``pca`` manifold name).  An
    ambiguity counts as resolved: it matches, just more than once."""
    from drowse.store.selectors import (
        AmbiguousSelectorError, resolve_manifold_label, resolve_manifold_name,
    )

    try:
        return (
            resolve_manifold_label(concept) is not None
            or resolve_manifold_name(concept) is not None
        )
    except AmbiguousSelectorError:
        return True


def ensure_vectors_installed(
    expression: str | None, *, strict: bool, dry_run: bool = False,
) -> list[str]:
    """Install every HF-namespaced manifold ``expression`` references.

    Walks the raw expression (:func:`referenced_selectors`) so ``bob/honest``
    keeps its install coordinate.  Readout atoms (the reserved ``jlens/`` /
    ``sae/`` … namespaces) name no manifold and are skipped.  Returns the
    references that could not be resolved or installed; ``strict`` raises on
    the first instead.  ``dry_run`` checks resolvability without installing
    (an uninstalled HF reference counts as installable).
    """
    from drowse.base.paths import manifold_dir
    from drowse.steering.grammar import referenced_selectors
    from drowse.store.bundled import materialize_bundled_artifacts
    from drowse.store.manifold_authoring import RESERVED_NAMESPACES

    if not expression:
        return []
    materialize_bundled_artifacts()

    missing: list[str] = []

    def miss(ref: str, why: str, cause: Exception | None = None) -> None:
        msg = f"vector {ref!r}: {why}"
        if strict:
            raise ConfigFileError(msg) from cause
        log.warning(msg)
        missing.append(ref)

    for ns, concept, _variant in referenced_selectors(expression):
        if ns in RESERVED_NAMESPACES:
            continue
        if ns is None:
            if not _bare_reference_resolves(concept):
                miss(concept, "no installed manifold matches; qualify it as "
                              "'<ns>/<name>' or extract it first")
            continue
        coord = f"{ns}/{concept}"
        if (manifold_dir(ns, concept) / "manifold.json").exists():
            continue
        if ns == "local":
            miss(coord, "local namespace, cannot auto-install")
        elif ns == "default":
            miss(coord, "bundled manifold missing from package data")
        elif not dry_run:
            try:
                from drowse.store.install import install_manifold
                install_manifold(coord, force=False)
            except Exception as e:
                miss(coord, f"install failed ({e})", e)
    return missing
