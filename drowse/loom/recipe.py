"""Generation recipes, cast members, and the one recipe-composition rule.

A :class:`Recipe` is the reproducibility receipt stamped on a generated loom
node: steering expression, sampling (the seed lives in ``sampling.seed`` and
nowhere else), thinking mode, system prompt, and the probe roster live at
generation time.  A :class:`CastMember` carries a standing recipe for a cast
label.

:func:`compose_call` is the single place a generation's effective recipe is
assembled from its tiers — the cast member (weakest), the call's own kwargs,
the regen anchor's recipe, and a regen modifier (strongest).  Sampling merges
field by field at every tier; see the function for the two exceptions
(observation knobs and the seed).
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Any, Callable, NamedTuple

from drowse.generation.sampling import SamplingConfig
from drowse.loom.errors import LoomTreeError, require_fields


class InheritSystemPrompt(Enum):
    """Sentinel: the recipe captured no system prompt; inherit the session's."""
    VALUE = "session_default"


INHERIT_SYSTEM_PROMPT = InheritSystemPrompt.VALUE

# The persisted recipe shape (tree_format 2 / transcript v2).  ``seed`` is a
# top-level key on disk and on the wire; in memory it is ``sampling.seed``.
_RECIPE_FIELDS = frozenset({
    "steering", "sampling", "thinking", "seed", "probes", "probe_hashes",
})
# Derived from the dataclass so a new sampling knob can't drift the reader
# away from what ``Recipe.to_dict`` writes.
_SAMPLING_FIELDS = frozenset(f.name for f in fields(SamplingConfig))
_CAST_MEMBER_FIELDS = frozenset({"recipe", "notes"})

#: Observation knobs: they choose what a call captures, never how it samples.
#: Every composition takes them from the live call — a stamped anchor or cast
#: recipe can never switch a consumer's capture on or off.
CALL_ONLY_SAMPLING_FIELDS = frozenset({
    "logprobs",
    "return_hidden",
    "return_top_k",
    "persist_per_layer_scores",
    "persist_subspace_coords",
    "return_probe_readings",
})
_DEFAULT_SAMPLING = SamplingConfig()


def _merge_sampling(
    weak: SamplingConfig | None, strong: SamplingConfig | None,
) -> SamplingConfig | None:
    """Field-wise merge: ``strong``'s non-default fields win over ``weak``."""
    if weak is None:
        return strong
    if strong is None:
        return weak
    return weak.merged_with(strong)


def _seed_of(recipe: Recipe | None) -> int | None:
    if recipe is None or recipe.sampling is None:
        return None
    return recipe.sampling.seed


@dataclass(frozen=True)
class Recipe:
    """Reproducibility receipt for a generated node.

    Captures everything needed to replay the same generation: steering
    expression, sampling parameters (including the RNG seed), thinking mode,
    the system prompt, and the probe set live at gen time (with per-probe
    content hashes so transcript replay can detect probe drift).
    """

    steering: str | None = None
    sampling: SamplingConfig | None = None
    thinking: bool | None = None
    # Names of probes active during the gen, plus their content hashes
    # (sha256 of baked tensor bytes).  ``probe_hashes`` is empty when the
    # session has no probes or could not hash them.
    probes: list[str] = field(default_factory=list)
    probe_hashes: dict[str, str] = field(default_factory=dict)
    system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT

    def resolved_system_prompt(self, default: str | None) -> str | None:
        return default if isinstance(self.system_prompt, InheritSystemPrompt) else self.system_prompt

    def to_dict(self) -> dict[str, Any]:
        """The persisted shape.

        The seed rides the top-level ``seed`` key.  A sampling block that
        carries nothing but the seed serializes as ``"sampling": null`` —
        the shape a seed-only recipe has always had on disk and on the wire.
        """
        seed = _seed_of(self)
        sampling = self.sampling
        sampling_out: dict[str, Any] | None = None
        if sampling is not None and (
            seed is None or sampling != SamplingConfig(seed=seed)
        ):
            sampling_out = {}
            for f in fields(SamplingConfig):
                v = getattr(sampling, f.name)
                if f.name == "stop" and v is not None:
                    v = list(v)
                elif f.name == "logit_bias" and v is not None:
                    v = {str(int(tid)): float(bias) for tid, bias in v.items()}
                sampling_out[f.name] = v
        return {
            "steering": self.steering,
            "sampling": sampling_out,
            "thinking": self.thinking,
            "seed": seed,
            "probes": list(self.probes),
            "probe_hashes": dict(self.probe_hashes),
            **({} if isinstance(self.system_prompt, InheritSystemPrompt) else {"system_prompt": self.system_prompt}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        """Read the complete current writer shape — strict, like its siblings.

        A recipe is nested inside both node and cast payloads, so a lenient
        read here would let a corrupt or future-format recipe slip through the
        strict outer gate and land as a partially-defaulted object.  Every
        field ``to_dict`` writes is required; unknown keys are rejected; the
        nested sampling block is checked against ``SamplingConfig``'s own
        fields so a bad key raises :class:`LoomTreeError` rather than a bare
        ``TypeError`` out of the constructor.  The top-level ``seed`` folds
        into ``sampling.seed`` (it wins when both are set).
        """
        require_fields(data, _RECIPE_FIELDS, "recipe", optional=frozenset({"system_prompt"}))
        if "system_prompt" in data and data["system_prompt"] is not None and not isinstance(data["system_prompt"], str):
            raise LoomTreeError("recipe system_prompt must be a string or null")
        seed = data["seed"]
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise LoomTreeError("recipe seed must be an integer or null")
        steering = data["steering"]
        if steering is not None and not isinstance(steering, str):
            raise LoomTreeError("recipe steering must be a string or null")
        if data["thinking"] is not None and not isinstance(data["thinking"], bool):
            raise LoomTreeError("recipe thinking must be a boolean or null")
        if not isinstance(data["probes"], list) or not all(
            isinstance(p, str) for p in data["probes"]
        ):
            raise LoomTreeError("recipe probes must be a list of strings")
        if not isinstance(data["probe_hashes"], dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in data["probe_hashes"].items()
        ):
            raise LoomTreeError("recipe probe_hashes must map strings to strings")
        sampling = None
        s = data["sampling"]
        if s is not None:
            if not isinstance(s, dict):
                raise LoomTreeError("recipe sampling must be an object or null")
            s = dict(s)
            require_fields(s, _SAMPLING_FIELDS, "recipe sampling")
            if s.get("logit_bias") is not None:
                try:
                    s["logit_bias"] = {
                        int(tid): float(bias)
                        for tid, bias in dict(s["logit_bias"]).items()
                    }
                except (TypeError, ValueError) as e:
                    raise LoomTreeError(f"recipe sampling logit_bias is invalid: {e}") from e
            try:
                sampling = SamplingConfig(
                    **{k: v for k, v in s.items() if v is not None}
                )
            except TypeError as e:
                raise LoomTreeError(f"recipe sampling is not constructible: {e}") from e
        if seed is not None:
            sampling = replace(sampling or SamplingConfig(), seed=seed)
        return cls(
            steering=steering,
            sampling=sampling,
            thinking=data["thinking"],
            probes=list(data["probes"]),
            probe_hashes=dict(data["probe_hashes"]),
            system_prompt=data.get("system_prompt", INHERIT_SYSTEM_PROMPT),
        )

    # ------------------------------------------------------------------
    # Regen modifiers
    # ------------------------------------------------------------------

    def invert_steering(self) -> "Recipe":
        """Return a partial Recipe with every steering term's α sign flipped.

        Used by the ``inverted`` regen mode.  The grammar carries triggers /
        projections / ablations through; only the numeric coefficient
        changes.  When this recipe has no steering the partial carries the
        empty (unsteered) expression, which composes as the no-op.
        """
        from drowse.steering.grammar import (
            AblationTerm, ManifoldTerm, ProjectedTerm, parse_expr, format_expr,
        )
        from drowse.steering.terms import Steering

        if not self.steering:
            return Recipe(steering="")
        parsed = parse_expr(self.steering)
        flipped: dict[str, Any] = {}
        for name, val in parsed.alphas.items():
            if isinstance(val, (ProjectedTerm, AblationTerm)):
                flipped[name] = replace(val, coeff=-val.coeff)
            elif isinstance(val, ManifoldTerm):
                # Flip the directional ``along`` (the representative coeff);
                # ``onto`` is a collapse fraction in [0, 1], not a signed
                # push, so it carries through unchanged.
                flipped[name] = replace(val, along=-val.along)
            elif isinstance(val, tuple):
                flipped[name] = (-float(val[0]), val[1])
            else:
                flipped[name] = -float(val)
        new = Steering(
            alphas=flipped,
            thinking=parsed.thinking,
            trigger=parsed.trigger,
        )
        return Recipe(steering=format_expr(new))

    def compose_modifier(self, mode: "str | Recipe") -> "Recipe":
        """Return the partial Recipe a regen ``mode`` applies to this anchor.

        Recognized string modes:

        - ``"unsteered"``: steering wiped (empty expression).
        - ``"inverted"``: flips every term's α sign — see
          :meth:`invert_steering`.
        - ``"reseed"``: a fresh entropy seed; everything else inherits.
        - ``"cool"``: temperature 0.3; everything else inherits.
        - ``"hot"``: temperature 1.2; everything else inherits.

        A :class:`Recipe` instance passes through unchanged. Other strings
        use the dashboard's comma-separated partial-recipe grammar, for
        example ``"seed=42, temperature=1.5"`` or ``"steering=0.5 calm"``.
        Unknown fields and malformed values raise ``ValueError``.
        """
        if isinstance(mode, Recipe):
            return mode
        if mode == "unsteered":
            return Recipe(steering="")
        if mode == "inverted":
            return self.invert_steering()
        if mode == "reseed":
            return Recipe(sampling=SamplingConfig(seed=secrets.randbits(31)))
        if mode == "cool":
            return Recipe(sampling=SamplingConfig(temperature=0.3))
        if mode == "hot":
            return Recipe(sampling=SamplingConfig(temperature=1.2))
        return _parse_recipe_modifier(mode)

    def with_probe_hashes(self, lookup: Callable[[str], str | None]) -> "Recipe":
        """Return a copy whose ``probe_hashes`` maps each probe through ``lookup``.

        Probes ``lookup`` cannot hash (``None``) are left out — a transcript
        reflects the session's view at stamp time.
        """
        hashes: dict[str, str] = {}
        for name in self.probes:
            digest = lookup(name)
            if digest is not None:
                hashes[name] = digest
        return replace(self, probe_hashes=hashes)


class ComposedCall(NamedTuple):
    """A generation call's effective kwargs after :func:`compose_call`."""

    steering: Any  # str | Steering | None — a programmatic Steering passes through
    sampling: SamplingConfig | None
    thinking: bool | None
    system_prompt: str | None | InheritSystemPrompt


def compose_call(
    *,
    steering: Any = None,
    sampling: SamplingConfig | None = None,
    thinking: bool | None = None,
    system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    cast: Recipe | None = None,
    anchor: Recipe | None = None,
    modifier: Recipe | None = None,
) -> ComposedCall:
    """Compose a generation's effective recipe — the one composition rule.

    Tiers, weakest first: the ``cast`` member's standing recipe, the call's
    own kwargs, the regen ``anchor`` (the recipe of the node a regen
    reproduces), and the regen ``modifier`` (``cool``, ``inverted``, a
    partial recipe …).  At each tier a set field wins over the tiers below
    it: ``steering``/``thinking`` are set when not ``None`` (``""`` is the
    explicit unsteered expression), ``system_prompt`` when not inherited, and
    ``sampling`` merges field by field (:meth:`SamplingConfig.merged_with` —
    a field at its default is unset, so a tier cannot reset a field to its
    default).

    Two sampling fields break the tier order:

    - **Observation knobs** (:data:`CALL_ONLY_SAMPLING_FIELDS` —
      ``return_top_k``, ``logprobs``, the persist flags …) always come from
      the call; a stamped recipe never decides what a consumer captures.
    - **The seed** resolves modifier > call > anchor > cast.  A seed the
      modifier pins (``reseed``, ``seed=42``) wins; otherwise a seed the call
      supplies — the per-sibling seed a fan derives — beats the anchor's
      inherited one, so N fanned siblings stay N distinct samples; an
      unseeded one-shot regen reproduces the anchor's seed (a controlled
      comparison).
    """
    out_steering = steering
    out_thinking = thinking
    out_prompt = system_prompt
    out_sampling = sampling
    if cast is not None:
        if out_steering is None:
            out_steering = cast.steering
        if out_thinking is None:
            out_thinking = cast.thinking
        if isinstance(out_prompt, InheritSystemPrompt):
            out_prompt = cast.system_prompt
        out_sampling = _merge_sampling(cast.sampling, out_sampling)
    for layer in (anchor, modifier):
        if layer is None:
            continue
        if layer.steering is not None:
            out_steering = layer.steering
        if layer.thinking is not None:
            out_thinking = layer.thinking
        if not isinstance(layer.system_prompt, InheritSystemPrompt):
            out_prompt = layer.system_prompt
        out_sampling = _merge_sampling(out_sampling, layer.sampling)

    if out_sampling is not None:
        call_sampling = sampling if sampling is not None else _DEFAULT_SAMPLING
        seed = next(
            (
                s for s in (
                    _seed_of(modifier),
                    call_sampling.seed,
                    _seed_of(anchor),
                    _seed_of(cast),
                )
                if s is not None
            ),
            None,
        )
        updates: dict[str, Any] = {
            name: getattr(call_sampling, name)
            for name in CALL_ONLY_SAMPLING_FIELDS
        }
        updates["seed"] = seed
        out_sampling = replace(out_sampling, **updates)
    return ComposedCall(out_steering, out_sampling, out_thinking, out_prompt)


# ---------------------------------------------------------------------------
# Partial-recipe modifier grammar
# ---------------------------------------------------------------------------

_RECIPE_MODIFIER_FIELD = re.compile(
    r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*="
)
_RECIPE_MODIFIER_FIELDS = {
    "steering",
    "thinking",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
}


def _parse_recipe_modifier(value: str) -> Recipe:
    text = value.strip()
    matches = list(_RECIPE_MODIFIER_FIELD.finditer(text))
    if not matches or matches[0].start() != 0:
        raise ValueError(
            f"unknown recipe-override mode {value!r}; valid: unsteered, "
            "inverted, reseed, cool, hot, or a partial recipe such as "
            "'seed=42, temperature=1.5'"
        )

    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key not in _RECIPE_MODIFIER_FIELDS:
            raise ValueError(f"unknown custom recipe field {key!r}")
        if key in parsed:
            raise ValueError(f"duplicate custom recipe field {key!r}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = text[match.end():end].strip()
        if key != "steering" and not raw:
            raise ValueError(f"custom recipe field {key!r} requires a value")
        parsed[key] = raw

    steering = parsed.get("steering")
    if steering:
        from drowse.steering.grammar import parse_expr

        parse_expr(steering)

    thinking: bool | None = None
    if "thinking" in parsed:
        raw_thinking = parsed["thinking"]
        if raw_thinking not in {"true", "false"}:
            raise ValueError("custom recipe field 'thinking' must be true or false")
        thinking = raw_thinking == "true"

    sampling_values: dict[str, Any] = {}
    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        number = _recipe_modifier_float(parsed, key)
        if number is not None:
            sampling_values[key] = number
    for key in ("top_k", "max_tokens", "seed"):
        number = _recipe_modifier_int(parsed, key)
        if number is not None:
            sampling_values[key] = number

    _recipe_modifier_range(sampling_values, "temperature", 0.0, None)
    _recipe_modifier_range(sampling_values, "top_p", 0.0, 1.0)
    _recipe_modifier_range(sampling_values, "top_k", 0, None)
    _recipe_modifier_range(sampling_values, "max_tokens", 1, None)
    _recipe_modifier_range(sampling_values, "presence_penalty", -2.0, 2.0)
    _recipe_modifier_range(sampling_values, "frequency_penalty", -2.0, 2.0)

    return Recipe(
        steering=steering,
        sampling=SamplingConfig(**sampling_values) if sampling_values else None,
        thinking=thinking,
    )


def _recipe_modifier_float(values: dict[str, str], key: str) -> float | None:
    if key not in values:
        return None
    try:
        result = float(values[key])
    except ValueError as error:
        raise ValueError(f"custom recipe field {key!r} must be a number") from error
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"custom recipe field {key!r} must be finite")
    return result


def _recipe_modifier_int(values: dict[str, str], key: str) -> int | None:
    if key not in values:
        return None
    try:
        result = int(values[key])
    except ValueError as error:
        raise ValueError(f"custom recipe field {key!r} must be an integer") from error
    return result


def _recipe_modifier_range(
    values: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float | None,
) -> None:
    if key not in values:
        return
    value = values[key]
    if value < minimum or (maximum is not None and value > maximum):
        requirement = (
            f"between {minimum:g} and {maximum:g}"
            if maximum is not None
            else f"{minimum:g} or greater"
        )
        raise ValueError(f"custom recipe field {key!r} must be {requirement}")


# ---------------------------------------------------------------------------
# Cast roster member
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CastMember:
    """One member of the tree's cast roster.

    The roster maps a cast *label* (the header label a turn is rendered
    with — ``"deer"``, ``"captain"``) to the member's standing steering
    :class:`Recipe`.  At generation time the member's recipe is the
    *weakest* tier of :func:`compose_call`.  The cast manager is a steering
    surface, not a chat feature — a member without a recipe is just a
    named label.
    """

    recipe: Recipe | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.to_dict() if self.recipe is not None else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CastMember":
        # Native effective-roster responses annotate whether the member was
        # structural, observed, or explicitly configured.  Origin is derived
        # and is not persisted as configuration.
        require_fields(
            data, _CAST_MEMBER_FIELDS, "cast member",
            optional=frozenset({"origin"}),
        )
        recipe = None
        if data["recipe"] is not None:
            if not isinstance(data["recipe"], dict):
                raise LoomTreeError("cast member field 'recipe' must be an object or null")
            recipe = Recipe.from_dict(data["recipe"])
        if not isinstance(data["notes"], str):
            raise LoomTreeError("cast member field 'notes' must be a string")
        return cls(recipe=recipe, notes=data["notes"])


# ---------------------------------------------------------------------------
# Seed schedule
# ---------------------------------------------------------------------------


def _mix_seed(base: int, i: int) -> int:
    """Deterministically derive a 31-bit seed from ``(base, i)``.

    Uses blake2b with an 8-byte digest as the avalanche function;
    cross-machine deterministic and well-mixed for small ``i`` (which
    a naive FNV-1a over little-endian bytes is not — the high bytes
    of small integers are zero, so consecutive ``i`` values differ in
    only one input byte and the FNV-1a output is nearly linear).
    """
    import hashlib
    import struct

    payload = struct.pack("<qq", int(base) & 0x7FFFFFFFFFFFFFFF, int(i))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    val = struct.unpack("<Q", digest)[0]
    return val & 0x7FFFFFFF


def derive_seed_schedule(base_seed: int | None, n: int) -> list[int]:
    """Return ``n`` deterministic seeds derived from ``base_seed``.

    When ``base_seed`` is None, resolves an entropy-derived base.  Each
    sibling's recipe stamps its own derived seed, so every run stays
    individually reproducible.  ``n=1`` returns the base seed verbatim so a
    single regen with no schedule looks unchanged.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if base_seed is None:
        # ``secrets.randbits`` gives us a 31-bit seed in a torch-friendly range.
        base_seed = secrets.randbits(31)
    if n == 1:
        return [int(base_seed) & 0x7FFFFFFF]
    return [_mix_seed(int(base_seed), i) for i in range(n)]


__all__ = [
    "CALL_ONLY_SAMPLING_FIELDS",
    "CastMember",
    "ComposedCall",
    "INHERIT_SYSTEM_PROMPT",
    "InheritSystemPrompt",
    "Recipe",
    "compose_call",
    "derive_seed_schedule",
]
