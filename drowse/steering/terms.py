"""Per-call steering configuration: the :class:`Steering` IR.

A frozen ``alphas: {key: entry}`` mapping plus an optional ``thinking``
override and a default ``trigger`` for bare-float entries.  Callers hand
:meth:`Steering.from_value` an expression string (parsed by
:mod:`drowse.steering.grammar`) or a pre-built :class:`Steering`; dicts are
not accepted.  Keys are registry keys — the grammar's canonical form — and
resolution to loaded artifacts happens once, in
:meth:`~drowse.steering.composer.SteeringComposer.resolve`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING, Union

from drowse.steering.triggers import Trigger

if TYPE_CHECKING:
    from drowse.steering.grammar import (
        AblationTerm,
        ManifoldTerm,
        ProjectedTerm,
    )

#: Accepted shapes for a single entry in ``Steering.alphas`` — a bare
#: alpha (inherits ``Steering.trigger``), a ``(alpha, Trigger)`` tuple, a
#: :class:`~drowse.steering.grammar.ProjectedTerm` (``~``/``|`` projection),
#: an :class:`~drowse.steering.grammar.AblationTerm` (``!``), or a
#: :class:`~drowse.steering.grammar.ManifoldTerm` (``%``).
AlphaEntry = Union[
    float,
    "tuple[float, Trigger]",
    "ProjectedTerm",
    "AblationTerm",
    "ManifoldTerm",
]

#: One entry of a pushed steering scope after resolution: vectors as
#: ``(alpha, Trigger)``; projection, ablation and manifold terms whole, with
#: their operands concretized to registry keys.  :func:`entry_trigger` /
#: :func:`entry_coeff` read every shape.
SteeringStackEntry = Union[
    "tuple[float, Trigger]",
    "ProjectedTerm",
    "AblationTerm",
    "ManifoldTerm",
]


def term_entry(
    val: AlphaEntry, default_trigger: Trigger = Trigger.BOTH,
) -> tuple[float, Trigger]:
    """``(coeff, trigger)`` of any ``Steering.alphas`` value or stack entry.

    A bare float takes ``default_trigger``; a manifold term reports its
    ``along`` coefficient.
    """
    if isinstance(val, tuple):
        return float(val[0]), val[1]
    if isinstance(val, (int, float)):
        return float(val), default_trigger
    return float(val.coeff), val.trigger


def entry_trigger(entry: SteeringStackEntry) -> Trigger:
    """The trigger a stack entry fires under, whatever shape it is."""
    return term_entry(entry)[1]


def entry_coeff(entry: SteeringStackEntry) -> float:
    """The scalar strength a stack entry carries (a manifold term's ``along``)."""
    return term_entry(entry)[0]


@dataclass(frozen=True)
class Steering:
    """Per-call steering configuration.

    alphas: registry key → :data:`AlphaEntry`.
    thinking: per-call thinking override; ``None`` falls through to the
        caller's ``thinking=`` kwarg / the session default.
    trigger: default trigger for bare-float entries (``Trigger.BOTH``).

    ``~`` / ``|`` projection terms always materialize through the
    closed-form LEACE projector against the session's
    :class:`~drowse.geometry.whitener.LayerWhitener` — there is no
    per-call metric override.
    """

    alphas: Mapping[str, AlphaEntry]
    thinking: bool | None = None
    trigger: Trigger = Trigger.BOTH

    @classmethod
    def from_value(
        cls, value: object, *, profile_names: set[str] | None = None,
    ) -> "Steering | None":
        """Coerce a string / Steering / None into a Steering or None.

        Strings parse through :mod:`drowse.steering.grammar`; ``None`` and
        :class:`Steering` instances pass through; anything else raises
        ``TypeError``.
        """
        if value is None:
            return None
        if isinstance(value, Steering):
            return value
        if isinstance(value, str):
            from drowse.steering.grammar import parse_expr
            return parse_expr(value, profile_names=profile_names)
        raise TypeError(
            f"Steering.from_value expects str | Steering | None, "
            f"got {type(value).__name__}"
        )

    def entries(self) -> dict[str, tuple[float, Trigger]]:
        """``{key: (coeff, trigger)}`` for every term, whatever its kind."""
        return {
            name: term_entry(val, self.trigger)
            for name, val in self.alphas.items()
        }

    def triggers(self) -> set[Trigger]:
        """Every trigger this steering's terms fire under."""
        return {trig for _coeff, trig in self.entries().values()}

    def __str__(self) -> str:
        from drowse.steering.grammar import format_expr
        return format_expr(self)
