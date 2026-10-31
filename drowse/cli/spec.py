"""The declarative verb table: flag specs, verb nodes, and parser construction.

Every verb module declares one :class:`Verb` group whose leaves carry their
flag specs next to their runner, so the parser and the dispatch cannot drift.
:func:`build_parser` walks the tree once; each leaf ``set_defaults(func=run)``,
each group defaults ``func`` to printing its own help (a bare group is help,
exit 0), and every parser stamps ``command`` with its verb path
(``"manifold fit"``) for error prefixes.

The shared flag specs below are the reserved short-flag table: one letter,
one meaning, tree-wide (``-k`` is the one double-booked letter — ``--api-key``
on ``serve``, ``--top-k`` on the lens readouts — and they never share a
parser).  ``pack push`` keeps ``--private`` / ``--dry-run`` long-only because
it is the one verb that publishes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Union


@dataclass(frozen=True, eq=False)
class Arg:
    """One ``add_argument`` call, held as data."""

    flags: tuple[str, ...]
    kwargs: dict[str, Any]

    def but(self, **overrides: Any) -> "Arg":
        """A copy with some ``add_argument`` keywords replaced."""
        return Arg(self.flags, {**self.kwargs, **overrides})


def arg(*flags: str, **kwargs: Any) -> Arg:
    return Arg(flags, kwargs)


@dataclass(frozen=True, eq=False)
class OneOf:
    """A mutually exclusive flag group."""

    args: tuple[Arg, ...]


Spec = Union[Arg, OneOf, tuple["Spec", ...]]


@dataclass(frozen=True, eq=False)
class Verb:
    """A leaf verb (``run`` set) or a group of verbs (``verbs`` set)."""

    name: str
    help: str
    run: Callable[[argparse.Namespace], None] | None = None
    args: tuple[Spec, ...] = ()
    description: str | None = None
    verbs: tuple["Verb", ...] = field(default=())


# ---------------------------------------------------------------------------
# Argument types
# ---------------------------------------------------------------------------

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def bounded_int(lo: int, hi: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if not lo <= parsed <= hi:
            raise argparse.ArgumentTypeError(f"must be in [{lo}, {hi}]")
        return parsed

    return parse


def smoothing(value: str) -> "str | float":
    """``auto`` (GCV-selected) or a nonnegative float λ."""
    if value == "auto":
        return value
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 'auto' or a number") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


# ---------------------------------------------------------------------------
# Shared flag specs
# ---------------------------------------------------------------------------

MODEL = arg("model", help="HuggingFace model ID or local path")
MODEL_OPT = arg("-m", "--model", default=None, metavar="MODEL_ID")
MODEL_REQ = arg("-m", "--model", required=True, metavar="MODEL_ID")

DEVICE = arg("-d", "--device", default="auto",
             help="Device: auto (detect), cuda, mps, or cpu (default: auto)")
QUANTIZE = arg("-q", "--quantize", choices=["4bit", "8bit"], default=None,
               help="Quantization mode (default: bf16/fp16)")
#: Every verb that loads a model takes these two.
MODEL_LOAD = (DEVICE, QUANTIZE)

PROBES = arg(
    "-p", "--probes", nargs="*", default=None,
    help="Probe categories: all, none, epistemic, alignment, register, "
         "cultural (default: all)",
)
JSON = arg("-j", "--json", dest="json_output", action="store_true",
           help="Emit machine-readable JSON on stdout")
FORCE = arg("-f", "--force", action="store_true")
YES = arg("-y", "--yes", action="store_true",
          help="Confirm removal (required)")
VERBOSE = arg("-v", "--verbose", action="store_true")
STEER = arg("-S", "--steer", default=None, metavar="EXPR",
            help='Steering expression, e.g. "0.5 honest + 0.3 warm"')
DESCRIPTION = arg("--description", default="", metavar="TEXT",
                  help="Human-readable description for the folder")

#: ``-c/--config`` + ``-s/--strict`` — the setup-YAML block.
CONFIG = (
    arg("-c", "--config", action="append", default=None, metavar="PATH",
        help="Load setup YAML (repeatable; later overrides earlier)"),
    arg("-s", "--strict", action="store_true",
        help="With -c: fail hard on steering references that cannot be "
             "resolved or installed"),
)

NO_DLS = arg(
    "--no-dls", dest="no_dls", action="store_true",
    help="Keep every extracted layer instead of dropping layers whose fitted "
         "poles do not straddle the neutral baseline.",
)
#: Generation-time engine options (``--no-dls`` plus compilation state).
INJECTION = (
    NO_DLS,
    arg("--compile", action="store_true",
        help="Enable torch.compile on CUDA or MPS (off by default: upfront "
             "cost and architecture-specific compiler failures). Unsteered "
             "and static-affine generations use the persistent compiled "
             "path; dynamic steering falls back to eager. YAML: compile."),
    arg("--cuda-graphs", dest="cuda_graphs", action="store_true",
        help="Add transformers.StaticCache + CUDA-graph capture on top of "
             "--compile (CUDA only; auto-skipped elsewhere and on "
             "architectures whose StaticCache fails). YAML: cuda_graphs."),
)
TOP_K_ALTS = arg(
    "--top-k-alts", dest="top_k_alts", type=nonnegative_int, default=None,
    metavar="N",
    help="Session default for top-K alternatives per generated token "
         "(0 = chosen-token logprob only). YAML: return_top_k.",
)


def layers_arg(help: str, metavar: str = "L1,L2,...|workspace|all") -> Arg:
    return arg("--layers", default=None, metavar=metavar, help=help)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _flatten(specs: tuple[Spec, ...]) -> Iterator[Arg | OneOf]:
    for spec in specs:
        if isinstance(spec, tuple):
            yield from _flatten(spec)
        else:
            yield spec


def _print_help(parser: argparse.ArgumentParser) -> Callable[[argparse.Namespace], None]:
    def run(_args: argparse.Namespace) -> None:
        parser.print_help()

    return run


def attach(parser: argparse.ArgumentParser, verb: Verb, path: tuple[str, ...]) -> None:
    """Add ``verb``'s flags (and its subverbs, recursively) to ``parser``."""
    for spec in _flatten(verb.args):
        if isinstance(spec, OneOf):
            group = parser.add_mutually_exclusive_group()
            for one in spec.args:
                group.add_argument(*one.flags, **one.kwargs)
        else:
            parser.add_argument(*spec.flags, **spec.kwargs)
    command = " ".join(path)
    if verb.verbs:
        sub = parser.add_subparsers(title="verbs", metavar="VERB")
        for child in verb.verbs:
            child_path = (*path, child.name)
            child_parser = sub.add_parser(
                child.name, help=child.help,
                description=child.description or child.help,
                epilog=(
                    f"Run `drowse {' '.join(child_path)} <verb> -h` for verb-specific options."
                    if child.verbs else None
                ),
            )
            attach(child_parser, child, child_path)
        parser.set_defaults(func=_print_help(parser), command=command)
    else:
        if verb.run is None:
            raise ValueError(f"leaf verb {command!r} has no runner")
        parser.set_defaults(func=verb.run, command=command)
