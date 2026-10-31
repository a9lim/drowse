"""The ``drowse`` command line: eight verb groups, one error boundary.

    drowse serve <model> [...]
    drowse manifold {extract,generate,from-template,fit,bake,merge,transfer,compare,why} ...
    drowse pack {ls,show,install,search,push,rm,clear,refresh,export} ...
    drowse experiment {fan,transcript,naturalness} ...
    drowse config {show,validate} ...
    drowse template {create,ls,show,score,rm} ...
    drowse lens {fit,fetch,ls,show,use,top,decompose,rm} ...
    drowse sae {train,fetch,ls,show,use,rm} ...

Each group lives in ``drowse.cli.verbs.<group>`` as a declarative
:class:`~drowse.cli.spec.Verb` tree.  Building the parser imports only those
modules — runners import the engine lazily — so ``drowse --help`` never loads
torch.  ``drowse`` alone, or a bare group, prints that level's help (exit 0).
"""

from __future__ import annotations

import argparse
import os
import sys

from drowse.cli.context import CliError, exit_code_for
from drowse.cli.spec import Verb, attach
from drowse.errors import DrowseError

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")


def _groups() -> tuple[Verb, ...]:
    from drowse.cli.verbs import (
        config, experiment, lens, manifold, pack, sae, serve, template,
    )

    return (
        serve.GROUP, manifold.GROUP, pack.GROUP, config.GROUP,
        experiment.GROUP, template.GROUP, lens.GROUP, sae.GROUP,
    )


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="drowse",
        description="Activation steering + trait monitoring for local HuggingFace models",
        epilog="Run `drowse <verb> -h` for verb-specific options.",
    )
    attach(root, Verb("drowse", "", verbs=_groups()), ())
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(sys.argv[1:] if argv is None else argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prefix = f"drowse {args.command}".rstrip()
    try:
        args.func(args)
    except CliError as e:
        print(f"{prefix}: {e}", file=sys.stderr)
        sys.exit(e.exit_code)
    except DrowseError as e:
        status, message = e.user_message()
        print(f"{prefix}: {message}", file=sys.stderr)
        sys.exit(exit_code_for(status))
    except (FileNotFoundError, FileExistsError) as e:
        print(f"{prefix}: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"{prefix}: interrupted", file=sys.stderr)
        sys.exit(130)
