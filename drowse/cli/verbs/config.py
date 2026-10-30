"""``drowse config <verb>`` — inspect and validate setup YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from drowse.cli.context import CliError
from drowse.cli.spec import Verb, arg


def run_show(args: argparse.Namespace) -> None:
    from drowse import __version__
    from drowse.cli.config_file import ConfigFile, apply_flag_overrides

    extras = [Path(p) for p in (args.config or [])]
    composed = ConfigFile.effective(extras, include_default=not args.no_default)
    composed = apply_flag_overrides(composed, model=args.model)
    sys.stdout.write(composed.to_yaml(
        header=f"# effective merged config for drowse {__version__}",
    ))


def run_validate(args: argparse.Namespace) -> None:
    """Parse the file and dry-run its ``vectors:`` resolution — the same check
    ``-c`` applies at startup, minus any install."""
    from drowse.cli.config_file import ConfigFile, ensure_vectors_installed

    path = Path(args.file)
    if not path.exists():
        raise CliError(f"{path}: file not found")
    cfg = ConfigFile.load(path)
    ensure_vectors_installed(cfg.vectors, strict=True, dry_run=True)
    print(f"{path}: ok")


GROUP = Verb(
    "config", "Inspect and validate drowse config files",
    description="Inspect / validate setup YAML.",
    verbs=(
        Verb("show", "Print the effective merged config", run=run_show, args=(
            arg("-c", "--config", action="append", default=None, metavar="PATH",
                help="Extra YAML files to compose on top of ~/.drowse/config.yaml"),
            arg("-m", "--model", default=None, help="Override the model field"),
            arg("--no-default", action="store_true",
                help="Skip loading ~/.drowse/config.yaml"),
        )),
        Verb("validate", "Validate a config file (CI hook; exit 2 when invalid)",
             run=run_validate, args=(arg("file", help="Path to a YAML config file"),)),
    ),
)
