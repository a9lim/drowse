"""The declarative verb table: eight groups, dispatch, help, short flags.

Parser *behavior* only — what a runner does with its namespace is tested in
``test_cli_flags`` / ``test_cli_manifold``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
from collections import defaultdict
from typing import Any

import pytest

from drowse.cli.main import build_parser, main, parse_args


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _all_paths() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []

    def walk(parser: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        out.append(path)
        for name, child in _subparsers(parser).items():
            walk(child, (*path, name))

    walk(build_parser(), ())
    return out


def test_exactly_eight_top_level_verbs() -> None:
    assert set(_subparsers(build_parser())) == {
        "serve", "manifold", "pack", "config", "experiment", "template", "lens", "sae",
    }


@pytest.mark.parametrize("path", _all_paths(), ids=lambda p: " ".join(p) or "root")
def test_every_parser_renders_help_and_dispatches(path: tuple[str, ...]) -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        parse_args([*path, "-h"])
    assert exc.value.code == 0
    assert "usage: drowse" in buf.getvalue()


def test_short_flags_keep_one_meaning_tree_wide() -> None:
    """The reserved short-flag table: one letter, one long form — except
    ``-k`` (``--api-key`` on serve, ``--top-k`` on the lens readouts)."""
    meanings: dict[str, set[str]] = defaultdict(set)

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            longs = [o for o in action.option_strings if o.startswith("--")]
            for short in (o for o in action.option_strings if len(o) == 2 and o != "-h"):
                meanings[short].add(longs[0])
        for child in _subparsers(parser).values():
            walk(child)

    walk(build_parser())
    shared = {flag: forms for flag, forms in meanings.items() if len(forms) > 1}
    assert shared == {"-k": {"--api-key", "--top-k"}}
    assert meanings["-m"] == {"--model"} and meanings["-j"] == {"--json"}


@pytest.mark.parametrize("argv", [
    [], ["manifold"], ["pack"], ["pack", "export"], ["template"], ["lens"],
    ["sae"], ["experiment"], ["experiment", "transcript"], ["config"],
])
def test_bare_group_prints_its_help_and_exits_zero(
    argv: list[str], capsys: pytest.CaptureFixture[str],
) -> None:
    main(argv)  # returns normally: exit 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: drowse{' ' if argv else ''}{' '.join(argv)}")
    for verb in _subparsers(_parser_at(argv)):
        assert verb in out


def _parser_at(argv: list[str]) -> argparse.ArgumentParser:
    parser = build_parser()
    for name in argv:
        parser = _subparsers(parser)[name]
    return parser


@pytest.mark.parametrize("argv", [
    ["google/gemma-2-2b-it"],              # no bare-model fallback
    ["tui", "m"], ["subspace", "extract", "x"], ["vector", "extract", "x"],
    ["manifold", "discover", "x"],
    ["serve", "m", "--port", "70000"],
    ["serve", "m", "--top-k-alts", "-1"],
    ["manifold", "fit", "m", "-m", "x", "--max-dim", "0"],
    ["manifold", "fit", "m", "-m", "x", "--k-nn", "-2"],
    ["manifold", "fit", "m", "-m", "x", "--smoothing", "warm"],
    ["manifold", "compare", "a.b"],        # -m is required
    ["manifold", "extract", "happy", "--sae", "r", "--role", "pirate", "-m", "x"],
    ["manifold", "extract", "happy", "-m", "x", "--namespace", "alice"],
    ["manifold", "generate", "x", "--concepts", "a", "b", "-m", "x", "--seed", "1"],
    ["manifold", "compare", "a", "-m", "x", "--ridge-scale", "2"],
    ["lens", "fit", "m", "--corpus", "f", "--hf-corpus", "org/data"],
    ["lens", "fit", "m", "--standard"], ["lens", "fit", "m", "--relp"],
    ["lens", "fit", "m", "--kind", "s"],
    ["experiment", "fan", "m", "p", "-g", "a=0", "--base-steering", "x"],
    ["sae", "fetch", "m", "saelens:r", "-d", "cpu"],
    ["sae", "fetch", "m", "saelens:r", "--layer", "-1"],
    ["sae", "rm", "m", "local:x", "-j"],
    ["pack", "push", "x", "-p"],
])
def test_rejected_argv(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(("argv", "expected"), [
    (["serve", "m"], {"command": "serve", "host": "127.0.0.1", "port": 8000,
                      "compile": False, "cuda_graphs": False, "steer": None}),
    (["serve", "m", "--top-k-alts", "262144"], {"top_k_alts": 262144}),
    (["manifold", "extract", "happy", "sad", "-m", "x", "-f"],
     {"command": "manifold extract", "concept": ["happy", "sad"], "force": True,
      "kind": "abstract", "device": "auto", "quantize": None}),
    (["manifold", "fit", "x", "-m", "m", "--smoothing", "0.25", "--method", "auto"],
     {"smoothing": 0.25, "method": "auto"}),
    (["manifold", "fit", "x", "-m", "m", "--smoothing", "auto"], {"smoothing": "auto"}),
    (["manifold", "from-template", "weekday"],
     {"fit_mode": "auto", "name": None, "max_dim": None}),
    (["manifold", "transfer", "c", "--from", "a", "--to", "b", "-j"],
     {"src_model": "a", "tgt_model": "b", "json_output": True}),
    (["pack", "export", "gguf", "x", "-m", "m"], {"command": "pack export gguf"}),
    (["pack", "push", "x"], {"variant": "raw", "private": False, "dry_run": False}),
    (["pack", "clear", "x"], {"variant": "all"}),
    (["lens", "fit", "m"], {"kind": "r", "prompts": 100}),
    (["lens", "fit", "m", "--kind", "j"], {"kind": "j"}),
    (["experiment", "fan", "m", "p", "-g", "a=0,1", "-S", "0.5 warm"],
     {"steer": "0.5 warm", "max_tokens": None}),
    (["experiment", "transcript", "run", "t.yaml"],
     {"command": "experiment transcript run", "model": None, "strict": False}),
    (["sae", "fetch", "m", "saelens:r", "--layer", "14", "--revision", "abc"],
     {"layer": 14, "revision": "abc"}),
])
def test_parsed_namespace(argv: list[str], expected: dict[str, Any]) -> None:
    args = parse_args(argv)
    for key, value in expected.items():
        assert getattr(args, key) == value, key


def test_help_does_not_import_torch() -> None:
    """Building the parser imports only the verb tables; runners import the
    engine lazily, so ``drowse --help`` never pays for torch."""
    code = (
        "import sys\n"
        "from drowse.cli.main import build_parser\n"
        "build_parser().format_help()\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"
