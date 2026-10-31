"""CLI runner behavior: config precedence, sessions, serve policies, ``-j``
purity, the error boundary, and the model-backed manifold / lens / SAE /
experiment runners (with a model-free ``FakeSession``)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from drowse.cli.context import CliError, effective_config, open_session
from drowse.cli.main import main, parse_args
from drowse.readout.types import LensLiveState, SaeLiveState
from drowse.store.selectors import SelectorError
from tests.support.cli import FakeSession, install_fake_session, poison_session


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from drowse.store import selectors

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    selectors.invalidate()


def _yaml(tmp_path: Path, body: str) -> str:
    path = tmp_path / "setup.yaml"
    path.write_text(body)
    return str(path)


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> Any:
    """``-j`` output must be exactly one JSON document — narration is stderr."""
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# Error boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("exc", "code"), [
    (CliError("bad input"), 2),
    (CliError("missing", exit_code=1), 1),
    (SelectorError("bad"), 2),
    (FileNotFoundError("gone"), 1),
])
def test_main_maps_errors_to_one_line_and_exit_code(
    exc: Exception, code: int, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import drowse.cli.main as cli_main

    def boom(_args: argparse.Namespace) -> None:
        raise exc

    args = parse_args(["pack", "ls"])
    args.func = boom
    monkeypatch.setattr(cli_main, "parse_args", lambda _argv=None: args)
    with pytest.raises(SystemExit) as exit_info:
        main(["pack", "ls"])
    assert exit_info.value.code == code
    err = capsys.readouterr().err
    assert err.startswith("drowse pack ls: ") and err.count("\n") == 1


def test_drowse_error_status_maps_404_to_one_and_4xx_to_two() -> None:
    from drowse.cli.context import exit_code_for

    assert [exit_code_for(s) for s in (400, 404, 409, 422, 500, 502)] == [2, 1, 2, 2, 1, 1]


# ---------------------------------------------------------------------------
# Effective config: CLI > YAML > runner default
# ---------------------------------------------------------------------------

def test_yaml_compile_and_cuda_graphs_fold_on_and_cli_wins(tmp_path: Path) -> None:
    on = _yaml(tmp_path, "model: m/x\ncompile: true\ncuda_graphs: true\n")
    args = parse_args(["serve", "-c", on])
    assert (args.compile, args.cuda_graphs) == (False, False)
    effective_config(args)
    assert (args.model, args.compile, args.cuda_graphs) == ("m/x", True, True)

    off = _yaml(tmp_path, "compile: false\n")
    args = parse_args(["serve", "m/x", "-c", off, "--compile"])
    effective_config(args)
    assert args.compile is True


_MAX_TOKEN_VERBS = [
    (["experiment", "fan", "m/x", "prompt", "-g", "a=0,1"], 256),
    (["experiment", "transcript", "run", "t.yaml", "m/x"], 256),
    (["experiment", "naturalness", "m/x", "p", "--manifold", "f", "-S", "0.5 a"], 128),
]


@pytest.mark.parametrize(("argv", "floor"), _MAX_TOKEN_VERBS, ids=["fan", "transcript", "naturalness"])
@pytest.mark.parametrize(("flag", "yaml_value", "expected"), [
    (["--max-tokens", "64"], "999", 64),   # CLI wins
    ([], "999", 999),                      # YAML fills an unset flag
    ([], None, None),                      # the runner's own floor
])
def test_max_tokens_precedence(
    argv: list[str], floor: int, flag: list[str], yaml_value: str | None,
    expected: int | None, tmp_path: Path,
) -> None:
    config = ["-c", _yaml(tmp_path, f"max_tokens: {yaml_value}\n")] if yaml_value else []
    args = parse_args([*argv, *config, *flag])
    effective_config(args, default_max_tokens=floor)
    assert args.max_tokens == (floor if expected is None else expected)


def test_serve_max_tokens_floor_is_1024() -> None:
    args = parse_args(["serve", "m/x"])
    effective_config(args)
    assert args.max_tokens == 1024


def test_yaml_vectors_fill_an_unset_steer_and_cli_steer_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        "drowse.cli.config_file.ensure_vectors_installed",
        lambda expr, **_kw: seen.append(expr) or [],
    )
    cfg = _yaml(tmp_path, 'vectors: "0.4 default/happy.sad"\n')
    args = parse_args(["serve", "m/x", "-c", cfg])
    effective_config(args, yaml_vectors=True)
    assert args.steer == "0.4 default/happy.sad"
    args = parse_args(["serve", "m/x", "-c", cfg, "-S", "0.2 bob/honest"])
    effective_config(args, yaml_vectors=True)
    assert args.steer == "0.2 bob/honest"
    # Installation runs over the *effective* expression, -S included.
    assert seen == ["0.4 default/happy.sad", "0.2 bob/honest"]


def test_fan_never_inherits_yaml_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML ``vectors:`` is serve's default steering; a global config must not
    silently become the base steering under every row of a research sweep."""
    monkeypatch.setattr(
        "drowse.cli.config_file.ensure_vectors_installed", lambda *_a, **_k: [],
    )
    cfg = _yaml(tmp_path, 'vectors: "0.4 default/happy.sad"\n')
    created = install_fake_session(monkeypatch, _FanSession)
    main(["experiment", "fan", "m/x", "p", "-g", "a=0,1", "-c", cfg])
    assert created[0].base_steering is None
    main(["experiment", "fan", "m/x", "p", "-g", "a=0,1", "-c", cfg, "-S", "0.2 x"])
    assert created[1].base_steering == "0.2 x"


def test_yaml_sampling_defaults_reach_the_session_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``temperature`` / ``top_p`` / ``thinking`` were validated and
    shown by ``config show`` but never applied to the session."""
    created = install_fake_session(monkeypatch)
    cfg = _yaml(tmp_path, "temperature: 0.3\ntop_p: 0.5\nthinking: false\nsystem_prompt: be terse\n")
    args = parse_args(["serve", "m/x", "-c", cfg])
    effective_config(args)
    with open_session(args, roster=True) as session:
        assert (session.config.temperature, session.config.top_p, session.config.thinking) == (0.3, 0.5, False)
    assert created[0].load_kwargs["system_prompt"] == "be terse"
    assert created[0].closed


@pytest.mark.parametrize(("argv", "probes"), [
    (["manifold", "fit", "x", "-m", "m/x"], []),
    (["template", "score", "t", "-m", "m/x"], []),
    (["lens", "top", "m/x", "p"], []),
    (["serve", "m/x"], None),
    (["serve", "m/x", "-p", "none"], []),
    (["serve", "m/x", "-p", "epistemic"], ["epistemic"]),
])
def test_only_reading_verbs_bootstrap_the_probe_roster(
    argv: list[str], probes: list[str] | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_session(monkeypatch)
    args = parse_args(argv)
    with open_session(args, roster=args.command == "serve"):
        pass
    assert created[0].load_kwargs["probes"] == probes


@pytest.mark.parametrize(("flag", "dls"), [([], True), (["--no-dls"], False)])
def test_no_dls_reaches_session_construction(
    flag: list[str], dls: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_session(monkeypatch)
    with open_session(parse_args(["manifold", "fit", "x", "-m", "m/x", *flag])):
        pass
    assert created[0].load_kwargs["dls"] is dls
    assert callable(created[0].load_kwargs["on_progress"])


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_show_composes_extras(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["config", "show", "--no-default", "-c", _yaml(tmp_path, "model: g/m\ntemperature: 0.7\n")])
    out = capsys.readouterr().out
    assert out.startswith("# effective merged config for drowse")
    assert "g/m" in out and "temperature: 0.7" in out


@pytest.mark.parametrize(("body", "code"), [
    ("model: g/m\n", 0),
    ('vectors: "0.3 jlens/orange + 0.2 sae/12"\n', 0),   # readout atoms name no manifold
    ('vectors: "0.4 alice/honest"\n', 0),                # HF refs are installable
    ('vectors: "0.4 local/nope"\n', 2),
    ('vectors: "0.4 nowhere_to_be_found"\n', 2),
    ("vectors:\n  local/nope: 0.5\n", 2),
])
def test_config_validate(body: str, code: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _yaml(tmp_path, body)
    if code == 0:
        main(["config", "validate", path])
        assert capsys.readouterr().out.strip().endswith(": ok")
    else:
        with pytest.raises(SystemExit) as exc:
            main(["config", "validate", path])
        assert exc.value.code == code


def test_config_validate_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["config", "validate", str(tmp_path / "nope.yaml")])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# serve: steering preflight, live readout policies, startup order
# ---------------------------------------------------------------------------

class _SteeringSession:
    """Records ``session.steering`` scopes; raises for configured misses."""

    def __init__(self, missing: set[str] | None = None) -> None:
        self.entered: list[Any] = []
        self.missing = missing or set()

    def steering(self, steering: Any) -> Any:
        from contextlib import contextmanager

        from drowse.errors import ProfileNotRegisteredError

        @contextmanager
        def scope() -> Any:
            for key in steering.alphas:
                if key in self.missing:
                    raise ProfileNotRegisteredError(f"profile {key!r} is not registered")
            self.entered.append(steering)
            yield

        return scope()

    def extract(self, *_a: Any, **_k: Any) -> Any:
        raise AssertionError("serve must never author a manifold")


def test_serve_preflight_composes_without_authoring() -> None:
    from drowse.cli.verbs.serve import preflight_steering

    session = _SteeringSession()
    steering = preflight_steering(session, "0.3 jlens/orange + 0.2 sae/12", "m/x")
    assert session.entered == [steering]
    assert {"jlens/orange", "sae/12"} <= set(steering.alphas)


def test_serve_preflight_miss_fails_with_an_extract_hint() -> None:
    from drowse.cli.verbs.serve import preflight_steering

    with pytest.raises(CliError) as exc:
        preflight_steering(_SteeringSession(missing={"agentic"}), "0.5 agentic", "m/x")
    assert exc.value.exit_code == 1
    assert "drowse manifold extract" in str(exc.value) and "-m m/x" in str(exc.value)


def _serve_harness(
    monkeypatch: pytest.MonkeyPatch, session: Any,
) -> dict[str, Any]:
    import uvicorn

    import drowse.cli.verbs.serve as serve
    import drowse.server as server

    captured: dict[str, Any] = {"warmed": False}
    install_fake_session(monkeypatch, lambda _m, **_k: session)
    monkeypatch.setattr(server, "create_app", lambda *_a, **kw: captured.update(app=kw) or "app")
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **kw: captured.update(run=kw))
    monkeypatch.setattr(serve, "enable_live_lens", lambda _s: captured.update(lens=True))
    monkeypatch.setattr(serve, "enable_live_sae", lambda _s: captured.update(sae=True))
    monkeypatch.setattr(serve, "warm_up", lambda _s: captured.update(warmed=True))
    return captured


class _ServeSession(FakeSession, _SteeringSession):
    def __init__(self) -> None:
        FakeSession.__init__(self)
        _SteeringSession.__init__(self)


@pytest.mark.parametrize("compile_flag", [[], ["--compile"]])
def test_serve_startup_uses_cli_steer_over_yaml_and_warms_only_compiled(
    compile_flag: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _ServeSession()
    captured = _serve_harness(monkeypatch, session)
    cfg = _yaml(tmp_path, 'vectors: "0.4 jlens/cat"\n')
    main(["serve", "m/x", "-c", cfg, "-S", "0.3 jlens/dog", "--no-web", *compile_flag])
    steering = captured["app"]["default_steering"]
    assert list(steering.alphas) == ["jlens/dog"] and session.entered == [steering]
    assert captured["warmed"] is bool(compile_flag)
    assert "sae" not in captured  # --no-web: no implicit provider download
    assert session.closed


def test_serve_rejects_a_bad_steer_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["serve", "m/x", "-S", "0.5 +"])
    assert exc.value.code == 2 and attempts == []


def _lens_session(compatible: set[str]) -> Any:
    class _Lens:
        live = False

        def __init__(self) -> None:
            self.selected: list[str] = []

        def set_live(self, enabled: bool, **_kw: Any) -> Any:
            self.live = enabled
            return LensLiveState(enabled=enabled, layers=(4, 5))

        def has_compatible(self) -> bool:
            from drowse.lens.sources import LENS_SOURCES

            active = LENS_SOURCES.active("org/model", "output")
            return active is not None and LENS_SOURCES.format(active) in compatible

        def select_source(self, source: str) -> None:
            from drowse.lens.sources import use_lens_source

            self.selected.append(source)
            use_lens_source("org/model", source)

    class _Session:
        model_id = "org/model"

        def __init__(self) -> None:
            self.lens = _Lens()
            self.selected = self.lens.selected

    return _Session()


def _prepare_local_lenses(model_id: str, *names: str) -> None:
    from drowse.lens.sources import LENS_SOURCES

    for name in names:
        folder = LENS_SOURCES.local_dir(model_id, name)
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text("{}")


def _active_lens(model_id: str) -> str | None:
    from drowse.lens.sources import LENS_SOURCES

    ref = LENS_SOURCES.active(model_id, "output")
    return None if ref is None else LENS_SOURCES.format(ref)


def test_serve_keeps_a_compatible_active_lens_selection() -> None:
    """Regression: serve re-pointed ``active.json`` to its preferred source,
    overwriting the user's ``lens use`` choice."""
    from drowse.cli.verbs.serve import enable_live_lens
    from drowse.lens.sources import LENS_SOURCES, use_lens_source

    _prepare_local_lenses("org/model", "r", "j")
    use_lens_source("org/model", "local:j")
    before = LENS_SOURCES.active_path("org/model").read_bytes()
    session = _lens_session({"local:j", "local:r"})
    assert enable_live_lens(session)
    assert session.selected == [] and session.lens.live
    assert LENS_SOURCES.active_path("org/model").read_bytes() == before


def test_serve_leaves_an_incompatible_active_lens_alone(capsys: pytest.CaptureFixture[str]) -> None:
    from drowse.cli.verbs.serve import enable_live_lens
    from drowse.lens.sources import LENS_SOURCES, use_lens_source

    _prepare_local_lenses("org/model", "r", "j")
    use_lens_source("org/model", "local:j")
    before = LENS_SOURCES.active_path("org/model").read_bytes()
    session = _lens_session({"local:r"})
    assert not enable_live_lens(session)
    assert session.selected == [] and not session.lens.live
    assert LENS_SOURCES.active_path("org/model").read_bytes() == before
    assert "lens use org/model" in capsys.readouterr().err


def test_serve_adopts_the_best_compatible_lens_when_none_is_selected() -> None:
    from drowse.cli.verbs.serve import enable_live_lens

    _prepare_local_lenses("org/model", "r", "j")
    session = _lens_session({"local:j"})
    assert enable_live_lens(session)
    assert session.selected == ["local:r", "local:j"]
    assert _active_lens("org/model") == "local:j"


def test_serve_clears_its_trial_selection_when_no_lens_is_compatible() -> None:
    from drowse.cli.verbs.serve import enable_live_lens

    _prepare_local_lenses("org/model", "r")
    assert not enable_live_lens(_lens_session(set()))
    assert _active_lens("org/model") is None


def test_best_sae_release_prefers_official_canonical_provider() -> None:
    from drowse.cli.verbs.serve import best_sae_release

    rows = [
        {"release": "third-party-canonical", "source": "saelens", "repo_id": "someone/sae", "neuronpedia": True},
        {"release": "gemma-scope-res-all", "source": "saelens", "repo_id": "google/gemma-scope", "neuronpedia": True},
        {"release": "gemma-scope-res-canonical", "source": "saelens", "repo_id": "google/gemma-scope", "neuronpedia": True},
    ]
    assert best_sae_release(rows) == "gemma-scope-res-canonical"


_PINNED_PROVIDER = {
    "source": "saelens:prov", "kind": "saelens", "active": True,
    "layer": 9, "revision": "c" * 40,
}


@pytest.mark.parametrize(("cached", "expected", "load_kwargs"), [
    ([], "gemma-scope-2-4b-it-res", {}),
    ([{"source": "local:cached", "kind": "local", "active": False}], "local:cached",
     {"layer": None, "revision": None}),
    # The active provider binding wins, reloaded at its pinned commit/layer.
    ([{"source": "local:cached", "kind": "local", "active": False}, _PINNED_PROVIDER],
     "prov", {"layer": 9, "revision": "c" * 40}),
])
def test_serve_live_sae_prefers_prepared_sources(
    cached: list[dict[str, Any]], expected: str, load_kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.cli.verbs.serve import enable_live_sae

    class _Sae:
        info = None
        loaded: str | None = None
        load_kwargs: dict[str, Any] | None = None

        def set_live(self, enabled: bool, **_k: Any) -> SaeLiveState:
            return SaeLiveState(enabled=enabled, layer=22)

        def load(self, release: str, **kwargs: Any) -> dict[str, Any]:
            self.loaded = release
            self.load_kwargs = kwargs
            return {"release": release}

    class _Session:
        model_id = "google/gemma-3-4b-it"
        sae = _Sae()

    monkeypatch.setattr("drowse.sae.sources.list_sae_sources", lambda _m: cached)
    monkeypatch.setattr(
        "drowse.sae.backend.list_sae_releases",
        lambda _m: [{"release": "gemma-scope-2-4b-it-res", "source": "saelens",
                     "repo_id": "google/gemma-scope-2-4b-it", "neuronpedia": True}],
    )
    session = _Session()
    assert enable_live_sae(session)
    assert session.sae.loaded == expected and session.sae.load_kwargs == load_kwargs


# ---------------------------------------------------------------------------
# -j purity: stdout is exactly one JSON document, narration goes to stderr
# ---------------------------------------------------------------------------

class _ReadoutLens:
    def resident(self) -> Any:
        return SimpleNamespace(source_layers=[3, 4])

    def readout(self, _prompt: str, **_kw: Any) -> Any:
        return {3: [[("a", 0.5)]], 4: [[("b", 0.25)]]}, [[("a", 0.4, 3.0, 0.5)]]

    def decompose(self, _selector: str, **_kw: Any) -> Any:
        return {3: (0.1, [("a", 0.2)]), 4: (0.3, [("b", 0.4)])}


class _ReadoutSession(FakeSession):
    layers = [object()] * 8
    lens = _ReadoutLens()
    sae = SimpleNamespace(
        train=lambda name, _docs, **kw: {
            "source": f"local:{name}", "artifact": "x", "layer": kw["layer"],
            "metrics": {"d_sae": 64, "tokens_trained": 10},
        },
    )


@pytest.mark.parametrize("argv", [
    ["lens", "top", "m/x", "prompt", "-j"],
    ["lens", "decompose", "happy.sad", "-m", "m/x", "-j"],
])
def test_lens_readouts_emit_pure_json(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    install_fake_session(monkeypatch, _ReadoutSession)
    main(argv)
    assert _stdout_json(capsys)


def test_sae_train_emits_pure_json_and_defaults_to_the_workspace_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    install_fake_session(monkeypatch, _ReadoutSession)
    corpus = tmp_path / "docs.txt"
    corpus.write_text("one\n{\"text\": \"two\"}\n")
    main(["sae", "train", "m/x", "mine", "--corpus", str(corpus), "-j"])
    payload = _stdout_json(capsys)
    assert payload["layer"] == 5  # nearest 65% depth of 8 layers


def _patch_sae_backend(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any] | None = None) -> None:
    class _Backend:
        revision = "deadbeef"
        fingerprint = "f" * 64
        layers = frozenset({12, 14, 16})
        sae_ids_by_layer = {"14": "layer_14"}
        repo_id = "provider/saes"
        neuronpedia_ids_by_layer = {"14": "np-14"}

        def feature_count(self, _idx: int) -> int:
            return 16384

        def feature_direction(self, _idx: int, _fid: int) -> Any:
            import torch
            return torch.zeros(2304)

    def load(release: str, **kwargs: Any) -> Any:
        if seen is not None:
            seen.update({"release": release, **kwargs})
        return _Backend()

    monkeypatch.setattr("drowse.model.arch.config_model_shape", lambda _m: (26, 2304))
    monkeypatch.setattr("drowse.sae.backend.load_sae_backend", load)


def test_sae_fetch_pins_the_binding_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from drowse.sae.sources import load_active_sae, load_sae_metadata

    seen: dict[str, Any] = {}
    _patch_sae_backend(monkeypatch, seen)
    poison_session(monkeypatch)
    main(["sae", "fetch", "m/x", "saelens:rel", "--layer", "14", "--revision", "abc", "-j"])
    assert _stdout_json(capsys)["layer"] == 14
    assert (seen["revision"], seen["model_id"]) == ("abc", "m/x")
    binding = load_sae_metadata("m/x", "rel")
    assert binding is not None and binding["neuronpedia_id"] == "np-14"
    active = load_active_sae("m/x")
    assert active is not None and active[0] == "rel"


def test_sae_fetch_rejects_an_uncovered_layer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_sae_backend(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["sae", "fetch", "m/x", "saelens:rel", "--layer", "3"])
    assert exc.value.code == 2
    assert "does not cover layer 3" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["pack", "ls", "-j"],
    ["template", "ls", "-j"],
    ["template", "show", "weekday", "-j"],
    ["template", "score", "weekday", "-m", "m/x", "-j"],
    ["lens", "ls", "m/x", "-j"],
    ["lens", "fetch", "m/x", "-j"],
    ["sae", "ls", "m/x", "-j"],
    ["pack", "show", "formal.casual", "-j"],
])
def test_inventory_verbs_emit_pure_json(
    argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from drowse.store.templates import create_template_folder

    create_template_folder(
        "local", "weekday", slot="[DAY]", values=["Monday", "Tuesday"],
        contexts=[{"turns": [{"role": "user", "content": "day?"}], "assistant": "It is [DAY]."}],
    )
    install_fake_session(monkeypatch)
    monkeypatch.setattr(
        FakeSession, "score_template",
        lambda _session, _tmpl, **_kw: [SimpleNamespace(to_dict=lambda: {"ok": True})],
        raising=False,
    )
    monkeypatch.setattr(
        "drowse.lens.sources.fetch_lens_source",
        lambda *_a, **_k: SimpleNamespace(to_json=lambda: {"name": "neuronpedia"}),
    )
    main(argv)
    assert _stdout_json(capsys) is not None


class _FanSession(FakeSession):
    def generate_sweep(self, _prompt: Any, sweep: dict[str, list[float]], **kwargs: Any) -> Any:
        import itertools

        from drowse.generation.results import GenerationResult, RunSet

        self.base_steering = kwargs["base_steering"]
        rows = [dict(zip(sweep, combo, strict=True)) for combo in itertools.product(*sweep.values())]
        results = [
            GenerationResult(text="", tokens=[], token_count=7 + i, tok_per_sec=0.0,
                             elapsed=0.0, finish_reason="stop")
            for i in range(len(rows))
        ]
        for i, (result, row) in enumerate(zip(results, rows, strict=True)):
            kwargs["on_result"](i, result, row)
        return RunSet(results, node_ids=[None] * len(rows), grid=rows, kind="fan")


def test_experiment_fan_reports_each_row_on_stderr_and_json_stays_pure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    created = install_fake_session(monkeypatch, _FanSession)
    main(["experiment", "fan", "m/x", "p", "-g", "a=0,0.5,1", "-g", "b=0,1", "-S", "0.2 x", "-j"])
    captured = capsys.readouterr()
    assert json.loads(captured.out)["kind"] == "fan"
    assert "  1/6: a=+0.000, b=+0.000" in captured.err
    assert "  6/6: a=+1.000, b=+1.000" in captured.err
    assert created[0].base_steering == "0.2 x"


def test_experiment_fan_rejects_a_bad_grid_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["experiment", "fan", "m/x", "p", "-g", "a=0,warm"])
    assert exc.value.code == 2 and attempts == []


def test_transcript_run_prints_the_engine_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The verb is a thin print of ``Transcript.replay`` (whose regeneration
    semantics ``tests/loom/test_transcript.py`` owns): the transcript's model, the
    ``--strict`` flag, and each step's recorded → replayed drift."""
    from drowse.loom.transcript import Transcript, Turn

    nodes = {
        "rec": SimpleNamespace(role="assistant", text="hello"),
        "rep": SimpleNamespace(role="assistant", text="again"),
    }

    class _Session(FakeSession):
        tree = SimpleNamespace(get=nodes.__getitem__)

    calls: list[dict[str, Any]] = []

    def replay(_self: Transcript, session: Any, **kwargs: Any) -> Any:
        calls.append({"session": session, **kwargs})
        delta = SimpleNamespace(name="warm", delta=-0.25, a_value=0.5, b_value=0.25)
        step = SimpleNamespace(recorded_id="rec", replay_id="rep",
                               diff=SimpleNamespace(readings=[delta]))
        return SimpleNamespace(leaf_id="leaf0000", steps=[step])

    path = tmp_path / "t.yaml"
    Transcript(model_id="m/x", system_prompt=None, probes=[],
               turns=[Turn("user", "hi"), Turn("assistant", "hello")]).save(path)
    monkeypatch.setattr(Transcript, "replay", replay)
    created = install_fake_session(monkeypatch, _Session)
    main(["experiment", "transcript", "run", str(path), "-s"])
    assert created[0].model_id == "m/x"
    assert calls == [{"session": created[0], "mode": "default", "strict": True}]
    out = capsys.readouterr().out
    assert "recorded: hello" in out and "replayed: again" in out
    assert "Δ -0.2500  (recorded +0.5000 → got +0.2500)" in out


def _authored_behavior_manifold() -> None:
    from drowse.store.manifold_authoring import create_manifold_folder

    domain = {"type": "box", "axes": [{"name": "t", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0}]}
    nodes = [{"label": label, "coords": [i / 2], "statements": [label]} for i, label in enumerate("abc")]
    create_manifold_folder("local", "behavior", "", domain, nodes)


def test_experiment_naturalness_prints_the_engine_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The verb is a thin print of ``fit.naturalness.evaluate`` (whose
    behavior-manifold semantics ``tests/fit/test_naturalness.py`` owns): the
    resolved folder, the steering, the floors, and a pure ``-j`` payload."""
    import drowse.fit.naturalness as nat

    calls: list[tuple[Any, ...]] = []

    def evaluate(session: Any, folder: Any, prompt: str, steering: str,
                 compare_linear: bool, **kwargs: Any) -> Any:
        calls.append((folder.name, prompt, steering, compare_linear, kwargs["max_tokens"]))
        kwargs["on_progress"]("fitting behavior manifold (3 nodes)...")
        return nat.NaturalnessReport(prompt=prompt, scores=(
            nat.NaturalnessScore("manifold", steering, "text", (0.1, 0.3)),
        ))

    _authored_behavior_manifold()
    created = install_fake_session(monkeypatch, FakeSession)
    monkeypatch.setattr(nat, "evaluate", evaluate)
    main(["experiment", "naturalness", "m/x", "p", "--manifold", "behavior", "-S", "0.5 x", "-j"])
    captured = capsys.readouterr()
    row = json.loads(captured.out)["results"][0]
    assert row["mean_bhattacharyya"] == pytest.approx(0.2)
    assert "fitting behavior manifold" in captured.err
    assert calls == [("behavior", "p", "0.5 x", False, 128)]
    assert created[0].load_kwargs["probes"] == []


@pytest.mark.parametrize(("argv", "message"), [
    (["--manifold", "behavior", "-S", "0.5 jlens/cat", "--compare-linear"], "single manifold"),
    (["--manifold", "nowhere", "-S", "0.5 x"], "neither a manifold folder"),
])
def test_experiment_naturalness_fails_before_loading(
    argv: list[str], message: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _authored_behavior_manifold()
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["experiment", "naturalness", "m/x", "p", *argv])
    assert exc.value.code == 2 and attempts == []
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# manifold extract / fit runners
# ---------------------------------------------------------------------------

class _AuthoringSession(FakeSession):
    def extract(self, *args: Any, **kwargs: Any) -> Any:
        self.extract_call = (args, kwargs)
        role = kwargs.get("role")
        return ("happy.sad" + (f":role-{role}" if role else "")), object()

    def fit(self, *_args: Any, **kwargs: Any) -> Any:
        from drowse.geometry.domains import CustomDomain

        self.fit_kwargs = kwargs
        return SimpleNamespace(
            name="roles", layers={0: object()}, node_labels=["a", "b"],
            domain=CustomDomain(1), feature_space="raw", metadata={"fit_mode": "auto"},
        )


def test_extract_narrates_and_reports_the_canonical_tensor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from drowse.base.paths import tensor_filename

    created = install_fake_session(monkeypatch, _AuthoringSession)
    main(["manifold", "extract", "alice/happy", "sad", "-m", "fake/model", "--role", "pirate"])
    args, kwargs = created[0].extract_call
    assert args == ("happy", "sad")
    assert (kwargs["namespace"], kwargs["role"], kwargs["kind"]) == ("alice", "pirate", "abstract")
    assert callable(kwargs["on_progress"])
    captured = capsys.readouterr()
    assert f"/alice/happy.sad/{tensor_filename('fake/model')}" in captured.out
    assert "Loading model: fake/model" in captured.err


def test_extract_corrupt_tensor_is_not_a_cache_hit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from drowse.base.paths import manifold_dir, tensor_filename

    folder = manifold_dir("local", "happy.sad")
    folder.mkdir(parents=True)
    (folder / tensor_filename("fake/model")).write_bytes(b"")
    created = install_fake_session(monkeypatch, _AuthoringSession)
    main(["manifold", "extract", "happy.sad", "-m", "fake/model"])
    assert hasattr(created[0], "extract_call")


@pytest.mark.parametrize("argv", [
    ["manifold", "extract", "january", "july", "-m", "m/x", "--kind", "custom"],
    ["manifold", "generate", "months", "--concepts", "jan", "jul", "-m", "m/x", "--kind", "custom"],
    ["manifold", "extract", "a", "b", "c", "-m", "m/x"],
    ["manifold", "generate", "solo", "--concepts", "one", "-m", "m/x"],
    ["manifold", "fit", "anything", "-m", "m/x", "--layers", "sample"],
    ["manifold", "fit", "anything", "-m", "m/x", "--layers", "1,x"],
])
def test_authoring_rejections_happen_before_the_model_loads(
    argv: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2 and attempts == []


def test_fit_delegates_overrides_to_the_fit_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    from drowse.store.manifold_authoring import create_discover_manifold_folder

    folder = create_discover_manifold_folder(
        "local", "roles", "", fit_mode="auto",
        node_corpora={"pirate": ["arrr"], "scholar": ["indeed"]},
        node_roles={"pirate": "pirate", "scholar": "scholar"},
    )
    created = install_fake_session(monkeypatch, _AuthoringSession)
    main(["manifold", "fit", str(folder), "-m", "fake/model", "--smoothing", "0.25"])
    assert created[0].fit_kwargs["fit_mode"] is None
    assert created[0].fit_kwargs["hyperparams"] == {"smoothing": 0.25}


def test_a_folder_path_without_manifold_json_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: an existing directory fell through to name resolution and
    was reported as a missing ``~/.drowse/manifolds/<path>`` manifold."""
    empty = tmp_path / "not-a-manifold"
    empty.mkdir()
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["manifold", "fit", str(empty), "-m", "m/x"])
    assert exc.value.code == 2 and attempts == []
    assert f"no manifold.json in {empty}" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["pack", "rm", "../../etc", "-y"],
    ["pack", "show", "../x"],
    ["pack", "clear", "/x", "-m", "m/x"],
    ["pack", "export", "archive", "a/b/c"],
    ["manifold", "extract", "../x", "-m", "m/x"],
    ["manifold", "generate", "a/", "--concepts", "x", "y", "-m", "m/x"],
    ["manifold", "fit", "/personas", "-m", "m/x"],
    ["manifold", "why", "../x", "-m", "m/x"],
    ["manifold", "from-template", "../t"],
    ["template", "create", "../t", "--slot", "X", "--values", "a", "b", "--contexts", "c.json"],
    ["template", "show", "../t"],
    ["template", "rm", "../t", "-y"],
    ["template", "score", "../t", "-m", "m/x"],
    ["sae", "show", "m/x", "local:../x"],
])
def test_malformed_references_are_one_line_usage_errors(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: an unsafe or empty ``ns/name`` escaped ``main`` as a raw
    ``ValueError`` traceback from the path-traversal barrier."""
    monkeypatch.chdir(Path(__file__).parents[2] / "drowse" / "data")
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(argv)
    err = capsys.readouterr().err
    assert exc.value.code == 2 and attempts == [], err
    assert err.count("\n") == 1


def test_fit_overrides_against_an_authored_folder_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _authored_behavior_manifold()
    attempts = poison_session(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["manifold", "fit", "behavior", "-m", "m/x", "--max-dim", "2"])
    assert exc.value.code == 2 and attempts == []


# ---------------------------------------------------------------------------
# lens fit
# ---------------------------------------------------------------------------

def test_lens_fit_pins_a_hub_corpus_once_for_preflight_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.cli.verbs.lens as lens_verbs
    import drowse.lens.corpus as lens_store
    from drowse.lens.corpus import HfLensCorpus, LensCorpusMix

    pinned = LensCorpusMix((
        HfLensCorpus("org/data", "default", "a" * 40, "body"),
        HfLensCorpus("org/zh", "cmn", "b" * 40, files=64),
    ), (3, 1))
    pins: list[Any] = []
    proofs: list[Any] = []
    streams: list[tuple[int, Any]] = []

    def stream(n: int, corpus: Any) -> Any:
        streams.append((n, corpus))
        raise SystemExit(0)  # stop before the model load

    monkeypatch.setattr(lens_store, "pin_lens_corpus", lambda c: pins.append(c) or pinned)
    monkeypatch.setattr(lens_store, "stream_lens_corpus", stream)
    monkeypatch.setattr(
        lens_verbs, "lens_fit_is_noop",
        lambda _args, _layers, **kw: proofs.append(kw["hf_corpus"]) or False,
    )
    with pytest.raises(SystemExit):
        main(["lens", "fit", "m", "--hf-corpus", "org/data@main,text_field=body,weight=3",
              "--hf-corpus", "org/zh:cmn,files=64", "--prompts", "7"])
    assert pins == [LensCorpusMix((
        HfLensCorpus("org/data", None, "main", "body"),
        HfLensCorpus("org/zh", "cmn", files=64),
    ), (3, 1))]
    assert proofs == [pinned] and streams == [(7, pinned)]


@pytest.mark.parametrize("argv", [
    ["lens", "fit", "m", "--hf-corpus", "org/data:"],
    ["lens", "fit", "m", "--hf-corpus", "org/data,weight=2"],
    ["lens", "fit", "m", "--hf-corpus", "org/data,files=0"],
    ["lens", "fit", "m", "--hf-corpus", "org/data", "--hf-corpus", "org/zh,seed=1"],
    ["lens", "fit", "m", "--layers", "sample"],
    ["lens", "fit", "m", "--layers", ""],
    ["lens", "fit", "m", "--layers", "band"],
])
def test_lens_fit_rejects_bad_options(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("drowse lens fit: ")


@pytest.mark.parametrize(("verb", "source"), [("lens", "local:r"), ("sae", "local:mine")])
def test_source_removal_requires_yes(
    verb: str, source: str, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main([verb, "rm", "m/x", source])
    assert exc.value.code == 2
    assert "-y/--yes" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# manifold compare / why output (folded vectors stubbed)
# ---------------------------------------------------------------------------

def _mock_profile(agg: Any, per_layer: Any) -> Any:
    class MockProfile:
        def cosine_similarity(self, other: Any, *, per_layer: bool = False, whitener: Any = None) -> Any:
            return per_layer_fn(other) if per_layer else agg(other)

        def items(self) -> Any:
            import torch
            return {layer: torch.full((1,), v) for layer, v in per_layer_fn(self).items()}.items()

    per_layer_fn = per_layer
    return MockProfile()


def _stub_folds(monkeypatch: pytest.MonkeyPatch, profiles: dict[str, Any]) -> None:
    """Serve ``profiles`` (keyed ``name`` or ``name:variant``, all under
    ``default/``) through the compare/why fold helpers."""
    import drowse.cli.verbs.manifold as verbs
    from drowse.store import neutral as neutral_store

    def fold(ref: str, model_id: str) -> Any:
        typed_ns, _, rest = ref.rpartition("/")
        name, _, variant = rest.partition(":")
        profile = profiles.get(rest)
        if profile is None:
            raise CliError(f"no fitted manifold for '{ref}'", exit_code=1)
        display = f"default/{rest}" if typed_ns else rest
        return verbs._Folded(display, "default", name, variant or "raw", profile)

    monkeypatch.setattr(verbs, "_fold", fold)
    monkeypatch.setattr(verbs, "_fold_all", lambda _m: {
        ("default", key): p for key, p in profiles.items() if ":" not in key
    })
    monkeypatch.setattr(neutral_store, "load_neutral_whitener", lambda _m, **_k: object())


def test_compare_rank_all_verbose(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    happy = _mock_profile(lambda o: None, lambda o: {})
    warm = _mock_profile(lambda o: None, lambda o: {})
    target = _mock_profile(
        lambda o: 0.3421 if o is happy else 0.1893,
        lambda o: {14: 0.5122, 15: 0.4891} if o is happy else {14: 0.1},
    )
    _stub_folds(monkeypatch, {"angry.calm": target, "happy.sad": happy, "warm.clinical": warm})
    main(["manifold", "compare", "angry.calm", "-m", "m/x", "-v"])
    out = capsys.readouterr().out
    assert "angry.calm vs all installed" in out and "per-layer (top 3)" in out and "0.5122" in out
    main(["manifold", "compare", "angry.calm", "-m", "m/x", "-v", "-j"])
    data = _stdout_json(capsys)
    assert data["target"] == "angry.calm"
    assert [row["name"] for row in data["similarities"]] == ["happy.sad", "warm.clinical"]
    assert data["per_layer_top3"]["happy.sad"]["14"] == pytest.approx(0.5122)


def test_compare_matrix(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    names = ["angry.calm", "happy.sad:sae-rel", "warm.clinical"]
    _stub_folds(monkeypatch, {n: _mock_profile(lambda o: 0.25, lambda o: {14: 0.3456}) for n in names})
    main(["manifold", "compare", *names, "-m", "m/x", "-v", "-j"])
    data = _stdout_json(capsys)
    assert data["concepts"] == names
    assert set(data["per_layer"]) == {
        "angry.calm|happy.sad:sae-rel", "angry.calm|warm.clinical", "happy.sad:sae-rel|warm.clinical",
    }
    main(["manifold", "compare", *names, "-m", "m/x", "-v"])
    assert "per-layer" not in capsys.readouterr().out


def test_why_histogram_and_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from drowse.cli.verbs.manifold import HIST_BUCKETS

    mags = {i: float(i + 1) * 0.01 for i in range(62)}
    _stub_folds(monkeypatch, {"angry.calm:sae-rel": _mock_profile(lambda o: 0, lambda o: mags)})
    main(["manifold", "why", "angry.calm:sae-rel", "-m", "m/x"])
    out = capsys.readouterr().out
    assert "angry.calm:sae-rel (62 layers" in out
    assert len([ln for ln in out.splitlines() if "█" in ln or "░" in ln]) == HIST_BUCKETS
    main(["manifold", "why", "angry.calm:sae-rel", "-m", "m/x", "-j"])
    data = _stdout_json(capsys)
    assert data["total_layers"] == 62 and data["layers"][0] == {"layer": 0, "magnitude": 0.01}
    with pytest.raises(SystemExit) as exc:
        main(["manifold", "why", "nope", "-m", "m/x"])
    assert exc.value.code == 1
