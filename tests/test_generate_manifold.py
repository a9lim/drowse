"""``generate_manifold``: the one discover-generate loop, over a plan.

The CLI ``manifold generate`` verb, ``POST /manifolds/generate`` and the
bundled regeneration script all plan with
:func:`~drowse.store.manifold_authoring.plan_discover_generation` and fill
through :meth:`DrowseSession.generate_manifold`
(:meth:`~drowse.fit.corpus.CorpusGenerator.fill`).  A scripted tiny model
answers each baseline prompt with ``resp::<prompt>``; the baseline set is
three prompts.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from tests.support.sessions import TinySessionFactory
from tests.test_generate_responses import _Answerer, _session


@pytest.fixture(autouse=True)
def _three_baseline_prompts(isolated_home: Path) -> None:
    (isolated_home / "baseline_prompts.json").write_text(json.dumps(["P0", "P1", "P2"]))


def _plan(folder: Path, labels: list[str], **kwargs: Any) -> Any:
    from drowse.store.manifold_authoring import plan_discover_generation

    return plan_discover_generation(
        folder, folder.name, "test", fit_mode="pca", labels=labels, **kwargs,
    )


def _node_corpus(folder: Path, index: int, label: str) -> list[str]:
    from drowse.store.manifold_folder import node_filename

    return json.loads((folder / "nodes" / node_filename(index, label)).read_text())


def test_writes_every_pending_node_under_its_manifest_frame(
    make_tiny_session: TinySessionFactory, tmp_path: Path,
) -> None:
    folder = tmp_path / "local" / "pair"
    plan = _plan(
        folder, ["pirate", "happy"],
        node_kinds={"pirate": "concrete", "happy": "abstract"},
        node_roles={"happy": "oracle"},
    )
    answerer = _Answerer()
    progress: list[str] = []
    _session(make_tiny_session, answerer).generate_manifold(plan, on_progress=progress.append)

    assert _node_corpus(folder, 0, "pirate") == ["resp::P0", "resp::P1", "resp::P2"]
    assert _node_corpus(folder, 1, "happy") == ["resp::P0", "resp::P1", "resp::P2"]
    assert [c["role"] for c in answerer.calls] == ["pirate"] * 3 + ["oracle"] * 3
    assert "Wrote node 'happy' (2/2, 3 responses)" in progress
    assert _plan(folder, ["pirate", "happy"]).pending == ()


def test_a_resumed_node_keeps_its_declared_kind(
    make_tiny_session: TinySessionFactory, tmp_path: Path,
) -> None:
    """Resuming with another ``kind`` generates under the manifest's kind, so a
    node's corpus always matches the provenance its manifest records."""
    folder = tmp_path / "local" / "pair"
    _plan(folder, ["happy", "sad"], node_kinds={"happy": "abstract", "sad": "abstract"})
    resumed = _plan(folder, ["happy", "sad"], node_kinds={"happy": "concrete", "sad": "concrete"})
    assert resumed.kinds == {"happy": "abstract", "sad": "abstract"}

    answerer = _Answerer()
    _session(make_tiny_session, answerer).generate_manifold(resumed)
    assert {c["role"] for c in answerer.calls} == {"someone happy", "someone sad"}


def test_seed_is_per_node_so_a_resumed_run_matches_a_full_run(
    make_tiny_session: TinySessionFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_authoring import append_discover_manifold_node

    folder = tmp_path / "local" / "trio"
    _plan(folder, ["a", "b", "c"])
    append_discover_manifold_node(folder, 0, "a", ["kept"])
    session = _session(make_tiny_session, _Answerer())
    seeds: list[int] = []
    real_seed = torch.manual_seed

    def _record(seed: int) -> Any:
        seeds.append(int(seed))
        return real_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", _record)
    session.generate_manifold(_plan(folder, ["a", "b", "c"]), seed=40)
    assert seeds == [41, 42]
    assert _node_corpus(folder, 0, "a") == ["kept"]


def test_unseeded_generation_leaves_the_rng_alone(
    make_tiny_session: TinySessionFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(make_tiny_session, _Answerer())
    seeds: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: seeds.append(seed))
    session.generate_manifold(_plan(tmp_path / "local" / "pair", ["a", "b"]))
    assert seeds == []


def test_holds_the_model_lock_refuses_steering_and_evicts_on_failure(tmp_path: Path) -> None:
    """The fill runs inside the exclusive-model guard after the unsteered check,
    and the folder is evicted even when the run stops part-way."""
    from drowse.events import EventBus
    from drowse.session.extraction import Extraction

    calls: list[Any] = []

    @contextmanager
    def exclusive(busy_msg: str, *, phase_msg: str) -> Iterator[None]:
        calls.append(("lock", "generate_manifold" in busy_msg, "generation" in phase_msg))
        try:
            yield
        finally:
            calls.append("unlock")

    class _Corpus:
        def fill(self, plan: Any, **kwargs: Any) -> None:
            calls.append(("fill", kwargs["seed"]))
            raise RuntimeError("model died")

    extraction = Extraction(
        object(), EventBus(), _Corpus(),  # type: ignore[arg-type]
        exclusive=exclusive,
        assert_unsteered=lambda: calls.append("unsteered"),
        adopt=lambda *_a: None,
        evict=lambda namespace, name, *, feature: calls.append(("evict", namespace, name, feature)),
        register_profile=lambda *_a: None,
        dls=True,
    )
    plan = _plan(tmp_path / "local" / "pair", ["a", "b"])
    with pytest.raises(RuntimeError, match="model died"):
        extraction.generate_manifold(plan, seed=3)
    assert calls == [
        ("lock", True, True), "unsteered", ("fill", 3),
        ("evict", "local", "pair", None), "unlock",
    ]


def test_cli_generate_plans_then_fills_with_the_seed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from drowse.cli.main import main
    from tests._cli_fakes import FakeSession, install_fake_session

    class _Generator(FakeSession):
        def generate_manifold(self, plan: Any, **kwargs: Any) -> None:
            self.generated = (plan, kwargs)

    created = install_fake_session(monkeypatch, _Generator)
    main([
        "manifold", "generate", "pair", "--concepts", "a", "b",
        "-m", "fake/model", "--seed", "7", "--samples-per-prompt", "2",
    ])
    plan, kwargs = created[0].generated
    assert plan.pending == ("a", "b") and plan.folder.name == "pair"
    assert kwargs["seed"] == 7 and kwargs["samples_per_prompt"] == 2
    assert "wrote local/pair (2 nodes)" in capsys.readouterr().out
