"""generate_responses / generate_neutral_responses on a real CPU session.

The conversational corpus generator: the model answers the shared baseline
prompts in character (concept in the system prompt plus a kind-derived
assistant-role label).  A scripted tiny model answers each prompt with
``resp::<prompt>`` and records the system turn, user turn and assistant
label it was actually rendered with; the baseline set is overridden to three
prompts through the user override file.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from drowse.fit.capture import _LENGTH_DIRECTIVE
from drowse.fit.corpus import CorpusGenerator, role_for, system_for
from tests.support.sessions import TinySessionFactory

_TURN = re.compile(
    r"<\|start_header_id\|>(.*?)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.DOTALL,
)


class _Answerer:
    """Scripted reply source: echo the user prompt and record the frame.

    ``blank`` prompts get an empty reply (immediate end-of-turn) on their
    first ``blank_attempts`` generations.
    """

    def __init__(self, *, blank: tuple[str, ...] = (), blank_attempts: int = 1,
                 prefix: str = "resp") -> None:
        self.calls: list[dict[str, Any]] = []
        self._blank = blank
        self._blank_attempts = blank_attempts
        self._prefix = prefix

    def __call__(self, label: str, prompt_text: str) -> str:
        turns = dict(_TURN.findall(prompt_text))
        prompt = turns["user"]
        self.calls.append({"system": turns.get("system"), "prompt": prompt, "role": label})
        seen = sum(call["prompt"] == prompt for call in self.calls)
        if prompt in self._blank and seen <= self._blank_attempts:
            return ""
        return f"{self._prefix}::{prompt}"


@pytest.fixture(autouse=True)
def _three_baseline_prompts(isolated_home: Path) -> None:
    (isolated_home / "baseline_prompts.json").write_text(json.dumps(["P0", "P1", "P2"]))


def _session(factory: TinySessionFactory, answerer: _Answerer) -> Any:
    session, _script = factory.scripted(answerer, warm=False)
    return session


def test_system_for_abstract_vs_concrete() -> None:
    assert system_for("happy", "abstract") == (
        "You are someone happy. Respond exactly as someone happy would."
    )
    assert system_for("alien", "concrete") == (
        "You are an alien. Respond exactly as an alien would."
    )
    # default kind is abstract
    assert system_for("happy", None).startswith("You are someone happy")


def test_system_for_custom_formats_the_callers_template() -> None:
    assert system_for("january", "custom", "I am {c}, {art} month. {{braces}}") == (
        "I am january, a month. {braces}"
    )


@pytest.mark.parametrize("template", ["I am {c", "I am {mood}", "I am {0}", None])
def test_system_for_custom_rejects_an_unusable_template_as_a_request_error(
    template: str | None,
) -> None:
    """Regression: a request's malformed ``custom_system`` escaped as the
    ``str.format`` error itself — a ``KeyError``/``IndexError`` 500, or a
    bare ``ValueError`` whose text never named the template."""
    from drowse.errors import RequestError

    with pytest.raises(RequestError, match="custom system template") as caught:
        system_for("january", "custom", template)
    assert caught.value.user_message()[0] == 400


def test_role_for_abstract_vs_concrete() -> None:
    assert role_for("happy", "abstract") == "someone_happy"
    assert role_for("pirate", "concrete") == "pirate"


# --- generate_responses ---------------------------------------------------


def test_round_robin_alignment_and_count(make_tiny_session: TinySessionFactory) -> None:
    answerer = _Answerer()
    out = _session(make_tiny_session, answerer).generate_responses(
        ["happy"], ["abstract"], samples_per_prompt=2,
    )
    # 2 samples x 3 prompts, samples-outer / prompts-inner: response[i]
    # answers baseline prompt[i % 3].
    assert out["happy"] == [
        "resp::P0", "resp::P1", "resp::P2",
        "resp::P0", "resp::P1", "resp::P2",
    ]
    assert [c["prompt"] for c in answerer.calls] == ["P0", "P1", "P2"] * 2
    # The length directive leads every node system; user turns stay bare.
    assert all(c["system"].startswith(_LENGTH_DIRECTIVE) for c in answerer.calls)


def test_kind_drives_system_and_role(make_tiny_session: TinySessionFactory) -> None:
    concrete = _Answerer()
    _session(make_tiny_session, concrete).generate_responses(["pirate"], ["concrete"])
    assert concrete.calls[0]["system"] == (
        f"{_LENGTH_DIRECTIVE} You are a pirate. Respond exactly as a pirate would."
    )
    assert concrete.calls[0]["role"] == "pirate"

    # The role slug ``someone_happy`` renders as the header label "someone happy".
    abstract = _Answerer()
    _session(make_tiny_session, abstract).generate_responses(["happy"], ["abstract"])
    assert abstract.calls[0]["role"] == "someone happy"


def test_slug_concept_is_humanized_in_the_system_prompt_and_label(
    make_tiny_session: TinySessionFactory,
) -> None:
    answerer = _Answerer()
    out = _session(make_tiny_session, answerer).generate_responses(
        ["artificial_intelligence"], ["abstract"],
    )
    assert out["artificial_intelligence"] == ["resp::P0", "resp::P1", "resp::P2"]
    # The slug is the identifier; the model reads it with spaces, both in the
    # persona system prompt and in the ``someone_<slug>`` assistant label.
    assert {c["system"] for c in answerer.calls} == {
        f"{_LENGTH_DIRECTIVE} You are someone artificial intelligence. "
        "Respond exactly as someone artificial intelligence would."
    }
    assert {c["role"] for c in answerer.calls} == {"someone artificial intelligence"}


def test_explicit_role_overrides_kind(make_tiny_session: TinySessionFactory) -> None:
    answerer = _Answerer()
    _session(make_tiny_session, answerer).generate_responses(
        ["happy"], ["abstract"], roles={"happy": "oracle"},
    )
    assert {c["role"] for c in answerer.calls} == {"oracle"}


def test_multiple_concepts_keyed_separately(make_tiny_session: TinySessionFactory) -> None:
    out = _session(make_tiny_session, _Answerer()).generate_responses(
        ["happy", "sad"], ["abstract", "abstract"],
    )
    assert out == {
        "happy": ["resp::P0", "resp::P1", "resp::P2"],
        "sad": ["resp::P0", "resp::P1", "resp::P2"],
    }


def test_blank_row_is_retried_in_place_with_same_elicitation_frame(
    make_tiny_session: TinySessionFactory,
) -> None:
    answerer = _Answerer(blank=("P1",))
    out = _session(make_tiny_session, answerer).generate_responses(["sad"], ["abstract"])

    assert out["sad"] == ["resp::P0", "resp::P1", "resp::P2"]
    frame = {
        "system": (
            f"{_LENGTH_DIRECTIVE} You are someone sad. "
            "Respond exactly as someone sad would."
        ),
        "role": "someone sad",
    }
    p1_calls = [c for c in answerer.calls if c["prompt"] == "P1"]
    assert p1_calls == [{**frame, "prompt": "P1"}] * 2
    assert [c["prompt"] for c in answerer.calls] == ["P0", "P1", "P2", "P1"]


def test_blank_row_retry_decodes_greedily_with_an_eos_floor(
    make_tiny_session: TinySessionFactory,
) -> None:
    """Only the blank row is re-decoded: greedy, with ``min_new_tokens`` set.

    Scripted one-hot logits make greedy and sampled decoding agree, so the
    retry's decode settings are observed on the ``generate`` call itself.
    """
    session = _session(make_tiny_session, _Answerer(blank=("P1",)))
    calls: list[dict[str, Any]] = []

    def generate(input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append({
            "rows": int(input_ids.shape[0]),
            **{key: kwargs.get(key) for key in (
                "do_sample", "min_new_tokens", "temperature", "top_p",
            )},
        })
        return session.model.generate(input_ids, **kwargs)

    corpus = CorpusGenerator(
        generate, session.tokenizer, device=session.device,
        traits=session.chat_traits, scene=lambda: session.scene_grammar,
        model_type=session.model.config.model_type,
    )
    assert corpus.responses(["sad"], ["abstract"]) == {
        "sad": ["resp::P0", "resp::P1", "resp::P2"],
    }
    assert calls == [
        {"rows": 3, "do_sample": True, "min_new_tokens": None,
         "temperature": 1.0, "top_p": 0.9},
        {"rows": 1, "do_sample": False, "min_new_tokens": 8,
         "temperature": None, "top_p": None},
    ]


def test_blank_deterministic_retry_fails_before_authoring(
    make_tiny_session: TinySessionFactory,
) -> None:
    session = _session(make_tiny_session, _Answerer(blank=("P1",), blank_attempts=2))
    with pytest.raises(RuntimeError, match="including the deterministic retry"):
        session.generate_responses(["sad"], ["abstract"])


# --- generate_neutral_responses -------------------------------------------


def test_neutral_responses_brevity_system_no_persona_no_role(
    make_tiny_session: TinySessionFactory,
) -> None:
    answerer = _Answerer()
    out = _session(make_tiny_session, answerer).generate_neutral_responses(
        samples_per_prompt=2,
    )
    assert out == ["resp::P0", "resp::P1", "resp::P2"] * 2
    # Neutral's only system is the shared length directive (no persona), the
    # user turn stays bare, and the assistant keeps its own label.
    assert all(c["system"] == _LENGTH_DIRECTIVE for c in answerer.calls)
    assert {c["role"] for c in answerer.calls} == {"assistant"}

# --- generate_manifold: the one discover-generate loop, over a plan --------
#
# The CLI ``manifold generate`` verb, ``POST /manifolds/generate`` and the
# bundled regeneration script all plan with ``plan_discover_generation`` and
# fill through ``DrowseSession.generate_manifold`` (``CorpusGenerator.fill``).


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


def test_a_resumed_node_with_no_recorded_kind_takes_the_callers(
    make_tiny_session: TinySessionFactory, tmp_path: Path,
) -> None:
    """A builder-authored node with ``kind=None`` resumes under the caller's
    kind (custom frame included) and records it; a node that already has its
    corpus keeps its unrecorded kind."""
    from drowse.store.manifold_authoring import append_discover_manifold_node

    folder = tmp_path / "local" / "pair"
    _plan(folder, ["kept", "late"])
    append_discover_manifold_node(folder, 0, "kept", ["done"])
    resumed = _plan(folder, ["kept", "late"], node_kinds={"kept": "custom", "late": "custom"})
    assert resumed.kinds == {"kept": None, "late": "custom"}
    assert resumed.pending == ("late",)
    manifest = json.loads((folder / "manifold.json").read_text())
    assert [node["kind"] for node in manifest["nodes"]] == [None, "custom"]

    answerer = _Answerer()
    _session(make_tiny_session, answerer).generate_manifold(
        resumed, custom_system="It is the month of {c}.",
    )
    assert {c["role"] for c in answerer.calls} == {"assistant"}
    assert all("It is the month of late." in c["system"] for c in answerer.calls)


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
