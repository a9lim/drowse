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
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.fit.capture import _LENGTH_DIRECTIVE
from drowse.generation.decode import GenerationState
from drowse.session.session import _SessionStopCriteria, _role_for, _system_for
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


@pytest.mark.parametrize("score_kind", ["none", "tensor", "tuple"])
def test_corpus_stop_criteria_accepts_transformers_score_shapes(score_kind: str) -> None:
    state = GenerationState()
    criterion = _SessionStopCriteria(state)
    input_ids = torch.LongTensor([[1, 2], [3, 4]])
    score = torch.FloatTensor([[0.1, 0.9], [0.8, 0.2]])
    scores = None if score_kind == "none" else score if score_kind == "tensor" else (score, score)
    assert criterion(input_ids, scores).tolist() == [False, False]
    state.request_stop()
    assert criterion(input_ids, scores).tolist() == [True, True]
    state.reset()
    assert criterion(input_ids, scores).tolist() == [False, False]


def test_system_for_abstract_vs_concrete() -> None:
    assert _system_for("happy", "abstract") == (
        "You are someone happy. Respond exactly as someone happy would."
    )
    assert _system_for("alien", "concrete") == (
        "You are an alien. Respond exactly as an alien would."
    )
    # default kind is abstract
    assert _system_for("happy", None).startswith("You are someone happy")


def test_role_for_abstract_vs_concrete() -> None:
    assert _role_for("happy", "abstract") == "someone_happy"
    assert _role_for("pirate", "concrete") == "pirate"


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
