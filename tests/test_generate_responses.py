"""generate_responses / generate_neutral_responses — CPU, mocked generator.

The 4.0 conversational (A2) corpus generator: the model answers the shared
baseline prompts in character (concept in the system prompt + a kind-derived
elicitation role).  These tests stub ``_run_generator`` and override the
baseline prompt set, so no real model is needed.
"""
from __future__ import annotations

from typing import Any

import pytest

from drowse.core import capture as V
from drowse.core.session import DrowseSession, _role_for, _system_for
from drowse.core.capture import _LENGTH_DIRECTIVE


class _FakeSession(DrowseSession):
    """Bypass real construction; record each generator call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def _run_generator_batch(
        self, system_msg: str, prompts: list[str], max_new_tokens: int,
        **kw: Any,
    ) -> list[str]:
        # The batched seam generate_responses / generate_neutral_responses call.
        # Record one entry per prompt (preserving order) so the alignment +
        # count assertions are identical to the old per-prompt seam.
        role = kw.get("role")
        out: list[str] = []
        for prompt in prompts:
            self.calls.append(
                {"system": system_msg, "prompt": prompt, "role": role},
            )
            # Echo so the caller can see which prompt produced which response.
            out.append(f"resp::{prompt}")
        return out


class _BlankThenContentSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.retry_calls: list[dict[str, Any]] = []

    def _run_generator_batch(
        self, system_msg: str, prompts: list[str], max_new_tokens: int,
        **kw: Any,
    ) -> list[str]:
        if kw.get("deterministic"):
            self.retry_calls.append({
                "system": system_msg,
                "prompts": list(prompts),
                "role": kw.get("role"),
                "min_new_tokens": kw.get("min_new_tokens"),
            })
            return [f"repaired::{prompt}" for prompt in prompts]
        return ["" if prompt == "P1" else f"sampled::{prompt}" for prompt in prompts]


class _AlwaysBlankSession(_BlankThenContentSession):
    def _run_generator_batch(
        self, system_msg: str, prompts: list[str], max_new_tokens: int,
        **kw: Any,
    ) -> list[str]:
        return ["" for _ in prompts]


@pytest.fixture(autouse=True)
def _small_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        V, "_load_baseline_prompts", lambda: ["P0", "P1", "P2"],
    )


# --- helpers --------------------------------------------------------------


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


def test_round_robin_alignment_and_count() -> None:
    s = _FakeSession()
    out = s.generate_responses(["happy"], ["abstract"], samples_per_prompt=2)
    # 2 samples x 3 prompts = 6 responses, samples-outer / prompts-inner.
    # The length directive rides the system prompt, so user turns stay bare.
    assert out["happy"] == [
        "resp::P0", "resp::P1", "resp::P2",
        "resp::P0", "resp::P1", "resp::P2",
    ]
    # response[i] was generated from baseline prompt[i % 3]
    prompts = [c["prompt"] for c in s.calls]
    assert prompts == ["P0", "P1", "P2", "P0", "P1", "P2"]
    # every node system leads with the shared brevity directive
    assert all(c["system"].startswith(_LENGTH_DIRECTIVE) for c in s.calls)


def test_kind_drives_system_and_role() -> None:
    s = _FakeSession()
    s.generate_responses(["pirate"], ["concrete"])
    call = s.calls[0]
    assert call["system"] == (
        f"{_LENGTH_DIRECTIVE} You are a pirate. Respond exactly as a pirate would."
    )
    assert call["role"] == "pirate"

    s2 = _FakeSession()
    s2.generate_responses(["happy"], ["abstract"])
    assert s2.calls[0]["role"] == "someone_happy"


def test_explicit_role_overrides_kind() -> None:
    s = _FakeSession()
    s.generate_responses(["happy"], ["abstract"], roles={"happy": "oracle"})
    assert all(c["role"] == "oracle" for c in s.calls)


def test_multiple_concepts_keyed_separately() -> None:
    s = _FakeSession()
    out = s.generate_responses(["happy", "sad"], ["abstract", "abstract"])
    assert set(out) == {"happy", "sad"}
    assert len(out["happy"]) == 3 and len(out["sad"]) == 3


def test_blank_row_is_retried_in_place_with_same_elicitation_frame() -> None:
    s = _BlankThenContentSession()
    out = s.generate_responses(["sad"], ["abstract"])

    assert out["sad"] == ["sampled::P0", "repaired::P1", "sampled::P2"]
    assert s.retry_calls == [{
        "system": (
            f"{_LENGTH_DIRECTIVE} You are someone sad. "
            "Respond exactly as someone sad would."
        ),
        "prompts": ["P1"],
        "role": "someone_sad",
        "min_new_tokens": 8,
    }]


def test_blank_deterministic_retry_fails_before_authoring() -> None:
    with pytest.raises(RuntimeError, match="including the deterministic retry"):
        _AlwaysBlankSession().generate_responses(["sad"], ["abstract"])


# --- generate_neutral_responses -------------------------------------------


def test_neutral_responses_brevity_system_no_persona_no_role() -> None:
    s = _FakeSession()
    out = s.generate_neutral_responses(samples_per_prompt=2)
    assert len(out) == 6  # 2 x 3 prompts
    # neutral's only system is the shared brevity directive (no persona); the
    # user turn stays bare and there is no role swap.
    assert all(c["system"] == _LENGTH_DIRECTIVE for c in s.calls)
    assert all(c["prompt"] in {"P0", "P1", "P2"} for c in s.calls)
    assert all(c["role"] is None for c in s.calls)
