"""The real tiny-session fixtures, end to end on CPU.

These pin the contract wave-2 tests build on: a ``DrowseSession`` loaded
through the real loader on a tiny checkpoint, warmed so steering and probes
work, with deterministic output available through ``ScriptedReplies``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from drowse import SamplingConfig, TokenEvent
from tests.support.sessions import TinyCheckpoint, TinySessionFactory
from tests.support.tiny_models import FAMILIES, byte_tokenizer


def test_tiny_session_is_real_and_warm(tiny_session: Any) -> None:
    assert len(tiny_session.layers) == 4
    assert tiny_session.whitener is not None
    assert set(tiny_session.layer_means) == {0, 1, 2, 3}
    assert tiny_session.scene_grammar is not None
    result = tiny_session.generate(
        "Say hi.", sampling=SamplingConfig(max_tokens=4, seed=0),
    ).first
    assert result.token_count == 4
    assert result.prompt_tokens > 0


def test_byte_tokenizer_round_trips_unicode() -> None:
    tokenizer = byte_tokenizer("llama")
    text = "naïve — “quoted” 🙂"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == len(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text


def test_scripted_reply_is_deterministic(make_tiny_session: TinySessionFactory) -> None:
    session, script = make_tiny_session.scripted("Hello there.")
    first = session.generate("Say hi.", stateless=True).first
    second = session.generate("Something else.", stateless=True).first
    assert first.text == second.text == "Hello there."
    assert first.finish_reason == "stop"
    assert "<|start_header_id|>user<|end_header_id|>\n\nSay hi.<|eot_id|>" in script.prompts[0]


def test_scripted_replies_follow_turn_index(make_tiny_session: TinySessionFactory) -> None:
    session, _script = make_tiny_session.scripted(["one", "two"])
    assert session.generate("a").first.text == "one"
    assert session.generate("b").first.text == "two"
    assert session.generate("c").first.text == "two"


def test_stream_merges_partial_utf8_tokens_into_one_event(
    make_tiny_session: TinySessionFactory,
) -> None:
    """A multi-byte character spans several tokens but streams as one event.

    Regression for the flaky ``test_session.py::test_generate_stream``: it
    asserted one event per token, which breaks whenever the model samples a
    character that needs more than one token (``é`` is two bytes here).
    """
    session, _script = make_tiny_session.scripted("café")
    events = list(session.generate_stream("Order?"))
    result = session.last_result
    assert result is not None
    assert all(isinstance(event, TokenEvent) for event in events)
    assert result.token_count == len("café".encode("utf-8")) == 5
    assert [event.text for event in events] == ["c", "a", "f", "é"]
    assert [event.index for event in events] == [0, 1, 2, 3]
    assert "".join(event.text for event in events) == result.text


def test_per_test_home_reuses_the_seeded_neutral_cache(
    isolated_home: Path, make_tiny_session: TinySessionFactory,
) -> None:
    from drowse.base.paths import model_dir

    make_tiny_session()  # computes the neutral cache once, or reuses the seed
    session = make_tiny_session()
    local = model_dir(session.model_id)
    assert local.is_relative_to(isolated_home)
    # Shard names carry a random generation id, so equal names mean the
    # cache was copied from the seed rather than recomputed in this home.
    names = {path.name for path in local.glob("neutral_activations*")}
    seeded = {path.name for path in make_tiny_session.seed_dir.glob("neutral_activations*")}
    assert names and names == seeded


def test_cold_session_defers_the_whitener(make_tiny_session: TinySessionFactory) -> None:
    session = make_tiny_session(warm=False)
    result = session.generate("hi", sampling=SamplingConfig(max_tokens=2, seed=0)).first
    assert result.token_count == 2


def test_app_client_serves_the_real_session(tiny_session: Any, app_client: Any) -> None:
    client = app_client(tiny_session)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == tiny_session.model_id


@pytest.mark.parametrize("family", sorted(set(FAMILIES) - {"llama"}))
def test_other_chat_families_load_and_generate(
    family: str, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    checkpoint = TinyCheckpoint.write(tmp_path_factory.mktemp(family), family)
    factory = TinySessionFactory(checkpoint, tmp_path_factory.mktemp(f"{family}-seed"))
    try:
        session, _script = factory.scripted("ok", warm=False)
        assert session.scene_grammar is not None
        assert session.generate("hi").first.text == "ok"
    finally:
        factory.close()
