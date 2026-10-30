"""The typed ``GenState`` lifecycle on a real ``DrowseSession`` (CPU)."""
from __future__ import annotations

from typing import Any

import pytest

from drowse import GenState, SamplingConfig
from drowse.errors import ConcurrentGenerationError
from drowse.session.session import ProfileNotRegisteredError
from tests.support.sessions import TinySessionFactory

_FOUR_TOKENS = SamplingConfig(max_tokens=4, seed=0)


@pytest.fixture
def session(make_tiny_session: TinySessionFactory) -> Any:
    session, _script = make_tiny_session.scripted("Hi there.")
    return session


class TestGenStateTransitions:
    def test_idle_at_construction(self, session: Any) -> None:
        assert session.gen_state is GenState.IDLE
        assert session.is_generating is False

    def test_running_during_generate(self, session: Any) -> None:
        """``on_token`` fires from inside the decode loop, so the session
        reports ``RUNNING`` there."""
        observed: list[GenState] = []

        def _tap(*args: Any, **kwargs: Any) -> None:
            observed.append(session.gen_state)

        session.generate("Say hi.", sampling=_FOUR_TOKENS, on_token=_tap)
        assert observed == [GenState.RUNNING] * 4

    def test_returns_to_idle_after_success(self, session: Any) -> None:
        session.generate("Say hi.", sampling=_FOUR_TOKENS)
        assert session.gen_state is GenState.IDLE
        assert session.is_generating is False

    def test_returns_to_idle_after_exception(self, session: Any) -> None:
        """A failure inside the generation transaction still drains the state
        machine and releases the lock, so the next call succeeds."""
        with pytest.raises(ProfileNotRegisteredError):
            session.generate(
                "Say hi.",
                steering="0.3 definitely_not_a_real_concept_xyz",
                sampling=_FOUR_TOKENS,
            )
        assert session.gen_state is GenState.IDLE
        assert session.is_generating is False

        session.generate("Say hi.", sampling=_FOUR_TOKENS)
        assert session.gen_state is GenState.IDLE


class TestConcurrentGuard:
    def test_concurrent_generate_rejected(self, session: Any) -> None:
        """Re-entering generation from inside ``on_token`` raises
        ``ConcurrentGenerationError``; the outer generation still finishes."""
        captured: list[BaseException] = []

        def _reentry(*args: Any, **kwargs: Any) -> None:
            try:
                session.generate("nope", sampling=SamplingConfig(max_tokens=2))
            except ConcurrentGenerationError as exc:
                captured.append(exc)

        result = session.generate("Say hi.", sampling=_FOUR_TOKENS, on_token=_reentry).first
        assert len(captured) == 4
        assert result.text == "Hi t"
        assert session.gen_state is GenState.IDLE
