"""Tests for transcript export / import (v2.3 phase 5)."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import pytest

from drowse import (
    LoomTree,
    ProbeRef,
    Recipe,
    SamplingConfig,
    Transcript,
    TranscriptFormatError,
    TranscriptModelMismatch,
    TranscriptProbeDriftError,
    TranscriptTurn,
)


# ---------------------------------------------------------------------------
# Synthetic session stub — avoids loading a real model.
# ---------------------------------------------------------------------------


class _Monitor:
    def __init__(self, probe_names: Iterable[str] = ()) -> None:
        self.probe_names = list(probe_names)
        self.profiles = {name: {0: f"prof:{name}"} for name in self.probe_names}


class _Config:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt


class StubSession:
    """Tiny replacement for DrowseSession exposing the transcript API surface."""

    def __init__(
        self,
        *,
        model_id: str = "m1",
        system_prompt: str | None = None,
        probe_names: Iterable[str] = ("angry.calm", "honest.deceptive"),
        probe_hashes: dict[str, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.tree = LoomTree(model_id=model_id)
        self.config = _Config(system_prompt=system_prompt)
        self.monitor = _Monitor(probe_names=probe_names)
        self._probe_digests = probe_hashes or {
            name: f"hash_of_{name}" for name in self.monitor.probe_names
        }

    def probe_hashes(self) -> dict[str, str]:
        return dict(self._probe_digests)


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


def test_schema_round_trip():
    t = Transcript(
        model_id="google/gemma-3-4b-it",
        system_prompt="You are a helpful assistant.",
        probes=[
            ProbeRef(name="angry.calm", sha256="abc123"),
            ProbeRef(name="honest.deceptive", sha256="def456"),
        ],
        turns=[
            TranscriptTurn(role="user", text="What makes a good day?"),
            TranscriptTurn(
                role="assistant",
                text="Sunshine.",
                recipe=Recipe(
                    steering="0.3 honest.deceptive",
                    sampling=SamplingConfig(temperature=0.7, max_tokens=256, seed=42),
                ),
                readings={"angry.calm": -0.12, "honest.deceptive": 0.41},
            ),
        ],
    )
    y = t.to_yaml()
    t2 = Transcript.from_yaml(y)
    assert t2.model_id == t.model_id
    assert t2.system_prompt == t.system_prompt
    assert len(t2.probes) == 2
    assert t2.probes[0].sha256 == "abc123"
    recipe = t2.turns[1].recipe
    assert recipe is not None
    assert recipe.steering == "0.3 honest.deceptive"
    assert t2.turns[1].readings["angry.calm"] == pytest.approx(-0.12)


@pytest.mark.parametrize("version", [1, 99])
def test_from_yaml_rejects_unsupported_version(version: int):
    with pytest.raises(TranscriptFormatError, match="version"):
        Transcript.from_yaml(f"drowse_transcript: {version}\nturns: []\n")


def test_from_yaml_rejects_non_mapping_root():
    with pytest.raises(TranscriptFormatError, match="must be a mapping"):
        Transcript.from_yaml("- foo\n- bar\n")


def test_save_load_round_trip(tmp_path: Path):
    t = Transcript(
        model_id="m", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="hi"),
            TranscriptTurn(role="assistant", text="hello"),
        ],
    )
    out = tmp_path / "trans.yaml"
    t.save(out)
    assert out.is_file()
    t2 = Transcript.load(out)
    assert len(t2.turns) == 2
    assert t2.turns[0].text == "hi"


# ---------------------------------------------------------------------------
# Build from session path
# ---------------------------------------------------------------------------


def test_from_path_walks_active_path():
    sess = StubSession()
    u = sess.tree.add_user_turn("hi there")
    a = sess.tree.begin_assistant(u, recipe=Recipe(steering="0.3 honest.deceptive"))
    sess.tree.finalize_assistant(
        a, text="hello back",
        aggregate_readings={"angry.calm": 0.1},
    )
    t = Transcript.from_path(None, sess)
    assert t.model_id == "m1"
    roles = [turn.role for turn in t.turns]
    assert roles == ["user", "assistant"]
    assert t.turns[1].readings == {"angry.calm": 0.1}
    assert len(t.probes) == 2
    assert t.probes[0].sha256 == "hash_of_angry.calm"


# ---------------------------------------------------------------------------
# Three import modes
# ---------------------------------------------------------------------------


def _simple_transcript(*, model_id: str = "m1", with_assistant: bool = True) -> Transcript:
    turns = [TranscriptTurn(role="user", text="hi")]
    if with_assistant:
        turns.append(TranscriptTurn(role="assistant", text="hello"))
    turns.append(TranscriptTurn(role="user", text="how"))
    if with_assistant:
        turns.append(TranscriptTurn(role="assistant", text="fine"))
    return Transcript(
        model_id=model_id, system_prompt=None,
        probes=[], turns=turns,
    )


def test_import_default_attaches_under_root():
    sess = StubSession()
    t = _simple_transcript()
    leaf = t.import_into(sess, mode="default")
    # Walk up — should hit root_id without crossing any existing user node.
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert sess.tree.root_id in chain


def test_import_here_attaches_under_active():
    sess = StubSession()
    # Seed an existing path so active != root.
    u0 = sess.tree.add_user_turn("seed")
    a0 = sess.tree.begin_assistant(u0, recipe=Recipe())
    sess.tree.finalize_assistant(a0, text="seed-ans")

    t = _simple_transcript()
    leaf = t.import_into(sess, mode="here")
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert a0 in chain


def test_import_merge_finds_user_prefix():
    sess = StubSession()
    # Pre-populate sess with the same first user-turn the transcript carries.
    u0 = sess.tree.add_user_turn("hi")
    a0 = sess.tree.begin_assistant(u0, recipe=Recipe())
    sess.tree.finalize_assistant(a0, text="prior-answer")

    t = _simple_transcript()
    leaf = t.import_into(sess, mode="merge")
    # Leaf path should descend from u0 (the shared "hi" user turn).
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert u0 in chain


def test_import_merge_skip_walks_existing_prefix_no_dupes():
    """Two-user-deep prefix merge: the imported tail must attach under
    the deepest matched user (u2), and the matched-prefix users (u1, u2)
    must NOT be re-attached as duplicates under it.

    Active path: ``root -> u1 -> a1 -> u2 -> a2``
    Transcript users: ``["u1_text", "u2_text", "u3_text"]`` (assistants
    between the matched users are dropped on the floor, matching the
    rest of the merge mode — the structural location is "already
    represented in the path", so the alternate assistant text would
    land at the wrong depth if attached under the merge anchor).
    Post-merge u3 should attach as a descendant of u2 (the deepest
    matched user), with no u1/u2 duplicates spawned under u2.
    """
    sess = StubSession()
    u1 = sess.tree.add_user_turn("u1_text")
    a1 = sess.tree.begin_assistant(u1, recipe=Recipe())
    sess.tree.finalize_assistant(a1, text="a1_text")
    u2 = sess.tree.add_user_turn("u2_text", parent_id=a1)
    a2 = sess.tree.begin_assistant(u2, recipe=Recipe())
    sess.tree.finalize_assistant(a2, text="a2_text")

    # Transcript: matched-prefix [u1, u2] plus a fresh u3 tail.  No
    # assistants in the matched-prefix region — keeping the structure
    # symmetric with the task description while still exercising the
    # tail-attach path (an assistant follows u3 so we have a real
    # leaf node to assert on).
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="u1_text"),
            TranscriptTurn(role="user", text="u2_text"),
            TranscriptTurn(role="user", text="u3_text"),
            TranscriptTurn(role="assistant", text="a3_text"),
        ],
    )
    leaf = t.import_into(sess, mode="merge")

    # u2's children: must contain a fresh user node "u3_text", with
    # NO duplicate "u1_text" / "u2_text" user nodes — the pre-fix
    # behaviour spawned those as children of the merge anchor when
    # the skip-walk re-called ``add_user_turn`` on matched prefixes.
    u2_children = sess.tree.children(u2)
    user_child_texts = [c.text for c in u2_children if c.role == "user"]
    assert "u3_text" in user_child_texts
    assert "u1_text" not in user_child_texts
    assert "u2_text" not in user_child_texts

    # The leaf assistant ("a3_text") lives at depth ``root -> u1 ->
    # a1 -> u2 -> u3_text -> a3_text`` — every matched-prefix node
    # is in the ancestor chain.
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert u1 in chain
    assert u2 in chain


def test_import_merge_falls_back_to_root_when_no_match():
    sess = StubSession()
    u0 = sess.tree.add_user_turn("different")
    sess.tree.begin_assistant(u0, recipe=Recipe())

    t = _simple_transcript()
    leaf = t.import_into(sess, mode="merge")
    chain = [n.id for n in sess.tree.path_to(leaf)]
    assert sess.tree.root_id in chain


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------


def test_guard_model_mismatch_refuses_merge():
    sess = StubSession(model_id="m1")
    t = _simple_transcript(model_id="m2")
    with (
        pytest.warns(UserWarning, match="differs from session"),
        pytest.raises(TranscriptModelMismatch),
    ):
        t.import_into(sess, mode="merge")


def test_guard_model_mismatch_warns_on_default():
    sess = StubSession(model_id="m1")
    t = _simple_transcript(model_id="m2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("model" in str(w.message) for w in caught)
    # First imported node carries a banner note.
    notes = []
    for n in sess.tree.nodes.values():
        if n.notes:
            notes.append(n.notes)
    assert any("model_mismatch" in n for n in notes)


def test_guard_system_prompt_mismatch_proceeds_with_warning():
    sess = StubSession(system_prompt="be helpful")
    t = Transcript(
        model_id="m1", system_prompt="be sarcastic",
        probes=[],
        turns=[TranscriptTurn(role="user", text="hi"),
               TranscriptTurn(role="assistant", text="ok")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("system" in str(w.message) for w in caught)


def test_guard_missing_probes_warns():
    sess = StubSession(probe_names=("angry.calm",))
    t = Transcript(
        model_id="m1", system_prompt=None,
        probes=[ProbeRef(name="never_loaded", sha256="abc")],
        turns=[TranscriptTurn(role="user", text="hi"),
               TranscriptTurn(role="assistant", text="ok")],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("probes" in str(w.message).lower() for w in caught)


def test_guard_probe_drift_warns_then_strict_raises():
    sess = StubSession(
        probe_names=("angry.calm",),
        probe_hashes={"angry.calm": "session_hash"},
    )
    t = Transcript(
        model_id="m1", system_prompt=None,
        probes=[ProbeRef(name="angry.calm", sha256="transcript_hash")],
        turns=[TranscriptTurn(role="user", text="hi"),
               TranscriptTurn(role="assistant", text="ok")],
    )
    # Non-strict warns + proceeds.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t.import_into(sess, mode="default")
    assert any("drift" in str(w.message).lower() for w in caught)

    # Strict raises.
    sess2 = StubSession(
        probe_names=("angry.calm",),
        probe_hashes={"angry.calm": "session_hash"},
    )
    with (
        pytest.warns(UserWarning, match="drift"),
        pytest.raises(TranscriptProbeDriftError),
    ):
        t.import_into(sess2, mode="default", strict=True)


# ---------------------------------------------------------------------------
# CLI verb dispatch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v2 — the cast model (speaker, cast block, seat-general import)
# ---------------------------------------------------------------------------


def test_v2_round_trip_speaker_and_cast(tmp_path: Path):
    from drowse import CastMember

    t = Transcript(
        model_id="m",
        system_prompt=None,
        probes=[],
        turns=[
            TranscriptTurn(role="user", text="hi", speaker="captain"),
            TranscriptTurn(
                role="assistant", text="rustle", speaker="deer",
                recipe=Recipe(steering="0.5 skittish"),
            ),
            # A generated *user-seat* turn — provenance rides the recipe.
            TranscriptTurn(
                role="user", text="who's there?", speaker="captain",
                recipe=Recipe(steering="0.2 formal"),
            ),
        ],
        cast={
            "deer": CastMember(recipe=Recipe(steering="0.5 skittish")),
            "captain": CastMember(notes="the human's usual mask"),
        },
    )
    out = tmp_path / "v2.yaml"
    t.save(out)
    t2 = Transcript.load(out)
    assert t2.turns[0].speaker == "captain"
    assert t2.turns[2].recipe is not None
    assert t2.cast == t.cast


def test_import_reattaches_generated_user_seat_turn():
    sess = StubSession()
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="hi", speaker="captain"),
            TranscriptTurn(
                role="user", text="hello?", speaker="captain",
                recipe=Recipe(steering="0.2 formal"),
            ),
            TranscriptTurn(
                role="assistant", text="a reply",
                recipe=Recipe(),
            ),
        ],
    )
    leaf = t.import_into(sess, mode="default")
    chain = list(sess.tree.path_to(leaf))
    committed, generated_user, generated_asst = chain[-3], chain[-2], chain[-1]
    assert committed.role == "user" and committed.recipe is None
    assert committed.role_label == "captain"
    # The generated user-seat turn keeps its seat AND its provenance.
    assert generated_user.role == "user"
    assert generated_user.recipe is not None
    assert generated_user.recipe.steering == "0.2 formal"
    assert generated_user.role_label == "captain"
    assert generated_asst.role == "assistant"


def test_import_merges_cast_and_flags_conflicts():
    from drowse import CastMember

    sess = StubSession()
    sess.tree.set_cast_member(
        "deer", CastMember(recipe=Recipe(steering="0.9 bold")),
    )
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[TranscriptTurn(role="user", text="hi")],
        cast={
            "deer": CastMember(recipe=Recipe(steering="0.5 skittish")),
            "narrator": CastMember(notes="stage voice"),
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        leaf = t.import_into(sess, mode="default")
    # Missing member lands; conflicting one keeps the session's version.
    assert "narrator" in sess.tree.cast
    assert sess.tree.cast["deer"].recipe is not None
    assert sess.tree.cast["deer"].recipe.steering == "0.9 bold"
    assert any("cast differs" in str(w.message) for w in caught)
    # Single-turn transcript: the leaf IS the first imported node, which
    # carries the guard notes.
    imported_node = sess.tree.path_to(leaf)[-1]
    assert "cast_conflict" in imported_node.notes


# ---------------------------------------------------------------------------
# Strict reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("turn, match", [
    ({"text": "hi"}, "missing: role"),
    ({"role": "user"}, "missing: text"),
    ({"role": "tool", "text": "x"}, "role must be"),
    ({"role": "user", "text": 3}, "text must be a string"),
    ({"role": "user", "text": "x", "extra": 1}, "unknown keys: extra"),
    ({"role": "user", "text": "x", "speaker": "Captain Hook"}, "speaker"),
    ({"role": "user", "text": "x", "thinking": 7}, "thinking must be a string"),
    ({"role": "assistant", "text": "x", "readings": {"a": "high"}}, "finite numbers"),
    ({"role": "assistant", "text": "x", "readings": {"a": float("nan")}}, "finite numbers"),
    ({"role": "assistant", "text": "x", "readings": {"a": True}}, "finite numbers"),
    ({"role": "assistant", "text": "x", "recipe": {"steering": None}}, "recipe"),
])
def test_turn_reader_is_strict(turn: dict, match: str):
    """Regression: missing roles defaulted to "user" and bad readings were
    silently dropped."""
    with pytest.raises(TranscriptFormatError, match=match):
        Transcript.from_dict({"drowse_transcript": 2, "turns": [turn]})


@pytest.mark.parametrize("extra, match", [
    ({"surprise": 1}, "unknown keys: surprise"),
    ({"cast": {"Bad Label": {"recipe": None, "notes": ""}}}, "cast.Bad Label"),
    ({"cast": {"deer": {"recipe": None, "notes": "", "origin": "observed"}}}, "origin"),
    ({"cast": {"deer": "skittish"}}, "must be a mapping"),
    ({"probes": [{"name": "calm"}]}, "missing: sha256"),
    ({"model_id": 3}, "model_id must be a string"),
])
def test_root_reader_is_strict(extra: dict, match: str):
    with pytest.raises(TranscriptFormatError, match=match):
        Transcript.from_dict({"drowse_transcript": 2, "turns": [], **extra})


def test_minimal_transcript_reads_with_optional_header_fields_absent():
    t = Transcript.from_dict({
        "drowse_transcript": 2,
        "turns": [{"role": "user", "text": "hi"}],
    })
    assert t.model_id is None and t.probes == [] and t.cast == {}


# ---------------------------------------------------------------------------
# Atomic, single-event import
# ---------------------------------------------------------------------------


def _event_session():
    from drowse.events import EventBus, LoomMutated

    sess = StubSession()
    events: list[LoomMutated] = []
    bus = EventBus()
    bus.subscribe(lambda e: events.append(e) if isinstance(e, LoomMutated) else None)
    sess.tree = LoomTree(model_id="m1", events=bus)
    return sess, events


def test_import_is_one_mutation_with_cast_additions():
    from drowse import CastMember

    sess, events = _event_session()
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="hi", speaker="captain"),
            TranscriptTurn(role="assistant", text="ahoy", recipe=Recipe(steering="0.3 calm"),
                           readings={"angry.calm": 0.2}),
            TranscriptTurn(role="user", text="more"),
        ],
        cast={"narrator": CastMember(notes="stage voice")},
    )
    leaf = t.import_into(sess)
    (event,) = events
    assert event.op == "import"
    assert event.rev == sess.tree.rev == 1
    assert event.added == tuple(n.id for n in sess.tree.path_to(leaf)[1:])
    assert event.active_node_id == leaf == sess.tree.active_node_id
    assert "narrator" in sess.tree.cast
    user, assistant, _ = sess.tree.path_to(leaf)[1:]
    assert user.role_label == "captain"
    assert assistant.applied_steering == "0.3 calm"
    assert assistant.aggregate_readings == {"angry.calm": 0.2}
    assert assistant.tokens is None and assistant.thinking_tokens is None


def test_failed_import_leaves_tree_and_cast_untouched():
    """Regression: a bad turn partway through used to leave a partial branch
    and a partial cast behind."""
    from drowse import CastMember
    from drowse.loom.errors import InvalidNodeOperationError

    sess, events = _event_session()
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="hi"),
            TranscriptTurn(role="assistant", text="x", speaker="Not A Slug"),
        ],
        cast={"narrator": CastMember(notes="stage voice")},
    )
    before = (sess.tree.rev, dict(sess.tree.nodes), dict(sess.tree.cast))
    with pytest.raises(InvalidNodeOperationError):
        t.import_into(sess)
    assert (sess.tree.rev, dict(sess.tree.nodes), dict(sess.tree.cast)) == before
    assert events == []


def test_merge_with_nothing_new_notes_the_anchor_in_one_update():
    sess, events = _event_session()
    u1 = sess.tree.add_user_turn("hi")
    u2 = sess.tree.add_user_turn("how", parent_id=u1)
    events.clear()
    t = Transcript(
        model_id="m2", system_prompt=None, probes=[],
        turns=[TranscriptTurn(role="user", text="hi"), TranscriptTurn(role="user", text="how")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        leaf = t.import_into(sess, mode="here")
    assert leaf == u2
    (event,) = events
    assert event.added == () and event.updated == (u2,)
    assert "model_mismatch" in sess.tree.get(u2).notes


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class _ReplaySession(StubSession):
    """Stub whose ``generate`` lands a finalized sibling, like the session."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def generate(self, input, **kwargs):
        from drowse.generation.results import GenerationResult, RunSet

        assert input is None
        self.calls.append(kwargs)
        node_id = self.tree.begin_assistant(
            kwargs["parent_node_id"], seat=kwargs["gen_seat"],
            recipe=Recipe(steering=kwargs["steering"], sampling=kwargs["sampling"]),
        )
        self.tree.finalize_assistant(
            node_id, text="Rain today.", aggregate_readings={"angry.calm": 0.1},
        )
        result = GenerationResult(
            text="Rain today.", tokens=[], token_count=0, tok_per_sec=0.0, elapsed=0.0,
        )
        return RunSet([result], node_ids=[node_id], kind="generation")


def test_replay_regenerates_each_generated_turn_under_its_history():
    """Regression: ``experiment transcript run`` replayed each user turn
    statelessly, without the conversation before it."""
    sess = _ReplaySession()
    sampling = SamplingConfig(temperature=0.7, seed=42)
    t = Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="Weather?"),
            TranscriptTurn(role="assistant", text="Sunshine.",
                           recipe=Recipe(steering="0.3 calm", sampling=sampling, thinking=False,
                                         system_prompt="Be brief."),
                           readings={"angry.calm": -0.12}),
            TranscriptTurn(role="user", text="Tomorrow?"),
            TranscriptTurn(role="assistant", text="Clouds.", recipe=Recipe(sampling=sampling)),
        ],
    )
    out = t.replay(sess)
    path = sess.tree.path_to(out.leaf_id)
    recorded = [n for n in path if n.recipe is not None]
    assert [step.recorded_id for step in out.steps] == [n.id for n in recorded]
    first, second = sess.calls
    # Each regeneration is a sibling under the recorded turn's parent, so
    # the model sees every earlier turn of the transcript.
    assert first["parent_node_id"] == recorded[0].parent_id
    assert second["parent_node_id"] == recorded[1].parent_id
    assert sess.tree.get(second["parent_node_id"]).text == "Tomorrow?"
    assert first["steering"] == "0.3 calm"
    assert second["steering"] == ""          # unsteered stays unsteered
    assert first["sampling"] == sampling
    assert first["thinking"] is False
    assert first["system_prompt"] == "Be brief."
    assert first["append_same_role"] is False
    assert first["gen_seat"] == "assistant"
    step = out.steps[0]
    assert sess.tree.get(step.replay_id).parent_id == recorded[0].parent_id
    (delta,) = step.diff.readings
    assert delta.name == "angry.calm"
    assert delta.delta == pytest.approx(0.1 - (-0.12))
    assert out.failures == []


def _two_turn_transcript() -> Transcript:
    sampling = SamplingConfig(seed=1)
    return Transcript(
        model_id="m1", system_prompt=None, probes=[],
        turns=[
            TranscriptTurn(role="user", text="Weather?"),
            TranscriptTurn(role="assistant", text="Sunshine.",
                           recipe=Recipe(steering="0.3 gone", sampling=sampling)),
            TranscriptTurn(role="user", text="Tomorrow?"),
            TranscriptTurn(role="assistant", text="Clouds.", recipe=Recipe(sampling=sampling)),
        ],
    )


class _FailingReplaySession(_ReplaySession):
    """Refuses any turn whose steering names the uninstalled ``gone`` vector."""

    def generate(self, input, **kwargs):
        from drowse.steering.grammar import SteeringExprError

        if "gone" in (kwargs.get("steering") or ""):
            raise SteeringExprError("unknown steering term 'gone'")
        return super().generate(input, **kwargs)


def test_replay_is_fail_fast_by_default():
    from drowse.steering.grammar import SteeringExprError

    with pytest.raises(SteeringExprError):
        _two_turn_transcript().replay(_FailingReplaySession())


def test_isolated_replay_records_a_failed_turn_and_keeps_going():
    sess = _FailingReplaySession()
    out = _two_turn_transcript().replay(sess, isolate=True)
    recorded = [n for n in sess.tree.path_to(out.leaf_id) if n.recipe is not None]
    (failure,) = out.failures
    assert failure.recorded_id == recorded[0].id
    assert "gone" in str(failure.error)
    assert [step.recorded_id for step in out.steps] == [recorded[1].id]
    assert sess.tree.get(out.steps[0].replay_id).text == "Rain today."


def test_isolated_replay_still_propagates_non_drowse_errors():
    class _Broken(_ReplaySession):
        def generate(self, input, **kwargs):
            raise RuntimeError("bug, not a user error")

    with pytest.raises(RuntimeError, match="bug"):
        _two_turn_transcript().replay(_Broken(), isolate=True)
