"""Tests for the shared server request/response helpers.

These cover the normalizations the three protocols (OpenAI, Ollama, native
WebSocket) share through :mod:`drowse.server.request_helpers` — the single
``SamplingConfig`` construction site — plus the streaming aggregate helpers.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from drowse.server.request_helpers import (
    build_sampling_config,
    logprob_count,
    normalize_stop,
)


@pytest.mark.parametrize("expression", [None, "0.5 local/pirate"])
def test_websocket_done_reports_applied_steering(expression: str | None):
    from drowse.generation.results import GenerationResult
    from drowse.server.ws_models import result_to_json

    result = GenerationResult(
        text="Hello", tokens=[1], token_count=1, tok_per_sec=1, elapsed=1,
        applied_steering=expression,
    )
    assert result_to_json(result)["applied_steering"] == expression


class TestNormalizeStop:
    def test_none_passes_through(self) -> None:
        assert normalize_stop(None) is None

    def test_bare_string_becomes_one_tuple(self) -> None:
        assert normalize_stop("###") == ("###",)

    def test_list_becomes_tuple(self) -> None:
        assert normalize_stop(["a", "b"]) == ("a", "b")

    def test_non_string_members_are_stringified(self) -> None:
        assert normalize_stop(cast(Any, [1, "b"])) == ("1", "b")

    def test_empty_sequence_is_none(self) -> None:
        assert normalize_stop([]) is None
        assert normalize_stop(()) is None


class TestLogprobCount:
    def test_none_disables(self) -> None:
        assert logprob_count(None) is None

    def test_chat_false_disables(self) -> None:
        """chat sends a bool; False means "no logprobs at all"."""
        assert logprob_count(False, 5) is None

    def test_chat_true_takes_top_logprobs(self) -> None:
        assert logprob_count(True, 5) == 5

    def test_chat_true_without_top_logprobs_is_chosen_only(self) -> None:
        assert logprob_count(True, None) == 0

    def test_completions_int_passes_through(self) -> None:
        """completions sends the alternative count directly."""
        assert logprob_count(3) == 3

    def test_completions_zero_is_chosen_only_not_disabled(self) -> None:
        assert logprob_count(0) == 0


class TestBuildSamplingConfig:
    def test_normalizes_stop_and_logprobs_in_one_place(self) -> None:
        sc = build_sampling_config(
            temperature=0.7,
            stop="###",
            logprobs=True,
            top_logprobs=4,
        )
        assert sc.temperature == 0.7
        assert sc.stop == ("###",)
        assert sc.logprobs == 4

    def test_penalty_none_collapses_to_zero(self) -> None:
        sc = build_sampling_config(
            presence_penalty=None,  # pyright: ignore[reportArgumentType]  # protocols hand through a possibly-None number
            frequency_penalty=None,  # pyright: ignore[reportArgumentType]
        )
        assert sc.presence_penalty == 0.0
        assert sc.frequency_penalty == 0.0

    def test_blank_role_labels_become_none(self) -> None:
        sc = build_sampling_config(user_role="", assistant_role="")
        assert sc.user_role is None
        assert sc.assistant_role is None

    def test_defaults_match_engine_neutral_sentinels(self) -> None:
        sc = build_sampling_config()
        assert sc.temperature is None
        assert sc.stop is None
        assert sc.logprobs is None
        assert sc.return_top_k == 0
        assert sc.return_probe_readings is True


class TestWsAdapterSharesTheConstructor:
    def test_ws_params_lower_onto_the_same_config(self) -> None:
        from drowse.server.ws_models import WSSamplingParams
        from drowse.server.ws_models import (
            build_sampling_config as ws_build_sampling_config,
        )

        body = WSSamplingParams(
            temperature=0.5,
            top_k=40,
            stop=["END"],
            return_top_k=3,
            persist_per_layer_scores=True,
        )
        sc = ws_build_sampling_config(body)
        assert sc is not None
        assert sc.temperature == 0.5
        assert sc.top_k == 40
        assert sc.stop == ("END",)
        assert sc.return_top_k == 3
        assert sc.persist_per_layer_scores is True

    def test_none_body_is_none(self) -> None:
        from drowse.server.ws_models import (
            build_sampling_config as ws_build_sampling_config,
        )

        assert ws_build_sampling_config(None) is None


class TestWSInputMessageLabel:
    """The wire message and a loom-derived message dict are the same shape.

    ``build_chat_input`` reads a per-turn ``"label"`` key (rendered into the
    constructed header by the scene stitcher; parity covered in
    tests/model/test_scene_wiring.py), and ``PromptRenderer.prepare_input`` forwards a list input
    to it verbatim — so accepting the field here is what makes the dashboard's
    labelled shadow replay render the prompt it is shadowing.
    """

    def test_label_survives_the_lowering(self) -> None:
        from drowse.server.ws_models import WSInputMessage, build_input

        lowered = build_input([
            WSInputMessage(role="user", content="ahoy", label="captain"),
            WSInputMessage(role="assistant", content="arr"),
        ])
        assert lowered == [
            {"role": "user", "content": "ahoy", "label": "captain"},
            {"role": "assistant", "content": "arr", "label": None},
        ]

    def test_label_is_optional_and_still_rejects_unknown_keys(self) -> None:
        import pydantic

        from drowse.server.ws_models import WSInputMessage

        assert WSInputMessage(role="user", content="hi").label is None
        with pytest.raises(pydantic.ValidationError):
            WSInputMessage(role="user", content="hi", name="old")  # type: ignore[call-arg]

    def test_string_and_none_inputs_pass_through(self) -> None:
        from drowse.server.ws_models import build_input

        assert build_input("plain prompt") == "plain prompt"
        assert build_input(None) is None


class TestWSGenerateSchemaValidation:
    """Mode consistency is a schema property of ``WSGenerateMessage``.

    These rules used to be a hand-rolled pass in the WS handler, so a
    programmatic construction could build a frame the wire would reject.
    """

    @staticmethod
    def _errors(**kwargs: Any) -> list[dict[str, Any]]:
        import pydantic

        from drowse.server.ws_models import WSGenerateMessage

        with pytest.raises(pydantic.ValidationError) as excinfo:
            WSGenerateMessage(type="generate", **kwargs)
        return [dict(error) for error in excinfo.value.errors()]

    def test_fork_requires_its_whole_field_group(self) -> None:
        errors = self._errors(fork_node_id="n1", fork_raw_index=3)
        assert errors[0]["type"] == "fork_fields"
        # ``PydanticCustomError`` keeps the message verbatim — a plain
        # ValueError would reach the wire as ``"Value error, fork ..."``.
        assert errors[0]["msg"].startswith("fork requires ")

    def test_fork_accepts_exactly_one_replacement_kind(self) -> None:
        from drowse.server.ws_models import WSGenerateMessage

        authored = WSGenerateMessage(
            type="generate",
            fork_node_id="n1",
            fork_raw_index=3,
            fork_replacement_text="a longer replacement",
        )
        assert authored.fork_replacement_text == "a longer replacement"
        for kwargs in (
            {
                "fork_node_id": "n1",
                "fork_raw_index": 3,
                "fork_replacement_text": "",
            },
            {
                "fork_node_id": "n1",
                "fork_raw_index": 3,
                "fork_alt_token_id": 7,
                "fork_replacement_text": "replacement",
            },
        ):
            assert self._errors(**kwargs)[0]["type"] == "fork_fields"

    def test_n_is_bounded_on_both_frames(self) -> None:
        import pydantic

        from drowse.server.ws_models import WSGenerateMessage, WSSubmitMessage

        assert self._errors(n=0)[0]["loc"] == ("n",)
        # ``submit`` needs the same bound: ``_normalize_submit`` forwards ``n``
        # into a ``WSGenerateMessage`` construction outside the handler's
        # error-frame guard, so an unbounded submit would close the socket.
        with pytest.raises(pydantic.ValidationError):
            WSSubmitMessage(type="submit", generated_role="assistant", n=0)
        assert WSGenerateMessage(type="generate", n=1).n == 1

    def test_well_formed_modes_construct(self) -> None:
        from drowse.server.ws_models import WSGenerateMessage

        assert WSGenerateMessage(
            type="generate",
            fork_node_id="n1", fork_raw_index=0, fork_alt_token_id=7,
        ).fork_node_id == "n1"

    def test_prefill_fields_are_not_part_of_the_frame(self) -> None:
        errors = self._errors(prefill_node_id="n1", prefill_text="Sure")
        assert {error["type"] for error in errors} == {"extra_forbidden"}


class TestDescribeError:
    """The one classifier: which failures are a client's, an extra's, or a bug."""

    def test_missing_third_party_module_is_a_logged_503(self, caplog: pytest.LogCaptureFixture) -> None:
        from drowse.server.errors import describe_error

        exc = ModuleNotFoundError("No module named 'sae_lens'", name="sae_lens")
        with caplog.at_level("ERROR", logger="drowse.api"):
            info = describe_error(exc)
        assert (info.status, info.message) == (503, "optional dependency not installed: sae_lens")
        assert caplog.records and caplog.records[0].exc_info is not None

    def test_install_hint_reraise_names_the_chained_module(self) -> None:
        from drowse.server.errors import describe_error

        try:
            try:
                raise ModuleNotFoundError("No module named 'pandas'", name="pandas")
            except ImportError as inner:
                raise ImportError("pandas is required: pip install pandas") from inner
        except ImportError as exc:
            info = describe_error(exc)
        assert (info.status, info.message) == (503, "optional dependency not installed: pandas")

    @pytest.mark.parametrize("exc", [
        ImportError("cannot import name 'gone' from 'drowse.lens.sources'", name="drowse.lens.sources"),
        ModuleNotFoundError("No module named 'drowse.moved'", name="drowse.moved"),
        ImportError("no name at all"),
    ])
    def test_internal_import_failure_is_a_logged_500(
        self, exc: ImportError, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Regression: every bare ImportError read as a missing extra (503,
        unlogged), so a moved drowse module blamed the user's install and left
        no traceback anywhere."""
        from drowse.server.errors import describe_error

        with caplog.at_level("ERROR", logger="drowse.api"):
            info = describe_error(exc, failure="Lens fetch failed")
        assert info.status == 500
        assert info.message == "Lens fetch failed. Check the server log for details."
        assert caplog.records and caplog.records[0].exc_info is not None

    def test_request_error_is_a_400_without_an_error_log(self, caplog: pytest.LogCaptureFixture) -> None:
        from drowse.errors import GenerationRequestError
        from drowse.server.errors import describe_error

        with caplog.at_level("DEBUG", logger="drowse.api"):
            info = describe_error(GenerationRequestError("n must be >= 1"), failure="Generation failed")
        assert (info.status, info.message, info.code) == (400, "n must be >= 1", "GenerationRequestError")
        assert not caplog.records

    def test_bare_value_error_is_a_logged_scrubbed_500(self, caplog: pytest.LogCaptureFixture) -> None:
        """A bare ``ValueError`` is an engine invariant: its text (often a
        path or a tensor shape) never reaches the client, and its traceback
        lands in the error log instead of a debug line blaming the client."""
        from drowse.server.errors import describe_error

        with caplog.at_level("DEBUG", logger="drowse.api"):
            info = describe_error(ValueError("/secret/cache: shape (3,) != (4,)"), failure="Generation failed")
        assert (info.status, info.message, info.code) == (
            500, "Generation failed. Check the server log for details.", "ValueError",
        )
        [record] = caplog.records
        assert record.levelname == "ERROR" and record.exc_info is not None

    def test_lock_timeout_is_a_503_without_an_error_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Regression: the timeout chained its ``TimeoutError``, and a chained
        DrowseError logs a traceback, so every expected 503 wrote an ERROR."""
        import asyncio
        from types import SimpleNamespace

        from drowse.server.errors import describe_error
        from drowse.server.runtime import SessionLockTimeout, session_lock

        async def contend() -> BaseException:
            lock = asyncio.Lock()
            await lock.acquire()
            app: Any = SimpleNamespace(state=SimpleNamespace(session_lock=lock))
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("drowse.server.runtime.SESSION_LOCK_TIMEOUT_SECONDS", 0.01)
                with pytest.raises(SessionLockTimeout) as caught:
                    async with session_lock(app):
                        pass
            return caught.value

        exc = asyncio.run(contend())
        assert exc.__cause__ is None
        with caplog.at_level("ERROR", logger="drowse.api"):
            assert describe_error(exc).status == 503
        assert not caplog.records


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__]))


@pytest.mark.parametrize(("model_type", "param_count", "expected"), [
    ("gemma2", 2_614_000_000, ["gemma2:2.6b"]),
    ("qwen3_moe", 30_500_000_000, ["qwen3:30b"]),
    ("llama", 1_000_000_000, ["llama:1b"]),
    ("llama", 360_000_000, ["llama:360m"]),
    ("llama", 500_000, []),
    ("llama", 0, []),
    ("", 2_000_000_000, []),
])
def test_inferred_ollama_aliases(
    model_type: str, param_count: int, expected: list[str],
) -> None:
    from drowse.server.compat.model_names import aliases_for

    info = {"model_type": model_type, "param_count": param_count}
    assert aliases_for("some/unlisted-model", info) == expected
