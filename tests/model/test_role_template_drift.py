"""Real-tokenizer drift tests for :data:`drowse.model.role_templates.SEAT_HEADERS`.

Complement to ``test_role_templates.py``.  That file uses synthetic Jinja
templates that mirror the *registered* per-family chat-template shapes,
so it validates the splice machinery against the assumed format.  This
file validates the *assumption itself* — that what we have registered
actually matches what the live tokenizer emits.

Catches the failure mode we hit when ``google/gemma-4-31b-it`` shipped a
new turn-boundary scheme (``<|turn>...<turn|>`` instead of the prior
``<start_of_turn>...<end_of_turn>``) and the registry's ``RoleHeader``
silently broke role substitution at extraction time.  Without this test,
the next chat-template drift will fail in user code rather than at
test-time.

Per-family tokenizers are tiny config-file downloads (no model weights),
but some require HF auth + license acceptance (Gemma, Llama).  Ordinary
test runs use only the local Hugging Face cache so CI never blocks on an
external service.  Set ``DROWSE_TEST_LIVE_TOKENIZERS=1`` to permit downloads
and refresh every family before a release.
"""
from __future__ import annotations

import functools
import os
from typing import Any

import pytest

from drowse.model.role_templates import SEAT_HEADERS, resolve_seat_headers

# Representative tokenizer per family.  Picked smallest available so the
# download is cheap.  Public/no-auth where possible; the gated ones
# (gemma, llama) skip per-family on download failure.
#
# When a new family is added to SEAT_HEADERS, add an entry here too.
# A family with a registered RoleHeader but no entry in this table
# is treated as untested and flagged by ``test_every_supported_family_is_tested``.
_REPRESENTATIVE_TOKENIZERS: dict[str, str] = {
    "qwen2": "Qwen/Qwen2-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3_text": "Qwen/Qwen3-0.6B",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_5": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_text": "Qwen/Qwen3.5-0.8B",
    "qwen3_5_moe": "Qwen/Qwen3.5-35B-A3B",
    "gemma2": "google/gemma-2-2b-it",
    "gemma3": "google/gemma-3-1b-it",
    "gemma3_text": "google/gemma-3-1b-it",
    "gemma4": "google/gemma-4-31b-it",
    "gemma4_text": "google/gemma-4-31b-it",
    "gemma4_unified": "google/gemma-4-12b-it",
    "gemma4_unified_text": "google/gemma-4-12b-it",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "glm": "THUDM/glm-4-9b-chat",
    "gpt_oss": "openai/gpt-oss-20b",
    # The HF-compatible fork ships a fast tokenizer.json + chat_template.jinja
    # that load without trust_remote_code; the upstream talkie-lm repo can't
    # instantiate its backend tokenizer.
    "talkie": "a9lim/talkie-1930-13b-it-hf-cached",
}


def _supported_families() -> list[str]:
    """Families with a non-None RoleHeader in the registry."""
    return [k for k, v in SEAT_HEADERS.items() if v.assistant is not None]


def test_every_supported_family_is_tested():
    """Every family with a registered RoleHeader has a representative
    tokenizer in this file.  If this fails after a registry edit, add the
    new family to ``_REPRESENTATIVE_TOKENIZERS``.
    """
    missing = [f for f in _supported_families() if f not in _REPRESENTATIVE_TOKENIZERS]
    assert not missing, (
        f"registered families lack a representative tokenizer for "
        f"drift testing: {missing}"
    )


@functools.cache
def _load_tokenizer(model_id: str) -> Any:
    """One tokenizer load per checkpoint (families share several)."""
    from transformers import AutoTokenizer

    allow_download = os.environ.get("DROWSE_TEST_LIVE_TOKENIZERS") == "1"
    return AutoTokenizer.from_pretrained(model_id, local_files_only=not allow_download)


@pytest.mark.parametrize("family", _supported_families())
def test_role_header_matches_live_template(family: str):
    """The registered ``RoleHeader`` splice key is present in the
    family's rendered chat template.

    The splice key is ``f"{before}{label}{after}"`` — the literal byte
    sequence that ``apply_with_role`` looks for in the rendered template
    and replaces ``label`` within.  If that key isn't found, role
    substitution can't fire and extraction/manifold-fit silently fail
    (or raise :class:`RoleTemplateDriftError` at runtime).
    """
    pytest.importorskip("transformers")

    model_id = _REPRESENTATIVE_TOKENIZERS[family]
    try:
        tok = _load_tokenizer(model_id)
    except Exception as e:
        pytest.skip(
            f"could not load tokenizer for {family} ({model_id}): {e}. "
            "set DROWSE_TEST_LIVE_TOKENIZERS=1 (plus HF_TOKEN and any gated "
            "license acceptance) to download and run this family."
        )
        return

    if tok.chat_template is None:
        pytest.skip(
            f"{model_id} has no chat_template — role substitution doesn't "
            f"apply to base/completion models"
        )

    rendered = tok.apply_chat_template(
        [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
        tokenize=False,
    )

    seats = resolve_seat_headers(tok, family)
    assert seats is not None and seats.assistant is not None
    header = seats.assistant
    splice_key = f"{header.before}{header.label}{header.after}"

    assert splice_key in rendered, (
        f"{family} ({model_id}): registered splice key {splice_key!r} "
        f"not found in rendered chat template. The chat template likely "
        f"changed upstream — update SEAT_HEADERS[{family!r}] in "
        f"drowse/model/role_templates.py to match. Rendered template "
        f"sample:\n{rendered[:300]!r}"
    )
