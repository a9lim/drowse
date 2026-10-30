"""Tiny real Hugging Face causal LMs for CPU tests.

Every model here is a genuine ``transformers`` architecture built with
``from_config`` (random weights, 4 layers x 32 hidden), paired with a real
``PreTrainedTokenizerFast`` and a chat template in the family's real wire
format.  Drowse's loader, scene grammar, role substitution, capture, steering
hooks and decode loop all run unmodified against them; only the weights are
meaningless.

The tokenizer is byte-level BPE with **no merges**: one token per UTF-8 byte,
plus the family's special tokens.  It round-trips any text exactly (the neutral
corpus has curly quotes and em dashes), and a non-ASCII character spans
several tokens, which is what exercises the decode loop's partial-UTF-8
buffering.

:class:`ScriptedReplies` makes output deterministic without faking the model:
a forward hook overwrites the last position's logits so the model spells a
scripted reply after each turn header, then the family's end-of-turn token.
Hidden states (and therefore steering, probes and capture) stay real.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import transformers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

__all__ = [
    "FAMILIES",
    "ChatFamily",
    "ScriptedReplies",
    "byte_tokenizer",
    "tiny_model",
    "write_tiny_checkpoint",
]


@dataclass(frozen=True)
class ChatFamily:
    """One chat wire format plus the architecture it ships on.

    ``header_open`` is the special token that starts every turn header and
    ``header_close`` the text that ends the role label; a reply starts right
    after it.  ``eos`` is the end-of-turn token generation stops on.
    """

    name: str
    model_type: str
    chat_template: str
    pad: str
    eos: str
    header_open: str
    header_close: str
    bos: str | None = None
    extra_specials: tuple[str, ...] = ()
    config_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def specials(self) -> tuple[str, ...]:
        bos = (self.bos,) if self.bos else ()
        ordered = (self.pad, *bos, self.eos, self.header_open, *self.extra_specials)
        return tuple(dict.fromkeys(ordered))


_LLAMA3_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)

_CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Gemma folds a system message into the first user turn and labels the
# assistant seat ``model``.
_GEMMA_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set prefix = messages[0]['content'] + '\n\n' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set prefix = '' %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for m in loop_messages %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "<start_of_turn>{{ role }}\n"
    "{{ prefix if loop.first else '' }}{{ m['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

FAMILIES: dict[str, ChatFamily] = {
    "llama": ChatFamily(
        name="llama",
        model_type="llama",
        chat_template=_LLAMA3_TEMPLATE,
        pad="<|finetune_right_pad_id|>",
        bos="<|begin_of_text|>",
        eos="<|eot_id|>",
        header_open="<|start_header_id|>",
        header_close="<|end_header_id|>\n\n",
        extra_specials=("<|end_header_id|>",),
    ),
    "qwen3": ChatFamily(
        name="qwen3",
        model_type="qwen3",
        chat_template=_CHATML_TEMPLATE,
        pad="<|endoftext|>",
        eos="<|im_end|>",
        header_open="<|im_start|>",
        header_close="\n",
    ),
    "gemma3_text": ChatFamily(
        name="gemma3_text",
        model_type="gemma3_text",
        chat_template=_GEMMA_TEMPLATE,
        pad="<pad>",
        bos="<bos>",
        eos="<end_of_turn>",
        header_open="<start_of_turn>",
        header_close="\n",
        config_overrides={"query_pre_attn_scalar": 8, "sliding_window": 4096},
    ),
}


def _family(family: str | ChatFamily) -> ChatFamily:
    return family if isinstance(family, ChatFamily) else FAMILIES[family]


def byte_tokenizer(family: str | ChatFamily = "llama") -> PreTrainedTokenizerFast:
    """A byte-level, merge-free tokenizer carrying ``family``'s chat template.

    Ids ``0..len(specials)-1`` are the family's special tokens (pad first);
    the 256 byte symbols follow.
    """
    spec = _family(family)
    vocab = {token: index for index, token in enumerate(spec.specials)}
    for symbol in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[symbol] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False,
    )
    backend.decoder = decoders.ByteLevel()
    named = {"pad": spec.pad, "eos": spec.eos, "bos": spec.bos}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=spec.pad,
        eos_token=spec.eos,
        bos_token=spec.bos,
        additional_special_tokens=[
            token for token in spec.specials if token not in named.values()
        ],
    )
    tokenizer.chat_template = spec.chat_template
    return tokenizer


def tiny_model(
    family: str | ChatFamily = "llama",
    *,
    tokenizer: PreTrainedTokenizerFast | None = None,
    num_layers: int = 4,
    hidden_size: int = 32,
    seed: int = 0,
    model_id: str = "drowse-tests/tiny",
) -> Any:
    """A randomly initialized ``family`` causal LM sized to ``tokenizer``.

    Eager attention, eval mode, grads off.  ``model_id`` lands in
    ``config._name_or_path`` (drowse keys per-model artifacts on it).
    """
    spec = _family(family)
    tok = tokenizer if tokenizer is not None else byte_tokenizer(spec)
    config = transformers.AutoConfig.for_model(
        spec.model_type,
        vocab_size=len(tok),
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        max_position_embeddings=4096,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        tie_word_embeddings=False,
        **spec.config_overrides,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.eval().requires_grad_(False)
    model.config._name_or_path = model_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.bos_token_id = tok.bos_token_id
    return model


def write_tiny_checkpoint(
    path: Path, family: str | ChatFamily = "llama", **model_kwargs: Any,
) -> Path:
    """Save a tiny model + tokenizer as a local HF checkpoint at ``path``.

    Loading it back through ``DrowseSession.from_pretrained(str(path))``
    exercises the real loader, which also stamps the source fingerprint the
    per-model artifact caches (neutral activations, fits) key on.
    """
    spec = _family(family)
    tokenizer = byte_tokenizer(spec)
    model = tiny_model(spec, tokenizer=tokenizer, model_id=str(path), **model_kwargs)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


ReplySource = str | Sequence[str] | Callable[[str, str], str]


class ScriptedReplies:
    """Force a model to spell scripted replies after each turn header.

    A forward hook on the top-level model rewrites the **last position's**
    logits of every row to a one-hot choice: inside a turn whose header label
    is not ``system``, the next token of the scripted reply, then the family's
    end-of-turn token.  Positions before a header (or after a finished turn)
    are left alone.  Teacher-forced positions other than the last are never
    touched, and hidden states are always the model's own.

    ``replies`` is a string (every turn), a sequence (the n-th non-system,
    non-user turn of the sequence gets ``replies[n]``, the last entry
    repeating), or a callable ``(label, prompt_text) -> reply``.

    ``prompts`` records the decoded input of every forward that started from
    an empty cache: the exact rendered prompts the model saw.

    Sequence tracking follows the KV cache length, so it assumes forwards
    that resume a cache continue the sequence the hook saw last (true for the
    decode loop; a prefix-cache entry reused after unrelated forwards is not
    tracked).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        replies: ReplySource,
        *,
        family: str | ChatFamily = "llama",
    ) -> None:
        spec = _family(family)
        self._model = model
        self._tokenizer = tokenizer
        self._replies = replies
        self._open_id = int(tokenizer.convert_tokens_to_ids(spec.header_open))
        self._close_ids = list(tokenizer.encode(spec.header_close, add_special_tokens=False))
        self._eos_id = int(tokenizer.convert_tokens_to_ids(spec.eos))
        self._seqs: list[list[int]] = []
        self._reply_cache: dict[tuple[int, ...], list[int]] = {}
        self.prompts: list[str] = []
        self._handle: Any = None

    def install(self) -> "ScriptedReplies":
        if self._handle is None:
            self._handle = self._model.register_forward_hook(
                self._hook, with_kwargs=True,
            )
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ScriptedReplies":
        return self.install()

    def __exit__(self, *_exc: Any) -> None:
        self.remove()

    # -- internals ---------------------------------------------------------

    def _reply_for(self, label: str, prompt_ids: list[int], turn: int) -> list[int]:
        key = tuple(prompt_ids)
        cached = self._reply_cache.get(key)
        if cached is not None:
            return cached
        source = self._replies
        if isinstance(source, str):
            text = source
        elif callable(source):
            text = source(label, self._tokenizer.decode(prompt_ids))
        else:
            text = source[min(turn, len(source) - 1)]
        ids = list(self._tokenizer.encode(text, add_special_tokens=False))
        self._reply_cache[key] = ids
        return ids

    def _find(self, seq: list[int], needle: list[int], start: int) -> int:
        width = len(needle)
        for index in range(start, len(seq) - width + 1):
            if seq[index:index + width] == needle:
                return index
        return -1

    def _target(self, seq: list[int]) -> int | None:
        opens = [i for i, tid in enumerate(seq) if tid == self._open_id]
        if not opens:
            return None
        head = opens[-1]
        close = self._find(seq, self._close_ids, head + 1)
        if close < 0:
            return None  # still inside the header
        start = close + len(self._close_ids)
        if self._eos_id in seq[start:]:
            return None  # this turn already ended
        label = self._tokenizer.decode(seq[head + 1:close]).strip()
        if label == "system":
            return None
        turn = 0
        for earlier in opens[:-1]:
            earlier_close = self._find(seq, self._close_ids, earlier + 1)
            earlier_label = (
                self._tokenizer.decode(seq[earlier + 1:earlier_close]).strip()
                if earlier_close >= 0 else ""
            )
            if earlier_label not in ("system", "user"):
                turn += 1
        reply = self._reply_for(label, seq[:start], turn)
        emitted = len(seq) - start
        return reply[emitted] if emitted < len(reply) else self._eos_id

    def _hook(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        logits = getattr(output, "logits", None)
        if not isinstance(input_ids, torch.Tensor) or logits is None:
            return output
        width = int(input_ids.shape[1])
        cache = getattr(output, "past_key_values", None) or kwargs.get("past_key_values")
        cached = int(cache.get_seq_length()) if cache is not None else width
        past = max(cached - width, 0)
        rows = input_ids.tolist()
        if past == 0 or len(self._seqs) != len(rows):
            self._seqs = [[] for _ in rows]
        for row, new_ids in enumerate(rows):
            seq = self._seqs[row][:past] + new_ids
            self._seqs[row] = seq
            if past == 0:
                self.prompts.append(self._tokenizer.decode(seq))
            target = self._target(seq)
            if target is None:
                continue
            last = logits[row, -1]
            last.fill_(-30.0)
            last[target] = 30.0
        return output
