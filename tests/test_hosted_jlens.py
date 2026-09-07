from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "browser-runtime"))

from fit_hosted_jlens import _capture_rows, _load_words, _stabilize_tensor_names


def test_capture_rows_require_one_contiguous_exact_closure() -> None:
    rows = _capture_rows(
        [
            {
                "input_ids": list(range(18)),
                "byte_offset": 0,
                "byte_length": 2 * 18 * 3 * 4,
            },
            {
                "input_ids": list(range(20)),
                "byte_offset": 2 * 18 * 3 * 4,
                "byte_length": 2 * 20 * 3 * 4,
            },
        ],
        2 * (18 + 20) * 3 * 4,
        layer_count=2,
        hidden_size=3,
        sequence_limit=20,
        skip_first=16,
    )
    assert [row["byte_offset"] for row in rows] == [0, 432]


@pytest.mark.parametrize(
    "field,value",
    [
        ("byte_offset", 4),
        ("byte_length", 8),
        ("input_ids", list(range(17))),
    ],
)
def test_capture_rows_reject_invalid_extent_or_short_prompt(
    field: str, value: object,
) -> None:
    row = {
        "input_ids": list(range(18)),
        "byte_offset": 0,
        "byte_length": 2 * 18 * 3 * 4,
    }
    row[field] = value
    with pytest.raises(ValueError, match="row 0"):
        _capture_rows(
            [row],
            2 * 18 * 3 * 4,
            layer_count=2,
            hidden_size=3,
            sequence_limit=20,
            skip_first=16,
        )


def test_words_must_be_unique_trimmed_strings(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text('["ocean", "calm"]')
    assert _load_words(valid) == ["ocean", "calm"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text('["ocean", "ocean"]')
    with pytest.raises(ValueError, match="unique trimmed"):
        _load_words(invalid)


def test_tensor_generation_names_are_stabilized(tmp_path: Path) -> None:
    (tmp_path / "random-a.safetensors").write_bytes(b"a")
    (tmp_path / "random-b.safetensors").write_bytes(b"b")
    manifest = {
        "source_layers": [2, 4],
        "tensor_files": {"2": "random-a.safetensors", "4": "random-b.safetensors"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _stabilize_tensor_names(tmp_path)
    rewritten = json.loads((tmp_path / "manifest.json").read_text())
    assert rewritten["tensor_files"] == {
        "2": "jlens.layer-2.safetensors",
        "4": "jlens.layer-4.safetensors",
    }
    assert (tmp_path / "jlens.layer-2.safetensors").read_bytes() == b"a"
    assert (tmp_path / "jlens.layer-4.safetensors").read_bytes() == b"b"
