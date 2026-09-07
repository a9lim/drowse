import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "browser-runtime"
    / "forks"
    / "create-tiny-webgpu-model.py"
)
SPEC = importlib.util.spec_from_file_location("drowse_tiny_webgpu_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parameter(shape: tuple[int, ...], dtype: str):
    return SimpleNamespace(shape=shape, dtype=dtype)


def test_fp32_fixture_has_nonzero_residual_and_unit_norms():
    embedding = MODULE.fixture_parameter(
        "model.embed_tokens.weight", parameter((16, 64), "float32")
    )
    norm = MODULE.fixture_parameter(
        "model.layers.0.input_layernorm.weight", parameter((64,), "float32")
    )

    np.testing.assert_array_equal(embedding[:, 0], np.full(16, 0.5, dtype=np.float32))
    assert embedding[0, 1] == np.float32(-0.25)
    assert embedding[-1, 1] == np.float32(0.25)
    np.testing.assert_array_equal(norm, np.ones(64, dtype=np.float32))


def test_q4_fixture_uses_offset_codes_and_matching_scales():
    weight = MODULE.fixture_parameter(
        "model.embed_tokens.q_weight", parameter((16, 8), "uint32")
    )
    scale = MODULE.fixture_parameter(
        "model.embed_tokens.q_scale", parameter((16, 2), "float16")
    )

    assert np.all(weight[:, 0] == np.uint32(0x777777AE))
    assert np.all(weight[:, 1:] == np.uint32(0x77777777))
    np.testing.assert_allclose(scale, np.float16(0.5 / 7), rtol=0, atol=0)


def test_quantized_linear_fixture_remains_zero():
    weight = MODULE.fixture_parameter(
        "model.layers.0.self_attn.o_proj.q_weight", parameter((64, 8), "uint32")
    )
    scale = MODULE.fixture_parameter(
        "model.layers.0.self_attn.o_proj.q_scale", parameter((64, 2), "float16")
    )

    assert not weight.any()
    assert not scale.any()
