from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch


_RUNTIME_ROOT = Path(__file__).parents[1] / "browser-runtime"
_MLC_SPEC = importlib.util.spec_from_file_location("mlc_q4_torch", _RUNTIME_ROOT / "mlc_q4_torch.py")
assert _MLC_SPEC is not None and _MLC_SPEC.loader is not None
_MLC_MODULE = importlib.util.module_from_spec(_MLC_SPEC)
sys.modules["mlc_q4_torch"] = _MLC_MODULE
_MLC_SPEC.loader.exec_module(_MLC_MODULE)
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_mlc_q4_parity", _RUNTIME_ROOT / "validate_mlc_q4_parity.py"
)
assert _VALIDATOR_SPEC is not None and _VALIDATOR_SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)


def capture_metadata(
    model_id: str = "gemma3-270m-instruct",
) -> dict[str, Any]:
    metadata = {
        "schema_version": 3,
        "produced_at": "2026-08-30T20:00:00.000Z",
        "model_id": model_id,
        "runtime_identity_sha256": "1" * 64,
        "runtime_lock": {
            "source": "repository",
            "sha256": "4" * 64,
            "identity_sha256": "5" * 64,
        },
        "layer_map": [0, 1],
        "hidden_size": 2,
        "input_ids": [1, 2],
        "positions": [1],
        "residual_sha256": "2" * 64,
        "rank_one_control_residual_sha256": "6" * 64,
        "steered_residual_sha256": "3" * 64,
        "rank_one_program": {
            "abiVersion": 1,
            "enabledLayer": 1,
            "direction": [0.6, 0.8],
            "target": 1.25,
            "along": 0.4,
            "collapse": 0.3,
        },
        "probe_measurements": [0.1, 0.2],
        "greedy": {
            "prompt": "hello",
            "input_ids": [1],
            "token_ids": [],
            "max_tokens": 4,
        },
    }
    if model_id == "smollm2-360m-instruct":
        metadata["named_role_capture"] = {
            "input_ids": list(_VALIDATOR.SMOL_NAMED_ROLE_INPUT_IDS),
            "position": 19,
        }
    return metadata


def test_capture_metadata_accepts_early_eos_and_exact_smol_named_role(
    tmp_path: Path,
) -> None:
    metadata = capture_metadata("smollm2-360m-instruct")
    (tmp_path / "capture.json").write_text(json.dumps(metadata))
    assert _VALIDATOR.load_capture_metadata(tmp_path) == metadata

    metadata["named_role_capture"]["input_ids"][-1] += 1
    (tmp_path / "capture.json").write_text(json.dumps(metadata))
    try:
        _VALIDATOR.load_capture_metadata(tmp_path)
    except ValueError as error:
        assert "pinned Python row" in str(error)
    else:
        raise AssertionError("mismatched named-role tokenization was accepted")


def test_smol_named_role_is_required_only_for_release_evidence(tmp_path: Path) -> None:
    metadata = capture_metadata("smollm2-360m-instruct")
    del metadata["named_role_capture"]
    (tmp_path / "capture.json").write_text(json.dumps(metadata))
    assert _VALIDATOR.load_capture_metadata(tmp_path) == metadata
    try:
        _VALIDATOR.load_capture_metadata(tmp_path, require_smol_named_role=True)
    except ValueError as error:
        assert "release evidence requires" in str(error)
    else:
        raise AssertionError("release evidence accepted a missing named-role row")


@pytest.mark.parametrize("model_id", ["gemma3-1b-instruct", "gemma3-4b-instruct"])
def test_gemma_parity_does_not_masquerade_as_270m_release_evidence(tmp_path: Path, model_id: str) -> None:
    metadata = capture_metadata(model_id)
    (tmp_path / "capture.json").write_text(json.dumps(metadata))
    assert _VALIDATOR.load_capture_metadata(tmp_path) == metadata
    try:
        _VALIDATOR.release_evidence_receipt(
            {"modelId": model_id}, metadata,
            "gemmaProductionQ4Parity", "candidate/capture.json",
        )
    except ValueError as error:
        assert "requires gemma3-270m-instruct" in str(error)
    else:
        raise AssertionError(f"{model_id} parity was accepted as 270M release evidence")


def test_cosine_stays_in_range_despite_float32_reduction_rounding() -> None:
    values = np.random.default_rng(7).normal(size=4096).astype(np.float32)
    assert 1 - 1e-12 <= _VALIDATOR.cosine(values, values) <= 1
    assert -1 <= _VALIDATOR.cosine(values, -values) <= -1 + 1e-12
    with pytest.raises(ValueError, match="no measurable residual delta"):
        _VALIDATOR.cosine(values, np.zeros_like(values))


def test_greedy_token_ids_honors_multi_eos_ids() -> None:
    class EarlyEosModel:
        config = types.SimpleNamespace(eos_token_id=[2, 106])

        def __call__(
            self,
            tokens: torch.Tensor,
            use_cache: bool = False,
        ) -> types.SimpleNamespace:
            assert use_cache is False
            logits = torch.zeros((1, tokens.shape[1], 128))
            logits[0, -1, 106] = 1
            return types.SimpleNamespace(logits=logits)

    assert _VALIDATOR.greedy_token_ids(EarlyEosModel(), [1], 4, "cpu") == []


def test_probe_error_is_normalized_to_the_activation_not_a_near_zero_projection() -> None:
    actual = np.asarray([0.0, 10.0], dtype=np.float32)
    expected = np.asarray([0.2, 10.1], dtype=np.float32)
    residuals = np.asarray([[100.0, 0.0], [100.0, 0.0]], dtype=np.float32)
    errors = _VALIDATOR.probe_relative_error_by_layer(
        actual,
        expected,
        residuals,
        residuals,
        np.asarray([1.0, 0.0], dtype=np.float32),
        [0, 1],
    )

    assert abs(errors["0"] - 0.002) < 1e-8
    assert abs(errors["1"] - 0.001) < 1e-8
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) > 0.01


def test_probe_error_accounts_for_direction_scale() -> None:
    errors = _VALIDATOR.probe_relative_error_by_layer(
        np.asarray([2.0], dtype=np.float32),
        np.asarray([2.2], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        np.asarray([0.6, 0.8], dtype=np.float32) * 2,
        [7],
    )

    assert abs(errors["7"] - 0.02) < 1e-7


def test_rank_one_delta_uses_the_disabled_same_vm_control() -> None:
    torch_baseline = np.zeros((2, 1, 2), dtype=np.float32)
    torch_steered = torch_baseline.copy()
    torch_steered[1, 0] = [0.6, 0.8]
    browser_control = np.asarray(
        [[[0.02, -0.03]], [[0.04, -0.01]]],
        dtype=np.float32,
    )
    browser_steered = browser_control.copy()
    browser_steered[1, 0] += [0.6, 0.8]

    assert abs(
        _VALIDATOR.steering_delta_cosine(
            torch_baseline,
            torch_steered,
            browser_control,
            browser_steered,
            1,
        ) - 1.0
    ) < 1e-7


def test_rank_one_program_ordering_detects_only_pre_enabled_changes() -> None:
    control = np.zeros((3, 1, 2), dtype=np.float32)
    steered = control.copy()
    steered[1:, 0, 0] = 1
    assert _VALIDATOR.maximum_pre_steering_absolute_delta(
        control,
        steered,
        [0, 1, 2],
        1,
    ) == 0

    steered[0, 0, 1] = 0.125
    assert _VALIDATOR.maximum_pre_steering_absolute_delta(
        control,
        steered,
        [0, 1, 2],
        1,
    ) == 0.125


def test_capture_metadata_requires_same_vm_rank_one_control(tmp_path: Path) -> None:
    metadata = capture_metadata()
    del metadata["rank_one_control_residual_sha256"]
    (tmp_path / "capture.json").write_text(json.dumps(metadata))
    try:
        _VALIDATOR.load_capture_metadata(tmp_path)
    except ValueError as error:
        assert "invalid schema" in str(error)
    else:
        raise AssertionError("schema-v3 capture accepted a missing rank-one control")


def test_release_receipt_retains_observed_runtime_identity() -> None:
    result = {
        "modelId": "gemma3-270m-instruct",
        "runtimeIdentitySha256": "a" * 64,
        "maxProbeRelativeError": 0.005,
        "maxCaptureRelativeError": 0.006,
        "steeringDirectionCosine": 0.999,
    }
    receipt = _VALIDATOR.release_evidence_receipt(
        result,
        capture_metadata(),
        "gemmaProductionQ4Parity",
        "browser-runtime/runtime-lock.json",
    )
    assert receipt["results"]["runtimeIdentitySha256"] == "a" * 64


def test_release_receipt_rejects_runtime_lock_override() -> None:
    metadata = capture_metadata()
    metadata["runtime_lock"]["source"] = "override"
    result = {
        "modelId": "gemma3-270m-instruct",
        "runtimeIdentitySha256": "a" * 64,
        "maxProbeRelativeError": 0.005,
        "maxCaptureRelativeError": 0.006,
        "steeringDirectionCosine": 0.999,
    }
    try:
        _VALIDATOR.release_evidence_receipt(
            result,
            metadata,
            "gemmaProductionQ4Parity",
            "browser-runtime/runtime-lock.json",
        )
    except ValueError as error:
        assert "checked-in repository runtime lock" in str(error)
    else:
        raise AssertionError("runtime-lock override was accepted as release evidence")
