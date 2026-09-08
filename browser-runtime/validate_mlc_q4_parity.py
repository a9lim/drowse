from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlc_q4_torch import build_transformers_model


EVIDENCE_MODEL_IDS = {
    "gemmaProductionQ4Parity": "gemma3-270m-instruct",
    "smolProductionQ4Parity": "smollm2-360m-instruct",
    "qwenProductionQ4Parity": "qwen3-1.7b",
}
PARITY_MODEL_IDS = {*EVIDENCE_MODEL_IDS.values(), "gemma3-1b-instruct", "gemma3-4b-instruct"}
SMOL_NAMED_ROLE_INPUT_IDS = [
    1,
    9690,
    198,
    1425,
    5453,
    30,
    2,
    198,
    1,
    36014,
    402,
    198,
    28120,
    2,
    198,
    1,
    8184,
    403,
    5274,
    198,
    2,
    198,
]


def validate(
    model_directory: Path,
    capture_directory: Path,
    *,
    device: str = "auto",
    require_smol_named_role: bool = False,
    maximum_capture_relative_error: float = 0.01,
    maximum_probe_relative_error: float = 0.01,
    maximum_probe_capture_relative_error: float = 1e-5,
    minimum_steering_direction_cosine: float = 0.99,
) -> dict[str, Any]:
    metadata = load_capture_metadata(
        capture_directory,
        require_smol_named_role=require_smol_named_role,
    )
    layers = [int(value) for value in metadata["layer_map"]]
    positions = [int(value) for value in metadata["positions"]]
    input_ids = [int(value) for value in metadata["input_ids"]]
    hidden_size = int(metadata["hidden_size"])
    browser = load_residuals(
        capture_directory / "residuals.f32",
        metadata["residual_sha256"],
        len(layers),
        len(positions),
        hidden_size,
    )
    browser_rank_one_control = load_residuals(
        capture_directory / "rank-one-control-residuals.f32",
        metadata["rank_one_control_residual_sha256"],
        len(layers),
        len(positions),
        hidden_size,
    )
    browser_steered = load_residuals(
        capture_directory / "steered-residuals.f32",
        metadata["steered_residual_sha256"],
        len(layers),
        len(positions),
        hidden_size,
    )

    selected_device = select_device(model_directory, device)
    model = build_transformers_model(model_directory, selected_device)
    torch_baseline, _ = capture_torch_residuals(
        model,
        layers,
        positions,
        input_ids,
        selected_device,
    )
    torch_steered, torch_probes = capture_torch_residuals(
        model,
        layers,
        positions,
        input_ids,
        selected_device,
        rank_one=metadata["rank_one_program"],
    )

    baseline_errors = relative_error_by_layer(torch_baseline, browser, layers)
    rank_one_control_errors = relative_error_by_layer(
        torch_baseline,
        browser_rank_one_control,
        layers,
    )
    steered_errors = relative_error_by_layer(torch_steered, browser_steered, layers)
    max_capture_relative_error = max([
        *baseline_errors.values(),
        *rank_one_control_errors.values(),
        *steered_errors.values(),
    ])
    if max_capture_relative_error > maximum_capture_relative_error:
        raise ValueError(
            "dequantized MLC q4 capture parity failed: "
            f"relative error {max_capture_relative_error:.6g} exceeds "
            f"{maximum_capture_relative_error:.6g}"
        )

    browser_probes = np.asarray(metadata["probe_measurements"], dtype=np.float32)
    probe_position = len(input_ids) - 1
    try:
        probe_position_slot = positions.index(probe_position)
    except ValueError as error:
        raise ValueError("browser residual fixture must capture the final input position used by probes") from error
    direction = np.zeros(hidden_size, dtype=np.float32)
    program_direction = np.asarray(metadata["rank_one_program"]["direction"], dtype=np.float32)
    direction[: len(program_direction)] = program_direction
    probe_errors = probe_relative_error_by_layer(
        torch_probes,
        browser_probes,
        torch_steered[:, probe_position_slot],
        browser_steered[:, probe_position_slot],
        direction,
        layers,
    )
    max_probe_relative_error = max(probe_errors.values())
    if max_probe_relative_error > maximum_probe_relative_error:
        raise ValueError(
            "dequantized MLC q4 probe parity failed: "
            f"activation-normalized error {max_probe_relative_error:.6g} exceeds "
            f"{maximum_probe_relative_error:.6g}"
        )
    browser_probe_capture_errors = probe_relative_error_by_layer(
        browser_probes,
        browser_steered[:, probe_position_slot] @ direction,
        browser_steered[:, probe_position_slot],
        browser_steered[:, probe_position_slot],
        direction,
        layers,
    )
    max_browser_probe_capture_relative_error = max(browser_probe_capture_errors.values())
    if max_browser_probe_capture_relative_error > maximum_probe_capture_relative_error:
        raise ValueError(
            "browser probe readback does not match its captured residual: "
            "activation-normalized error "
            f"{max_browser_probe_capture_relative_error:.6g} exceeds "
            f"{maximum_probe_capture_relative_error:.6g}"
        )
    probe_trace_relative_error = float(
        np.linalg.norm(torch_probes - browser_probes)
        / max(np.linalg.norm(browser_probes), np.linalg.norm(torch_probes), 1e-12)
    )

    enabled_layer = int(metadata["rank_one_program"]["enabledLayer"])
    try:
        enabled_slot = layers.index(enabled_layer)
    except ValueError as error:
        raise ValueError("rank-one capture layer is absent from the layer map") from error
    max_pre_steering_absolute_delta = maximum_pre_steering_absolute_delta(
        browser_rank_one_control,
        browser_steered,
        layers,
        enabled_layer,
    )
    if max_pre_steering_absolute_delta != 0:
        raise ValueError(
            "dequantized MLC q4 rank-one program ordering failed: "
            "steered residuals changed before the enabled layer "
            f"(maximum absolute delta {max_pre_steering_absolute_delta:.6g})"
        )
    steering_direction_cosine = steering_delta_cosine(
        torch_baseline,
        torch_steered,
        browser_rank_one_control,
        browser_steered,
        enabled_slot,
    )
    if steering_direction_cosine < minimum_steering_direction_cosine:
        raise ValueError(
            "dequantized MLC q4 steering parity failed: "
            f"direction cosine {steering_direction_cosine:.6g} is below "
            f"{minimum_steering_direction_cosine:.6g}"
        )

    greedy = metadata["greedy"]
    torch_greedy_ids = greedy_token_ids(
        model,
        [int(value) for value in greedy["input_ids"]],
        int(greedy["max_tokens"]),
        selected_device,
    )
    browser_greedy_ids = [int(value) for value in greedy["token_ids"]]
    hook_disabled_greedy_matches_baseline = torch_greedy_ids == browser_greedy_ids
    if not hook_disabled_greedy_matches_baseline:
        raise ValueError(
            f"dequantized MLC q4 greedy-token parity failed: browser={browser_greedy_ids}, torch={torch_greedy_ids}"
        )

    return {
        "modelId": metadata["model_id"],
        "runtimeIdentitySha256": metadata["runtime_identity_sha256"],
        "device": selected_device,
        "hookDisabledGreedyMatchesBaseline": True,
        "maxProbeRelativeError": max_probe_relative_error,
        "probeTraceRelativeError": probe_trace_relative_error,
        "maxBrowserProbeCaptureRelativeError": (max_browser_probe_capture_relative_error),
        "maxCaptureRelativeError": max_capture_relative_error,
        "maxPreSteeringAbsoluteDelta": max_pre_steering_absolute_delta,
        "rankOneProgramOrderingPreserved": True,
        "steeringDirectionCosine": steering_direction_cosine,
        "baselineRelativeErrorByLayer": baseline_errors,
        "rankOneControlRelativeErrorByLayer": rank_one_control_errors,
        "steeredRelativeErrorByLayer": steered_errors,
        "probeRelativeErrorByLayer": probe_errors,
        "browserProbeCaptureRelativeErrorByLayer": browser_probe_capture_errors,
        "browserProbeMeasurements": browser_probes.tolist(),
        "torchProbeMeasurements": torch_probes.tolist(),
        "browserGreedyTokenIds": browser_greedy_ids,
        "torchGreedyTokenIds": torch_greedy_ids,
    }


def load_capture_metadata(
    capture_directory: Path,
    *,
    require_smol_named_role: bool = False,
) -> dict[str, Any]:
    metadata = json.loads((capture_directory / "capture.json").read_text())
    required = {
        "schema_version",
        "produced_at",
        "model_id",
        "runtime_identity_sha256",
        "layer_map",
        "hidden_size",
        "input_ids",
        "positions",
        "residual_sha256",
        "rank_one_control_residual_sha256",
        "steered_residual_sha256",
        "rank_one_program",
        "probe_measurements",
        "greedy",
    }
    allowed = required | {"named_role_capture", "runtime_lock"}
    if set(metadata) - allowed or not required.issubset(metadata) or metadata["schema_version"] != 3:
        raise ValueError("browser residual fixture has an invalid schema")
    if (
        metadata["model_id"] not in PARITY_MODEL_IDS
        or not valid_sha256(metadata["runtime_identity_sha256"])
        or not valid_sha256(metadata["residual_sha256"])
        or not valid_sha256(metadata["rank_one_control_residual_sha256"])
        or not valid_sha256(metadata["steered_residual_sha256"])
        or not isinstance(metadata["produced_at"], str)
        or not metadata["produced_at"]
    ):
        raise ValueError("browser residual fixture identity is invalid")
    layers = metadata["layer_map"]
    positions = metadata["positions"]
    input_ids = metadata["input_ids"]
    hidden_size = metadata["hidden_size"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(value, int) or value < 0 for value in layers)
        or not isinstance(input_ids, list)
        or not input_ids
        or any(not isinstance(value, int) or value < 0 for value in input_ids)
        or not isinstance(positions, list)
        or not positions
        or any(
            not isinstance(value, int)
            or value < 0
            or value >= len(input_ids)
            or (index > 0 and value <= positions[index - 1])
            for index, value in enumerate(positions)
        )
        or not isinstance(hidden_size, int)
        or hidden_size < 2
    ):
        raise ValueError("browser residual fixture dimensions are invalid")
    program = metadata["rank_one_program"]
    if (
        not isinstance(program, dict)
        or set(program)
        != {
            "abiVersion",
            "enabledLayer",
            "direction",
            "target",
            "along",
            "collapse",
        }
        or program["abiVersion"] != 1
        or program["enabledLayer"] not in layers
        or not isinstance(program["direction"], list)
        or len(program["direction"]) != hidden_size
        or not all(finite_number(value) for value in program["direction"])
        or not all(finite_number(program[field]) for field in ("target", "along", "collapse"))
    ):
        raise ValueError("browser residual fixture rank-one program is invalid")
    probes = metadata["probe_measurements"]
    if not isinstance(probes, list) or len(probes) != len(layers) or not all(finite_number(value) for value in probes):
        raise ValueError("browser residual fixture probe measurements are invalid")
    greedy = metadata["greedy"]
    if (
        not isinstance(greedy, dict)
        or set(greedy) != {"prompt", "input_ids", "token_ids", "max_tokens"}
        or not isinstance(greedy["prompt"], str)
        or not greedy["prompt"]
        or not valid_token_ids(greedy["input_ids"], allow_empty=False)
        or not valid_token_ids(greedy["token_ids"], allow_empty=True)
        or not isinstance(greedy["max_tokens"], int)
        or not 1 <= greedy["max_tokens"] <= 32
        or len(greedy["token_ids"]) > greedy["max_tokens"]
    ):
        raise ValueError("browser residual fixture greedy observation is invalid")
    named_role = metadata.get("named_role_capture")
    if named_role is not None and (
        not isinstance(named_role, dict)
        or set(named_role) != {"input_ids", "position"}
        or not valid_token_ids(named_role["input_ids"], allow_empty=False)
        or not isinstance(named_role["position"], int)
        or named_role["position"] < 0
        or named_role["position"] >= len(named_role["input_ids"])
    ):
        raise ValueError("browser residual fixture named-role observation is invalid")
    if (
        metadata["model_id"] == "smollm2-360m-instruct"
        and named_role is not None
        and (named_role["input_ids"] != SMOL_NAMED_ROLE_INPUT_IDS or named_role["position"] != 19)
    ):
        raise ValueError("SmolLM2 named-role tokenization does not match the pinned Python row")
    if require_smol_named_role and metadata["model_id"] == "smollm2-360m-instruct" and named_role is None:
        raise ValueError("SmolLM2 release evidence requires the pinned named-role row")
    runtime_lock = metadata.get("runtime_lock")
    if runtime_lock is not None and (
        not isinstance(runtime_lock, dict)
        or set(runtime_lock) != {"source", "sha256", "identity_sha256"}
        or runtime_lock["source"] not in {"repository", "override"}
        or not valid_sha256(runtime_lock["sha256"])
        or not valid_sha256(runtime_lock["identity_sha256"])
    ):
        raise ValueError("browser residual fixture runtime-lock provenance is invalid")
    return metadata


def capture_torch_residuals(
    model: torch.nn.Module,
    layers: list[int],
    positions: list[int],
    input_ids: list[int],
    device: str,
    *,
    rank_one: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    captured: dict[int, torch.Tensor] = {}
    probes: dict[int, float] = {}
    handles = []
    enabled_layer = None if rank_one is None else int(rank_one["enabledLayer"])
    direction = None
    if rank_one is not None:
        direction = torch.zeros(int(model.config.hidden_size), dtype=torch.float32, device=device)
        direction[:] = torch.tensor(rank_one["direction"], dtype=torch.float32, device=device)

    def capture_layer(layer: int):
        def hook(_module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
            residual = output[0] if isinstance(output, tuple) else output
            if not isinstance(residual, torch.Tensor):
                raise ValueError(f"Transformers layer {layer} returned a non-tensor residual")
            steered = residual
            if layer == enabled_layer:
                residual_f32 = residual.to(torch.float32)
                coordinate = torch.matmul(residual_f32, direction)
                delta = float(rank_one["along"]) * (
                    float(rank_one["target"]) - float(rank_one["collapse"]) * coordinate
                )
                steered = (residual_f32 + delta.unsqueeze(-1) * direction).to(residual.dtype)
            captured[layer] = steered.detach().cpu()
            if direction is not None:
                probes[layer] = float(torch.matmul(steered[0, -1].to(torch.float32), direction).item())
            if steered is residual:
                return None
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        return hook

    model_layers = model.model.layers
    for layer in layers:
        handles.append(model_layers[layer].register_forward_hook(capture_layer(layer)))
    try:
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            model(tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    residuals = np.stack([captured[layer][0, positions].to(torch.float32).numpy() for layer in layers])
    probe_values = np.asarray(
        [probes[layer] for layer in layers] if direction is not None else [],
        dtype=np.float32,
    )
    return residuals, probe_values


def greedy_token_ids(
    model: torch.nn.Module,
    input_ids: list[int],
    max_tokens: int,
    device: str,
) -> list[int]:
    generated: list[int] = []
    eos = model.config.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in eos or []}
    with torch.inference_mode():
        for _ in range(max_tokens):
            tokens = torch.tensor([input_ids + generated], dtype=torch.long, device=device)
            logits = model(tokens, use_cache=False).logits[0, -1]
            token_id = int(torch.argmax(logits).item())
            if token_id in eos_ids:
                break
            generated.append(token_id)
    return generated


def load_residuals(
    path: Path,
    expected_sha256: str,
    layers: int,
    positions: int,
    hidden_size: int,
) -> np.ndarray:
    if hash_file(path) != expected_sha256:
        raise ValueError(f"browser residual fixture digest does not match: {path.name}")
    expected_values = layers * positions * hidden_size
    if path.stat().st_size != expected_values * 4:
        raise ValueError(f"browser residual fixture has the wrong size: {path.name}")
    values = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        shape=(layers, positions, hidden_size),
    )
    if not np.isfinite(values).all():
        raise ValueError(f"browser residual fixture contains non-finite values: {path.name}")
    return np.asarray(values)


def relative_error_by_layer(
    actual: np.ndarray,
    expected: np.ndarray,
    layers: list[int],
) -> dict[str, float]:
    return {
        str(layer): float(np.linalg.norm(actual[index] - expected[index]) / max(np.linalg.norm(expected[index]), 1e-12))
        for index, layer in enumerate(layers)
    }


def probe_relative_error_by_layer(
    actual: np.ndarray,
    expected: np.ndarray,
    actual_residuals: np.ndarray,
    expected_residuals: np.ndarray,
    direction: np.ndarray,
    layers: list[int],
) -> dict[str, float]:
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ValueError("probe direction has zero norm")
    return {
        str(layer): float(
            abs(float(actual[index]) - float(expected[index]))
            / max(
                float(np.linalg.norm(actual_residuals[index])) * direction_norm,
                float(np.linalg.norm(expected_residuals[index])) * direction_norm,
                1e-12,
            )
        )
        for index, layer in enumerate(layers)
    }


def maximum_pre_steering_absolute_delta(
    control: np.ndarray,
    steered: np.ndarray,
    layers: list[int],
    enabled_layer: int,
) -> float:
    pre_steering_slots = [
        index for index, layer in enumerate(layers) if layer < enabled_layer
    ]
    if not pre_steering_slots:
        return 0.0
    return float(
        np.max(
            np.abs(
                steered[np.asarray(pre_steering_slots)]
                - control[np.asarray(pre_steering_slots)]
            )
        )
    )


def steering_delta_cosine(
    torch_baseline: np.ndarray,
    torch_steered: np.ndarray,
    browser_control: np.ndarray,
    browser_steered: np.ndarray,
    enabled_slot: int,
) -> float:
    torch_delta = torch_steered[enabled_slot] - torch_baseline[enabled_slot]
    browser_delta = browser_steered[enabled_slot] - browser_control[enabled_slot]
    return cosine(torch_delta, browser_delta)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        raise ValueError("rank-one steering produced no measurable residual delta")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def select_device(model_directory: Path, requested: str) -> str:
    chat = json.loads((model_directory / "mlc-chat-config.json").read_text())
    quantization = chat.get("quantization")
    selected = "mps" if requested == "auto" and torch.backends.mps.is_available() else requested
    if selected == "auto":
        selected = "cpu"
    if selected == "mps" and not torch.backends.mps.is_available():
        raise ValueError("the requested MPS q4 comparator is unavailable")
    if quantization == "q4f16_1" and selected != "mps":
        raise ValueError(
            "exact q4f16_1 comparison requires the MPS half-FMA comparator; "
            "CPU float32 accumulation is not release evidence"
        )
    return selected


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def valid_token_ids(value: object, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or len(value) > 0)
        and all(isinstance(token, int) and token >= 0 for token in value)
    )


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def release_evidence_receipt(
    result: dict[str, Any],
    metadata: dict[str, Any],
    evidence_type: str,
    fixture_path: str,
) -> dict[str, Any]:
    if metadata.get("runtime_lock", {}).get("source") != "repository":
        raise ValueError("release evidence requires the checked-in repository runtime lock")
    expected_model = EVIDENCE_MODEL_IDS[evidence_type]
    if result["modelId"] != expected_model:
        raise ValueError(f"{evidence_type} requires {expected_model}, found {result['modelId']}")
    return {
        "schemaVersion": 1,
        "evidenceType": evidence_type,
        "fixturePath": fixture_path,
        "producedAt": metadata["produced_at"],
        "passed": True,
        "results": {
            "modelId": result["modelId"],
            "runtimeIdentitySha256": result["runtimeIdentitySha256"],
            "hookDisabledGreedyMatchesBaseline": True,
            "maxProbeRelativeError": result["maxProbeRelativeError"],
            "maxCaptureRelativeError": result["maxCaptureRelativeError"],
            "steeringDirectionCosine": result["steeringDirectionCosine"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--maximum-capture-relative-error", type=float, default=0.01)
    parser.add_argument("--maximum-probe-relative-error", type=float, default=0.01)
    parser.add_argument("--maximum-probe-capture-relative-error", type=float, default=1e-5)
    parser.add_argument("--minimum-steering-direction-cosine", type=float, default=0.99)
    parser.add_argument("--evidence-type", choices=sorted(EVIDENCE_MODEL_IDS))
    parser.add_argument("--fixture-path")
    arguments = parser.parse_args()
    result = validate(
        arguments.model_directory,
        arguments.capture_directory,
        device=arguments.device,
        require_smol_named_role=arguments.evidence_type == "smolProductionQ4Parity",
        maximum_capture_relative_error=arguments.maximum_capture_relative_error,
        maximum_probe_relative_error=arguments.maximum_probe_relative_error,
        maximum_probe_capture_relative_error=(arguments.maximum_probe_capture_relative_error),
        minimum_steering_direction_cosine=arguments.minimum_steering_direction_cosine,
    )
    if (arguments.evidence_type is None) != (arguments.fixture_path is None):
        raise ValueError("--evidence-type and --fixture-path must be supplied together")
    if arguments.evidence_type is not None:
        metadata = load_capture_metadata(
            arguments.capture_directory,
            require_smol_named_role=arguments.evidence_type == "smolProductionQ4Parity",
        )
        result = release_evidence_receipt(
            result,
            metadata,
            arguments.evidence_type,
            arguments.fixture_path,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
