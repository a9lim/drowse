"""Local-cache-only numerical steering smoke test on a real causal LM."""

import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from drowse.core.mahalanobis import LayerWhitener
from drowse.core.manifold import (
    BoxAxis, BoxDomain, CustomDomain, Manifold, fit_affine_subspace,
    fit_layer_subspace, subspace_inject,
)
from drowse.core.monitor import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32,
    ).to(args.device).eval()
    layer = len(model.model.layers) // 2
    module = model.model.layers[layer]
    prompts = [
        "The garden has green leaves and red flowers.",
        "The train arrived at the station on time.",
        "She read the book and put it on the shelf.",
        "The ocean is deep and the waves move slowly.",
        "A scientist recorded the temperature of the sample.",
        "The musician played a quiet melody on the piano.",
        "They walked through the forest before sunrise.",
        "The city library opens every morning at nine.",
        "He measured the table before building the chair.",
        "The baker mixed flour and water in a bowl.",
        "A bird landed near the window.",
        "The teacher wrote a question on the board.",
    ]
    rows = []

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any):
        hidden = output[0] if isinstance(output, tuple) else output
        rows.append(hidden[0, -1].detach().float().cpu())

    with torch.inference_mode():
        handle = module.register_forward_hook(capture)
        try:
            for text in prompts:
                model(**tokenizer(text, return_tensors="pt").to(args.device), use_cache=False)
        finally:
            handle.remove()
        activations = torch.stack(rows)
        whitener = LayerWhitener.from_neutral_activations(
            {layer: activations}, {layer: activations.mean(0)},
        )
        flat, _, _ = fit_affine_subspace(
            activations[:8], neutral_mean=activations.mean(0), n_components=2,
            whitener=whitener, layer=layer,
        )
        domain = BoxDomain([BoxAxis("phase", periodic=True, period=1)])
        coords = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
        embedded = domain.embed(coords)
        radius = torch.linalg.vector_norm(activations - activations.mean(0), dim=1).mean() * 0.1
        points = activations.mean(0) + radius * embedded @ flat.basis[:2]
        curved, _ = fit_layer_subspace(points, embedded)
        curved.sigma_rbf_weights = torch.zeros(8, 1)
        curved.sigma_poly_coeffs = torch.zeros(3, 1)
        manifold = Manifold(
            name="runtime-circle", domain=domain, node_labels=[f"n{i}" for i in range(8)],
            node_coords=coords, layers={layer: curved}, mahalanobis_share={layer: 1.0},
            origin={layer: torch.zeros(1)},
        )
        monitor = Monitor(whitener=whitener)
        monitor.add_probe("runtime-circle", manifold)
        monitor.score_single_token({layer: activations[-1]})
        prompt = tokenizer("Explain why leaves change color in autumn.", return_tensors="pt").to(args.device)

        def forward():
            return model(**prompt, use_cache=False).logits[0, -1].float().cpu()

        baseline = forward()
        results = []
        for kind, sub, shape, target in [
            ("flat", flat, CustomDomain(flat.rank), torch.tensor([0.3, -0.2])),
            ("curved", curved, domain, torch.tensor([0.25])),
        ]:
            resident = sub.to(device=torch.device(args.device), dtype=torch.float32)
            target = target.to(args.device)
            for strength in [0.0, 0.3, -0.3]:
                def steer(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any):
                    hidden = output[0] if isinstance(output, tuple) else output
                    changed, _ = subspace_inject(
                        hidden, resident, shape, target,
                        torch.zeros((*hidden.shape[:-1], shape.intrinsic_dim), device=hidden.device),
                        strength, 0.0, gn_steps=3,
                    )
                    assert torch.isfinite(changed).all()
                    return (changed, *output[1:]) if isinstance(output, tuple) else changed

                handle = module.register_forward_hook(steer)
                try:
                    logits = forward()
                finally:
                    handle.remove()
                assert torch.isfinite(logits).all()
                change = float(torch.linalg.vector_norm(logits - baseline))
                if strength == 0:
                    torch.testing.assert_close(logits, baseline, atol=1e-4, rtol=1e-5)
                else:
                    assert change > 1e-5
                torch.testing.assert_close(forward(), baseline, atol=0, rtol=0)
                results.append({"kind": kind, "strength": strength, "logit_l2_change": change})
        print(json.dumps({"model": args.model, "device": args.device, "layer": layer, "checks": results, "note": "Numerical integration check; the circle is a controlled test geometry, not a semantic discovery claim."}, indent=2))


if __name__ == "__main__":
    main()
