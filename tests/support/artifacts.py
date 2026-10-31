"""Small on-disk manifold artifacts for store/install/distribution tests.

:func:`author_manifold` writes a two-node discover folder under ``local/``
(fitted for :data:`MODEL` by default); :func:`write_fitted_manifold` writes
one identity-consistent fitted 1-D affine tensor pair into a folder.
"""

from __future__ import annotations

from pathlib import Path

from drowse.base.paths import tensor_filename
from drowse.store.manifold_authoring import create_discover_manifold_folder
from drowse.store.manifold_folder import ManifoldFolder

__all__ = ["MODEL", "author_manifold", "write_fitted_manifold"]

MODEL = "google/gemma-2-2b-it"


def write_fitted_manifold(
    folder: Path, model_id: str, *, publish_manifest: bool = True,
) -> Path:
    """Write one identity-consistent fitted 1-D affine pair into ``folder``."""
    import torch

    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION, Manifold
    from drowse.geometry.subspace import LayerSubspace
    from drowse.store.manifold_tensors import save_manifold

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    node_coords = torch.tensor([[-1.0], [1.0]])
    manifold = Manifold(
        name=mf.name,
        domain=CustomDomain(1),
        node_labels=list(mf.node_labels),
        node_coords=node_coords,
        layers={
            0: LayerSubspace.affine(
                mean=torch.zeros(2), basis=torch.tensor([[1.0, 0.0]]),
                node_coords=node_coords.clone(),
            ),
        },
        feature_space="raw",
        mahalanobis_share={0: 1.0},
    )
    path = folder / tensor_filename(model_id)
    save_manifold(manifold, path, {
        "method": "manifold_discover_pca",
        "fit_mode": "pca",
        "hyperparams": dict(mf.hyperparams),
        "nodes_sha256": mf.nodes_sha256(),
        "model_fingerprint": f"fp:{model_id}",
        "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
    })
    if publish_manifest:
        mf.update_file_hashes(path, path.with_suffix(".json"))
    return path


def author_manifold(name: str = "mood", *, fitted: bool = True) -> Path:
    folder = create_discover_manifold_folder(
        "local", name, "a mood axis", fit_mode="pca",
        node_corpora={"calm": ["easy"], "afraid": ["tense"]},
        hyperparams={"max_dim": 1},
    )
    if fitted:
        write_fitted_manifold(folder, MODEL)
    return folder
