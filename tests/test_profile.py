"""Unit tests for the Profile wrapper class."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pytest
import torch

from drowse.geometry import profile as V
from drowse.geometry.profile import Profile, ProfileError, load_profile, save_profile
from drowse.base.integrity import PROFILE_FORMAT_VERSION


def _mk(layers: Any = (0, 5, 10), dim: int = 8, dtype: Any = torch.float32) -> dict[int, torch.Tensor]:
    return {i: torch.randn(dim, dtype=dtype) for i in layers}


def test_construct_and_dict_surface():
    p = Profile(_mk())
    assert p.layers == [0, 5, 10]
    assert len(p) == 3
    assert 5 in p
    assert 99 not in p
    # items/keys/values
    assert set(p.keys()) == {0, 5, 10}
    assert [t.shape for t in p.values()] == [torch.Size([8])] * 3
    for idx, t in p.items():
        assert isinstance(idx, int) and isinstance(t, torch.Tensor)
    # __getitem__
    assert p[0].shape == (8,)


def test_empty_profile_rejected():
    with pytest.raises(ProfileError, match="at least one layer"):
        Profile({})


def test_bad_key_type_rejected():
    with pytest.raises(ProfileError, match="layer key must be int"):
        Profile({"0": torch.zeros(4)})  # pyright: ignore[reportArgumentType]  # str key, intentionally wrong type for the test


def test_bad_value_type_rejected():
    with pytest.raises(ProfileError, match="must be torch.Tensor"):
        Profile({0: [1.0, 2.0]})  # pyright: ignore[reportArgumentType]  # list value, intentionally wrong type for the test


def test_metadata_is_copy():
    meta = {"method": "profile", "drowse_version": "1.4.0"}
    p = Profile(_mk(), metadata=meta)
    out = p.metadata
    out["method"] = "mutated"
    assert p.metadata["method"] == "profile"


def test_save_load_roundtrip(tmp_path: Path):
    p = Profile(_mk(layers=(0, 3)))
    path = tmp_path / "cv.safetensors"
    p.save(path, metadata={"method": "manifold_pca"})
    assert path.exists()
    assert path.with_suffix(".json").exists()

    loaded = Profile.load(path)
    assert loaded.layers == [0, 3]
    for idx in loaded.layers:
        assert torch.allclose(loaded[idx], p[idx])
    assert loaded.metadata["method"] == "manifold_pca"
    assert loaded.metadata["format_version"] == PROFILE_FORMAT_VERSION


def test_merged_union_semantics():
    a = Profile({0: torch.ones(4), 1: torch.ones(4), 2: torch.ones(4)})
    b = Profile({1: torch.ones(4) * 2, 2: torch.ones(4) * 2, 3: torch.ones(4) * 2})
    merged = Profile.merged([(a, 1.0), (b, 0.5)])
    # Live expression parity: missing terms contribute zero on union layers.
    assert merged.layers == [0, 1, 2, 3]
    assert torch.allclose(merged[0], torch.ones(4))
    assert torch.allclose(merged[3], torch.ones(4))
    # 1*1 + 0.5*2 = 2
    assert torch.allclose(merged[1], torch.full((4,), 2.0))
    assert torch.allclose(merged[2], torch.full((4,), 2.0))


def test_merged_strict_refuses_drop():
    a = Profile({0: torch.ones(4), 1: torch.ones(4)})
    b = Profile({1: torch.ones(4), 2: torch.ones(4)})
    from drowse.errors import MergeError
    with pytest.raises(MergeError):
        Profile.merged([(a, 1.0), (b, 1.0)], strict=True)


def test_merged_with_binary_convenience():
    a = Profile({0: torch.ones(4)})
    b = Profile({0: torch.ones(4) * 3})
    out = a.merged_with(b, weights=(1.0, 2.0))
    # 1*1 + 2*3 = 7
    assert torch.allclose(out[0], torch.full((4,), 7.0))


def test_merged_requires_two_components():
    p = Profile({0: torch.ones(4)})
    with pytest.raises(ProfileError, match="at least two"):
        Profile.merged([(p, 1.0)])


def test_repr_contains_layer_info():
    p = Profile(_mk(layers=range(10)))
    r = repr(p)
    assert "Profile(" in r
    assert "10 layers" in r


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

# Mahalanobis-only (4.0 collapse): ``cosine_similarity`` requires a covering
# whitener — there is no Euclidean path.  The whitener is built with zero
# neutral means so it doesn't recenter the test vectors; the parallel /
# anti-parallel ±1 invariants hold under any positive-definite metric, while
# the orthogonal case is rewritten against *Mahalanobis* orthogonality.

def _cov_whitener(layers: Any, dim: int):
    from tests._whitener import synthetic_whitener, synthetic_means
    return synthetic_whitener(
        layers, dim, means=synthetic_means(layers, dim, seed=0),
    )


def test_cosine_similarity_identical_profiles():
    """Identical profiles should have whitened cosine 1.0 (metric-invariant)."""
    tensors = _mk(layers=(0, 1, 2), dim=16)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    w = _cov_whitener((0, 1, 2), 16)
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_opposite_profiles():
    """Negated profiles should have whitened cosine -1.0 (metric-invariant)."""
    tensors = _mk(layers=(0, 1, 2), dim=16)
    a = Profile(tensors)
    b = Profile({k: -v for k, v in tensors.items()})
    w = _cov_whitener((0, 1, 2), 16)
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_similarity_mahalanobis_orthogonal_profiles():
    """Profiles M-orthogonal per layer have whitened cosine 0.0.

    Build ``b_L`` Euclidean-orthogonal to ``Σ⁻¹ a_L`` so the Mahalanobis dot
    ``a_L^T Σ⁻¹ b_L`` vanishes — the whitened analogue of the old
    Euclidean-orthogonal construction.
    """
    layers = (0, 1, 2)
    w = _cov_whitener(layers, 16)
    a_tensors, b_tensors = {}, {}
    for L in layers:
        v = torch.randn(16)
        sv = w.apply_inv(L, v)          # Σ⁻¹ a_L
        u = torch.randn(16)
        # Remove the component of u along Σ⁻¹a so <u, Σ⁻¹a> = 0, i.e.
        # <a, u>_M = a^T Σ⁻¹ u = 0 (Σ⁻¹ symmetric).
        u = u - (u @ sv) / (sv @ sv) * sv
        a_tensors[L] = v
        b_tensors[L] = u
    a = Profile(a_tensors)
    b = Profile(b_tensors)
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(0.0, abs=1e-4)


def test_cosine_similarity_partial_layer_overlap():
    """Only shared layers contribute to the similarity."""
    shared = torch.randn(8)
    a = Profile({0: shared.clone(), 1: torch.randn(8)})
    b = Profile({0: shared.clone(), 2: torch.randn(8)})
    w = _cov_whitener((0,), 8)
    # Only layer 0 overlaps, and it's identical -> 1.0 (metric-invariant).
    assert a.cosine_similarity(b, whitener=w) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_empty_intersection_raises():
    """No shared layers should raise ProfileError."""
    a = Profile({0: torch.randn(8)})
    b = Profile({1: torch.randn(8)})
    w = _cov_whitener((0, 1), 8)
    with pytest.raises(ProfileError, match="no shared layers"):
        a.cosine_similarity(b, whitener=w)


def test_cosine_similarity_missing_whitener_raises():
    """Mahalanobis-only: a missing / non-covering whitener is a hard error."""
    from drowse.errors import WhitenerError

    tensors = _mk(layers=(0, 1), dim=8)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    with pytest.raises(WhitenerError, match="whitener"):
        a.cosine_similarity(b)
    # A whitener covering only some shared layers is also rejected.
    partial = _cov_whitener((0,), 8)
    with pytest.raises(WhitenerError, match="whitener"):
        a.cosine_similarity(b, whitener=partial)


def test_cosine_similarity_per_layer():
    """per_layer=True returns a dict of per-layer whitened cosines."""
    tensors = _mk(layers=(0, 5, 10), dim=16)
    a = Profile(tensors)
    b = Profile({k: v.clone() for k, v in tensors.items()})
    w = _cov_whitener((0, 5, 10), 16)
    result = a.cosine_similarity(b, per_layer=True, whitener=w)
    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 5, 10}
    for v in result.values():
        assert v == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_per_layer_partial_overlap():
    """per_layer=True only includes shared layers."""
    a = Profile({0: torch.randn(8), 1: torch.randn(8)})
    b = Profile({1: torch.randn(8), 2: torch.randn(8)})
    w = _cov_whitener((1,), 8)
    result: dict[int, float] = a.cosine_similarity(b, per_layer=True, whitener=w)
    assert set(result.keys()) == {1}


# ---------------------------------------------------------------------------
# Profile.projected_away
# ---------------------------------------------------------------------------

def test_projected_away_orthogonality():
    """Result at each shared layer should be orthogonal to other."""
    a = Profile({
        0: torch.tensor([1.0, 2.0, 0.0]),
        1: torch.tensor([3.0, 1.0, 5.0]),
    })
    b = Profile({
        0: torch.tensor([1.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0]),
    })
    result = a.projected_away(b)
    for layer in [0, 1]:
        r_f = result[layer].float()
        b_f = b[layer].float()
        dot = torch.dot(r_f, b_f).item()
        assert abs(dot) < 1e-5, f"Layer {layer} not orthogonal: dot={dot}"


def test_projected_away_layer_not_in_other_unchanged():
    """Layers in self but not in other are included unchanged."""
    a = Profile({0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])})
    b = Profile({1: torch.tensor([1.0, 0.0])})
    result = a.projected_away(b)
    assert set(result.keys()) == {0, 1}
    assert torch.allclose(result[0], a[0])


def test_projected_away_near_zero_b_skipped():
    """Near-zero b layer (dot < 1e-12) is skipped; self layer copied as-is."""
    a = Profile({0: torch.tensor([1.0, 2.0])})
    b = Profile({0: torch.tensor([0.0, 0.0])})
    result = a.projected_away(b)
    assert torch.allclose(result[0], a[0])


def test_projected_away_empty_intersection_raises():
    """No shared layers should raise ProfileError."""
    a = Profile({0: torch.randn(4)})
    b = Profile({1: torch.randn(4)})
    with pytest.raises(ProfileError, match="no shared layers"):
        a.projected_away(b)


def test_projected_away_full_projection():
    """Projecting a vector fully aligned with b yields near-zero result."""
    v = torch.tensor([3.0, 0.0, 0.0])
    a = Profile({0: v.clone()})
    b = Profile({0: torch.tensor([1.0, 0.0, 0.0])})
    result = a.projected_away(b)
    assert torch.allclose(result[0].float(), torch.zeros(3), atol=1e-6)


def test_projected_away_preserves_metadata():
    """Metadata from self is preserved in the output Profile."""
    a = Profile({0: torch.tensor([1.0, 0.0])}, metadata={"method": "test"})
    b = Profile({0: torch.tensor([1.0, 0.0])})
    result = a.projected_away(b)
    assert result.metadata.get("method") == "test"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr_short_form_lists_layers():
    """<=4 layers renders the layer list verbatim."""
    p = Profile(_mk(layers=(0, 3)))
    r = repr(p)
    assert "[0, 3]" in r


def test_repr_long_form_shows_range_and_count():
    """>4 layers collapses to ``[first..last] (n layers)``."""
    p = Profile(_mk(layers=range(8)))
    r = repr(p)
    assert "0..7" in r
    assert "8 layers" in r


# ---------------------------------------------------------------------------
# save metadata override
# ---------------------------------------------------------------------------

def test_save_metadata_override_merges_on_top_of_self_metadata(tmp_path: Path):
    """``metadata=`` kwarg to ``save`` overrides matching keys in ``self.metadata``.

    Verified via ``method`` (override wins) and ``statements_sha256``
    (self-metadata that the override doesn't touch and that rides the
    free-form provenance blob).
    """
    p = Profile(
        _mk(layers=(0, 1)),
        metadata={"method": "manifold_pca", "statements_sha256": "a" * 64},
    )
    path = tmp_path / "cv.safetensors"
    p.save(path, metadata={"method": "profile"})
    loaded = Profile.load(path)
    assert loaded.metadata["method"] == "profile"
    assert loaded.metadata["statements_sha256"] == "a" * 64


@pytest.mark.parametrize("metadata", [
    {"method": "invented"},                                   # unknown producer
    {"method": "contrastive_pca"},                            # retired producer
    {"method": "profile", "components": {0: {}}},             # non-string key
    {"method": "profile", "score": float("nan")},             # not JSON
    {"method": "profile", "tensor": torch.ones(2)},           # not JSON
])
def test_invalid_metadata_rejects_before_publication(
    tmp_path: Path, metadata: dict[str, Any],
) -> None:
    path = tmp_path / "invalid.safetensors"
    with pytest.raises(ProfileError):
        Profile(_mk(layers=(0, 1))).save(path, metadata=metadata)
    assert not path.exists()
    assert not path.with_suffix(".json").exists()


def test_profile_digest_binds_tensor_pair(tmp_path: Path) -> None:
    path = tmp_path / "profile.safetensors"
    Profile(_mk(layers=(0,))).save(path, metadata={"method": "profile"})
    sidecar = json.loads(path.with_suffix(".json").read_text())
    sidecar["tensor_sha256"] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ProfileError, match="tensor sha256"):
        Profile.load(path)


def test_sidecar_schema_is_exactly_five_fields(tmp_path: Path) -> None:
    """The wire schema is closed: four identity fields + one provenance blob."""
    path = tmp_path / "schema.safetensors"
    Profile(_mk(layers=(0,))).save(path, metadata={
        "method": "profile", "note": "hello",
    })
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert set(sidecar) == {
        "format_version", "drowse_version", "method",
        "tensor_sha256", "provenance",
    }
    assert sidecar["provenance"] == {"note": "hello"}


def test_extract_shaped_profile_round_trips(tmp_path: Path) -> None:
    """The Profile ``DrowseSession.extract`` returns saves and loads.

    ``extract`` stamps ``{"method": "manifold_pca", "name": ..., "share_metric":
    ...}`` on the folded 2-node manifold view; the schema has to take that shape
    verbatim, and the loaded metadata has to save again unchanged.
    """
    extract_metadata = {
        "method": "manifold_pca",
        "name": "confident.uncertain",
        "share_metric": "mahalanobis",
    }
    profile = Profile(_mk(layers=(0, 3)), metadata=extract_metadata)
    first = tmp_path / "extract.safetensors"
    profile.save(first)

    loaded = Profile.load(first)
    assert loaded.metadata["method"] == "manifold_pca"
    assert loaded.metadata["name"] == "confident.uncertain"
    assert loaded.metadata["share_metric"] == "mahalanobis"

    # load -> save round-trip: the writer-stamped identity fields the reader
    # hands back must not read as unknown metadata on the way in again.
    second = tmp_path / "extract-again.safetensors"
    loaded.save(second)
    again = Profile.load(second)
    assert again.metadata["method"] == "manifold_pca"
    assert again.metadata["name"] == "confident.uncertain"
    assert again.layers == [0, 3]


def test_merged_profile_saves(tmp_path: Path) -> None:
    """``Profile.merged`` claims ``method="merge"``; that must be savable."""
    merged = Profile.merged([
        (Profile({0: torch.ones(4)}), 1.0),
        (Profile({0: torch.ones(4) * 2}), 0.5),
    ])
    path = tmp_path / "merged.safetensors"
    merged.save(path)
    assert Profile.load(path).metadata["method"] == "merge"


# ---------------------------------------------------------------------------
# DrowseError hierarchy — the v2 guarantee that every custom error
# reparents to DrowseError while preserving its stdlib MRO.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persistence: format-version gate, provenance round trip
# ---------------------------------------------------------------------------


def test_load_profile_rejects_missing_format_version(tmp_path: Path):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})

    # The current reader never infers a version for an unstamped sidecar.
    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data.pop("format_version", None)
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="need exactly"):
        load_profile(str(path))


@pytest.mark.parametrize("version", [1, PROFILE_FORMAT_VERSION + 1])
def test_load_profile_rejects_non_current_format_version(
    tmp_path: Path, version: int,
):
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})

    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data["format_version"] = version
    sc_path.write_text(json.dumps(data))

    # A stale artifact says so by version, not by whichever field the current
    # exact-set schema happens to miss first, and names the remedy.
    with pytest.raises(ProfileError, match="need exactly"):
        load_profile(str(path))


def test_load_profile_rejects_pre_cut_sidecar_by_version(tmp_path: Path):
    """The pre-5.x fourteen-key sidecar is rejected on version, not shape.

    The field set was cut without a bump, so a file stamped ``5`` already
    failed the exact-set check; ``PROFILE_FORMAT_VERSION = 6`` makes that
    rejection say what actually happened.
    """
    profile = {0: torch.zeros(4)}
    path = tmp_path / "x.safetensors"
    save_profile(profile, str(path), {"method": "profile"})

    sc_path = path.with_suffix(".json")
    data = json.loads(sc_path.read_text())
    data.pop("provenance")
    data.update({
        "format_version": 5, "statements_sha256": "x", "components": None,
        "bake": None, "sae_release": None, "sae_revision": None,
        "sae_ids_by_layer": {}, "source_model_id": None,
        "alignment_map_hash": None, "transfer_quality_estimate": None,
        "diagnostics_by_layer": {},
    })
    sc_path.write_text(json.dumps(data))

    with pytest.raises(ProfileError, match="format_version=5"):
        load_profile(str(path))


def test_save_load_profile_roundtrip_slim_sidecar(tmp_path: Path):
    import torch
    from drowse.geometry.profile import load_profile, save_profile
    profile = {
        0: torch.randn(8),
        14: torch.randn(8),
    }
    path = tmp_path / "google__gemma-2-2b-it.safetensors"
    save_profile(profile, str(path), {
        "method": "profile",
        "statements_sha256": "a" * 64,
    })
    loaded, meta = load_profile(str(path))
    assert sorted(loaded.keys()) == [0, 14]
    assert meta["method"] == "profile"
    assert meta["statements_sha256"] == "a" * 64
    assert "drowse_version" in meta
    # Scores no longer live on disk — shares are baked into tensor magnitudes.
    assert "scores" not in meta
    # No legacy keys:
    assert "concept" not in meta
    assert "model_id" not in meta
    assert "num_pairs" not in meta
    # Round-trip preserves tensor values bit-for-bit.
    for idx in profile:
        assert torch.allclose(profile[idx], loaded[idx])


class TestProvenanceRoundTrip:
    def test_save_load_preserves_diagnostics_provenance(self, tmp_path: Path) -> None:
        profile_dict = {0: torch.ones(4), 2: torch.ones(4) * 0.5}
        # JSON object keys must be strings, so a producer stringifies its own
        # layer keys; the schema refuses a non-string key rather than silently
        # rewriting one.
        diagnostics = {
            "0": {"evr": 0.62, "inter_pair_alignment": 0.78},
            "2": {"evr": 0.41, "inter_pair_alignment": 0.55},
        }
        path = tmp_path / "test.safetensors"
        V.save_profile(profile_dict, str(path), {
            "method": "profile",
            "diagnostics": diagnostics,
        })

        # The blob lands nested under ``provenance`` on the wire; the exact
        # sidecar schema stays five keys wide.
        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert set(raw) == {
            "format_version", "drowse_version", "method",
            "tensor_sha256", "provenance",
        }
        assert raw["provenance"]["diagnostics"] == diagnostics

        # ...and flattens back to the top level on load, so the metadata the
        # reader gets is the shape the writer accepts.
        _, meta = V.load_profile(str(path))
        assert meta["method"] == "profile"
        assert meta["diagnostics"]["0"]["evr"] == pytest.approx(0.62)

    def test_int_layer_keys_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "int-keys.safetensors"
        with pytest.raises(V.ProfileError, match="object keys must be strings"):
            V.save_profile({0: torch.ones(4)}, str(path), {
                "method": "profile", "diagnostics": {0: {"evr": 0.5}},
            })
        assert not path.exists()

    def test_sidecar_without_diagnostics_loads_clean(self, tmp_path: Path) -> None:
        profile_dict = {0: torch.ones(4)}
        path = tmp_path / "plain.safetensors"
        V.save_profile(profile_dict, str(path), {"method": "profile"})

        with open(path.with_suffix(".json")) as f:
            raw = json.load(f)
        assert raw["provenance"] == {}

        _, meta = V.load_profile(str(path))
        assert "diagnostics" not in meta
