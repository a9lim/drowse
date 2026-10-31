"""``base.source_registry`` — the source family shared by lens + SAE.

The two families are documented as mirrors and used to diverge on every
load-bearing axis — locking, name validation, the source grammar, the
lifecycle around the selection.  These tests pin the shared contract and
assert both families are actually built on it.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest

from drowse.base.source_registry import SourceFamily, SourceRef


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


def _family(**overrides: Any) -> SourceFamily:
    kwargs: dict[str, Any] = {
        "label": "test",
        "dirname": "fam",
        "format_version": 1,
        "provider_kind": "external",
        "provider_prefix": "ext:",
        "local_name": re.compile(r"^[a-z]+$"),
        "provider_name_ok": lambda name: bool(name.strip()),
    }
    kwargs.update(overrides)
    return SourceFamily(**kwargs)


def _prepare(family: SourceFamily, ref: SourceRef) -> None:
    path = family.path_of("org/model", ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def test_parse_and_format_are_inverse() -> None:
    fam = _family()
    assert fam.parse("local:mine") == SourceRef("local", "mine")
    assert fam.parse("  ext:Org/Rel.v2  ") == SourceRef("external", "Org/Rel.v2")
    for source in ("local:mine", "ext:Org/Rel.v2"):
        assert fam.format(fam.parse(source)) == source
    with pytest.raises(ValueError, match="local:NAME or ext:NAME"):
        fam.parse("bare")
    with pytest.raises(ValueError, match="invalid local test name"):
        fam.parse("local:MiXeD")
    bare = _family(provider_prefix=None, provider_name_ok=lambda n: n == "neuro")
    assert bare.parse("neuro") == SourceRef("external", "neuro")
    assert bare.format(SourceRef("external", "neuro")) == "neuro"
    with pytest.raises(ValueError, match="invalid test external source name"):
        bare.parse("other")


def test_select_then_active_round_trips() -> None:
    fam = _family()
    ref = SourceRef("local", "mine")
    _prepare(fam, ref)
    path = fam.select("org/model", ref)
    assert path == fam.active_path("org/model")
    assert json.loads(path.read_text()) == {
        "format_version": 1, "model_id": "org/model",
        "kind": "local", "name": "mine",
    }
    assert fam.active("org/model") == ref


def test_select_rejects_unknown_kind_bad_name_and_absent_source() -> None:
    """A selection must never point at a source that is not on disk."""
    fam = _family()
    with pytest.raises(ValueError, match="unknown test source kind"):
        fam.select("org/model", SourceRef("bogus", "mine"))
    with pytest.raises(ValueError, match="invalid local test name"):
        fam.select("org/model", SourceRef("local", "MiXeD"))
    with pytest.raises(FileNotFoundError, match="local:absent"):
        fam.select("org/model", SourceRef("local", "absent"))
    assert not fam.active_path("org/model").exists()


@pytest.mark.parametrize("mutate", [
    {"format_version": 2},
    {"model_id": "other/model"},
    {"kind": "bogus"},
    {"name": "MiXeD"},
    {"name": ""},
    {"extra": 1},
])
def test_read_rejects_any_deviation(mutate: dict[str, Any]) -> None:
    """A stale or hand-edited selection reads as 'none', never raises."""
    fam = _family()
    path = fam.active_path("org/model")
    path.parent.mkdir(parents=True)
    payload = {
        "format_version": 1, "model_id": "org/model",
        "kind": "local", "name": "mine", **mutate,
    }
    path.write_text(json.dumps(payload))
    assert fam.active("org/model") is None


def test_read_survives_corrupt_json() -> None:
    fam = _family()
    path = fam.active_path("org/model")
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    assert fam.active("org/model") is None


def test_clear_if_active_only_matches_exactly() -> None:
    fam = _family()
    mine = SourceRef("local", "mine")
    _prepare(fam, mine)
    fam.select("org/model", mine)
    assert fam.clear_if_active("org/model", SourceRef("local", "other")) is False
    assert fam.clear_if_active("org/model", SourceRef("external", "mine")) is False
    assert fam.active("org/model") == mine
    assert fam.clear_if_active("org/model", mine) is True
    assert fam.active("org/model") is None
    # Idempotent — nothing left to clear.
    assert fam.clear_if_active("org/model", mine) is False


def test_slotted_family_keeps_one_selection_per_slot() -> None:
    fam = _family(
        slots=("left", "right"),
        slot_of=lambda _model, ref: "left" if ref.name.startswith("l") else "right",
    )
    left, right = SourceRef("local", "lone"), SourceRef("local", "rone")
    for ref in (left, right):
        _prepare(fam, ref)
        fam.select("org/model", ref)
    assert json.loads(fam.active_path("org/model").read_text()) == {
        "format_version": 1, "model_id": "org/model",
        "left": {"kind": "local", "name": "lone"},
        "right": {"kind": "local", "name": "rone"},
    }
    assert fam.active("org/model", "left") == left
    assert fam.active("org/model", "right") == right
    assert fam.clear_if_active("org/model", left)
    assert fam.active("org/model", "left") is None
    assert fam.active("org/model", "right") == right
    assert fam.clear_if_active("org/model", right)
    assert not fam.active_path("org/model").exists()


def test_both_families_use_the_shared_family() -> None:
    """The mirror claim has to be structural, not a comment."""
    from drowse.lens.sources import LENS_SOURCES
    from drowse.sae.sources import SAE_SOURCES

    for fam in (LENS_SOURCES, SAE_SOURCES):
        assert isinstance(fam, SourceFamily)
    assert LENS_SOURCES.provider_kind == "huggingface"
    assert LENS_SOURCES.slots == ("output", "input")
    assert SAE_SOURCES.provider_kind == "saelens"
    assert SAE_SOURCES.slots is None


def test_sae_selection_is_validated_and_precondition_checked() -> None:
    from drowse.sae.sources import SAE_SOURCES

    with pytest.raises(ValueError, match="unknown SAE source kind"):
        SAE_SOURCES.select("org/model", SourceRef("bogus", "x"))
    with pytest.raises(ValueError, match="invalid local SAE name"):
        SAE_SOURCES.select("org/model", SourceRef("local", "Not A Slug"))
    with pytest.raises(ValueError, match="invalid SAE saelens source name"):
        SAE_SOURCES.select("org/model", SourceRef("saelens", "  "))
    # A well-formed name for a source that isn't on disk is still refused.
    with pytest.raises(FileNotFoundError, match="local:mine"):
        SAE_SOURCES.select("org/model", SourceRef("local", "mine"))


def _weights() -> dict[str, Any]:
    import torch

    return {
        "W_enc": torch.eye(3, 5),
        "W_dec": torch.eye(5, 3),
        "b_enc": torch.zeros(5),
        "b_dec": torch.ones(3),
    }


def _save_local(name: str = "mine", *, activate: bool = True):
    from drowse.sae.store import save_local_sae

    return save_local_sae(
        "org/model", name, _weights(),
        model_fingerprint="model-fp", model_source_fingerprint="source-fp",
        layer=2, corpus_spec="test", corpus_sha256="a" * 64,
        tokens_trained=100, seq_len=16, batch_size=2,
        learning_rate=1e-3, l1_coefficient=1e-3, dead_feature_threshold=1e-6,
        activate=activate,
    )


def _active_sae_source() -> SourceRef:
    from drowse.sae.sources import SAE_SOURCES

    source = SAE_SOURCES.active("org/model")
    assert source is not None
    return source


_BINDING = {
    "layer": 3, "width": 16, "revision": "rev", "fingerprint": "fp",
    "sae_id": None, "repo_id": None, "neuronpedia_id": None,
}


def test_save_local_sae_activate_opt_out() -> None:
    """Publishing an artifact and selecting it are separable, like the lens fetch."""
    _save_local("first")
    assert _active_sae_source().name == "first"
    _save_local("second", activate=False)
    assert _active_sae_source().name == "first"


def test_save_sae_metadata_activate_opt_out() -> None:
    from drowse.sae.sources import save_sae_metadata

    save_sae_metadata("org/model", "rel/one", _BINDING)
    assert _active_sae_source().name == "rel/one"
    save_sae_metadata("org/model", "rel/two", _BINDING, activate=False)
    assert _active_sae_source().name == "rel/one"


def test_load_active_sae_owns_the_release_grammar() -> None:
    """``load_active_sae`` is the SAE mirror of ``lens.store.load_lens``."""
    from drowse.sae.sources import load_active_sae, save_sae_metadata

    assert load_active_sae("org/model") is None

    _save_local("mine")
    active = load_active_sae("org/model")
    assert active is not None
    release, metadata = active
    assert release == "local:mine"
    assert metadata is None, "a local artifact's manifest is its binding"

    save_sae_metadata("org/model", "rel/one", _BINDING)
    active = load_active_sae("org/model")
    assert active is not None
    release, metadata = active
    assert release == "rel/one"
    assert metadata is not None and metadata["layer"] == 3


def test_sae_release_and_source_grammars_share_one_parser() -> None:
    from drowse.sae.sources import SAE_SOURCES, parse_sae_release, sae_release

    assert parse_sae_release("  local:mine ") == SourceRef("local", "mine")
    assert parse_sae_release("Org/Rel") == SourceRef("saelens", "Org/Rel")
    assert sae_release(SourceRef("saelens", "Org/Rel")) == "Org/Rel"
    assert sae_release(SourceRef("local", "mine")) == "local:mine"
    assert SAE_SOURCES.format(parse_sae_release("Org/Rel")) == "saelens:Org/Rel"


def test_removing_a_source_unpublishes_the_selection() -> None:
    """The selection can never outlive what it points at."""
    from drowse.sae.sources import SAE_SOURCES, remove_sae_source, save_sae_metadata

    _save_local("mine")
    assert remove_sae_source("org/model", SourceRef("local", "mine")) is True
    assert SAE_SOURCES.active("org/model") is None

    save_sae_metadata("org/model", "rel/one", _BINDING)
    assert remove_sae_source("org/model", SourceRef("saelens", "rel/one")) is True
    assert SAE_SOURCES.active("org/model") is None


def test_removing_a_different_source_keeps_the_selection() -> None:
    from drowse.sae.store import remove_local_sae

    _save_local("keep")
    _save_local("drop", activate=False)
    assert remove_local_sae("org/model", "drop") is True
    assert _active_sae_source().name == "keep"


def test_the_engine_does_not_respell_the_local_source_prefix() -> None:
    """The engine reads the source grammars; the families own spelling them.

    No module that resolves an artifact needs a literal ``local:`` — the
    family's parse/format (and the SAE release helpers built on them) are the
    only spelling, so the convention cannot drift in one place and not the
    others.
    """
    import drowse.lens.instrument as lens_instrument_mod
    import drowse.sae.backend as sae_mod
    import drowse.sae.instrument as sae_instrument_mod
    import drowse.session.session as session_mod

    offenders = []
    for mod in (sae_mod, session_mod, sae_instrument_mod, lens_instrument_mod):
        module_path = mod.__file__
        assert module_path is not None
        for lineno, line in enumerate(
            Path(module_path).read_text(encoding="utf-8").splitlines(), start=1,
        ):
            if '"local:' in line or "'local:" in line or '"saelens:' in line:
                offenders.append(f"{Path(module_path).name}:{lineno}: {line.strip()}")
    assert offenders == []
