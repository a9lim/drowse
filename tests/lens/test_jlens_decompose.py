"""CPU tests for the dictionary-generic sparse lens decomposition."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from drowse.lens.decompose import TokensOut, sparse_decompose
from tests.support.lens import LENS_PROMPTS as _PROMPTS
from tests.support.lens import LensHarness

_D = 32
_VOCAB = 60


def _dictionary(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    jacobian = torch.randn(_D, _D, generator=gen) / _D**0.5
    unembed = torch.randn(_VOCAB, _D, generator=gen)
    return jacobian, unembed


def test_recovers_planted_sparse_combination() -> None:
    jacobian, unembed = _dictionary()
    atoms = {7: 3.0, 21: 2.0, 40: 1.2}
    target = torch.zeros(_D)
    for v, c in atoms.items():
        target = target + c * (unembed[v].float() @ jacobian)

    dec = sparse_decompose(target, TokensOut(jacobian, unembed), layer=0, k=8)

    got = dict(dec.atoms)
    assert set(atoms) <= set(got)
    for v, c in atoms.items():
        assert abs(got[v] - c) < 0.05, (v, got[v], c)
    assert dec.share > 0.99


def test_positive_tiny_solve_skips_projected_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = 8
    jacobian = torch.eye(d)
    unembed = torch.eye(d)
    target = 2.0 * unembed[2] + 1.5 * unembed[5]

    def _explode(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("positive NNLS case should not need eigvalsh/PGD")

    monkeypatch.setattr(torch.linalg, "eigvalsh", _explode)
    dec = sparse_decompose(target, TokensOut(jacobian, unembed), layer=0, k=2)

    got = dict(dec.atoms)
    assert got[2] == pytest.approx(2.0, abs=1e-5)
    assert got[5] == pytest.approx(1.5, abs=1e-5)
    assert dec.share == pytest.approx(1.0)


def test_share_low_for_offspace_direction() -> None:
    jacobian, unembed = _dictionary()
    # a direction orthogonal to every atom's positive span is unreachable
    # with nonneg coefficients from a *random* dictionary only approximately;
    # instead check that pure noise explains much less than a planted target.
    gen = torch.Generator().manual_seed(3)
    noise = torch.randn(_D, generator=gen)
    dec = sparse_decompose(noise, TokensOut(jacobian, unembed), layer=0, k=4)
    assert dec.share < 0.9


def test_zero_target_yields_empty() -> None:
    jacobian, unembed = _dictionary()
    dec = sparse_decompose(torch.zeros(_D), TokensOut(jacobian, unembed), layer=0, k=4)
    assert dec.share == 0.0 and dec.atoms == []


def test_respects_k_budget() -> None:
    jacobian, unembed = _dictionary()
    gen = torch.Generator().manual_seed(5)
    target = torch.randn(_D, generator=gen)
    dec = sparse_decompose(target, TokensOut(jacobian, unembed), layer=0, k=3)
    assert len(dec.atoms) <= 3


@pytest.fixture()
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))


def test_lens_decompose_on_registered_profile(_isolated_home: None) -> None:
    session = LensHarness()
    lens = session.lens.fit(_PROMPTS)
    # register the direction for toy-vocab token 'g', then decompose it —
    # the decomposition should attribute it (near-)fully to that atom.
    name = session.lens.register_direction("g")

    out = session.lens.decompose(name, k=4)
    del lens
    assert set(out) == {0, 1}
    for share, tokens in out.values():
        assert share > 0.99
        assert tokens and tokens[0][0] == "g"


class _SignedDictionary:
    """A neurons-shaped dictionary: signed atoms (two poles per unit)."""

    kind = "test-signed"
    nonneg = False
    device = torch.device("cpu")

    def __init__(self, atoms: torch.Tensor) -> None:
        self._atoms = atoms

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        return self._atoms @ residual

    def atom(self, index: int) -> torch.Tensor:
        return self._atoms[index]

    def norms(self) -> torch.Tensor:
        return self._atoms.norm(dim=-1)


def test_signed_dictionary_recovers_negative_coefficients() -> None:
    """The dictionary hook carries signed atoms (the future ``neurons``)."""
    gen = torch.Generator().manual_seed(11)
    atoms = torch.randn(_VOCAB, _D, generator=gen)
    target = 2.0 * atoms[4] - 1.5 * atoms[17]

    dec = sparse_decompose(target, _SignedDictionary(atoms), layer=3, k=4)

    got = dict(dec.atoms)
    assert got[4] == pytest.approx(2.0, abs=1e-3)
    assert got[17] == pytest.approx(-1.5, abs=1e-3)
    assert dec.share == pytest.approx(1.0, abs=1e-5)
    assert dec.dictionary == "test-signed" and dec.layer == 3


def test_token_dictionary_matches_the_composed_lens_matvec() -> None:
    jacobian, unembed = _dictionary()
    words = TokensOut(jacobian, unembed)
    residual = torch.randn(_D, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(words.scores(residual), unembed @ (jacobian @ residual))
    assert torch.allclose(words.atom(9), unembed[9] @ jacobian)
    assert torch.allclose(words.norms(), (unembed @ jacobian).norm(dim=-1), atol=1e-5)
