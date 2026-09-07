from __future__ import annotations

import pytest

from tests._gpu_model import _is_access_error, load_or_skip_inaccessible


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("403 Client Error: access to this gated repo is not authorized"),
        RuntimeError("401 Client Error: authentication token is required"),
        RuntimeError("Cannot access gated repository"),
    ],
)
def test_model_access_errors_are_classified(error: Exception) -> None:
    assert _is_access_error(error)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("MPS ran out of memory"),
        RuntimeError("tensor shape mismatch"),
        RuntimeError("403 Client Error: rate limit exceeded"),
    ],
)
def test_runtime_and_transport_failures_are_not_access_errors(error: Exception) -> None:
    assert not _is_access_error(error)


def test_access_error_in_cause_chain_skips() -> None:
    cause = RuntimeError("403 Client Error: gated access is not authorized")
    wrapper = OSError("model could not be loaded")
    wrapper.__cause__ = cause

    with pytest.raises(pytest.skip.Exception):
        load_or_skip_inaccessible(lambda: (_ for _ in ()).throw(wrapper), "gated/model")


def test_non_access_error_is_raised() -> None:
    error = RuntimeError("invalid tensor shape")
    with pytest.raises(RuntimeError, match="invalid tensor shape"):
        load_or_skip_inaccessible(lambda: (_ for _ in ()).throw(error), "public/model")


def test_missing_public_repository_is_not_treated_as_gated() -> None:
    error = RuntimeError("404 Client Error: Repository Not Found")
    assert not _is_access_error(error)
