from __future__ import annotations

import json
from pathlib import Path

import pytest

from drowse.core.steering_expr import (
    SteeringExprError,
    parse_expr,
    referenced_selectors,
)


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "browser_steering_expression.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["accepted"])
def test_browser_steering_fixture_accepts_python_expressions(
    case: dict[str, object],
) -> None:
    expression = case["expression"]
    assert isinstance(expression, str)
    parse_expr(expression)
    assert [list(item) for item in referenced_selectors(expression)] == case["selectors"]


@pytest.mark.parametrize("expression", FIXTURE["rejected"])
def test_browser_steering_fixture_rejects_python_expressions(expression: str) -> None:
    with pytest.raises(SteeringExprError):
        parse_expr(expression)
