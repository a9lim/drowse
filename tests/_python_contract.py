from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import re
from typing import Any
from unittest.mock import MagicMock


def python_contract(package: str = "drowse") -> dict[str, Any]:
    module = importlib.import_module(package)
    callables: dict[str, Any] = {}
    properties: list[str] = []

    def record(name: str, value: Any) -> None:
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return
        if inspect.isclass(value) and issubclass(value, enum.Enum):
            # EnumMeta's introspection signature differs across Python versions.
            signature = inspect.signature(value.__new__)
            signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        callables[name] = [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "required": parameter.default is inspect.Parameter.empty,
                "default": None if parameter.default is inspect.Parameter.empty else re.sub(
                    r" at 0x[0-9a-f]+", "", repr(parameter.default),
                ),
            }
            for parameter in signature.parameters.values()
        ]

    for name in module.__all__:
        value = getattr(module, name)
        if callable(value):
            record(name, value)
        if inspect.isclass(value):
            for member, descriptor in vars(value).items():
                if member.startswith("_"):
                    continue
                if isinstance(descriptor, property):
                    properties.append(f"{name}.{member}")
                elif inspect.isfunction(descriptor) or isinstance(descriptor, (classmethod, staticmethod)):
                    record(f"{name}.{member}", getattr(value, member))

    parsers = importlib.import_module(f"{package}.cli.parsers")
    cli: dict[str, Any] = {}

    def visit(parser: argparse.ArgumentParser, path: str) -> None:
        actions = []
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for verb, subparser in action.choices.items():
                    visit(subparser, f"{path} {verb}".strip())
            else:
                actions.append({
                    "dest": action.dest,
                    "options": list(action.option_strings),
                    "required": action.required,
                    "nargs": action.nargs,
                    "choices": list(action.choices) if action.choices is not None else None,
                })
        cli[path] = actions

    visit(parsers._build_root_parser(), "")
    server = importlib.import_module(f"{package}.server")
    app = server.create_app(MagicMock(), default_steering=None)
    routes = sorted(
        f"{method.upper()} {path}"
        for path, methods in app.openapi()["paths"].items()
        for method in methods
    )
    routes.extend(sorted(
        f"WS {route.path}" for route in app.routes
        if type(route).__name__ == "APIWebSocketRoute"
    ))
    return {
        "exports": sorted(module.__all__),
        "callables": dict(sorted(callables.items())),
        "properties": sorted(properties),
        "cli": dict(sorted(cli.items())),
        "routes": routes,
    }
