from __future__ import annotations

import json
from pathlib import Path

import pytest

import drowse.core.loom as loom_module
from drowse.core.events import EventBus, LoomMutated
from drowse.core.loom import CastMember, LoomTree, Recipe


FIXTURE = (
    Path(__file__).parents[1]
    / "browser-runtime"
    / "fixtures"
    / "loom-operation-transcript-v1.json"
)


def _event_dict(event: LoomMutated) -> dict[str, object]:
    return {
        "op": event.op,
        "rev": event.rev,
        "added": list(event.added),
        "removed": list(event.removed),
        "updated": list(event.updated),
        "active_node_id": event.active_node_id,
    }


def _canonical_tree(tree: LoomTree) -> dict[str, object]:
    return {
        "rev": tree.rev,
        "root_id": tree.root_id,
        "active_node_id": tree.active_node_id,
        "nodes": [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "role": node.role,
                "text": node.text,
                "starred": node.starred,
                "notes": node.notes,
                "edit_count": node.edit_count,
            }
            for node in tree.nodes.values()
        ],
        "children_of": {
            node_id: list(children)
            for node_id, children in tree.children_of.items()
        },
        "cast": {
            label: member.to_dict()
            for label, member in tree.cast.items()
        },
    }


def test_shared_loom_operation_transcript_matches_python_source_of_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(FIXTURE.read_text())
    generated_ids = iter(
        operation["createdId"]
        for operation in fixture["operations"]
        if "createdId" in operation
    )
    monkeypatch.setattr(loom_module, "_ulid", lambda: next(generated_ids))

    events = EventBus()
    mutations: list[LoomMutated] = []
    events.subscribe(
        lambda event: mutations.append(event)
        if isinstance(event, LoomMutated)
        else None
    )
    tree = LoomTree.from_dict(fixture["initial"], events=events)
    preserved_measurements = None

    for operation in fixture["operations"]:
        kind = operation["op"]
        if kind == "edit":
            tree.edit(operation["node"], operation["text"])
        elif kind == "branch":
            created = tree.branch(
                operation["node"],
                operation["text"],
                role=operation["role"],
            )
            assert created == operation["createdId"]
        elif kind == "star":
            tree.star(operation["node"], operation["value"])
        elif kind == "note":
            tree.annotate(operation["node"], operation["text"])
        elif kind == "navigate":
            tree.navigate(operation["node"])
        elif kind == "castPut":
            tree.set_cast_member(
                operation["label"],
                CastMember(
                    recipe=Recipe(
                        steering=operation["steering"],
                        thinking=operation["thinking"],
                        seed=operation["seed"],
                    ),
                    notes=operation["notes"],
                ),
            )
        elif kind == "delete":
            assert tree.delete_subtree(operation["node"]) == 1
        elif kind == "reset":
            tokens = tree.get("assistant-1").tokens
            assert tokens is not None
            preserved_measurements = tokens[0][
                "measurements"
            ]
            tree.reset()
        else:  # pragma: no cover - fixture schema is intentionally closed
            raise AssertionError(f"unknown shared loom operation {kind!r}")

    assert [_event_dict(event) for event in mutations] == fixture["expected"][
        "events"
    ]
    assert preserved_measurements == fixture["expected"][
        "preservedMeasurementsBeforeReset"
    ]
    assert _canonical_tree(tree) == fixture["expected"]["final"]
