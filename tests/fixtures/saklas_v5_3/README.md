# Original Python compatibility fixtures

Written by unmodified Saklas 5.3.0 at commit
`e32e368b` (the dev branch before the local browser work).
The small baked manifold and Profile hold direction `[1, 2, 3, 4]` at layer 0
for `stub/model`. The tree holds one authored user turn. These are synthetic
fixtures; no user state or model weights are included.

`contract.json` was captured from the same unmodified package with
`tests/_python_contract.py::python_contract("saklas")`, then normalized only
for the Saklas-to-Drowse spelling change. It records public exports, method
signatures, properties, CLI parser actions, and HTTP/WebSocket routes.

Keep the original serialized bytes and version keys. Recreating these files
with the current writer would stop testing compatibility with the original.
