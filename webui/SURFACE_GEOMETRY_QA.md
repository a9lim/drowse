# Surface geometry UI verification — 2026-09-05

## Available paths

- All tools → Inspect surface geometry (also searchable by topology, Klein, or projective).
- Manifold catalog → Inspect surface geometry.
- Native inspection → Create a manifold with coordinates → Custom → Klein bottle / Projective plane (RP²).
- Authored geometry is labeled separately from topology evidence. Position controls describe the identified seams and use angular ranges in radians.

The hosted browser release shows native-runtime guidance instead of an inspection form or unsupported quotient authoring choices. This change does not enable hosted quotient steering or automatic chart recovery.

## Executed checks

- 195 Python tests: surface API, generated response-schema parity, and manifold persistence.
- 34 quotient-domain regressions; 14 existing manifold API regressions.
- Svelte: zero errors and warnings. Python type check of the route: zero errors and warnings.
- Interface policy, shared popup material, favicon, surface input/labels/hosted guards, artifact service, hook capabilities, and runtime-boundary checks passed.
- Native and hosted production builds passed, including bundle isolation. Existing large-chunk warnings remain.

## Browser checks

Used the in-app browser and an isolated native test server with a mock model session, real geometry computation, and a temporary artifact directory. No real-model generation was tested or required here; saved user chats and model files were not used.

- Opened the tool through the command palette using search and Enter.
- Empty input produced an announced error, `aria-invalid`, and focus on the labeled textarea.
- The 96-point sphere example called the actual native detector and displayed Sphere as a topology suggestion, with focus on the result.
- A 32-point line returned Unresolved surface, not a guessed surface.
- The creation link opened the native authoring form. RP² selection required six nodes and focused Add node when incomplete.
- Filled and saved a six-node RP² artifact through the form; verified it appeared in the unfitted catalog. Fixed and regression-tested the incorrect 0D domain label exposed by this check.
- Inspected light-mode native and dark-mode hosted layouts: no horizontal overflow at the tested desktop viewport.
- Hosted guidance was reachable from the palette and contained no unsupported inspection action.
- Tab stayed inside the hosted dialog; Escape closed it and returned focus to All tools.

Not covered by this UI pass: manual screen-reader testing, small-device viewport testing, real-model fitting/generation, and hosted quotient computation.
