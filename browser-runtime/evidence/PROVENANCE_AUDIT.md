# Browser runtime provenance audit

Audit date: 2026-09-04.

The repository has exact, independently reproducible hashes for the overlay
manifest, compiler recipe, and vendored WebLLM package:

- `forks/manifest.json`: `a2d316eb02b30733f92474d6a00925a4ca43accff93f7dd1ec5d08fcbe452f7a`
- `compiler/Dockerfile`: `afaa21db01a3049256787ab6ce1b773811de0db4e199105bb45c9aa8473e9d59`
- `vendor/polythetic-web-llm-0.2.84-polythetic.32.tgz`: `1d28468f22a3c51f9463527fcdb2fecfb2a1d1beb03e2f6ebac9460e2784e881`

The overlay manifest pins MLC-LLM base commit
`9fa644f54b04983adea4d0168f49fc6af4a893ba`, WebLLM base commit
`90f67096b68d3b77509c938f2221e4cef03b7d76`, every patch digest, and the
result-file digests. Those facts reproduce the working-tree overlay, but they
do not establish an immutable commit in either intended fork.

The source-provenance fields for local-only fork and compiler steps remain
`null`:

- MLC-LLM fork commit is not recorded.
- WebLLM fork and package source commits are not recorded.
- Compiler image reference and digest are not recorded; the recipe hash is
  pinned.

Converted model revisions are commit-pinned for every model in the lock.

A local image ID, a dirty checkout, a mutable branch head, or an empty Hugging
Face repository is not substituted for any of these immutable identities. The
release checker must continue to distinguish those identities from immutable
source and compiler provenance.
